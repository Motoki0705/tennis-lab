"""WASB ball detection predictor for tennis analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import cv2
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn

from src.base.api.predictor import BasePredictor

if TYPE_CHECKING:
    from numpy.typing import NDArray

# WASB source directory
_WASB_SRC_DIR = Path(__file__).parents[3] / "third_party" / "WASB-SBDT" / "src"


def _ensure_wasb_path() -> None:
    """Ensure WASB source directory is in sys.path."""
    wasb_path = str(_WASB_SRC_DIR)
    if wasb_path not in sys.path:
        sys.path.insert(0, wasb_path)


class WASBPredictor(BasePredictor):
    """WASB ball detection inference predictor.

    Detects tennis ball 2D positions from image batches.

    Attributes:
        model: The WASB detection model.
        device: The inference device (CUDA only).

    Example:
        >>> predictor = WASBPredictor.load_from_checkpoint(
        ...     "third_party/WASB-SBDT/pretrained/wasb_tennis_best.pth.tar"
        ... )
        >>> results = predictor.predict(frames)
        >>> print(results["ball_uv"].shape)  # (N, 2)

    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cfg: DictConfig,
    ) -> None:
        """Initialize the predictor.

        Use load_from_checkpoint to create instances in most cases.

        Args:
            model: Initialized WASB model.
            device: Inference device.
            cfg: WASB configuration (OmegaConf DictConfig).

        """
        self.model = model
        self.device = device
        self._cfg = cfg

        # Initialize components from WASB
        _ensure_wasb_path()
        from dataloaders import build_img_transforms
        from detectors.postprocessor import TracknetV2Postprocessor
        from trackers import build_tracker

        _, self._transform = build_img_transforms(cfg)
        self._postprocessor = TracknetV2Postprocessor(cfg)
        self._tracker = build_tracker(cfg)

        # Model parameters
        self._frames_in = cfg["model"]["frames_in"]
        self._input_wh = (cfg["model"]["inp_width"], cfg["model"]["inp_height"])

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cuda",
        **kwargs: Any,
    ) -> Self:
        """Create a WASBPredictor from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file (.pth.tar).
            device: Inference device. Only "cuda" is supported.
            **kwargs: Additional settings:
                - gpus: List of GPU IDs (default: [0])
                - score_threshold: Detection threshold (default: 0.5)
                - max_disp: Maximum tracker displacement (default: 300)

        Returns:
            Initialized WASBPredictor instance.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            ValueError: If device is not "cuda".

        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("WASBPredictor requires CUDA device")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        # Build configuration
        cfg = cls._build_config(
            checkpoint_path=checkpoint_path,
            gpus=kwargs.get("gpus", [0]),
            score_threshold=kwargs.get("score_threshold", 0.5),
            max_disp=kwargs.get("max_disp", 300),
        )

        # Build and load model
        _ensure_wasb_path()
        from models import build_model

        model = build_model(cfg)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=cfg["runner"]["gpus"])
        model.eval()

        return cls(model=model, device=device, cfg=cfg)

    @classmethod
    def _build_config(
        cls,
        checkpoint_path: Path,
        gpus: list[int],
        score_threshold: float,
        max_disp: int,
    ) -> DictConfig:
        """Build configuration for WASB."""
        config_dir = _WASB_SRC_DIR / "configs"

        # Load base configs
        model_cfg = OmegaConf.load(config_dir / "model" / "wasb.yaml")
        detector_cfg = OmegaConf.load(config_dir / "detector" / "tracknetv2.yaml")
        tracker_cfg = OmegaConf.load(config_dir / "tracker" / "online.yaml")
        transform_cfg = OmegaConf.load(config_dir / "transform" / "default.yaml")
        dataloader_cfg = OmegaConf.load(config_dir / "dataloader" / "default.yaml")

        # Override settings
        detector_cfg["model_path"] = str(checkpoint_path)
        detector_cfg["postprocessor"]["score_threshold"] = score_threshold
        tracker_cfg["max_disp"] = max_disp

        # Build combined config (keep as DictConfig for attribute access)
        cfg = OmegaConf.create(
            {
                "model": model_cfg,
                "detector": detector_cfg,
                "tracker": tracker_cfg,
                "transform": transform_cfg,
                "dataloader": dataloader_cfg,
                "runner": {
                    "device": "cuda",
                    "gpus": gpus,
                },
            }
        )

        return cfg

    @torch.no_grad()
    def predict(
        self,
        frames: NDArray[np.uint8],
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Detect ball positions from image batch.

        Args:
            frames: RGB image array. Shape (T, H, W, 3).
                    T must be at least frames_in (=3).

        Returns:
            Inference results dictionary:
                - ball_uv: Normalized coordinates (N, 2), range [0, 1]
                - ball_xy_px: Pixel coordinates (N, 2)
                - visibility: Visibility flags (N,)
                - score: Detection scores (N,)
                - frame_indices: Frame indices (N,)

        Raises:
            ValueError: If number of frames is less than frames_in.

        """
        num_frames = len(frames)
        if num_frames < self._frames_in:
            raise ValueError(
                f"Expected at least {self._frames_in} frames, got {num_frames}"
            )

        height, width = frames[0].shape[:2]

        # Process frames through detector
        det_results = self._run_detection(frames)

        # Process through tracker
        self._tracker.refresh()
        results = self._process_tracking(det_results, num_frames)

        # Convert to output format
        frame_indices = np.array(results["frame_indices"], dtype=np.int64)
        ball_xy_px = np.array(results["xy"], dtype=np.float32)
        visibility = np.array(results["visibility"], dtype=bool)
        score = np.array(results["score"], dtype=np.float32)

        # Normalize coordinates
        ball_uv = ball_xy_px.copy()
        ball_uv[:, 0] /= width
        ball_uv[:, 1] /= height

        return {
            "ball_uv": ball_uv,
            "ball_xy_px": ball_xy_px,
            "visibility": visibility,
            "score": score,
            "frame_indices": frame_indices,
        }

    def _run_detection(
        self,
        frames: NDArray[np.uint8],
    ) -> dict[int, list[dict[str, Any]]]:
        """Run detection on frames."""
        _ensure_wasb_path()

        num_frames = len(frames)
        det_results: dict[int, list[dict[str, Any]]] = {}

        # Process in sliding windows
        for start_idx in range(num_frames - self._frames_in + 1):
            window_frames = frames[start_idx : start_idx + self._frames_in]

            # Prepare input tensors
            imgs_t, trans_inv = self._preprocess_frames(window_frames)

            # Run model
            imgs_t = imgs_t.to(self.device)
            preds = self.model(imgs_t)

            # Postprocess
            trans_dict = {0: torch.from_numpy(trans_inv).unsqueeze(0)}
            pp_results = self._postprocessor.run(preds, trans_dict)

            # Extract results for the last frame in window
            frame_idx = start_idx + self._frames_in - 1
            batch_idx = 0
            elem_idx = self._frames_in - 1  # Last element

            if batch_idx in pp_results and elem_idx in pp_results[batch_idx]:
                frame_preds = []
                for scale_data in pp_results[batch_idx][elem_idx].values():
                    for xy, sc in zip(
                        scale_data["xys"], scale_data["scores"], strict=True
                    ):
                        frame_preds.append({"xy": xy, "score": sc})
                det_results[frame_idx] = frame_preds

        return det_results

    def _preprocess_frames(
        self,
        frames: NDArray[np.uint8],
    ) -> tuple[torch.Tensor, NDArray[np.floating[Any]]]:
        """Preprocess frames to model input format."""
        _ensure_wasb_path()
        from utils.image import get_affine_transform

        input_w, input_h = self._input_wh
        first_frame = frames[0]
        h, w = first_frame.shape[:2]

        # Compute affine transform
        center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
        scale = max(h, w) * 1.0
        trans = get_affine_transform(center, scale, 0, [input_w, input_h])
        trans_inv = get_affine_transform(center, scale, 0, [input_w, input_h], inv=1)

        # Process each frame
        processed = []
        for frame in frames:
            # Warp to input size
            warped = cv2.warpAffine(
                frame, trans, (input_w, input_h), flags=cv2.INTER_LINEAR
            )
            img_pil = Image.fromarray(warped)
            img_t = self._transform(img_pil)
            processed.append(img_t)

        # Stack and add batch dimension
        imgs_t = torch.cat(processed, dim=0).unsqueeze(0)

        return imgs_t, trans_inv

    def _process_tracking(
        self,
        det_results: dict[int, list[dict[str, Any]]],
        num_frames: int,
    ) -> dict[str, list[Any]]:
        """Process detection results through tracker."""
        results: dict[str, list[Any]] = {
            "frame_indices": [],
            "xy": [],
            "visibility": [],
            "score": [],
        }

        # Process frames in order
        for frame_idx in range(num_frames):
            preds = det_results.get(frame_idx, [])
            track_result = self._tracker.update(preds)

            results["frame_indices"].append(frame_idx)
            results["xy"].append([track_result["x"], track_result["y"]])
            results["visibility"].append(track_result["visi"])
            results["score"].append(track_result["score"])

        return results

    # =========================================================================
    # Streaming inference methods
    # =========================================================================

    def reset_tracker(self) -> None:
        """Reset tracker state for new video processing."""
        self._tracker.refresh()
        self._streaming_buffer: list[NDArray[np.uint8]] = []
        self._streaming_frame_offset = 0

    @torch.no_grad()
    def predict_batch(
        self,
        frames: NDArray[np.uint8],
        frame_indices: list[int],
    ) -> dict[str, NDArray[np.floating[Any]]]:
        """Process a batch of frames in streaming mode.

        Unlike predict(), this method maintains tracker state across calls,
        enabling streaming inference over long videos.

        Args:
            frames: RGB image array. Shape (B, H, W, 3).
            frame_indices: Original frame indices in the video.

        Returns:
            Inference results dictionary:
                - ball_uv: Normalized coordinates (B, 2), range [0, 1]
                - ball_xy_px: Pixel coordinates (B, 2)
                - visibility: Visibility flags (B,)
                - score: Detection scores (B,)
                - frame_indices: Frame indices (B,)

        Note:
            Call reset_tracker() before processing a new video.

        """
        if len(frames) == 0:
            return {
                "ball_uv": np.array([], dtype=np.float32).reshape(0, 2),
                "ball_xy_px": np.array([], dtype=np.float32).reshape(0, 2),
                "visibility": np.array([], dtype=bool),
                "score": np.array([], dtype=np.float32),
                "frame_indices": np.array([], dtype=np.int64),
            }

        height, width = frames[0].shape[:2]

        # Add frames to buffer for sliding window
        self._streaming_buffer.extend(list(frames))

        # Process detection on available windows
        det_results: dict[int, list[dict[str, Any]]] = {}
        buffer_len = len(self._streaming_buffer)

        # Process windows that have enough frames
        while buffer_len >= self._frames_in:
            window_frames = np.stack(self._streaming_buffer[: self._frames_in], axis=0)

            # Run detection
            imgs_t, trans_inv = self._preprocess_frames(window_frames)
            imgs_t = imgs_t.to(self.device)
            preds = self.model(imgs_t)

            # Postprocess
            trans_dict = {0: torch.from_numpy(trans_inv).unsqueeze(0)}
            pp_results = self._postprocessor.run(preds, trans_dict)

            # Extract results for the last frame in window
            frame_idx = self._streaming_frame_offset + self._frames_in - 1
            batch_idx = 0
            elem_idx = self._frames_in - 1

            if batch_idx in pp_results and elem_idx in pp_results[batch_idx]:
                frame_preds = []
                for scale_data in pp_results[batch_idx][elem_idx].values():
                    for xy, sc in zip(
                        scale_data["xys"], scale_data["scores"], strict=True
                    ):
                        frame_preds.append({"xy": xy, "score": sc})
                det_results[frame_idx] = frame_preds

            # Slide window by 1
            self._streaming_buffer.pop(0)
            self._streaming_frame_offset += 1
            buffer_len -= 1

        # Process tracking for this batch
        results: dict[str, list[Any]] = {
            "frame_indices": [],
            "xy": [],
            "visibility": [],
            "score": [],
        }

        for idx in frame_indices:
            preds = det_results.get(idx, [])
            track_result = self._tracker.update(preds)

            results["frame_indices"].append(idx)
            results["xy"].append([track_result["x"], track_result["y"]])
            results["visibility"].append(track_result["visi"])
            results["score"].append(track_result["score"])

        # Convert to output format
        frame_indices_arr = np.array(results["frame_indices"], dtype=np.int64)
        ball_xy_px = np.array(results["xy"], dtype=np.float32)
        visibility = np.array(results["visibility"], dtype=bool)
        score = np.array(results["score"], dtype=np.float32)

        # Normalize coordinates
        ball_uv = ball_xy_px.copy()
        if len(ball_uv) > 0:
            ball_uv[:, 0] /= width
            ball_uv[:, 1] /= height

        return {
            "ball_uv": ball_uv,
            "ball_xy_px": ball_xy_px,
            "visibility": visibility,
            "score": score,
            "frame_indices": frame_indices_arr,
        }
