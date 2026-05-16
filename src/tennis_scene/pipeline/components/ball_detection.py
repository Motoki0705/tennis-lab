"""Ball detection module for the tennis scene pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from src.tasks.ball_detection.data.argumentation import normalize_tensor_images_imagenet
from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tennis_scene.pipeline.components.base import BasePipelineModule

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass
class BallDetectionConfig:
    """Configuration for the scene ball-detection module.

    Attributes:
        checkpoint: Path to a ``src.tasks.ball_detection`` Lightning checkpoint.
        batch_size: Batch size for inference.
        device: Inference device.
        image_size: Model input size as ``(height, width)``.
        normalize_imagenet: Whether to apply ImageNet normalization.
        score_threshold: Minimum peak confidence for visible detections.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint: str | Path
    batch_size: int = 64
    device: str = "cuda"
    image_size: tuple[int, int] = (288, 512)
    normalize_imagenet: bool = True
    score_threshold: float = 0.5
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class BallDetectionResult:
    """Result of scene-level ball detection.

    Attributes:
        ball_uv: Ball 2D position (T, 2), normalized [0, 1].
        ball_uv_px: Ball 2D position (T, 2), in pixels.
        visibility: Ball visibility mask (T,).
        score: Detection confidence score (T,).

    """

    ball_uv: NDArray[np.float32]
    ball_uv_px: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    score: NDArray[np.float32]

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "ball_uv": self.ball_uv.tolist(),
            "ball_uv_px": self.ball_uv_px.tolist(),
            "visibility": self.visibility.tolist(),
            "score": self.score.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> BallDetectionResult:
        """Create result from dict."""
        return cls(
            ball_uv=np.array(data["ball_uv"], dtype=np.float32),
            ball_uv_px=np.array(data["ball_uv_px"], dtype=np.float32),
            visibility=np.array(data["visibility"], dtype=np.bool_),
            score=np.array(data["score"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved ball detection result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.ball_uv.ndim != 2 or self.ball_uv.shape[1] != 2:
            errors.append(f"ball_uv shape must be (T, 2), got {self.ball_uv.shape}")
        if self.ball_uv_px.ndim != 2 or self.ball_uv_px.shape[1] != 2:
            errors.append(
                f"ball_uv_px shape must be (T, 2), got {self.ball_uv_px.shape}"
            )
        if self.visibility.ndim != 1:
            errors.append(f"visibility shape must be (T,), got {self.visibility.shape}")
        if self.score.ndim != 1:
            errors.append(f"score shape must be (T,), got {self.score.shape}")

        t_uv = self.ball_uv.shape[0]
        if self.ball_uv_px.shape[0] != t_uv:
            errors.append("ball_uv_px length does not match ball_uv length")
        if self.visibility.shape[0] != t_uv:
            errors.append("visibility length does not match ball_uv length")
        if self.score.shape[0] != t_uv:
            errors.append("score length does not match ball_uv length")

        if not np.isfinite(self.ball_uv).all():
            errors.append("ball_uv contains non-finite values")
        if not np.isfinite(self.ball_uv_px).all():
            errors.append("ball_uv_px contains non-finite values")
        if not np.isfinite(self.score).all():
            errors.append("score contains non-finite values")

        if not np.isin(self.visibility, [0, 1, False, True]).all():
            errors.append("visibility must contain only 0 or 1")

        tol = 1e-6
        if np.any(self.ball_uv < -tol) or np.any(self.ball_uv > 1.0 + tol):
            errors.append("ball_uv must be normalized to [0, 1]")
        if np.any(self.ball_uv_px < -tol):
            errors.append("ball_uv_px must be non-negative")
        if np.any(self.score < -tol):
            errors.append("score must be non-negative")

        if self.visibility.size:
            invalid = ~self.visibility.astype(bool)
            if invalid.any():
                if np.any(np.abs(self.ball_uv[invalid]) > tol):
                    errors.append("ball_uv must be zero for invalid frames")
                if np.any(np.abs(self.ball_uv_px[invalid]) > tol):
                    errors.append("ball_uv_px must be zero for invalid frames")
                if np.any(np.abs(self.score[invalid]) > tol):
                    errors.append("score must be zero for invalid frames")

        return len(errors) == 0, errors

    @classmethod
    def load(cls, path: str | Path) -> BallDetectionResult:
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class BallDetectionModule(BasePipelineModule):
    """Scene pipeline module for ball detection.

    Detects ball positions in video frames using the task-local ball detection
    predictor.

    """

    def __init__(self, config: BallDetectionConfig) -> None:
        """Initialize the module.

        Args:
            config: Ball detection configuration.

        """
        self.config = config
        self._pipeline = None

    def load(self) -> None:
        """Load the ball detection predictor."""
        if self._pipeline is not None:
            return

        LOGGER.info(f"Loading ball detection model from {self.config.checkpoint}")
        self._pipeline = BallDetectionPredictor.load_from_checkpoint(
            self.config.checkpoint, device=self.config.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None

    def process(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> BallDetectionResult:
        """Run ball detection on video.

        Args:
            video_path: Path to input video.
            max_frames: Maximum frames to process.
            image_width: Image width for normalization.
            image_height: Image height for normalization.

        Returns:
            BallDetectionResult with ball positions.

        """
        # Check if we should load from pre-computed result
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(
                    f"Loading ball detection result from {load_path} "
                    "(skipping inference)"
                )
                return BallDetectionResult.load(load_path)
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running inference"
            )

        if not self.is_loaded:
            self.load()

        LOGGER.info("Running ball detection...")
        frames, original_size = self._read_video_frames(video_path, max_frames=max_frames)
        original_width = image_width if image_width is not None else original_size[0]
        original_height = image_height if image_height is not None else original_size[1]
        ball_uv, score = self._predict_frames(frames)

        ball_uv_px = ball_uv.copy()
        ball_uv_px[..., 0] *= max(original_width - 1, 1)
        ball_uv_px[..., 1] *= max(original_height - 1, 1)

        finite_uv = np.isfinite(ball_uv).all(axis=-1)
        finite_px = np.isfinite(ball_uv_px).all(axis=-1)
        finite_score = np.isfinite(score)
        valid_mask = (
            finite_uv
            & finite_px
            & finite_score
            & (score >= float(self.config.score_threshold))
        )

        # Replace invalid values with zeros to keep JSON strictly numeric
        ball_uv[~valid_mask] = 0.0
        ball_uv_px[~valid_mask] = 0.0
        score[~valid_mask] = 0.0

        ball_detection_result = BallDetectionResult(
            ball_uv=ball_uv,
            ball_uv_px=ball_uv_px,
            visibility=valid_mask.astype(np.bool_),
            score=score,
        )

        if self.config.save_result and self.config.output_path is not None:
            ball_detection_result.save(self.config.output_path)

        return ball_detection_result

    def _read_video_frames(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Read and resize video frames into ``(T, 3, H, W)`` float tensors."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        image_h, image_w = self.config.image_size
        frames: list[np.ndarray] = []
        original_size: tuple[int, int] | None = None
        try:
            while max_frames is None or len(frames) < max_frames:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if original_size is None:
                    original_size = (frame_bgr.shape[1], frame_bgr.shape[0])
                frame_bgr = cv2.resize(frame_bgr, (image_w, image_h))
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb.astype(np.float32) / 255.0)
        finally:
            cap.release()

        if not frames:
            raise RuntimeError(f"No frames were read from video: {video_path}")

        frame_tensor = (
            torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()
        )
        if self.config.normalize_imagenet:
            frame_tensor = normalize_tensor_images_imagenet(frame_tensor)
        if original_size is None:
            original_size = (image_w, image_h)
        return frame_tensor, original_size

    def _predict_frames(
        self,
        frames: torch.Tensor,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Run fixed-window prediction and return normalized coords and scores."""
        if self._pipeline is None:
            raise RuntimeError("Ball detection predictor is not loaded.")

        total_frames = int(frames.shape[0])
        model_config = getattr(self._pipeline, "model_config", {}) or {}
        window = int(model_config.get("num_frames", 8))
        if window <= 0:
            raise ValueError(f"model.num_frames must be positive, got {window}")

        sequences: list[torch.Tensor] = []
        valid_lengths: list[int] = []
        for start in range(0, total_frames, window):
            sequence = frames[start : start + window]
            valid_lengths.append(int(sequence.shape[0]))
            if sequence.shape[0] < window:
                pad = sequence[-1:].repeat(window - sequence.shape[0], 1, 1, 1)
                sequence = torch.cat([sequence, pad], dim=0)
            sequences.append(sequence)

        coords_chunks: list[np.ndarray] = []
        score_chunks: list[np.ndarray] = []
        for batch_start in range(0, len(sequences), int(self.config.batch_size)):
            batch_sequences = sequences[
                batch_start : batch_start + int(self.config.batch_size)
            ]
            batch = torch.stack(batch_sequences, dim=0)
            prediction = self._pipeline.predict(batch)
            coords = prediction["coords"].numpy().astype(np.float32)
            scores = prediction["visibility"].numpy().astype(np.float32)
            for index, valid_length in enumerate(
                valid_lengths[batch_start : batch_start + len(batch_sequences)]
            ):
                coords_chunks.append(coords[index, :valid_length])
                score_chunks.append(scores[index, :valid_length])

        ball_uv = np.concatenate(coords_chunks, axis=0)
        score = np.concatenate(score_chunks, axis=0)
        ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32)
        score = np.clip(score, 0.0, 1.0).astype(np.float32)
        return ball_uv, score


if __name__ == "__main__":
    # Quick smoke test for module instantiation
    print("BallDetectionModule: scene ball detection module")
    print("Use BallDetectionModule(BallDetectionConfig(...))")

    # Test config creation
    config = BallDetectionConfig(
        checkpoint="test.ckpt",
        device="cpu",
        save_result=True,
        output_path="test_output.json",
    )
    print(f"Config: {config}")
    assert config.device == "cpu"
    print("Smoke test passed.")
