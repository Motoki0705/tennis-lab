"""Court keypoint detection module."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.video import OpenCVVideoFrameReader, probe_video_info, read_video_frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

NUM_COURT_KEYPOINTS = 14


@dataclass
class CourtKPConfig:
    """Configuration for CourtKP module.

    Attributes:
        checkpoint_path: Path to model checkpoint.
        mode: Detection mode ("model", "manual_ui").
        device: Inference device.
        num_keypoints: Number of court keypoints expected from the predictor/UI.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint_path: str | Path
    mode: Literal["model", "manual_ui"] = "model"
    device: str = "cuda"
    num_keypoints: int = NUM_COURT_KEYPOINTS
    save_result: bool = False
    output_path: str | Path | None = None
    load_path: str | Path | None = None


@dataclass
class CourtKPResult:
    """Result of court keypoint detection over a video sequence.

    Attributes:
        keypoints: Court keypoints (T, 14, 2), normalized [0, 1].
        visibility: Court keypoint visibility flags (T, 14).
        frame_indices: Source frame indices aligned with T.

    """

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    frame_indices: NDArray[np.int32]

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        return {
            "keypoints": self.keypoints.tolist(),
            "visibility": self.visibility.tolist(),
            "frame_indices": self.frame_indices.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CourtKPResult:
        """Create result from dict."""
        keypoints = np.array(data["keypoints"], dtype=np.float32)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
        if "visibility" in data:
            visibility = np.array(data["visibility"], dtype=np.float32)
        else:
            visibility = np.ones(keypoints.shape[:2], dtype=np.float32)
        if visibility.ndim == 1:
            visibility = visibility[None, ...]
        if "frame_indices" in data:
            frame_indices = np.array(data["frame_indices"], dtype=np.int32)
        elif "frame_index" in data:
            frame_indices = np.array([data["frame_index"]], dtype=np.int32)
        else:
            frame_indices = np.arange(keypoints.shape[0], dtype=np.int32)
        return cls(
            keypoints=keypoints,
            visibility=visibility,
            frame_indices=frame_indices,
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        LOGGER.info(f"Saved CourtKP result to {path}")

    def validate(
        self,
        *,
        num_keypoints: int | None = NUM_COURT_KEYPOINTS,
    ) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if num_keypoints is not None and num_keypoints <= 0:
            errors.append(f"num_keypoints must be positive, got {num_keypoints}")
        expected_shape = None
        if num_keypoints is not None and num_keypoints > 0:
            expected_shape = (num_keypoints, 2)
        if self.keypoints.ndim != 3 or (
            expected_shape is not None and self.keypoints.shape[1:] != expected_shape
        ):
            expected_text = (
                f"(T, {num_keypoints}, 2)"
                if num_keypoints is not None
                else "(T, K, 2)"
            )
            errors.append(
                f"keypoints shape must be {expected_text}, got {self.keypoints.shape}"
            )
        if self.visibility.shape != self.keypoints.shape[:2]:
            errors.append(
                "visibility shape must match keypoints[:2], "
                f"got {self.visibility.shape} for {self.keypoints.shape}"
            )
        if self.frame_indices.shape != (self.keypoints.shape[0],):
            errors.append(
                "frame_indices shape must match (T,), "
                f"got {self.frame_indices.shape} for T={self.keypoints.shape[0]}"
            )
        if not np.isfinite(self.keypoints).all():
            errors.append("keypoints contain non-finite values")
        if not np.isfinite(self.visibility).all():
            errors.append("visibility contains non-finite values")
        tol = 1e-6
        if np.any(self.keypoints < -tol) or np.any(self.keypoints > 1.0 + tol):
            errors.append("keypoints must be normalized to [0, 1]")
        if np.any(self.visibility < -tol) or np.any(self.visibility > 1.0 + tol):
            errors.append("visibility must be normalized to [0, 1]")
        return len(errors) == 0, errors

    @classmethod
    def load(cls, path: str | Path) -> CourtKPResult:
        """Load result from JSON file."""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class CourtKPModule(BasePipelineModule):
    """Court keypoint detection module.

    Detects 14 court keypoints for each frame in a video.

    Attributes:
        config: Configuration for the module.
        checkpoint_path: Path to model checkpoint.
        device: Inference device.
        mode: "model" for predictor or "manual_ui" for interactive input.

    """

    def __init__(
        self,
        config: CourtKPConfig,
    ) -> None:
        """Initialize the module.

        Args:
            config: CourtKP configuration.

        """
        self.config = config
        self.checkpoint_path = Path(self.config.checkpoint_path)
        self.mode = self.config.mode
        self.device = self.config.device
        self.num_keypoints = int(self.config.num_keypoints)
        if self.num_keypoints <= 0:
            raise ValueError(f"num_keypoints must be positive, got {self.num_keypoints}")
        self._predictor = None
        self._manual_keypoints: NDArray[np.float32] | None = None
        self._manual_needs_normalization = False

    def load(self) -> None:
        """Load the court keypoint predictor."""
        if self.mode == "manual_ui":
            return

        if self._predictor is not None:
            return

        LOGGER.info(f"Loading Court KP model from {self.checkpoint_path}")
        from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor

        self._predictor = CourtKeypointPredictor.load_from_checkpoint(
            self.checkpoint_path, device=self.device
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        if self.mode == "manual_ui":
            return self._manual_keypoints is not None
        return self._predictor is not None

    def _collect_manual_keypoints_ui(self, frame: NDArray[np.uint8]) -> None:
        """Collect manual keypoints via an interactive UI.

        Args:
            frame: RGB frame array (H, W, 3).
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required for manual_ui mode.") from exc

        keypoints = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        placed = np.zeros(self.num_keypoints, dtype=bool)
        current_idx = 0

        def draw_overlay(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
            overlay = image.copy()
            for idx in range(self.num_keypoints):
                if not placed[idx]:
                    continue
                x, y = keypoints[idx]
                color = (0, 255, 0)
                cv2.circle(overlay, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    str(idx),
                    (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                overlay,
                f"Keypoint {current_idx}/{self.num_keypoints - 1}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                "LMB: place | N/P: next/prev | S: save | Q: quit",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            return overlay

        def on_mouse(event: int, mx: int, my: int, _flags: int, _param: object) -> None:
            nonlocal current_idx
            if event == cv2.EVENT_LBUTTONDOWN:
                keypoints[current_idx] = [mx, my]
                placed[current_idx] = True
                current_idx = (current_idx + 1) % self.num_keypoints

        window_name = "Court Keypoints (manual_ui)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        try:
            while True:
                overlay = draw_overlay(frame)
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(30) & 0xFF

                if key in {ord("n"), ord(" "), 82}:
                    current_idx = (current_idx + 1) % self.num_keypoints
                elif key in {ord("p"), 84}:
                    current_idx = (current_idx - 1) % self.num_keypoints
                elif key == ord("c"):
                    keypoints[current_idx] = [0.0, 0.0]
                    placed[current_idx] = False
                elif key == ord("s") or key in {ord("q"), 27}:
                    break
        finally:
            cv2.destroyWindow(window_name)

        self._manual_keypoints = keypoints.astype(np.float32)
        self._manual_needs_normalization = True

    def process(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        annotation_frame_index: int = 0,
    ) -> CourtKPResult:
        """Detect court keypoints over a video sequence.

        Args:
            video_path: Path to input video.
            max_frames: Maximum frames to return from the beginning of the video.
            annotation_frame_index: Frame to annotate in manual UI mode.

        Returns:
            CourtKPResult with normalized keypoints shaped (T, K, 2).

        """
        if self.config.load_path is not None:
            load_path = Path(self.config.load_path)
            if load_path.exists():
                LOGGER.info(
                    f"Loading CourtKP result from {load_path} (skipping detection)"
                )
                result = CourtKPResult.load(load_path)
                is_valid, errors = result.validate(num_keypoints=self.num_keypoints)
                if not is_valid:
                    raise ValueError(f"Invalid CourtKP result: {errors}")
                return result
            LOGGER.warning(
                f"load_path specified but not found: {load_path}, running detection"
            )

        if self.mode == "manual_ui":
            result = self._process_manual_video(
                video_path,
                max_frames=max_frames,
                annotation_frame_index=annotation_frame_index,
            )
        else:
            result = self._process_model_video(video_path, max_frames=max_frames)

        is_valid, errors = result.validate(num_keypoints=self.num_keypoints)
        if not is_valid:
            raise ValueError(f"Invalid CourtKP result: {errors}")

        if self.config.save_result and self.config.output_path is not None:
            result.save(self.config.output_path)

        return result

    def _process_manual_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None,
        annotation_frame_index: int,
    ) -> CourtKPResult:
        """Annotate one frame and repeat it across the selected sequence."""
        if annotation_frame_index < 0:
            raise ValueError(
                "annotation_frame_index must be non-negative, "
                f"got {annotation_frame_index}"
            )

        video_info = probe_video_info(video_path)
        if annotation_frame_index >= video_info.frame_count:
            raise ValueError(
                f"annotation_frame_index={annotation_frame_index} is outside video "
                f"with {video_info.frame_count} frames"
            )

        num_frames = video_info.frame_count
        if max_frames is not None:
            num_frames = min(num_frames, int(max_frames))
        if num_frames <= 0:
            raise ValueError(f"No frames selected for CourtKP result: {video_path}")

        packet = read_video_frame(video_path, annotation_frame_index)
        frame_rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)

        if not self.is_loaded:
            self.load()

        if self._manual_keypoints is None:
            self._collect_manual_keypoints_ui(frame_rgb)

        keypoints = np.array(self._manual_keypoints, copy=True)

        if self._manual_needs_normalization:
            image_width, image_height = packet.original_size
            keypoints[..., 0] /= max(image_width - 1, 1)
            keypoints[..., 1] /= max(image_height - 1, 1)

        keypoints = np.repeat(
            keypoints[None, ...].astype(np.float32),
            repeats=num_frames,
            axis=0,
        )
        return CourtKPResult(
            keypoints=keypoints,
            visibility=np.ones(keypoints.shape[:2], dtype=np.float32),
            frame_indices=np.arange(num_frames, dtype=np.int32),
        )

    def _process_model_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None,
    ) -> CourtKPResult:
        """Run model inference for every selected video frame."""
        if not self.is_loaded:
            self.load()

        keypoints: list[NDArray[np.float32]] = []
        frame_indices: list[int] = []
        for packet in OpenCVVideoFrameReader(video_path, max_frames=max_frames):
            frame_rgb = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB)
            frame_keypoints = self._predict_frame(
                frame_rgb,
                image_width=packet.original_size[0],
                image_height=packet.original_size[1],
            )
            keypoints.append(frame_keypoints)
            frame_indices.append(packet.index)

        if not keypoints:
            raise RuntimeError(f"No frames were read from video: {video_path}")

        stacked = np.stack(keypoints, axis=0).astype(np.float32)
        return CourtKPResult(
            keypoints=stacked,
            visibility=np.ones(stacked.shape[:2], dtype=np.float32),
            frame_indices=np.array(frame_indices, dtype=np.int32),
        )

    def _predict_frame(
        self,
        frame_rgb: NDArray[np.uint8],
        *,
        image_width: int,
        image_height: int,
    ) -> NDArray[np.float32]:
        """Run the loaded model on one RGB frame and return normalized keypoints."""
        if self._predictor is None:
            raise RuntimeError("Court KP predictor is not loaded.")
        pred = self._predictor.predict(frame_rgb)

        raw_keypoints = pred["keypoints"]
        if hasattr(raw_keypoints, "detach"):
            raw_keypoints = raw_keypoints.detach().cpu().numpy()
        elif hasattr(raw_keypoints, "numpy"):
            raw_keypoints = raw_keypoints.numpy()
        keypoints = np.asarray(raw_keypoints, dtype=np.float32)

        keypoints[..., 0] /= max(image_width - 1, 1)
        keypoints[..., 1] /= max(image_height - 1, 1)
        return keypoints.astype(np.float32)


if __name__ == "__main__":
    # Quick smoke test for module instantiation
    print("CourtKPModule: court keypoint detection module")
    print("Use CourtKPModule(CourtKPConfig(...)) to create")

    # Test config creation
    config = CourtKPConfig(
        checkpoint_path="test.ckpt",
        mode="manual_ui",
        device="cpu",
        save_result=True,
        output_path="test_output.json",
    )
    print(f"Config: {config}")
    assert config.device == "cpu"
    assert config.mode == "manual_ui"
    print("Smoke test passed.")
