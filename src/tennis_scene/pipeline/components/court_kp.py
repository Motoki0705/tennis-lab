"""Court keypoint detection module."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import cv2
import numpy as np

from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.configuration import PathResolver
from src.utils.io import load_json, save_json
from src.utils.video import OpenCVVideoFrameReader, probe_video_info, read_video_frame

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor

LOGGER = logging.getLogger(__name__)

NUM_COURT_KEYPOINTS = 14


@dataclass(frozen=True, slots=True)
class CourtKPPostprocessConfig:
    """Configuration for model-mode court keypoint post-processing."""

    enabled: bool
    min_score: float
    ransac_reproj_threshold: float
    temporal_median_window: int


@dataclass(frozen=True, slots=True)
class CourtKPConfig:
    """Configuration for CourtKP module.

    Attributes:
        checkpoint: Path to model checkpoint.
        mode: Detection mode ("model", "manual_ui").
        device: Inference device.
        num_keypoints: Number of court keypoints expected from the predictor/UI.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint: Path
    source: Literal["execute", "load"]
    mode: Literal["model", "manual_ui"]
    device: str
    subpixel_refine: bool
    num_keypoints: int
    save_result: bool
    output_path: Path
    load_path: Path | None
    postprocess: CourtKPPostprocessConfig
    resolver: PathResolver

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "CourtKP source='load' requires load_path; execute forbids it"
            )


@dataclass
class CourtKPResult:
    """Result of court keypoint detection over a video sequence."""

    keypoints: NDArray[np.float32]
    visibility: NDArray[np.float32]
    frame_indices: NDArray[np.int32]
    diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        """Convert result to JSON-serializable dict."""
        data: dict[str, Any] = {
            "keypoints": self.keypoints.tolist(),
            "visibility": self.visibility.tolist(),
            "frame_indices": self.frame_indices.tolist(),
        }
        if self.diagnostics is not None:
            data["diagnostics"] = self.diagnostics
        return data

    @classmethod
    def from_dict(cls, data: dict) -> CourtKPResult:
        """Create result from dict."""
        missing = {"keypoints", "visibility", "frame_indices"} - set(data)
        if missing:
            raise ValueError(
                f"CourtKP result is missing required fields: {sorted(missing)}"
            )
        keypoints = np.array(data["keypoints"], dtype=np.float32)
        visibility = np.array(data["visibility"], dtype=np.float32)
        frame_indices = np.array(data["frame_indices"], dtype=np.int32)
        diagnostics = data.get("diagnostics")
        if diagnostics is not None and not isinstance(diagnostics, dict):
            raise TypeError("CourtKP result diagnostics must be an object when present")
        return cls(
            keypoints=keypoints,
            visibility=visibility,
            frame_indices=frame_indices,
            diagnostics=diagnostics,
        )

    def save(self, path: str | Path) -> None:
        """Save result to JSON file."""
        save_json(self.to_dict(), path)
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
        if self.keypoints.ndim != 4 or (
            expected_shape is not None and self.keypoints.shape[2:] != expected_shape
        ):
            expected_text = (
                f"(N, T, {num_keypoints}, 2)"
                if num_keypoints is not None
                else "(N, T, K, 2)"
            )
            errors.append(
                f"keypoints shape must be {expected_text}, got {self.keypoints.shape}"
            )
        if self.visibility.shape != self.keypoints.shape[:3]:
            errors.append(
                "visibility shape must match keypoints[:3], "
                f"got {self.visibility.shape} for {self.keypoints.shape}"
            )
        if self.keypoints.ndim >= 2 and self.frame_indices.shape != (
            self.keypoints.shape[1],
        ):
            errors.append(
                "frame_indices shape must match (T,), "
                f"got {self.frame_indices.shape} for T={self.keypoints.shape[1]}"
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
        return cls.from_dict(load_json(path))


class CourtKPModule(BasePipelineModule):
    """Court keypoint detection module.

    Detects court keypoints for each synchronized camera video.
    """

    def __init__(self, config: CourtKPConfig) -> None:
        """Initialize the module."""
        self.config = config
        self.checkpoint = self.config.checkpoint
        self.mode = self.config.mode
        self.device = self.config.device
        self.num_keypoints = int(self.config.num_keypoints)
        self.postprocess = self.config.postprocess
        if self.num_keypoints <= 0:
            raise ValueError(
                f"num_keypoints must be positive, got {self.num_keypoints}"
            )
        self._predictor: CourtKeypointPredictor | None = None
        self._manual_keypoints: NDArray[np.float32] | None = None
        self._manual_needs_normalization = False

    def load(self) -> None:
        """Load the court keypoint predictor."""
        if self.mode == "manual_ui":
            return

        if self._predictor is not None:
            return

        LOGGER.info(f"Loading Court KP model from {self.checkpoint}")
        from src.tasks.court_detection.inference.predictor import CourtKeypointPredictor

        self._predictor = CourtKeypointPredictor.load_from_checkpoint(
            self.checkpoint,
            resolver=self.config.resolver,
            device=self.device,
            subpixel_refine=self.config.subpixel_refine,
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
        keypoints: NDArray[np.float32] = np.zeros(
            (self.num_keypoints, 2),
            dtype=np.float32,
        )
        placed: NDArray[np.bool_] = np.zeros(self.num_keypoints, dtype=bool)
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
        video_paths: Sequence[Path],
        max_frames: int | None = None,
        annotation_frame_index: int = 0,
    ) -> CourtKPResult:
        """Detect court keypoints over synchronized video sequences."""
        video_paths = list(video_paths)
        if not video_paths:
            raise ValueError("video_paths must contain at least one video")

        if self.config.source == "load":
            load_path = self.config.load_path
            if load_path is None:
                raise RuntimeError("Validated load source is missing load_path")
            if not load_path.is_file():
                raise FileNotFoundError(f"CourtKP artifact not found: {load_path}")
            LOGGER.info(f"Loading CourtKP result from {load_path}")
            result = CourtKPResult.load(load_path)
            is_valid, errors = result.validate(num_keypoints=self.num_keypoints)
            if not is_valid:
                raise ValueError(f"Invalid CourtKP result: {errors}")
            if result.keypoints.shape[0] != len(video_paths):
                raise ValueError(
                    "Loaded CourtKP result camera count must match video_paths, "
                    f"got {result.keypoints.shape[0]} and {len(video_paths)}"
                )
            return result

        if self.mode == "manual_ui":
            result = self._process_manual_video(
                video_paths,
                max_frames=max_frames,
                annotation_frame_index=annotation_frame_index,
            )
        else:
            result = self._process_model_video(video_paths, max_frames=max_frames)

        is_valid, errors = result.validate(num_keypoints=self.num_keypoints)
        if not is_valid:
            raise ValueError(f"Invalid CourtKP result: {errors}")

        if self.config.save_result:
            result.save(self.config.output_path)

        return result

    def _process_manual_video(
        self,
        video_paths: Sequence[Path],
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

        first_info = probe_video_info(video_paths[0])
        if annotation_frame_index >= first_info.frame_count:
            raise ValueError(
                f"annotation_frame_index={annotation_frame_index} is outside video "
                f"with {first_info.frame_count} frames"
            )

        num_frames = first_info.frame_count
        if max_frames is not None:
            num_frames = min(num_frames, int(max_frames))
        if num_frames <= 0:
            raise ValueError("No frames selected for CourtKP result")

        if not self.is_loaded:
            self.load()

        per_camera_keypoints: list[NDArray[np.float32]] = []
        for camera_index, video_path in enumerate(video_paths):
            video_info = probe_video_info(video_path)
            if video_info.frame_count < num_frames:
                raise ValueError(
                    f"video_paths[{camera_index}] has {video_info.frame_count} frames, "
                    f"expected at least {num_frames}"
                )
            if annotation_frame_index >= video_info.frame_count:
                raise ValueError(
                    f"annotation_frame_index={annotation_frame_index} is outside "
                    f"video_paths[{camera_index}] with {video_info.frame_count} frames"
                )

            packet = read_video_frame(video_path, annotation_frame_index)
            frame_rgb = cast(
                "NDArray[np.uint8]",
                cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB),
            )
            self._manual_keypoints = None
            self._collect_manual_keypoints_ui(frame_rgb)

            manual_keypoints = cast(
                "NDArray[np.float32] | None", self._manual_keypoints
            )
            if manual_keypoints is None:
                raise RuntimeError("Manual court keypoint UI did not produce keypoints")
            keypoints = np.array(manual_keypoints, copy=True)

            if self._manual_needs_normalization:
                image_width, image_height = packet.original_size
                keypoints[..., 0] /= max(image_width - 1, 1)
                keypoints[..., 1] /= max(image_height - 1, 1)
            per_camera_keypoints.append(keypoints.astype(np.float32))

        keypoints = np.stack(
            [
                np.repeat(kp[None, ...], repeats=num_frames, axis=0)
                for kp in per_camera_keypoints
            ],
            axis=0,
        ).astype(np.float32)
        return CourtKPResult(
            keypoints=keypoints,
            visibility=np.ones(keypoints.shape[:3], dtype=np.float32),
            frame_indices=np.arange(num_frames, dtype=np.int32),
        )

    def _process_model_video(
        self,
        video_paths: Sequence[Path],
        *,
        max_frames: int | None,
    ) -> CourtKPResult:
        """Run model inference for every selected frame in each camera video."""
        if not self.is_loaded:
            self.load()

        per_camera_keypoints: list[NDArray[np.float32]] = []
        per_camera_visibility: list[NDArray[np.float32]] = []
        postprocess_diagnostics: list[dict[str, Any]] = []
        expected_frame_indices: NDArray[np.int32] | None = None
        for camera_index, video_path in enumerate(video_paths):
            keypoints_px: list[NDArray[np.float32]] = []
            scores: list[NDArray[np.float32]] = []
            validities: list[NDArray[np.bool_]] = []
            frame_indices: list[int] = []
            image_width: int | None = None
            image_height: int | None = None
            for packet in OpenCVVideoFrameReader(video_path, max_frames=max_frames):
                frame_rgb = cast(
                    "NDArray[np.uint8]",
                    cv2.cvtColor(packet.frame, cv2.COLOR_BGR2RGB),
                )
                frame_keypoints_px, frame_scores, frame_valid = (
                    self._predict_frame_pixels(frame_rgb)
                )
                keypoints_px.append(frame_keypoints_px)
                scores.append(frame_scores)
                validities.append(frame_valid)
                frame_indices.append(packet.index)
                width, height = packet.original_size
                if image_width is None:
                    image_width = width
                    image_height = height
                elif image_width != width or image_height != height:
                    raise ValueError(
                        f"video_paths[{camera_index}] changed resolution within the "
                        f"stream: got {(width, height)}, expected "
                        f"{(image_width, image_height)}"
                    )

            if not keypoints_px:
                raise RuntimeError(f"No frames were read from video: {video_path}")
            if image_width is None or image_height is None:
                raise RuntimeError(f"Could not determine video size for {video_path}")

            camera_frame_indices = np.array(frame_indices, dtype=np.int32)
            if expected_frame_indices is None:
                expected_frame_indices = camera_frame_indices
            elif not np.array_equal(expected_frame_indices, camera_frame_indices):
                raise ValueError(
                    f"video_paths[{camera_index}] frame indices do not match "
                    "the first camera"
                )

            camera_keypoints_px = np.stack(keypoints_px, axis=0).astype(np.float32)
            camera_scores = np.stack(scores, axis=0).astype(np.float32)
            camera_validity = np.stack(validities, axis=0)
            camera_visibility = camera_validity.astype(np.float32)
            if self.postprocess.enabled:
                from src.tasks.court_detection.geometry import (
                    refine_court_keypoints_with_homography,
                )

                postprocess_result = refine_court_keypoints_with_homography(
                    camera_keypoints_px,
                    np.where(camera_validity, camera_scores, 0.0),
                    min_score=float(self.postprocess.min_score),
                    ransac_reproj_threshold=float(
                        self.postprocess.ransac_reproj_threshold
                    ),
                    temporal_median_window=int(self.postprocess.temporal_median_window),
                )
                camera_keypoints_px = postprocess_result.keypoints
                camera_visibility = (
                    postprocess_result.visibility * camera_validity
                ).astype(np.float32)
                camera_diagnostics = dict(postprocess_result.diagnostics)
                camera_diagnostics["camera_index"] = int(camera_index)
                camera_diagnostics["video_path"] = str(video_path)
                postprocess_diagnostics.append(camera_diagnostics)

            per_camera_keypoints.append(
                _normalize_keypoints(
                    camera_keypoints_px,
                    image_width=image_width,
                    image_height=image_height,
                )
            )
            per_camera_visibility.append(camera_visibility)

        if expected_frame_indices is None:
            raise RuntimeError("No frames were read from video_paths")

        stacked = np.stack(per_camera_keypoints, axis=0).astype(np.float32)
        visibility = np.stack(per_camera_visibility, axis=0).astype(np.float32)
        diagnostics = None
        if self.postprocess.enabled:
            diagnostics = {
                "postprocess": {
                    "enabled": True,
                    "min_score": float(self.postprocess.min_score),
                    "ransac_reproj_threshold": float(
                        self.postprocess.ransac_reproj_threshold
                    ),
                    "temporal_median_window": int(
                        self.postprocess.temporal_median_window
                    ),
                    "cameras": postprocess_diagnostics,
                }
            }
        return CourtKPResult(
            keypoints=stacked,
            visibility=visibility,
            frame_indices=expected_frame_indices,
            diagnostics=diagnostics,
        )

    def _predict_frame_pixels(
        self,
        frame_rgb: NDArray[np.uint8],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.bool_],
    ]:
        """Return one explicit ordered peak, score, and validity per KP channel."""
        if self._predictor is None:
            raise RuntimeError("Court KP predictor is not loaded.")
        prediction = self._predictor.predict(frame_rgb)
        keypoints = prediction.keypoints.numpy().astype(np.float32)
        scores = prediction.scores.numpy().astype(np.float32)
        valid = prediction.valid.numpy().astype(np.bool_)

        if keypoints.shape != (self.num_keypoints, 1, 2):
            raise ValueError(
                f"Predicted court keypoints must have shape "
                f"({self.num_keypoints}, 1, 2), got {keypoints.shape}."
            )
        if scores.shape != (self.num_keypoints, 1):
            raise ValueError(
                f"Predicted court scores must have shape ({self.num_keypoints}, 1), "
                f"got {scores.shape}."
            )
        if valid.shape != (self.num_keypoints, 1):
            raise ValueError(
                f"Predicted court validity must have shape ({self.num_keypoints}, 1), "
                f"got {valid.shape}."
            )
        return (
            keypoints[:, 0].astype(np.float32),
            scores[:, 0].astype(np.float32),
            valid[:, 0],
        )


def _normalize_keypoints(
    keypoints_px: NDArray[np.float32],
    *,
    image_width: int,
    image_height: int,
) -> NDArray[np.float32]:
    keypoints = np.array(keypoints_px, copy=True, dtype=np.float32)
    keypoints[..., 0] /= max(image_width - 1, 1)
    keypoints[..., 1] /= max(image_height - 1, 1)
    return keypoints.astype(np.float32)
