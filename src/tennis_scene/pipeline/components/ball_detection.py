"""Ball detection module for the tennis scene pipeline."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.inference.trajectory_gate import (
    TrajectoryGateConfig,
    apply_trajectory_gate,
)
from src.tennis_scene.pipeline.components.base import BasePipelineModule
from src.utils.configuration import PathResolver
from src.utils.io import load_json, save_json
from src.utils.video import (
    BgrToTensorTransform,
    FramePacket,
    OpenCVVideoFrameReader,
    PrefetchIterator,
    iter_temporal_batches,
    iter_temporal_windows,
    probe_video_info,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BallDetectionConfig:
    """Configuration for the scene ball-detection module.

    Attributes:
        checkpoint: Path to a ``src.tasks.ball_detection`` Lightning checkpoint.
        batch_size: Batch size for inference.
        device: Inference device.
        image_size: Model input size as ``(height, width)``.
        normalize_imagenet: Whether to apply ImageNet normalization.
        score_threshold: Minimum peak confidence for visible detections.
        subpixel_refine: Whether peak coordinates are refined to sub-cell
            precision instead of raw heatmap-lattice argmax.
        prefetch_batches: Number of preprocessed inference batches to queue.
        window_stride: Temporal window stride. Defaults to model sequence length.
        tail_policy: Final-window policy for partial tails.
        overlap_aggregation: How duplicate frame predictions are resolved.
        pin_memory: Whether to pin preprocessed batch tensors before inference.
        trajectory_gate: Optional local trajectory-consistency postprocess gate.
        save_result: Whether to save result to file.
        output_path: Path to save result JSON file.
        load_path: Path to load pre-computed result from (skips inference).

    """

    checkpoint: Path
    source: Literal["execute", "load"]
    batch_size: int
    device: str
    image_size: tuple[int, int]
    normalize_imagenet: bool
    score_threshold: float
    subpixel_refine: bool
    allow_device_fallback: bool
    checkpoint_strict: bool
    checkpoint_weights_only: bool
    prefetch_batches: int
    window_stride: int | None
    tail_policy: str
    overlap_aggregation: str
    pin_memory: bool
    trajectory_gate: TrajectoryGateConfig
    save_result: bool
    output_path: Path
    load_path: Path | None
    resolver: PathResolver

    def __post_init__(self) -> None:
        if (self.source == "load") != (self.load_path is not None):
            raise ValueError(
                "BallDetection source='load' requires load_path; execute forbids it"
            )


@dataclass
class BallDetectionResult:
    """Result of scene-level ball detection.

    Attributes:
        ball_uv: Ball 2D position (N, T, 2), normalized [0, 1].
        ball_uv_px: Ball 2D position (N, T, 2), in pixels.
        visibility: Ball visibility mask (N, T).
        score: Detection confidence score (N, T).

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
        save_json(self.to_dict(), path)
        LOGGER.info(f"Saved ball detection result to {path}")

    def validate(self) -> tuple[bool, list[str]]:
        """Validate result content.

        Returns:
            Tuple of (is_valid, errors).
        """
        errors: list[str] = []
        if self.ball_uv.ndim != 3 or self.ball_uv.shape[2] != 2:
            errors.append(f"ball_uv shape must be (N, T, 2), got {self.ball_uv.shape}")
        if self.ball_uv_px.ndim != 3 or self.ball_uv_px.shape[2] != 2:
            errors.append(
                f"ball_uv_px shape must be (N, T, 2), got {self.ball_uv_px.shape}"
            )
        if self.visibility.ndim != 2:
            errors.append(
                f"visibility shape must be (N, T), got {self.visibility.shape}"
            )
        if self.score.ndim != 2:
            errors.append(f"score shape must be (N, T), got {self.score.shape}")

        nt_uv = self.ball_uv.shape[:2]
        if self.ball_uv_px.shape[:2] != nt_uv:
            errors.append("ball_uv_px shape does not match ball_uv on (N, T)")
        if self.visibility.shape != nt_uv:
            errors.append("visibility shape does not match ball_uv on (N, T)")
        if self.score.shape != nt_uv:
            errors.append("score shape does not match ball_uv on (N, T)")

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
        return cls.from_dict(load_json(path))


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
        self._pipeline: BallDetectionPredictor | None = None

    def load(self) -> None:
        """Load the ball detection predictor."""
        if self._pipeline is not None:
            return

        LOGGER.info(f"Loading ball detection model from {self.config.checkpoint}")
        self._pipeline = BallDetectionPredictor.load_from_checkpoint(
            self.config.checkpoint,
            resolver=self.config.resolver,
            device=self.config.device,
            allow_device_fallback=self.config.allow_device_fallback,
            subpixel_refine=self.config.subpixel_refine,
            strict=self.config.checkpoint_strict,
            weights_only=self.config.checkpoint_weights_only,
        )

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None

    def process(
        self,
        video_paths: Sequence[Path],
        max_frames: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> BallDetectionResult:
        """Run ball detection on synchronized videos.

        Args:
            video_paths: Paths to synchronized input videos.
            max_frames: Maximum frames to process.
            image_width: Image width for normalization.
            image_height: Image height for normalization.

        Returns:
            BallDetectionResult with ball positions shaped (N, T, ...).

        """
        video_paths = list(video_paths)
        if not video_paths:
            raise ValueError("video_paths must contain at least one video")

        # Check if we should load from pre-computed result
        if self.config.source == "load":
            assert self.config.load_path is not None
            load_path = self.config.load_path
            if load_path.is_file():
                LOGGER.info(
                    f"Loading ball detection result from {load_path} "
                    "(skipping inference)"
                )
                result = BallDetectionResult.load(load_path)
                is_valid, errors = result.validate()
                if not is_valid:
                    raise ValueError(f"Invalid ball detection result: {errors}")
                if result.ball_uv.shape[0] != len(video_paths):
                    raise ValueError(
                        "Loaded ball detection result camera count must match "
                        f"video_paths, got {result.ball_uv.shape[0]} and "
                        f"{len(video_paths)}"
                    )
                result = self._apply_trajectory_gate(result)
                is_valid, errors = result.validate()
                if not is_valid:
                    raise ValueError(f"Invalid gated ball detection result: {errors}")
                return result
            raise FileNotFoundError(f"Ball detection artifact not found: {load_path}")

        if not self.is_loaded:
            self.load()

        LOGGER.info("Running ball detection...")
        per_camera_uv: list[NDArray[np.float32]] = []
        per_camera_uv_px: list[NDArray[np.float32]] = []
        per_camera_visibility: list[NDArray[np.bool_]] = []
        per_camera_score: list[NDArray[np.float32]] = []
        expected_frames: int | None = None

        for camera_index, video_path in enumerate(video_paths):
            video_info = probe_video_info(video_path)
            original_width = (
                image_width if image_width is not None else video_info.width
            )
            original_height = (
                image_height if image_height is not None else video_info.height
            )
            ball_uv, score = self._predict_video(video_path, max_frames=max_frames)

            if expected_frames is None:
                expected_frames = ball_uv.shape[0]
            elif ball_uv.shape[0] != expected_frames:
                raise ValueError(
                    f"video_paths[{camera_index}] produced T={ball_uv.shape[0]}, "
                    f"expected T={expected_frames}"
                )

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

            per_camera_uv.append(ball_uv.astype(np.float32))
            per_camera_uv_px.append(ball_uv_px.astype(np.float32))
            per_camera_visibility.append(valid_mask.astype(np.bool_))
            per_camera_score.append(score.astype(np.float32))

        ball_detection_result = BallDetectionResult(
            ball_uv=np.stack(per_camera_uv, axis=0).astype(np.float32),
            ball_uv_px=np.stack(per_camera_uv_px, axis=0).astype(np.float32),
            visibility=np.stack(per_camera_visibility, axis=0).astype(np.bool_),
            score=np.stack(per_camera_score, axis=0).astype(np.float32),
        )
        ball_detection_result = self._apply_trajectory_gate(ball_detection_result)
        is_valid, errors = ball_detection_result.validate()
        if not is_valid:
            raise ValueError(f"Invalid ball detection result: {errors}")

        if self.config.save_result:
            ball_detection_result.save(self.config.output_path)

        return ball_detection_result

    def _apply_trajectory_gate(
        self,
        result: BallDetectionResult,
    ) -> BallDetectionResult:
        """Apply the optional trajectory gate and zero rejected detections."""
        gate_config = self.config.trajectory_gate
        if not gate_config.enabled:
            return result

        ball_uv = result.ball_uv.copy()
        ball_uv_px = result.ball_uv_px.copy()
        visibility = result.visibility.copy()
        score = result.score.copy()
        rejected_by_camera: list[list[int]] = []

        for camera_index in range(result.ball_uv.shape[0]):
            gated_visibility, diagnostics = apply_trajectory_gate(
                positions_px=result.ball_uv_px[camera_index],
                visibility=result.visibility[camera_index],
                score=result.score[camera_index],
                max_residual_px=gate_config.max_residual_px,
                k_support=gate_config.k_support,
                max_support_gap=gate_config.max_support_gap,
                max_passes=gate_config.max_passes,
            )
            rejected_mask = result.visibility[camera_index] & ~gated_visibility
            ball_uv[camera_index, rejected_mask] = 0.0
            ball_uv_px[camera_index, rejected_mask] = 0.0
            score[camera_index, rejected_mask] = 0.0
            visibility[camera_index] = gated_visibility
            rejected_by_camera.append(diagnostics.rejected_indices)

        total_rejected = sum(len(indices) for indices in rejected_by_camera)
        LOGGER.info(
            "Ball trajectory gate rejected %d frame(s) across %d camera(s)",
            total_rejected,
            len(rejected_by_camera),
        )
        if total_rejected:
            LOGGER.info(
                "Ball trajectory gate rejected frames by camera: %s",
                {
                    f"cam{camera_index}": indices
                    for camera_index, indices in enumerate(rejected_by_camera)
                    if indices
                },
            )

        return BallDetectionResult(
            ball_uv=ball_uv.astype(np.float32),
            ball_uv_px=ball_uv_px.astype(np.float32),
            visibility=visibility.astype(np.bool_),
            score=score.astype(np.float32),
        )

    def _predict_video(
        self,
        video_path: Path,
        *,
        max_frames: int | None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Stream video windows through the predictor."""
        if self._pipeline is None:
            raise RuntimeError("Ball detection predictor is not loaded.")

        model_config = self._pipeline.model_config
        sequence_length = cast("int", model_config["num_frames"])
        if sequence_length <= 0:
            raise ValueError(
                f"model.num_frames must be positive, got {sequence_length}"
            )
        stride = (
            sequence_length
            if self.config.window_stride is None
            else int(self.config.window_stride)
        )
        if stride <= 0:
            raise ValueError(f"window_stride must be positive, got {stride}")

        transform = BgrToTensorTransform(
            image_size=self.config.image_size,
            normalize_imagenet=self.config.normalize_imagenet,
        )
        frame_stream = (
            FramePacket(
                index=packet.index,
                frame=transform(packet.frame),
                original_size=packet.original_size,
            )
            for packet in OpenCVVideoFrameReader(video_path, max_frames=max_frames)
        )
        windows = iter_temporal_windows(
            frame_stream,
            sequence_length=sequence_length,
            stride=stride,
            tail_policy=self.config.tail_policy,
        )
        batches = iter_temporal_batches(
            windows,
            batch_size=int(self.config.batch_size),
            pin_memory=bool(self.config.pin_memory),
        )
        prefetched_batches = PrefetchIterator(
            batches,
            max_prefetch=int(self.config.prefetch_batches),
        )

        coords_by_frame: dict[int, np.ndarray] = {}
        score_by_frame: dict[int, float] = {}
        max_frame_index = -1

        for batch in prefetched_batches:
            prediction = self._pipeline.predict(batch.tensor)
            coords = prediction["coords"].numpy().astype(np.float32)
            scores = prediction["visibility"].numpy().astype(np.float32)
            for window_index, window in enumerate(batch.windows):
                for time_index, frame_index in enumerate(window.frame_indices):
                    max_frame_index = max(max_frame_index, int(frame_index))
                    self._accumulate_frame_prediction(
                        coords_by_frame=coords_by_frame,
                        score_by_frame=score_by_frame,
                        frame_index=int(frame_index),
                        coord=coords[window_index, time_index],
                        score=float(scores[window_index, time_index]),
                    )

        if max_frame_index < 0:
            raise RuntimeError(f"No frames were read from video: {video_path}")

        total_frames = max_frame_index + 1
        ball_uv: NDArray[np.float32] = np.zeros((total_frames, 2), dtype=np.float32)
        score: NDArray[np.float32] = np.zeros((total_frames,), dtype=np.float32)
        for frame_index in range(total_frames):
            if frame_index in coords_by_frame:
                ball_uv[frame_index] = coords_by_frame[frame_index]
                score[frame_index] = score_by_frame[frame_index]

        ball_uv = np.clip(ball_uv, 0.0, 1.0).astype(np.float32)
        score = np.clip(score, 0.0, 1.0).astype(np.float32)
        return ball_uv, score

    def _accumulate_frame_prediction(
        self,
        *,
        coords_by_frame: dict[int, np.ndarray],
        score_by_frame: dict[int, float],
        frame_index: int,
        coord: NDArray[np.float32],
        score: float,
    ) -> None:
        """Resolve duplicate frame predictions from overlapping tail windows."""
        if self.config.overlap_aggregation == "last_window_wins":
            coords_by_frame[frame_index] = coord
            score_by_frame[frame_index] = score
            return
        if self.config.overlap_aggregation == "max_score":
            old_score = score_by_frame.get(frame_index)
            if old_score is None or score >= old_score:
                coords_by_frame[frame_index] = coord
                score_by_frame[frame_index] = score
            return
        raise ValueError(
            "overlap_aggregation must be one of ['last_window_wins', 'max_score'], "
            f"got '{self.config.overlap_aggregation}'."
        )
