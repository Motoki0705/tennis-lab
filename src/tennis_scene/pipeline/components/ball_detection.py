"""Per-camera ball detection for the tennis scene pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.tasks.ball_detection.inference import BallDetectionPredictor
from src.tasks.ball_detection.inference.trajectory_gate import (
    TrajectoryGateConfig,
    apply_trajectory_gate,
)
from src.tennis_scene.pipeline.components.base import (
    BasePipelineModule,
    release_inference_memory,
)
from src.tennis_scene.pipeline.contracts import ComponentIO, SourceVideo
from src.utils.configuration import PathResolver
from src.utils.geometry.keypoints import denormalize_grid_keypoints
from src.utils.video import (
    BgrToTensorTransform,
    FramePacket,
    OpenCVVideoFrameReader,
    PrefetchIterator,
    iter_temporal_batches,
    iter_temporal_windows,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BallDetectionInput:
    video: SourceVideo


@dataclass(frozen=True)
class BallDetectionOutput:
    camera_id: str
    frame_indices: NDArray[np.int64]
    uv_px: NDArray[np.float32]
    confidence: NDArray[np.float32]
    observed: NDArray[np.bool_]
    point_kind: NDArray[np.uint8]  # 0 absent, 1 observed, 2 interpolated, 3 occlusion estimate
    score_semantics: str

    def __post_init__(self) -> None:
        count = len(self.frame_indices)
        if not self.camera_id or self.frame_indices.dtype != np.int64 or not np.array_equal(self.frame_indices, np.arange(count)):
            raise ValueError("Ball artifact must preserve the complete source timeline")
        if self.uv_px.shape != (count, 2) or any(x.shape != (count,) for x in (self.confidence, self.observed, self.point_kind)):
            raise ValueError("Invalid ball artifact shapes")
        if self.uv_px.dtype != np.float32 or self.confidence.dtype != np.float32 or self.observed.dtype != np.bool_ or self.point_kind.dtype != np.uint8:
            raise TypeError("Invalid ball artifact dtypes")
        if not np.isfinite(self.uv_px).all() or not np.isfinite(self.confidence).all() or (self.confidence < 0).any() or (self.confidence > 1).any():
            raise ValueError("Invalid ball coordinates or acceptance weights")
        if not np.array_equal(self.observed, self.point_kind == 1):
            raise ValueError("Only directly observed ball points may be observations")
        if (self.point_kind > 3).any():
            raise ValueError("Unknown ball point provenance")


@dataclass(frozen=True, slots=True)
class BallDetectionConfig:
    """Configuration for the scene ball-detection module.

    Attributes:
        checkpoint: Path to a ``src.tasks.ball_detection`` Lightning checkpoint.
        batch_size: Batch size for inference.
        device: Inference device.
        image_size: Model input size as ``(height, width)``.
        normalize_imagenet: Expected checkpoint normalization flag; checked on load.
        score_threshold: Minimum peak confidence for visible detections.
        subpixel_refine: Whether peak coordinates are refined to sub-cell
            precision instead of raw heatmap-lattice argmax.
        prefetch_batches: Number of preprocessed inference batches to queue.
        window_stride: Temporal window stride. Defaults to model sequence length.
        tail_policy: Final-window policy for partial tails.
        overlap_aggregation: How duplicate frame predictions are resolved.
        pin_memory: Whether to pin preprocessed batch tensors before inference.
        trajectory_gate: Local trajectory-consistency postprocess gate.

    """

    checkpoint: Path
    batch_size: int
    device: str
    image_size: tuple[int, int]
    normalize_imagenet: bool
    score_threshold: float
    subpixel_refine: bool
    checkpoint_strict: bool
    checkpoint_weights_only: bool
    prefetch_batches: int
    window_stride: int | None
    tail_policy: str
    overlap_aggregation: str
    pin_memory: bool
    trajectory_gate: TrajectoryGateConfig
    resolver: PathResolver


class BallDetectionModule(BasePipelineModule):
    """Detect one unidentified ball observation stream for one camera video."""

    io = ComponentIO("ball_detection", BallDetectionInput, BallDetectionOutput, {}, "ball_detections")

    def __init__(self, config: BallDetectionConfig, *, enabled: bool = True) -> None:
        self.config = config
        self.enabled = enabled
        self._pipeline: BallDetectionPredictor | None = None

    def load(self) -> None:
        """Load the ball detection predictor."""
        if self._pipeline is not None:
            return

        LOGGER.info(f"Loading ball detection model from {self.config.checkpoint}")
        predictor = BallDetectionPredictor.load_from_checkpoint(
            self.config.checkpoint,
            resolver=self.config.resolver,
            device=self.config.device,
            subpixel_refine=self.config.subpixel_refine,
            strict=self.config.checkpoint_strict,
            weights_only=self.config.checkpoint_weights_only,
        )
        if self.config.normalize_imagenet != predictor.image_normalization.enabled:
            raise ValueError(
                "ball_detection.normalize_imagenet does not match the saved "
                f"checkpoint preprocessing ({predictor.image_normalization.enabled})."
            )
        self._pipeline = predictor

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipeline is not None

    def unload(self) -> None:
        self._pipeline = None
        release_inference_memory(self.config.device)

    def process(self, inputs: BallDetectionInput) -> BallDetectionOutput:
        video = inputs.video
        frames: NDArray[np.int64] = np.arange(video.num_frames, dtype=np.int64)
        if not self.enabled:
            return BallDetectionOutput(video.camera_id, frames, np.zeros((video.num_frames, 2), np.float32),
                np.zeros(video.num_frames, np.float32), np.zeros(video.num_frames, bool), np.zeros(video.num_frames, np.uint8), "disabled")
        try:
            self.load()
            ball_uv, score = self._predict_video(video.path, max_frames=video.num_frames)
        finally:
            self.unload()
        if len(ball_uv) != video.num_frames:
            raise ValueError(f"Ball detector covered {len(ball_uv)} of {video.num_frames} frames of {video.camera_id}; "
                             f"tail_policy={self.config.tail_policy!r} cannot produce the complete source timeline")
        uv_px, score, observed = self._accept_detections(denormalize_grid_keypoints(ball_uv, video.width, video.height), score)
        return BallDetectionOutput(video.camera_id, frames, uv_px, score, observed, observed.astype(np.uint8), "model_score")

    def _accept_detections(
        self, uv_px: NDArray[np.float32], score: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]:
        """Threshold, apply the trajectory gate and zero every rejected frame."""
        observed = np.isfinite(uv_px).all(axis=-1) & np.isfinite(score) & (score >= float(self.config.score_threshold))
        gate = self.config.trajectory_gate
        if gate.enabled:
            gated, diagnostics = apply_trajectory_gate(
                positions_px=np.where(observed[:, None], uv_px, 0.0).astype(np.float32),
                visibility=observed,
                score=np.where(observed, score, 0.0).astype(np.float32),
                max_residual_px=gate.max_residual_px,
                k_support=gate.k_support,
                max_support_gap=gate.max_support_gap,
                max_passes=gate.max_passes,
            )
            LOGGER.info("Ball trajectory gate rejected %d frame(s): %s", len(diagnostics.rejected_indices), diagnostics.rejected_indices)
            observed = gated
        uv_px = np.where(observed[:, None], uv_px, 0.0).astype(np.float32)
        score = np.where(observed, score, 0.0).astype(np.float32)
        return uv_px, score, observed.astype(np.bool_)

    def _predict_video(
        self,
        video_path: Path,
        *,
        max_frames: int | None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Stream video windows through the predictor."""
        if self._pipeline is None:
            raise RuntimeError("Ball detection predictor is not loaded.")

        sequence_length = self._pipeline.configured_frames
        stride = (
            sequence_length
            if self.config.window_stride is None
            else int(self.config.window_stride)
        )
        if stride <= 0:
            raise ValueError(f"window_stride must be positive, got {stride}")

        transform = BgrToTensorTransform(
            image_size=self.config.image_size,
            normalize_imagenet=False,
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
            coords = prediction.coords.numpy().astype(np.float32)
            scores = prediction.confidence.numpy().astype(np.float32)
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
