"""Motion-consistent, LOESS-smoothed detector points with explicit provenance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.contracts import ComponentIO, InputPort
from src.utils.geometry.temporal_smoothing import (
    DEFAULT_TEMPORAL_SMOOTHING,
    TemporalSmoothingConfig,
    TemporalSmoothingResult,
    smooth_2d_trajectory,
)

BALL_POINTS_PORT = InputPort("ball_points", 4)


@dataclass(frozen=True)
class BallPointsInput:
    detections: BallDetectionOutput
    source_size_wh: tuple[int, int]
    fps: float


@dataclass(frozen=True)
class BallPointsOutput:
    """Full frame axis with explicit validity; zero-filled missing entries are not observations."""
    camera_id: str
    source_size_wh: tuple[int, int]
    frame_indices: NDArray[np.int64]
    uv_px: NDArray[np.float32]
    confidence: NDArray[np.float32]
    observed: NDArray[np.bool_]
    smoothing: TemporalSmoothingResult

    def __post_init__(self) -> None:
        n = len(self.frame_indices)
        if (not self.camera_id or n < 1 or self.frame_indices.dtype != np.int64
                or not np.array_equal(self.frame_indices, np.arange(n))):
            raise ValueError("Ball points require a camera and the full source frame axis")
        if len(self.source_size_wh) != 2 or any(type(x) is not int or x < 2 for x in self.source_size_wh):
            raise ValueError("Ball points require source image size")
        if self.uv_px.shape != (n, 2) or self.confidence.shape != (n,) or self.observed.shape != (n,):
            raise ValueError("Ball point array shape mismatch")
        if self.uv_px.dtype != np.float32 or self.confidence.dtype != np.float32 or self.observed.dtype != bool:
            raise TypeError("Ball points require float32 coordinates/confidence and bool observed")
        if not np.isfinite(self.uv_px).all() or not np.isfinite(self.confidence).all():
            raise ValueError("Ball points and confidence must be finite")
        if np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("Ball point confidence must be in [0,1]")
        if np.any(self.uv_px[~self.observed] != 0) or np.any(self.confidence[~self.observed] != 0):
            raise ValueError("Missing points must have zero coordinates/confidence")
        if not np.array_equal(self.observed, self.smoothing.accepted):
            raise ValueError("Ball observation mask must equal accepted source measurements")
        if not np.array_equal(self.uv_px[self.observed], self.smoothing.coordinates[self.observed]):
            raise ValueError("Accepted ball coordinates must match the smoothing result")


class BallPointsModule:
    io = ComponentIO("ball_points", BallPointsInput, BallPointsOutput,
                     {"detections": InputPort("ball_detections", 2)}, BALL_POINTS_PORT.schema, version=BALL_POINTS_PORT.version)

    def __init__(self, config: TemporalSmoothingConfig = DEFAULT_TEMPORAL_SMOOTHING) -> None:
        self.config = config

    def process(self, inputs: BallPointsInput) -> BallPointsOutput:
        row = inputs.detections
        smoothing = smooth_2d_trajectory(row.uv_px, row.confidence, row.observed,
            fps=inputs.fps, image_size_wh=inputs.source_size_wh, config=self.config)
        observed = smoothing.accepted
        return BallPointsOutput(row.camera_id, inputs.source_size_wh, row.frame_indices.copy(),
                                np.where(observed[:, None], smoothing.coordinates, 0).astype(np.float32),
                                np.where(observed, row.confidence, 0).astype(np.float32), observed.copy(), smoothing)
