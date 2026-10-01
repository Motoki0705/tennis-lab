"""Triangulate the single ball supplied by each camera's 2D detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.observation_types import (
    GroupedObservations,
    ObjectObservations,
    group_observations,
)
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointTriangulationConfig,
    TriangulatedPoints,
    reject_excessive_speed,
    triangulate_points,
)


@dataclass(frozen=True)
class BallReconstructionResult:
    uv_px: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    trajectory: TriangulatedPoints
    status: str


def single_ball_observations(observations: ObjectObservations, *, threshold: float) -> GroupedObservations:
    """Group the single detector stream without tracking or identity inference."""
    if observations.uv_px.shape[2:4] not in ((0, 1), (1, 1)):
        raise ValueError("Ball input must contain at most one 2D detection per camera/frame")
    visible = observations.visibility(threshold).any(-1)
    ids = np.where(visible, 0, -1).astype(np.int64)
    return group_observations(observations, ids, threshold=threshold)


def reconstruct_ball(
    observations: GroupedObservations, cameras: tuple[PinholeCamera, ...], *,
    fps: float, reprojection_px: float, min_frames: int = 5,
) -> BallReconstructionResult:
    count, views, frames, joints, _ = observations.uv_px.shape
    if count > 1 or joints != 1:
        raise ValueError("Ball reconstruction requires a single 2D detector stream")
    uv = observations.uv_px[0, :, :, 0] if count else np.zeros((views, frames, 2), np.float32)
    visible = observations.visibility[0, :, :, 0] if count else np.zeros((views, frames), bool)
    confidence = observations.confidence[0, :, :, 0] if count else np.zeros((views, frames), np.float32)
    trajectory = reject_excessive_speed(
        triangulate_points(uv, visible, cameras, config=PointTriangulationConfig(reprojection_px, (-.2, 20.)), confidence=confidence),
        fps=fps, max_speed_mps=65.,
    )
    if int(trajectory.valid.sum()) < min_frames:
        trajectory = TriangulatedPoints(np.zeros((frames, 3), np.float32), np.zeros(frames, bool), np.ones(frames, np.uint8), np.zeros((views, frames), bool), np.zeros((views, frames), np.float32))
        return BallReconstructionResult(uv, visible, trajectory, "ball_insufficient_support")
    return BallReconstructionResult(uv, visible, trajectory, "ok")
