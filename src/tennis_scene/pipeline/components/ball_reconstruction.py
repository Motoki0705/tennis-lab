"""Choose one clip-global rally ball after identity-aware triangulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.model_io.observations import GroupedObservations
from src.utils.geometry.triangulation import (
    PinholeCamera,
    PointTriangulationConfig,
    TriangulatedPoints,
    reject_excessive_speed,
    triangulate_points,
)


@dataclass(frozen=True)
class BallReconstructionResult:
    identity: int | None
    uv_px: NDArray[np.float32]
    visibility: NDArray[np.bool_]
    trajectory: TriangulatedPoints
    status: str
    candidate_counts: dict[int, int]


def reconstruct_rally_ball(
    observations: GroupedObservations, cameras: tuple[PinholeCamera, ...], *,
    fps: float, reprojection_px: float, min_frames: int = 5, ambiguity_ratio: float = .8,
) -> BallReconstructionResult:
    _, views, frames, joints, _ = observations.uv_px.shape
    if joints != 1:
        raise ValueError("Ball reconstruction requires one point per object")
    trajectories = [
        reject_excessive_speed(triangulate_points(
            observations.uv_px[row, :, :, 0], observations.visibility[row, :, :, 0],
            cameras, config=PointTriangulationConfig(reprojection_px, (-.2, 20.)),
            confidence=observations.confidence[row, :, :, 0],
        ), fps=fps, max_speed_mps=65.)
        for row in range(len(observations.identities))
    ]
    ranking = sorted(range(len(trajectories)), key=lambda i: (-int(trajectories[i].valid.sum()), int(observations.identities[i])))
    counts = {int(identity): int(result.valid.sum()) for identity, result in zip(observations.identities, trajectories, strict=True)}
    ambiguous = len(ranking) > 1 and counts[int(observations.identities[ranking[1]])] >= min_frames and counts[int(observations.identities[ranking[1]])] >= ambiguity_ratio * counts[int(observations.identities[ranking[0]])]
    if not ranking or counts[int(observations.identities[ranking[0]])] < min_frames or ambiguous:
        empty = TriangulatedPoints(np.zeros((frames, 3), np.float32), np.zeros(frames, bool), np.ones(frames, np.uint8), np.zeros((views, frames), bool), np.zeros((views, frames), np.float32))
        return BallReconstructionResult(None, np.zeros((views, frames, 2), np.float32), np.zeros((views, frames), bool), empty, "ball_identity_ambiguous" if ambiguous else "ball_insufficient_support", counts)
    row = ranking[0]
    return BallReconstructionResult(int(observations.identities[row]), observations.uv_px[row, :, :, 0], observations.visibility[row, :, :, 0], trajectories[row], "ok", counts)
