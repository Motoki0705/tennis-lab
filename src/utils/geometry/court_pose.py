"""Court-space pose geometry (torch).

Conversions between normalized court positions/canonical poses and world/court
coordinates in meters, using the court coordinate scales from
:mod:`src.utils.schema.court`. Shared by PLCS training, visualization and
analysis.
"""

from __future__ import annotations

from torch import Tensor

from src.utils.schema.court import (
    COURT_COORD_SCALE_X,
    COURT_COORD_SCALE_Y,
    COURT_COORD_SCALE_Z,
)


def court_position_to_world_translation(position: Tensor) -> Tensor:
    """Convert a normalized court position into world/court translation (meters)."""
    scale = position.new_tensor(
        (COURT_COORD_SCALE_X, COURT_COORD_SCALE_Y, COURT_COORD_SCALE_Z)
    )
    return position * scale


def canonical_pose_to_world_pose(
    canonical_pose: Tensor,
    position: Tensor,
    rotation: Tensor,
) -> Tensor:
    """Place canonical joints into court coordinates using translation and yaw."""
    translation = court_position_to_world_translation(position)
    cos_yaw = rotation[..., 0].unsqueeze(-1)
    sin_yaw = rotation[..., 1].unsqueeze(-1)

    world_pose = canonical_pose.clone()
    world_pose[..., 0] = (
        canonical_pose[..., 0] * cos_yaw
        - canonical_pose[..., 1] * sin_yaw
        + translation[..., 0].unsqueeze(-1)
    )
    world_pose[..., 1] = (
        canonical_pose[..., 0] * sin_yaw
        + canonical_pose[..., 1] * cos_yaw
        + translation[..., 1].unsqueeze(-1)
    )
    world_pose[..., 2] = canonical_pose[..., 2] + translation[..., 2].unsqueeze(-1)
    return world_pose


def world_pose_to_canonical_pose(
    world_pose: Tensor,
    position: Tensor,
    rotation: Tensor,
) -> Tensor:
    """Invert court placement and recover canonical joints from a world/court pose."""
    translation = court_position_to_world_translation(position)
    centered_x = world_pose[..., 0] - translation[..., 0].unsqueeze(-1)
    centered_y = world_pose[..., 1] - translation[..., 1].unsqueeze(-1)

    cos_yaw = rotation[..., 0].unsqueeze(-1)
    sin_yaw = rotation[..., 1].unsqueeze(-1)

    canonical_pose = world_pose.clone()
    canonical_pose[..., 0] = centered_x * cos_yaw + centered_y * sin_yaw
    canonical_pose[..., 1] = -centered_x * sin_yaw + centered_y * cos_yaw
    canonical_pose[..., 2] = world_pose[..., 2] - translation[..., 2].unsqueeze(-1)
    return canonical_pose


__all__ = [
    "canonical_pose_to_world_pose",
    "court_position_to_world_translation",
    "world_pose_to_canonical_pose",
]
