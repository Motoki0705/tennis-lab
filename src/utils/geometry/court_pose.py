"""Court-space pose geometry (torch).

Conversions between normalized court positions/canonical poses and world/court
coordinates in meters. Only the global translation is normalized; canonical
root-relative pose values remain metres. Shared by PLCS training,
visualization and analysis.
"""

from __future__ import annotations

from torch import Tensor

from src.utils.schema.court_normalization import (
    CourtCoordinateNormalization,
    resolve_court_coordinate_normalization,
)


def _resolve_contract(
    normalization: CourtCoordinateNormalization | str,
) -> CourtCoordinateNormalization:
    if isinstance(normalization, CourtCoordinateNormalization):
        return normalization
    return resolve_court_coordinate_normalization(normalization)


def court_position_to_world_translation(
    position: Tensor,
    *,
    normalization: CourtCoordinateNormalization | str = "v1",
) -> Tensor:
    """Convert a normalized court position into world/court translation (meters)."""
    result = _resolve_contract(normalization).denormalize_position(position)
    if not isinstance(result, Tensor):
        raise TypeError("Torch court translation conversion returned a non-tensor.")
    return result


def canonical_pose_to_world_pose(
    canonical_pose: Tensor,
    position: Tensor,
    rotation: Tensor,
    *,
    normalization: CourtCoordinateNormalization | str = "v1",
) -> Tensor:
    """Place canonical joints into court coordinates using translation and yaw."""
    translation = court_position_to_world_translation(
        position,
        normalization=normalization,
    )
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
    *,
    normalization: CourtCoordinateNormalization | str = "v1",
) -> Tensor:
    """Invert court placement and recover canonical joints from a world/court pose."""
    translation = court_position_to_world_translation(
        position,
        normalization=normalization,
    )
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
