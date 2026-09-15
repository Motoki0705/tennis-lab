"""Deterministic court placement for canonical COCO-17 motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.plcs.motion.contracts import Coco17MotionClip
from src.utils.schema.court_normalization import normalize_court_position


@dataclass(frozen=True, slots=True)
class PlacedCoco17Motion:
    """A canonical clip rigidly placed into the physical court frame."""

    position: NDArray[np.float32]
    rotation: NDArray[np.float32]
    canonical_pose_3d: NDArray[np.float32]
    world_joints_3d: NDArray[np.float32]
    root_rotation: NDArray[np.float32]


def _yaw_rotation(yaw: float | NDArray[np.float32]) -> NDArray[np.float32]:
    values = np.asarray(yaw, dtype=np.float32)
    cos = np.cos(values).astype(np.float32)
    sin = np.sin(values).astype(np.float32)
    output = np.zeros(values.shape + (3, 3), dtype=np.float32)
    output[..., 0, 0] = cos
    output[..., 0, 1] = -sin
    output[..., 1, 0] = sin
    output[..., 1, 1] = cos
    output[..., 2, 2] = 1.0
    return output


def root_yaw(rotation: NDArray[np.float32]) -> NDArray[np.float32]:
    """Extract Z-up yaw from full root rotations."""
    matrices = np.asarray(rotation, dtype=np.float32)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("rotation must have shape (T, 3, 3).")
    return np.arctan2(matrices[:, 1, 0], matrices[:, 0, 0]).astype(np.float32)


def place_motion_on_court(
    clip: Coco17MotionClip,
    *,
    initial_x_m: float,
    initial_y_m: float,
    initial_yaw_rad: float,
) -> PlacedCoco17Motion:
    """Preserve relative motion while choosing one court-space start transform."""
    if not all(
        np.isfinite(value) for value in (initial_x_m, initial_y_m, initial_yaw_rad)
    ):
        raise ValueError("Court placement values must be finite.")

    source_yaw = root_yaw(clip.root_rotation)
    yaw_offset = float(
        np.arctan2(
            np.sin(initial_yaw_rad - float(source_yaw[0])),
            np.cos(initial_yaw_rad - float(source_yaw[0])),
        )
    )
    offset_rotation = _yaw_rotation(yaw_offset)

    centered_root = clip.root_translation_m.copy()
    centered_root[:, :2] -= clip.root_translation_m[0, :2]
    court_root = centered_root @ offset_rotation.T
    court_root[:, 0] += initial_x_m
    court_root[:, 1] += initial_y_m

    root_relative = clip.joints_3d_m - clip.root_translation_m[:, None, :]
    court_relative = np.einsum(
        "ij,tkj->tki", offset_rotation, root_relative, optimize=True
    ).astype(np.float32)
    world_joints = (court_relative + court_root[:, None, :]).astype(np.float32)
    court_root_rotation = np.einsum(
        "ij,tjk->tik", offset_rotation, clip.root_rotation, optimize=True
    ).astype(np.float32)
    world_yaw = root_yaw(court_root_rotation)
    inverse_yaw = _yaw_rotation(-world_yaw)
    canonical = np.einsum(
        "tij,tkj->tki",
        inverse_yaw,
        world_joints - court_root[:, None, :],
        optimize=True,
    ).astype(np.float32)

    rotations = np.stack((np.cos(world_yaw), np.sin(world_yaw)), axis=-1).astype(
        np.float32
    )
    position = np.asarray(normalize_court_position(court_root), dtype=np.float32)
    return PlacedCoco17Motion(
        position=position,
        rotation=rotations,
        canonical_pose_3d=canonical,
        world_joints_3d=world_joints,
        root_rotation=court_root_rotation,
    )


__all__ = ["PlacedCoco17Motion", "place_motion_on_court", "root_yaw"]
