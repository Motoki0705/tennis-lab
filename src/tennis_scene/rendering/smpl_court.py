"""Place GVHMR/SMPL body-local vertices into tennis-court coordinates.

``SceneResult`` stores each player's SMPL mesh in the body-local, Y-up GVHMR
frame (``smpl_vertices_local`` + ``smpl_global_orient``) and the PLCS placement
separately (``player_position`` root position + ``player_yaw`` about court +Z).
This module owns the single composition of those pieces so every consumer
(matplotlib scene renderer, Blender export) places bodies identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.utils.geometry.matrices import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
    smpl_y_up_to_court_z_up,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def load_smpl_joint_regressor(path: Path) -> NDArray[np.float32]:
    """Load a ``(joints, vertices)`` SMPL joint regressor from a torch file."""
    import torch

    reg = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(reg, torch.Tensor):
        result = reg.cpu().numpy().astype(np.float32)
    else:
        result = np.asarray(reg, dtype=np.float32)
    if result.ndim != 2:
        raise ValueError(
            "SMPL joint-regressor asset must contain a 2D (joints, vertices) "
            f"array, got shape {result.shape}."
        )
    return result


def smpl_vertices_to_court(
    verts_local: NDArray[np.float32],
    global_orient: NDArray[np.float32],
    players_position: NDArray[np.float32],
    players_yaw: NDArray[np.float32],
    joint_regressor: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return ``(P, T, V, 3)`` court-frame vertices for every player and frame.

    The SMPL root joint (regressor row 0) is moved to the origin, the body is
    rotated by its global orientation, converted from SMPL Y-up to court Z-up,
    rotated by the PLCS yaw, and finally translated to ``players_position``.
    """
    if verts_local.ndim != 4 or verts_local.shape[-1] != 3:
        raise ValueError(
            f"smpl_vertices_local must have shape (P, T, V, 3), got {verts_local.shape}."
        )
    if global_orient.shape != verts_local.shape[:2] + (3,):
        raise ValueError(
            f"smpl_global_orient must have shape (P, T, 3), got {global_orient.shape}."
        )
    if players_position.shape != verts_local.shape[:2] + (3,):
        raise ValueError(
            f"player_position must have shape (P, T, 3), got {players_position.shape}."
        )
    if players_yaw.shape != verts_local.shape[:2]:
        raise ValueError(f"player_yaw must have shape (P, T), got {players_yaw.shape}.")
    if joint_regressor.ndim != 2 or joint_regressor.shape[1] != verts_local.shape[2]:
        raise ValueError(
            "joint_regressor must have shape (J, V) matching the vertex count, "
            f"got {joint_regressor.shape} for V={verts_local.shape[2]}."
        )

    roots = np.einsum("jv,ptvc->ptjc", joint_regressor, verts_local)[:, :, 0, :]
    verts_centered = verts_local - roots[:, :, None, :]

    orient_rot = axis_angle_to_rotation_matrix(global_orient)
    verts_pose_local = np.einsum("ptji,ptvj->ptvi", orient_rot, verts_centered)

    verts_court_local = smpl_y_up_to_court_z_up(verts_pose_local)
    plcs_rot = rotation_matrix_z(players_yaw)
    verts_court = np.einsum("ptij,ptvj->ptvi", plcs_rot, verts_court_local)
    verts_court = verts_court + players_position[:, :, None, :]
    return np.asarray(verts_court, dtype=np.float32)


__all__ = ["load_smpl_joint_regressor", "smpl_vertices_to_court"]
