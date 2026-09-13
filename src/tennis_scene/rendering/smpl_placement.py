"""Place cached GVHMR meshes using PLCS root position and court yaw."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.matrices import (
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
    smpl_y_up_to_court_z_up,
)


def place_smpl_vertices(
    verts_local: NDArray[np.float32],
    global_orient: NDArray[np.float32],
    players_position: NDArray[np.float32],
    players_yaw: NDArray[np.float32],
    joint_regressor: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Return (players, frames, vertices, 3) court vertices in metres.

    Regress and remove the SMPL pelvis, undo GVHMR's global orientation,
    convert Y-up to Z-up, then apply PLCS yaw and translation. The body
    articulation and metric scale are preserved; no ground snapping is used.
    """
    if verts_local.ndim != 4 or verts_local.shape[-1] != 3:
        raise RuntimeError(
            "smpl_vertices_local must have shape (P, T, V, 3), "
            f"got {verts_local.shape}."
        )
    if global_orient.shape != verts_local.shape[:2] + (3,):
        raise RuntimeError(
            f"smpl_global_orient must have shape (P, T, 3), got {global_orient.shape}."
        )
    if players_position.shape != verts_local.shape[:2] + (3,):
        raise RuntimeError(
            f"player_position must have shape (P, T, 3), got {players_position.shape}."
        )
    if players_yaw.shape != verts_local.shape[:2]:
        raise RuntimeError(
            f"player_yaw must have shape (P, T), got {players_yaw.shape}."
        )

    roots = np.einsum("jv,ptvc->ptjc", joint_regressor, verts_local)[:, :, 0, :]
    verts_centered = verts_local - roots[:, :, None, :]

    orient_rot = axis_angle_to_rotation_matrix(global_orient)
    verts_pose_local = np.einsum("ptji,ptvj->ptvi", orient_rot, verts_centered)

    verts_court_local = smpl_y_up_to_court_z_up(verts_pose_local)
    plcs_rot = rotation_matrix_z(players_yaw)
    verts_court = np.einsum("ptij,ptvj->ptvi", plcs_rot, verts_court_local)
    verts_court = verts_court + players_position[:, :, None, :]
    verts_court = cast("NDArray[np.float32]", verts_court.astype(np.float32))

    return verts_court
