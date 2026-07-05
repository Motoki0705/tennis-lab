"""Rotation matrices and rigid SMPL court transforms (numpy).

Consolidates the rotation-matrix builders that were defined privately in
``tennis_scene`` (a scalar Y-axis matrix in ``utils/transforms`` and batched
Z-axis / axis-angle matrices in the renderer).

Convention note: :func:`rotation_matrix_y` is retained for historical callers.
:func:`rotation_matrix_z` and :func:`axis_angle_to_rotation_matrix` are batched
and accept array-shaped inputs, returning ``(..., 3, 3)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

SMPL_Y_UP_TO_COURT_Z_UP: NDArray[np.float32] = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def rotation_matrix_y(yaw: float) -> NDArray[np.float32]:
    """Create a ``(3, 3)`` Y-axis rotation matrix from a scalar yaw (radians)."""
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    return np.array(
        [
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ],
        dtype=np.float32,
    )


def rotation_matrix_z(yaw: NDArray[np.float32]) -> NDArray[np.float32]:
    """Create batched ``(..., 3, 3)`` Z-axis rotation matrices from yaw angles."""
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot: NDArray[np.float32] = np.zeros((*yaw.shape, 3, 3), dtype=np.float32)
    rot[..., 0, 0] = cos_y
    rot[..., 0, 1] = -sin_y
    rot[..., 1, 0] = sin_y
    rot[..., 1, 1] = cos_y
    rot[..., 2, 2] = 1.0
    return rot


def smpl_y_up_to_court_z_up(
    points: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Rotate SMPL/GVHMR Y-up points into court-local Z-up coordinates.

    GVHMR/SMPL vertices use a Y-up body coordinate convention. Tennis court
    coordinates use the XY plane as the ground and +Z as up. This applies the
    proper +90 degree X-axis rotation ``[x, y, z] -> [x, -z, y]`` to any
    array whose last dimension is XYZ.
    """
    if points.shape[-1:] != (3,):
        raise ValueError(
            "points must have XYZ coordinates in the last dimension, "
            f"got shape {points.shape}."
        )
    rotated: NDArray[np.float32] = points @ SMPL_Y_UP_TO_COURT_Z_UP.T
    return rotated.astype(np.float32, copy=False)


def axis_angle_to_rotation_matrix(
    axis_angle: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Convert batched axis-angle vectors ``(..., 3)`` to ``(..., 3, 3)`` matrices.

    Uses Rodrigues' formula; the zero-rotation case is handled by clamping the
    angle's denominator.
    """
    theta = np.linalg.norm(axis_angle, axis=-1)
    denom = np.where(theta > 1e-8, theta, 1.0)
    axis = axis_angle / denom[..., None]
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    c = np.cos(theta)
    s = np.sin(theta)
    one_minus_c = 1.0 - c

    rot: NDArray[np.float32] = np.empty((*axis_angle.shape[:-1], 3, 3), dtype=np.float32)
    rot[..., 0, 0] = c + x * x * one_minus_c
    rot[..., 0, 1] = x * y * one_minus_c - z * s
    rot[..., 0, 2] = x * z * one_minus_c + y * s
    rot[..., 1, 0] = y * x * one_minus_c + z * s
    rot[..., 1, 1] = c + y * y * one_minus_c
    rot[..., 1, 2] = y * z * one_minus_c - x * s
    rot[..., 2, 0] = z * x * one_minus_c - y * s
    rot[..., 2, 1] = z * y * one_minus_c + x * s
    rot[..., 2, 2] = c + z * z * one_minus_c
    return rot


def apply_plcs_transform(
    vertices: NDArray[np.float32],
    position: NDArray[np.float32],
    yaw: float,
) -> NDArray[np.float32]:
    """Apply PLCS position and yaw to SMPL/GVHMR vertices (single frame).

    Converts Y-up SMPL vertices (``(V, 3)``) to court-local Z-up, rotates them
    about court +Z by ``yaw``, then translates by court ``position`` (``(3,)``).
    """
    vertices_court_local = smpl_y_up_to_court_z_up(vertices)
    rot_mat = rotation_matrix_z(np.asarray(yaw, dtype=np.float32))
    rotated = vertices_court_local @ rot_mat.T
    return rotated + position


def apply_plcs_transform_batch(
    vertices: NDArray[np.float32],
    positions: NDArray[np.float32],
    yaws: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply PLCS position and yaw to batched SMPL vertices.

    Args:
        vertices: Local SMPL/GVHMR vertices ``(T, V, 3)`` in Y-up coordinates.
        positions: 3D positions in court coords ``(T, 3)``, meters.
        yaws: Yaw angles ``(T,)``, radians.

    Returns:
        Transformed vertices ``(T, V, 3)`` in court coordinates.
    """
    vertices_court_local = smpl_y_up_to_court_z_up(vertices)
    rot = rotation_matrix_z(yaws)
    # (T, V, 3) @ (T, 3, 3)^T -> (T, V, 3) via batched matmul.
    rotated = vertices_court_local @ rot.transpose(0, 2, 1)
    return rotated + positions[:, None, :]


__all__ = [
    "SMPL_Y_UP_TO_COURT_Z_UP",
    "apply_plcs_transform",
    "apply_plcs_transform_batch",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "smpl_y_up_to_court_z_up",
]
