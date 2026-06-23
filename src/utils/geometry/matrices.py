"""Rotation matrices and rigid SMPL court transforms (numpy).

Consolidates the rotation-matrix builders that were defined privately in
``tennis_scene`` (a scalar Y-axis matrix in ``utils/transforms`` and batched
Z-axis / axis-angle matrices in the renderer).

Convention note: :func:`rotation_matrix_y` takes a scalar yaw and returns a
single ``(3, 3)`` matrix (used for SMPL vertices, Y-up). :func:`rotation_matrix_z`
and :func:`axis_angle_to_rotation_matrix` are batched and accept array-shaped
inputs, returning ``(..., 3, 3)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
    """Apply PLCS position and yaw to SMPL vertices (single frame).

    Rotates ``vertices`` (``(V, 3)``) about the Y-axis by ``yaw`` then translates
    by ``position`` (``(3,)``), returning court-space vertices ``(V, 3)``.
    """
    rot_mat = rotation_matrix_y(yaw)
    rotated = vertices @ rot_mat.T
    return rotated + position


def apply_plcs_transform_batch(
    vertices: NDArray[np.float32],
    positions: NDArray[np.float32],
    yaws: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Apply PLCS position and yaw to batched SMPL vertices.

    Args:
        vertices: Local SMPL vertices ``(T, V, 3)``.
        positions: 3D positions in court coords ``(T, 3)``, meters.
        yaws: Yaw angles ``(T,)``, radians.

    Returns:
        Transformed vertices ``(T, V, 3)`` in court coordinates.
    """
    num_frames = len(yaws)
    cos_y = np.cos(yaws).astype(np.float32)
    sin_y = np.sin(yaws).astype(np.float32)
    # Build (T, 3, 3) Y-axis rotation matrices without a Python loop.
    rot: NDArray[np.float32] = np.zeros((num_frames, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_y
    rot[:, 0, 2] = sin_y
    rot[:, 1, 1] = 1.0
    rot[:, 2, 0] = -sin_y
    rot[:, 2, 2] = cos_y
    # (T, V, 3) @ (T, 3, 3)^T -> (T, V, 3) via batched matmul.
    rotated = vertices @ rot.transpose(0, 2, 1)
    return rotated + positions[:, None, :]


__all__ = [
    "apply_plcs_transform",
    "apply_plcs_transform_batch",
    "axis_angle_to_rotation_matrix",
    "rotation_matrix_y",
    "rotation_matrix_z",
]
