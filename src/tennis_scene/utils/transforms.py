"""Coordinate transformations for tennis scene reconstruction.

This module provides utilities to transform SMPL vertices from local space
to global court coordinate space using PLCS position and yaw.

Example:
    >>> vertices_global = apply_plcs_transform(
    ...     vertices_local, position, yaw
    ... )
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rotation_matrix_y(yaw: float) -> NDArray[np.float32]:
    """Create a Y-axis rotation matrix from yaw angle.

    Args:
        yaw: Yaw angle in radians.

    Returns:
        3x3 rotation matrix.

    """
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


def apply_plcs_transform(
    vertices: NDArray[np.float32],
    position: NDArray[np.float32],
    yaw: float,
) -> NDArray[np.float32]:
    """Apply PLCS position and yaw to SMPL vertices.

    Transforms local SMPL vertices to global court coordinates by:
    1. Rotating around Y-axis by yaw
    2. Translating by position

    Args:
        vertices: Local SMPL vertices (V, 3).
        position: 3D position in court coords (3,), meters.
        yaw: Yaw angle in radians.

    Returns:
        Transformed vertices (V, 3) in court coordinates.

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
        vertices: Local SMPL vertices (T, V, 3).
        positions: 3D positions in court coords (T, 3), meters.
        yaws: Yaw angles (T,), radians.

    Returns:
        Transformed vertices (T, V, 3) in court coordinates.

    """
    T = len(yaws)
    cos_y = np.cos(yaws).astype(np.float32)
    sin_y = np.sin(yaws).astype(np.float32)
    # Build (T, 3, 3) Y-axis rotation matrices without a Python loop.
    rot = np.zeros((T, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = cos_y
    rot[:, 0, 2] = sin_y
    rot[:, 1, 1] = 1.0
    rot[:, 2, 0] = -sin_y
    rot[:, 2, 2] = cos_y
    # (T, V, 3) @ (T, 3, 3)^T -> (T, V, 3) via batched matmul
    rotated = vertices @ rot.transpose(0, 2, 1)
    return rotated + positions[:, None, :]


def normalize_keypoints(
    keypoints: NDArray[np.float32],
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Normalize pixel keypoints to [0, 1] range.

    Args:
        keypoints: Keypoints in pixel coords (..., 2).
        width: Image width.
        height: Image height.

    Returns:
        Normalized keypoints (..., 2).

    """
    result = keypoints.copy()
    result[..., 0] /= width
    result[..., 1] /= height
    return result


def denormalize_keypoints(
    keypoints: NDArray[np.float32],
    width: int,
    height: int,
) -> NDArray[np.float32]:
    """Denormalize keypoints from [0, 1] to pixel coords.

    Args:
        keypoints: Normalized keypoints (..., 2).
        width: Image width.
        height: Image height.

    Returns:
        Pixel keypoints (..., 2).

    """
    result = keypoints.copy()
    result[..., 0] *= width
    result[..., 1] *= height
    return result


if __name__ == "__main__":
    # Smoke test
    V = 6890  # SMPL vertex count
    T = 10

    vertices_local = np.random.rand(T, V, 3).astype(np.float32)
    positions = np.random.rand(T, 3).astype(np.float32) * 10
    yaws = np.random.rand(T).astype(np.float32) * 2 * np.pi

    vertices_global = apply_plcs_transform_batch(vertices_local, positions, yaws)

    assert vertices_global.shape == (T, V, 3)
    assert vertices_global.dtype == np.float32

    # Check single frame transform
    v_single = apply_plcs_transform(vertices_local[0], positions[0], yaws[0])
    assert np.allclose(v_single, vertices_global[0])

    print("transforms.py smoke test passed!")
