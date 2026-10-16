"""CPU transforms of already saved SMPL vertices; no body-model inference."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.utils.geometry.matrices import (
    SMPL_Y_UP_TO_COURT_Z_UP,
    axis_angle_to_rotation_matrix,
    rotation_matrix_z,
)


def placed_vertices(
    vertices: NDArray[np.float32],
    orientation: NDArray[np.float32],
    yaw: float,
    position: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Decode the placed_bodies v1 transform (vertices already centered/scaled).

    The encoder in motion_alignment/mesh_placement.py stores canonical vertices
    and R_encoded = R_court.T @ Rz(yaw) @ A. Consequently the court rotation is
    Rz(yaw) @ A @ R_encoded.T. Do not apply scale or a hip offset a second time.
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3 or orientation.shape != (3,) or position.shape != (3,):
        raise ValueError("Invalid saved placement geometry dimensions")
    if not all(np.isfinite(value).all() for value in (vertices, orientation, position)) or not np.isfinite(yaw):
        raise ValueError("Saved placement must contain finite coordinates")
    encoded = axis_angle_to_rotation_matrix(orientation)
    rotation = rotation_matrix_z(np.asarray(yaw, np.float32)) @ SMPL_Y_UP_TO_COURT_Z_UP @ encoded.T
    result: NDArray[np.float32] = (vertices @ rotation.T + position).astype(np.float32)
    return result
