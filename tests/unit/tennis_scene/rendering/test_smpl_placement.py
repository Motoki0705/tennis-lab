"""Checks that PLCS replaces global placement without altering body pose."""

import numpy as np

from src.tennis_scene.rendering.smpl_placement import place_smpl_vertices
from src.utils.geometry.matrices import axis_angle_to_rotation_matrix


def test_gvhmr_orientation_is_removed_before_plcs_placement() -> None:
    # A root and three anatomical offsets, globally rotated/translated by GVHMR.
    local = np.array(
        [[0, 0, 0], [0, 1.7, 0], [0.3, 0.8, 0.4], [-0.2, 0.5, 0]], dtype=np.float32
    )
    orient = np.array([[[0.3, 0.7, -0.2]]], dtype=np.float32)
    rot = axis_angle_to_rotation_matrix(orient)[0, 0]
    incam = (local @ rot.T + np.array([2, -1, 7], dtype=np.float32))[None, None]
    regressor = np.array([[1, 0, 0, 0]], dtype=np.float32)
    position = np.array([[[4, 5, 0.9]]], dtype=np.float32)
    yaw = np.array([[np.pi / 2]], dtype=np.float32)
    result = place_smpl_vertices(incam, orient, position, yaw, regressor)[0, 0]
    # Y-up -> Z-up followed by 90deg court yaw: [x,y,z] -> [z,x,y].
    np.testing.assert_allclose(result, local[:, [2, 0, 1]] + position[0, 0], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(result[1:] - result[0], axis=-1),
        np.linalg.norm(local[1:], axis=-1),
        atol=1e-6,
    )
    other = position + np.array([[[1, -2, 0.3]]], dtype=np.float32)
    shifted = place_smpl_vertices(incam, orient, other, yaw, regressor)[0, 0]
    np.testing.assert_allclose(
        shifted - result, np.broadcast_to([1, -2, 0.3], result.shape), atol=1e-6
    )
