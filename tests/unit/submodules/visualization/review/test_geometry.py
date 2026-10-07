from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from src.submodules.visualization.review.geometry import placed_vertices
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP


def test_decodes_saved_orientation_without_double_scale_or_hip_offset() -> None:
    # Construct the exact encoder identity independently, including tilt and yaw.
    court = Rotation.from_euler("xyz", [.23, -.42, .8]).as_matrix()
    yaw = .61
    rz = Rotation.from_euler("z", yaw).as_matrix()
    encoded = court.T @ rz @ SMPL_Y_UP_TO_COURT_Z_UP
    orientation = Rotation.from_matrix(encoded).as_rotvec().astype(np.float32)
    vertices = np.array([[.4, 1.3, -.2], [-.2, .7, .3]], np.float32)
    root = np.array([2.1, -6.3, .9], np.float32)
    result = placed_vertices(vertices, orientation, yaw, root)
    np.testing.assert_allclose(result, vertices @ court.T + root, atol=1e-6)


def test_zero_root_is_a_valid_coordinate_when_mask_is_true() -> None:
    vertices = np.array([[0, 1, 0]], np.float32)
    result = placed_vertices(vertices, np.zeros(3, np.float32), 0, np.zeros(3, np.float32))
    np.testing.assert_allclose(result, [[0, 0, 1]], atol=1e-6)
