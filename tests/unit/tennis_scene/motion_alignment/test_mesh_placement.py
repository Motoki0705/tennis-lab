"""Court placement preserves metric geometry and the existing renderer equation."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.tennis_scene.motion_alignment.mesh_placement import (
    canonicalize_incam_body,
    interpolate_rotations,
    place_canonical_body,
)
from src.utils.geometry.matrices import SMPL_Y_UP_TO_COURT_Z_UP
from src.utils.geometry.triangulation import PinholeCamera


def test_hip_anchor_is_not_assumed_to_be_smpl_root() -> None:
    rng = np.random.default_rng(78)
    rotation = Rotation.from_euler("xyz", [0.4, -0.7, 0.2]).as_matrix()
    camera = PinholeCamera("camera", np.eye(3), rotation, np.array([1.0, 2.0, 4.0]))
    vertices = rng.normal(size=(3, 7, 3)).astype(np.float32)
    joints = rng.normal(size=(3, 17, 3)).astype(np.float32)
    orient = Rotation.random(3, random_state=rng).as_rotvec().astype(np.float32)
    hips = np.array([[[1.0, -4.0, 1.0], [1.4, -4.0, 1.0]]] * 3, np.float32)
    root_regressor = np.array([1.0, 0, 0, 0, 0, 0, 0])
    canonical = canonicalize_incam_body(
        vertices, joints, orient, camera, root_regressor
    )
    delta = np.array([0.2, -0.4, 0.8])
    scale = 1.1
    result = place_canonical_body(
        canonical, hips.mean(1).astype(np.float64), delta, scale
    )
    np.testing.assert_allclose(
        result.joints_court[:, 11:13].mean(1), hips.mean(1), atol=1e-6
    )
    assert not np.allclose(result.position, hips.mean(1))
    correction = Rotation.from_euler("z", delta[:, None]).as_matrix()
    direct = (
        scale
        * np.einsum(
            "tij,tvj->tvi", correction, (vertices - vertices[:, :1]) @ camera.rotation
        )
        + result.position[:, None]
    )
    g = Rotation.from_rotvec(result.global_orient).as_matrix()
    z = Rotation.from_euler("z", result.yaw[:, None]).as_matrix()
    matrix = z @ SMPL_Y_UP_TO_COURT_Z_UP @ g.transpose(0, 2, 1)
    encoded = (
        np.einsum("tij,tvj->tvi", matrix, result.vertices_local)
        + result.position[:, None]
    )
    np.testing.assert_allclose(encoded, direct, atol=2e-6)


def test_pose_interpolation_takes_the_short_rotation_arc() -> None:
    angles = np.array(
        [[0.0, 0.0, np.deg2rad(179)], [0.0, 0.0, np.deg2rad(-179)]], np.float32
    )
    interpolated = interpolate_rotations(angles, np.array([0, 2]), np.array([1]))
    assert abs(interpolated[0, 2]) > 3.0
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_rotations(angles, np.array([0, 2]), np.array([3]))
