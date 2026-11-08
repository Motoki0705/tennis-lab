"""Tests for the pinhole camera and its Blender camera conversion."""

from __future__ import annotations

import numpy as np
import pytest

from src.tennis_scene.visualization.blender.export.camera import (
    PinholeCamera,
    to_blender_camera,
)
from src.tennis_scene.visualization.player_view.head_camera import (
    HeadCameraConfig,
    build_head_camera_series,
)
from src.utils.schema.player import SMPLH_BODY_JOINT_NAMES, SMPLH_SKELETON

WIDTH, HEIGHT = 1920, 1080


def _look_camera(center: np.ndarray, target: np.ndarray, focal: float) -> PinholeCamera:
    forward = (target - center) / np.linalg.norm(target - center)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    K = np.array([[focal, 0.0, WIDTH / 2], [0.0, focal, HEIGHT / 2], [0.0, 0.0, 1.0]])
    return PinholeCamera(K=K, R=R, t=-R @ center, width=WIDTH, height=HEIGHT)


def _blender_project(
    location: np.ndarray,
    rotation: np.ndarray,
    lens_mm: float,
    sensor_mm: float,
    point: np.ndarray,
) -> np.ndarray:
    """Project like a Blender camera: local -Z forward, +Y up, horizontal sensor fit."""
    local = rotation.T @ (point - location)
    depth = -local[2]
    focal_px = lens_mm / sensor_mm * WIDTH
    projected: np.ndarray = np.array(
        [
            WIDTH / 2 + focal_px * local[0] / depth,
            HEIGHT / 2 - focal_px * local[1] / depth,
        ]
    )
    return projected


@pytest.mark.parametrize(
    "neck_turn", [None, 0.0, 0.63], ids=["court-fit", "head-neutral", "head-neck-turn"]
)
def test_blender_pose_reproduces_opencv_projection(neck_turn: float | None) -> None:
    if neck_turn is None:
        camera = _look_camera(
            np.array([-7.0, -18.2, 2.7]), np.array([1.0, 3.0, 0.0]), 1044.3
        )
        points = np.array([[0.0, 0.0, 0.0], [5.485, 11.885, 0.0], [-3.0, -9.0, 1.8]])
        expected_center = [-7.0, -18.2, 2.7]
    else:
        parents: np.ndarray = np.full(22, -1, np.int32)
        for parent, child in SMPLH_SKELETON:
            parents[child] = parent
        pose = np.zeros((1, 63))
        pose[0, 34] = neck_turn  # change the neck while keeping the body fixed
        pose[0, 44] = 0.21  # retain a head roll
        regressor = np.tile([0.0, 1.0], (24, 1))
        series = build_head_camera_series(
            placed_vertices=np.array([[[0, 0, 1], [2, -4, 1.8]]]),
            body_pose=pose,
            body_to_court_rotation=np.array([[[1, 0, 0], [0, 0, -1], [0, 1, 0]]]),
            mesh_valid=np.array([True]),
            scale=1.2,
            joint_regressor=regressor,
            parents=parents,
            body_joint_names=SMPLH_BODY_JOINT_NAMES,
            mesh_joint_names=(*SMPLH_BODY_JOINT_NAMES, "left_hand", "right_hand"),
            config=HeadCameraConfig(
                width=WIDTH, height=HEIGHT, face_offset_head_m=(0, 0, 0)
            ),
        )
        camera = series.pinhole(0)
        # The third point represents an off-screen ball; do not clip it into view.
        points = (
            np.array([[-1, 0.4, 5], [0.8, -0.3, 2], [5, -2, 1]]) @ camera.R
            + camera.center
        )
        np.testing.assert_allclose(
            camera.project(points),
            [[768, 616.8], [1344, 396], [5760, -1380]],
            atol=1e-6,
        )
        expected_center = [2, -4, 1.8]
    pose = to_blender_camera(camera, sensor_width_mm=36.0)

    for point in points:
        expected = camera.project(point)
        actual = _blender_project(
            pose.location,
            pose.rotation_matrix,
            pose.lens_mm,
            pose.sensor_width_mm,
            point,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-9)
    np.testing.assert_allclose(pose.location, expected_center, atol=1e-12)
    assert np.isclose(np.linalg.det(pose.rotation_matrix), 1.0)


def test_off_center_principal_point_is_rejected() -> None:
    camera = _look_camera(np.array([0.0, -20.0, 3.0]), np.zeros(3), 1000.0)
    shifted = PinholeCamera(
        K=camera.K + np.array([[0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        R=camera.R,
        t=camera.t,
        width=WIDTH,
        height=HEIGHT,
    )
    with pytest.raises(ValueError, match="principal point"):
        to_blender_camera(shifted, sensor_width_mm=36.0)


def test_non_rotation_matrix_is_rejected() -> None:
    camera = _look_camera(np.array([0.0, -20.0, 3.0]), np.zeros(3), 1000.0)
    with pytest.raises(ValueError, match="orthonormal"):
        PinholeCamera(
            K=camera.K, R=camera.R * 2.0, t=camera.t, width=WIDTH, height=HEIGHT
        )


def test_from_scene_metadata_selects_camera_by_id() -> None:
    first = _look_camera(np.array([0.0, -20.0, 3.0]), np.zeros(3), 900.0)
    second = _look_camera(np.array([-7.0, -18.0, 2.7]), np.zeros(3), 1044.0)
    metadata = {
        "court_reference": {
            "camera_ids": ["cam0", "cam1"],
            "camera_fits": [
                {"K": c.K.tolist(), "R": c.R.tolist(), "t": c.t.tolist()}
                for c in (first, second)
            ],
        }
    }

    loaded = PinholeCamera.from_scene_metadata(
        metadata, "cam1", width=WIDTH, height=HEIGHT
    )

    np.testing.assert_allclose(loaded.center, second.center)
    with pytest.raises(ValueError, match="cam9"):
        PinholeCamera.from_scene_metadata(metadata, "cam9", width=WIDTH, height=HEIGHT)


def test_project_rejects_points_behind_camera() -> None:
    camera = _look_camera(np.array([0.0, -20.0, 3.0]), np.zeros(3), 1000.0)
    with pytest.raises(ValueError, match="behind"):
        camera.project(np.array([0.0, -40.0, 3.0]))
