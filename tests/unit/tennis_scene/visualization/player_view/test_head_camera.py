"""Independent head axes/centres and explicit missing-frame contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.tennis_scene.visualization.player_view.head_camera import (
    CameraRejectionCode,
    HeadCameraConfig,
    build_head_camera_series,
    head_joint_chain,
)
from src.utils.schema.player import SMPLH_BODY_JOINT_NAMES, SMPLH_SKELETON


@pytest.fixture
def head_inputs() -> dict:
    parents: np.ndarray = np.full(22, -1, np.int32)
    for parent, child in SMPLH_SKELETON:
        parents[child] = parent
    regressor: np.ndarray = np.zeros((24, 4), np.float32)
    regressor[:, 0] = 1
    regressor[15] = [0, 0.25, 0.75, 0]
    return {
        "placed_vertices": np.tile(
            [[5, -3, 0.9], [4, -2, 2], [6, -4, 3], [8, 2, 1]], (3, 1, 1)
        ),
        "body_pose": np.zeros((3, 63)),
        "body_to_court_rotation": np.tile(np.eye(3), (3, 1, 1)),
        "mesh_valid": np.ones(3, bool),
        "scale": 1.7,
        "joint_regressor": regressor,
        "parents": parents,
        "body_joint_names": SMPLH_BODY_JOINT_NAMES,
        "mesh_joint_names": (*SMPLH_BODY_JOINT_NAMES, "left_hand", "right_hand"),
        "config": HeadCameraConfig(face_offset_head_m=(0.04, 0.08, 0.12)),
    }


def test_scaled_root_centered_placement_and_noncommuting_neck_head_rotations(
    head_inputs: dict,
) -> None:
    # A nonzero local root; weighted head row differs from any vertex and hip.
    # Pose/placement constructed by hand: body X->Y, Y->Z, Z->X;
    # neck +90 around Y and head +90 around Z => head X->Z,Y->X,Z->Y.
    body = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    local = np.array([[1, 2, 3], [1.2, 3.1, 3.3], [1.2, 3.1, 3.3], [3, 5, 8]]) * 1.7
    world = (local - local[0]) @ body.T + [5, -3, 0.9]
    head_inputs["placed_vertices"] = np.tile(world, (3, 1, 1))
    head_inputs["body_to_court_rotation"][:] = body
    pose = head_inputs["body_pose"].reshape(3, 21, 3)
    pose[:, 11, 1] = np.pi / 2  # named neck joint 12 -> pose triple 11
    pose[:, 14, 2] = np.pi / 2  # named head joint 15 -> pose triple 14
    series = build_head_camera_series(**head_inputs)

    np.testing.assert_allclose(
        series.head_position, [[5.51, -2.66, 2.77]] * 3, atol=1e-6
    )
    np.testing.assert_allclose(
        series.camera_to_court[:, :3, 3], [[5.646, -2.456, 2.838]] * 3, atol=1e-6
    )
    np.testing.assert_allclose(
        series.camera_to_court[0, :3, :3],
        [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
        atol=1e-6,
    )
    # Known asymmetric points: up/down, left/right and an off-screen ball.
    camera = series.pinhole(0)
    points = camera.center + np.array([[-1, 4, -2], [1.5, 2, 0.25], [0, 1, -3]])
    np.testing.assert_allclose(
        camera.project(points), [[960, 520], [560, -120], [2560, 360]], atol=1e-4
    )
    np.testing.assert_allclose(series.K, [[640, 0, 640], [0, 640, 360], [0, 0, 1]])


def test_body_yaw_and_neck_turn_are_independent(head_inputs: dict) -> None:
    a = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    head_inputs["body_to_court_rotation"][:] = a
    head_inputs["body_to_court_rotation"][1] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    head_inputs["body_pose"][2, 34] = np.pi / 2  # neck +Y only; body stays fixed
    series = build_head_camera_series(**head_inputs)
    np.testing.assert_allclose(
        series.camera_to_court[:, :3, 2], [[0, -1, 0], [1, 0, 0], [1, 0, 0]], atol=1e-6
    )
    np.testing.assert_allclose(series.camera_to_court[0, :3, 0], [-1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(
        series.camera_to_court[1:, :3, 0], [[0, -1, 0]] * 2, atol=1e-6
    )


def test_spine_neck_head_and_roll_survive_tilted_body(head_inputs: dict) -> None:
    pose = head_inputs["body_pose"].reshape(3, 21, 3)
    pose[:, 8, 0] = 0.31  # spine3
    pose[:, 11, 1] = -0.47
    pose[:, 14, 2] = 0.62
    body = Rotation.from_euler("zyx", [0.81, -0.28, 0.19])
    head_inputs["body_to_court_rotation"][:] = body.as_matrix()
    series = build_head_camera_series(**head_inputs)
    # Independent scipy application to three explicitly signed camera axes.
    axes = np.diag([-1.0, -1.0, 1.0])
    for axis, angle in [("z", 0.62), ("y", -0.47), ("x", 0.31)]:
        axes = Rotation.from_euler(axis, angle).apply(axes)
    axes = body.apply(axes)
    np.testing.assert_allclose(series.camera_to_court[0, :3, :3].T, axes, atol=1e-6)
    assert abs(series.camera_to_court[0, 2, 0]) > 0.1  # no world-up roll reset


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("mesh", CameraRejectionCode.MESH_INVALID),
        ("pose", CameraRejectionCode.HEAD_POSE_NONFINITE),
        ("rotation", CameraRejectionCode.PLACEMENT_ROTATION_INVALID),
        ("reflection", CameraRejectionCode.PLACEMENT_ROTATION_INVALID),
        ("position", CameraRejectionCode.HEAD_POSITION_NONFINITE),
    ],
)
def test_isolated_invalid_frame_has_reason_and_cannot_be_projected(
    head_inputs: dict, failure: str, expected: int
) -> None:
    if failure == "mesh":
        head_inputs["mesh_valid"][1] = False
        head_inputs["placed_vertices"][1] = 0
        head_inputs["body_pose"][1] = np.nan  # source mask takes precedence
        head_inputs["body_to_court_rotation"][1] = 0
    elif failure == "pose":
        head_inputs["body_pose"][1, 34] = np.nan
    elif failure == "rotation":
        head_inputs["body_to_court_rotation"][1] = 0
    elif failure == "reflection":
        head_inputs["body_to_court_rotation"][1, 0, 0] = -1
    else:
        head_inputs["placed_vertices"] = head_inputs["placed_vertices"].astype(float)
        head_inputs["placed_vertices"][1, 1] = np.nan
    series = build_head_camera_series(**head_inputs)
    np.testing.assert_array_equal(series.camera_valid, [True, False, True])
    np.testing.assert_array_equal(series.camera_rejection_code, [0, expected, 0])
    assert np.isnan(series.camera_to_court[1]).all()
    assert np.isnan(series.head_position[1]).all()
    with pytest.raises(ValueError, match="invalid"):
        series.pinhole(1)
    assert np.isfinite(series.pinhole(2).center).all()


def test_entirely_invalid_player_needs_no_invented_scale(head_inputs: dict) -> None:
    head_inputs["scale"] = None
    head_inputs["mesh_valid"][:] = False
    series = build_head_camera_series(**head_inputs)
    assert not series.camera_valid.any()
    assert np.isnan(series.camera_to_court).all()
    np.testing.assert_array_equal(series.camera_rejection_code, [1, 1, 1])


@pytest.mark.parametrize(
    "failure",
    ["parents", "body-names", "mesh-names", "scale", "regressor", "dimensions"],
)
def test_inconsistent_static_inputs_are_rejected(
    head_inputs: dict, failure: str
) -> None:
    if failure == "parents":
        head_inputs["parents"][15] = 9  # a plausible but incorrect head chain
    elif failure == "body-names":
        names = list(SMPLH_BODY_JOINT_NAMES)
        names[12], names[15] = names[15], names[12]
        head_inputs["body_joint_names"] = names
    elif failure == "mesh-names":
        head_inputs["mesh_joint_names"] = SMPLH_BODY_JOINT_NAMES
    elif failure == "scale":
        head_inputs["scale"] = None
    elif failure == "regressor":
        head_inputs["joint_regressor"][15] *= 2
    else:
        head_inputs["body_pose"] = np.zeros((3, 69))
    with pytest.raises(ValueError):
        build_head_camera_series(**head_inputs)


def test_named_head_chain_is_complete(head_inputs: dict) -> None:
    chain = head_joint_chain(
        head_inputs["parents"],
        head_inputs["body_joint_names"],
        head_inputs["mesh_joint_names"],
    )
    assert [SMPLH_BODY_JOINT_NAMES[j] for j in chain] == [
        "spine1",
        "spine2",
        "spine3",
        "neck",
        "head",
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": 1.5},
        {"horizontal_fov_degrees": 180},
        {"horizontal_fov_degrees": np.nan},
        {"near_m": 0},
        {"far_m": 0.01},
        {"face_offset_head_m": (0, np.nan, 0)},
    ],
)
def test_invalid_camera_settings_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        replace(HeadCameraConfig(), **kwargs)
