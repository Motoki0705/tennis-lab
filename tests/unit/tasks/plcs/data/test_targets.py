"""Tests for public PLCS target conversion boundaries."""

import numpy as np
import pytest

from src.tasks.plcs.data.targets import (
    build_coco17_world_targets,
    smplh_joints_to_coco17,
)
from src.utils.schema.court_normalization import resolve_court_coordinate_normalization
from src.utils.schema.player import FACE_KEYPOINT_OFFSETS, SMPLH_TO_COCO17_MAPPING


def test_smplh_joints_to_coco17_maps_joints_and_rotates_face_offsets() -> None:
    joints: np.ndarray = np.arange(2 * 52 * 3, dtype=np.float32).reshape(
        2, 52, 3
    )
    yaw = np.array([0.0, np.pi / 2.0], dtype=np.float32)

    result = smplh_joints_to_coco17(joints, yaw)

    assert result.shape == (2, 17, 3)
    assert result.dtype == np.float32
    for coco_index, smplh_index in SMPLH_TO_COCO17_MAPPING.items():
        if coco_index not in FACE_KEYPOINT_OFFSETS:
            np.testing.assert_array_equal(
                result[:, coco_index], joints[:, smplh_index]
            )
    head = joints[:, 15]
    nose_offset = np.asarray(FACE_KEYPOINT_OFFSETS[0], dtype=np.float32)
    np.testing.assert_allclose(result[0, 0], head[0] + nose_offset)
    np.testing.assert_allclose(
        result[1, 0],
        head[1]
        + np.array([-nose_offset[1], nose_offset[0], nose_offset[2]]),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    ("joints", "error", "message"),
    [
        (np.zeros((1, 52, 3), dtype=np.float64), TypeError, "float32"),
        (np.zeros((52, 3), dtype=np.float32), ValueError, "shape"),
        (np.zeros((1, 51, 3), dtype=np.float32), ValueError, "shape"),
    ],
)
def test_smplh_joints_to_coco17_rejects_invalid_arrays(
    joints: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        smplh_joints_to_coco17(joints, 0.0)


def test_smplh_joints_to_coco17_rejects_non_finite_values() -> None:
    joints: np.ndarray = np.zeros((1, 52, 3), dtype=np.float32)
    joints[0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinity"):
        smplh_joints_to_coco17(joints, 0.0)


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_world_targets_scale_only_position_translation(version: str) -> None:
    contract = resolve_court_coordinate_normalization(version)
    physical_position = np.array([[2.0, -3.0, 1.25]], dtype=np.float32)
    canonical: np.ndarray = np.zeros((1, 52, 3), dtype=np.float32)
    canonical[0, :, 0] = np.linspace(-0.5, 0.5, 52)
    canonical[0, :, 2] = np.linspace(0.0, 2.0, 52)
    scene = {
        "position": contract.normalize_position(physical_position),
        "rotation": np.array([[1.0, 0.0]], dtype=np.float32),
        "canonical_pose_3d": canonical,
        "meta": {"initial_yaw": 0.0},
    }

    result = build_coco17_world_targets(scene, normalization=contract)

    for coco_index, smplh_index in SMPLH_TO_COCO17_MAPPING.items():
        if coco_index not in FACE_KEYPOINT_OFFSETS:
            np.testing.assert_allclose(
                result[0, coco_index],
                canonical[0, smplh_index] + physical_position[0],
                atol=1.0e-5,
                rtol=0.0,
            )


def test_v1_v2_world_targets_are_identical_for_same_physical_scene() -> None:
    physical_position = np.array([[1.2, -5.4, 0.9]], dtype=np.float32)
    canonical = np.arange(52 * 3, dtype=np.float32).reshape(1, 52, 3) / 100.0
    results = []
    for version in ("v1", "v2"):
        contract = resolve_court_coordinate_normalization(version)
        results.append(
            build_coco17_world_targets(
                {
                    "position": contract.normalize_position(physical_position),
                    "rotation": np.array([[0.0, 1.0]], dtype=np.float32),
                    "canonical_pose_3d": canonical,
                    "meta": {"initial_yaw": np.pi / 2.0},
                },
                normalization=contract,
            )
        )

    np.testing.assert_allclose(results[0], results[1], atol=1.0e-5, rtol=0.0)
