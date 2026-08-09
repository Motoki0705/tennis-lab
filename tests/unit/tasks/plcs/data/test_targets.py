"""Tests for public PLCS target conversion boundaries."""

import numpy as np
import pytest

from src.tasks.plcs.data.targets import smplh_joints_to_coco17
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
