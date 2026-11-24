"""Test the RTMPose comparison's COCO17 and unavailable-condition contracts."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray

from src.utils.inference_benchmark import Backend, ConditionUnavailable, Status
from src.utils.schema.player import COCO_KP_NAMES

RUN = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-meiji-topdown-pose-comparison"
)


@pytest.fixture
def rtmpose(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return import_module("rtmpose")


def test_coco17_projection_keeps_side_row_and_valid_raw_scores(
    rtmpose: ModuleType,
) -> None:
    coords: NDArray[np.float32] = np.full((2, 26, 2), (144, 192), dtype=np.float32)
    coords[0, 9:11] = ((72, 288), (216, 96))
    coords[0, 15] = (0, 0)
    coords[1, 2] = (-0.5, -0.5)  # Official invalid SimCC peak sentinel.
    coords[1, 3] = (-0.5, -0.5)
    scores: NDArray[np.float32] = np.ones((2, 26), dtype=np.float32)
    scores[0, 9:11] = (0.625, 0.75)
    scores[1, 2:4] = (-0.25, 0)
    scores[1, 7] = np.inf
    finite: NDArray[np.bool_] = np.ones((2, 26), dtype=np.bool_)
    finite[0, 5] = False

    poses = rtmpose._to_native_coco17(
        coords,
        scores,
        np.array([[80, 90], [210, 30]], np.float32),
        np.array([[144, 192], [72, 96]], np.float32),
        finite,
    ).keypoints_xyc

    assert poses.dtype == np.float32
    assert poses.shape == (2, 17, 3)
    np.testing.assert_array_equal(poses[0, 9:11], [[44, 138, 0.625], [116, 42, 0.75]])
    np.testing.assert_array_equal(poses[0, 15], [8, -6, 1])
    np.testing.assert_array_equal(poses[1, 0], [210, 30, 1])
    assert np.isnan(poses[0, 5, :2]).all() and poses[0, 5, 2] == 1
    assert np.isnan(poses[1, 2:4, :2]).all()
    np.testing.assert_array_equal(poses[1, 2:4, 2], [-0.25, 0])
    assert np.isnan(poses[1, 7]).all()


def test_halpe_schema_rejects_a_left_right_swap(rtmpose: ModuleType) -> None:
    names = dict(enumerate((*COCO_KP_NAMES, *(f"halpe_{i}" for i in range(9)))))
    rtmpose._validate_joint_order(names)
    names[1], names[2] = names[2], names[1]
    with pytest.raises(ConditionUnavailable, match="Halpe26 order") as caught:
        rtmpose._validate_joint_order(names)
    assert caught.value.status is Status.UNSUPPORTED


def test_trt_load_needs_an_engine_and_pt_rejects_unpinned_weights(
    rtmpose: ModuleType, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "rtmpose.pth"
    checkpoint.write_bytes(b"a different checkpoint")
    trt = rtmpose.make_adapter(
        model_id="rtmpose-m",
        backend=Backend.TRT16,
        checkpoint=checkpoint,
        device="cuda:0",
    )
    with pytest.raises(ConditionUnavailable, match="TensorRT bundle") as missing_engine:
        trt.load()
    assert missing_engine.value.status is Status.UNAVAILABLE

    eager = rtmpose.make_adapter(
        model_id="rtmpose-m", backend=Backend.PT32, checkpoint=checkpoint, device="cpu"
    )
    with pytest.raises(
        ConditionUnavailable, match=rtmpose.MODELS["rtmpose-m"].sha256
    ) as wrong_weights:
        eager.load()
    assert wrong_weights.value.status is Status.UNAVAILABLE
