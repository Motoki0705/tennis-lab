"""Test the RTMPose comparison's COCO17 and unavailable-condition contracts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
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


@pytest.mark.usefixtures("rtmpose")
def test_backend_delta_counts_missing_and_measures_native_pixel_distance() -> None:
    summarize = import_module("rtmpose_accuracy").paired_delta
    pt32: NDArray[np.float32] = np.zeros((1, 17, 3), dtype=np.float32)
    fp16 = pt32.copy()
    fp16[..., 2] = -0.5
    fp16[0, 9, :2] = (3, 4)
    pt32[0, [0, 1], :2] = np.nan
    fp16[0, [0, 2], :2] = np.nan

    summary = summarize(pt32, fp16)

    assert summary["paired_joints"] == 14
    assert (
        summary["missing_in_candidate_only"]
        == summary["missing_in_reference_only"]
        == 1
    )
    assert summary["missing_in_both"] == 1
    assert summary["xy_px"]["max"] == 5
    assert summary["xy_above_1px"] == 1 and summary["xy_above_5px"] == 0
    assert summary["score_abs_p95"] == 0.5
    assert summary["worst"] == {"row_index": 0, "joint": "left_wrist"}


@pytest.mark.usefixtures("rtmpose")
def test_rescore_comparison_sees_pose_changes_with_fixed_box_recall() -> None:
    compare_scores = import_module("rtmpose_accuracy").score_changes
    previous: dict[str, Any] = {
        "policy": {"pose_match_iou": 0.5},
        "overall": {
            "detection": {
                "references": 2,
                "matched_at_iou_0_50": 2,
                "recall_by_iou": {"0.50": 1.0, "0.75": 1.0},
            },
            "pose": {"with_detection_coverage": {"oks": {"mean": 0.8}}},
        },
    }
    rescored = deepcopy(previous)
    rescored["policy"]["candidate_order_for_iou_ties"] = ["detection_score_descending"]
    added_policy = compare_scores(previous, rescored)
    assert added_policy["changed_metric_sections"] == []
    assert added_policy["changed_policy_fields"] == ["candidate_order_for_iou_ties"]

    rescored["overall"]["pose"]["with_detection_coverage"]["oks"]["mean"] = 0.7
    changed_pose = compare_scores(previous, rescored)
    assert changed_pose["changed_metric_sections"] == ["overall"]
    assert (
        changed_pose["recall_by_iou_before"]
        == changed_pose["recall_by_iou_after"]
        == {"0.50": 1.0, "0.75": 1.0}
    )


def test_measured_forwards_preserve_original_then_flipped_view(
    rtmpose: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = import_module("pose_contract")
    stages = import_module("evaluate").AccuracyStages(Backend.PT32)
    measured = stages.measure_forward
    calls: list[tuple[Backend, str]] = []
    crop_ends: list[list[float]] = []

    def measure(backend: Backend, op: Callable[[], Any], *, name: str = "") -> Any:
        calls.append((backend, name))
        return measured(backend, op, name=name)

    monkeypatch.setattr(stages, "measure_forward", measure)

    def affine(sample: dict[str, Any]) -> dict[str, Any]:
        return {**sample, "input_center": (20, 30), "input_scale": (144, 192)}

    crop_coordinates = np.broadcast_to(np.array([144, 192], np.float32), (1, 26, 2))
    scores: NDArray[np.float32] = np.ones((1, 26), np.float32)
    modules = {
        "mmpose.datasets.transforms": SimpleNamespace(
            GetBBoxCenterScale=lambda: lambda sample: sample,
            TopdownAffine=lambda **_: affine,
        ),
        "mmpose.registry": SimpleNamespace(
            KEYPOINT_CODECS=SimpleNamespace(
                build=lambda _: SimpleNamespace(
                    decode=lambda x, y: (crop_coordinates, scores)
                )
            )
        ),
        "mmpose.models.utils.tta": SimpleNamespace(
            flip_vectors=lambda x, y, **_: (x.flip(-1), y)
        ),
    }
    monkeypatch.setattr(rtmpose, "import_module", modules.__getitem__)
    config = SimpleNamespace(
        model=SimpleNamespace(
            data_preprocessor=SimpleNamespace(mean=[0, 0, 0], std=[1, 1, 1])
        ),
        codec={},
    )
    recipe = rtmpose.Recipe(
        Path("weights"), Path("config"), config, {"flip_indices": list(range(26))}
    )

    def forward(pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        crop_ends.append(pixels[0, 0, 0, [0, -1]].tolist())
        return torch.ones((1, 26, 2)), torch.ones((1, 26, 2))

    image: NDArray[np.uint8] = np.zeros((384, 288, 3), np.uint8)
    image[:, 0, 2], image[:, -1, 2] = 41, 43
    row = contract.PoseInput(image, np.array([1, 2, 10, 12], np.float32))
    infer = rtmpose._make_infer(recipe, Backend.PT32, forward, torch.device("cpu"))
    result = infer((row,), stages)

    assert calls == [(Backend.PT32, "original"), (Backend.PT32, "horizontal_flip")]
    assert crop_ends == [[41, 43], [43, 41]]
    np.testing.assert_array_equal(result.keypoints_xyc[0, 9], [20, 30, 1])
