"""Test the RF-DETR experiment adapter without upstream models or weights."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend, ConditionUnavailable, Status
from src.utils.io import write_json_atomic

_SCRIPT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "knowledge/runs/run-meiji-player-model-comparison/rfdetr.py"
    )
)


def test_preprocess_keeps_native_height_width_and_converts_bgr_to_rgb() -> None:
    image: NDArray[np.uint8] = np.full((3, 5, 3), [51, 102, 204], dtype=np.uint8)
    images, shape = _SCRIPT["prepare_images"]((image,), 2)

    assert shape.tolist() == [[3, 5]]
    torch.testing.assert_close(
        images,
        torch.tensor([1.375546, -0.25, -0.915556]).view(1, 3, 1, 1).expand(1, 3, 2, 2),
    )


def test_person_filter_keeps_coco_label_one_and_aligned_native_boxes() -> None:
    rows = np.array(
        [
            [10, 20, 70, 85, 0.99, 0],  # Not a COCO person.
            [10, 30, 10, 60, 0.98, 1],  # Collapsed after clipping.
            [20, 60, 70, 90, 0.95, 2],  # Bicycle.
            [13, 41, 79, 96, 0.25, 1],  # Exactly on the inclusive cutoff.
            [15, 18, 23, 29, 0.24, 1],
        ],
        np.float32,
    )
    result = _SCRIPT["person_predictions"](rows)
    np.testing.assert_array_equal(result.boxes_xyxy, [[13, 41, 79, 96]])
    np.testing.assert_array_equal(result.scores, [0.25])
    assert result.keypoints_xyc is None
    assert _SCRIPT["person_predictions"](rows[:3]).boxes_xyxy.shape == (0, 4)


def test_parity_matches_reordered_boxes_and_detects_a_missing_person() -> None:
    expected = Predictions(
        np.array([[10, 20, 30, 60], [70, 40, 120, 90]]), np.array([0.8, 0.6])
    )
    observed = Predictions(
        np.array([[70, 40, 119, 90], [10, 20, 30, 60]]), np.array([0.62, 0.79])
    )
    compare = _SCRIPT["prediction_parity"]

    parity = compare([(expected, observed)])
    assert parity.passed
    assert parity.metrics["matched_iou_p05"] == pytest.approx(0.981)
    assert parity.metrics["abs_score_difference"]["p95"] == pytest.approx(0.0195)

    incomplete = Predictions(observed.boxes_xyxy[:1], observed.scores[:1])
    parity = compare([(expected, incomplete)])
    assert not parity.passed
    assert parity.metrics["pt32_retained_fraction"] == 0.5


def test_pt32_jsonl_pins_empty_frames_and_rejects_modified_predictions(
    tmp_path: Path,
) -> None:
    model = _SCRIPT["Model"](
        _SCRIPT["VARIANTS"]["rfdetr-l"], tmp_path / "weights.pth", "a" * 64
    )
    identity = _SCRIPT["header"]("meiji/cohort", "b" * 64, model, Backend.PT32)
    key = FrameKey("video_000/clip_003", "cam0", 120)
    raw = tmp_path / "predictions.jsonl"
    _SCRIPT["write_predictions"](
        raw, identity, {key: Predictions(np.empty((0, 4)), np.empty(0))}
    )
    write_json_atomic(
        tmp_path / "accuracy.json",
        {"weights_sha256": model.sha256, "predictions_sha256": dual_sha256(raw)},
    )
    raw_text = raw.read_text()
    _, row = map(json.loads, raw_text.splitlines())
    assert row == {
        "clip_id": "video_000/clip_003",
        "camera_id": "cam0",
        "source_frame": 120,
        "boxes_xyxy": [],
        "scores": [],
        "keypoints_xyc": None,
    }
    predictions = _SCRIPT["read_pt32_predictions"](tmp_path, identity, model.sha256)
    assert predictions[key].boxes_xyxy.shape == (0, 4)
    raw.write_text(raw_text.replace("120", "121"))
    with pytest.raises(ValueError, match="do not match"):
        _SCRIPT["read_pt32_predictions"](tmp_path, identity, model.sha256)


def test_2xl_refuses_access_before_weight_open_without_consent(tmp_path: Path) -> None:
    with pytest.raises(ConditionUnavailable, match="user acceptance") as caught:
        _SCRIPT["Model"].open("rfdetr-2xl", tmp_path / "absent.pth")
    assert caught.value.status is Status.UNAVAILABLE
