"""Test that timing parity uses the pinned image IDs and PT32 output bytes."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.evaluation.player_pose.scoring import Predictions
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend, BenchmarkCase, ConditionUnavailable
from src.utils.io import write_json_atomic


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    runs = Path(__file__).resolve().parents[3] / "knowledge/runs"
    baseline = runs / "run-person-model-speed-baseline"
    comparison = runs / "run-meiji-player-model-comparison"
    monkeypatch.syspath_prepend(str(baseline))
    monkeypatch.syspath_prepend(str(comparison))
    return runpy.run_path(str(comparison / "edgecrafter_speed.py"))


def test_speed_rows_require_equal_image_ids(script: dict[str, Any]) -> None:
    frame = Predictions(np.array([[10.0, 20.0, 30.0, 50.0]]), np.array([0.8]))
    compare = script["compare_speed_rows"]
    assert compare({"frame0": frame}, {"frame0": frame}, has_pose=False).passed
    report = compare({"frame0": frame}, {"frame1": frame}, has_pose=False)
    assert not report.passed
    assert report.reason == "Speed input IDs differ from PT32"


def test_speed_reference_validates_manifest_and_prediction_digest(
    script: dict[str, Any], tmp_path: Path
) -> None:
    settings = {
        "official_source_commit": "a" * 40,
        "weight_sha256": "b" * 64,
        "dataset_id": "speed-example",
        "image_size_hw": [640, 640],
        "pose_box_margin": None,
        "person_score_exclusive": 0.0,
    }
    case = BenchmarkCase("ecdet-l-coco", Backend.PT16, 2, "image", "c" * 64, settings)
    predictions_path = tmp_path / "speed-b1-predictions.json"
    content = {
        "input_manifest_sha256": case.input_manifest_sha256,
        "rows": {
            "clip/cam/23": {
                "boxes_xyxy": [[1.0, 4.0, 11.0, 16.0]],
                "scores": [0.8],
                "keypoints_xyc": None,
            }
        },
    }
    write_json_atomic(predictions_path, content)
    report = {
        "status": "ok",
        "case": {
            "name": case.name,
            "backend": "PT32",
            "batch_size": 1,
            "input_manifest_sha256": case.input_manifest_sha256,
            "settings": settings,
        },
        "parity": {"metrics": {"predictions_sha256": dual_sha256(predictions_path)}},
    }
    write_json_atomic(tmp_path / "speed-b1.json", report)
    read = script["read_pt32_reference"]
    result = read(tmp_path, case=case, has_pose=False)
    np.testing.assert_array_equal(
        result["clip/cam/23"].boxes_xyxy, [[1.0, 4.0, 11.0, 16.0]]
    )
    with pytest.raises(ConditionUnavailable, match="input or model"):
        wrong = BenchmarkCase(
            case.name, case.backend, case.batch_size, "image", "d" * 64, settings
        )
        read(tmp_path, case=wrong, has_pose=False)

    content["rows"]["clip/cam/23"]["scores"] = [0.9]
    write_json_atomic(predictions_path, content)
    with pytest.raises(ConditionUnavailable, match="checksum differs"):
        read(tmp_path, case=case, has_pose=False)
