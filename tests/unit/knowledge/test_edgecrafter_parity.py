"""CPU tests for parity of matched EdgeCrafter prediction rows."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.evaluation.player_pose.scoring import Predictions
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    directory = (
        Path(__file__).resolve().parents[3]
        / "knowledge/runs/run-meiji-player-model-comparison"
    )
    monkeypatch.syspath_prepend(str(directory))
    return runpy.run_path(str(directory / "edgecrafter_parity.py"))


def test_parity_matches_reordered_people_and_reports_a_missing_row(
    script: dict[str, Any],
) -> None:
    reference = Predictions(
        np.array([[10, 20, 30, 60], [70, 40, 120, 90], [70, 70, 90, 90]], dtype=float),
        np.array([0.8, 0.6, 0.24]),
    )
    candidate = Predictions(
        np.array([[70, 40, 119, 90], [10, 20, 30, 60]], dtype=float),
        np.array([0.62, 0.79]),
    )
    compare = script["prediction_parity"]

    parity = compare([(reference, candidate)], has_pose=False)
    assert parity.passed
    assert parity.metrics["matched_iou_p05"] == pytest.approx(0.981)
    assert parity.metrics["abs_score_error"]["p95"] == pytest.approx(0.0195)

    incomplete = Predictions(candidate.boxes_xyxy[:1], candidate.scores[:1])
    parity = compare([(reference, incomplete)], has_pose=False)
    assert not parity.passed
    assert parity.metrics["matched_fraction_both_directions"] == 0.5


@pytest.mark.parametrize("displacement,missing", [(9.0, False), (0.0, True)])
def test_parity_checks_pose_displacement_and_joint_availability(
    script: dict[str, Any],
    displacement: float,
    missing: bool,
) -> None:
    boxes = np.array([[20.0, 10.0, 80.0, 90.0]])
    confidence = np.array([0.8])
    joints = np.tile([30.0, 45.0, 1.0], (1, 17, 1))
    candidate = joints.copy()
    candidate[..., 0] += displacement
    if missing:
        candidate[0, 9] = 0.0
    parity = script["prediction_parity"](
        [
            (
                Predictions(boxes, confidence, joints),
                Predictions(boxes, confidence, candidate),
            )
        ],
        has_pose=True,
    )
    assert not parity.passed
    assert parity.metrics["joint_error_px"]["p95"] == displacement
    assert parity.metrics["joint_availability_agreement"] == (
        16 / 17 if missing else 1.0
    )


def test_file_comparison_rejects_different_source_frames(
    script: dict[str, Any], tmp_path: Path
) -> None:
    header = {
        "schema": "player_model_predictions_v1",
        "benchmark_id": "example",
        "manifest_sha256": "a" * 64,
        "model_id": "ecdet-l-coco",
        "has_pose": False,
    }
    for backend, frame in (("pt32", 23), ("pt16", 24)):
        directory = tmp_path / backend
        directory.mkdir()
        path = directory / "predictions.jsonl"
        record = {
            "clip_id": "video_000/clip_003",
            "camera_id": "cam0",
            "source_frame": frame,
            "boxes_xyxy": [[1, 4, 10, 15]],
            "scores": [0.9],
            "keypoints_xyc": None,
        }
        fields = {**header, "backend": backend}
        path.write_text(
            json.dumps(fields) + "\n" + json.dumps(record) + "\n", encoding="utf-8"
        )
        write_json_atomic(
            directory / "metrics.json",
            {
                **fields,
                "weight_sha256": "b" * 64,
                "source_commit": "c" * 40,
                "input_size_hw": [640, 640],
                "image_batch_size": 1,
                "person_confidence_exclusive": 0.0,
                "pose_bbox_joint_margin": None,
                "score": {"images": 1},
                "predictions_sha256": dual_sha256(path),
            },
        )
    with pytest.raises(ValueError, match="misaligned"):
        script["compare_files"](tmp_path / "pt32", tmp_path / "pt16")
