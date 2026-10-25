"""Exercise domain-gap reporting with mismatched and complete fixed-budget evidence."""

from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from tests.unit.tasks.court_detection.scaling.test_comparison import _report

pytestmark = pytest.mark.unit


def _report_module():
    path = (
        Path(__file__).resolve().parents[5]
        / "knowledge/runs/run-court-scene-diversity-preparation-s42/report.py"
    )
    spec = importlib.util.spec_from_file_location("court_scene_campaign_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reports():
    reports = []
    for n in range(1, 5):
        report = _report(images=5500)
        report["global_step"] = 33160
        val_counts = {f"B{i:02d}": 500 // n + int(i < 500 % n) for i in range(n)}
        report["data_condition"]["scene_images"] = {
            scene: 6000 // n - count for scene, count in val_counts.items()
        }
        report["data_condition"]["synthetic_validation"] = {
            "images": 500,
            "scene_images": val_counts,
        }
        real = report["splits"]["tennis_court_detector-val"]
        real["count"] = 2211
        synthetic = deepcopy(real)
        synthetic.update(
            source="synthetic_court",
            count=500,
            pose_supervised=True,
            ordered_sample_ids_sha256=f"scenes-{n}",
        )
        synthetic["metrics"] = {metric: 0.3 for metric in synthetic["metrics"]}
        report["splits"]["synthetic_court-val"] = synthetic
        reports.append(report)
    return reports


def test_report_keeps_domains_and_signed_gaps_separate():
    rows = _report_module().summarize(list(reversed(_reports())))
    assert [row["scenes"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["synthetic/line_dice"] == 0.3
    assert rows[0]["real/line_dice"] == 0.42
    assert rows[0]["real_minus_synthetic/line_dice"] == pytest.approx(0.12)


@pytest.mark.parametrize(
    "mutation",
    ["real_order", "step", "train_count", "synthetic_count", "duplicate", "nonfinite"],
)
def test_report_rejects_incompatible_or_incomplete_runs(mutation):
    reports = _reports()
    changed = reports[-1]
    if mutation == "real_order":
        changed["splits"]["tennis_court_detector-val"]["ordered_sample_ids_sha256"] = (
            "changed"
        )
    elif mutation == "step":
        changed["global_step"] = 33159
    elif mutation == "train_count":
        changed["data_condition"]["accepted_images"] = 6000
    elif mutation == "synthetic_count":
        changed["splits"]["synthetic_court-val"]["count"] = 499
    elif mutation == "duplicate":
        reports[-1] = deepcopy(reports[0])
    else:
        changed["splits"]["synthetic_court-val"]["metrics"]["line_dice"] = float("nan")
    with pytest.raises(ValueError):
        _report_module().summarize(reports)
