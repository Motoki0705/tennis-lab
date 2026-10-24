"""Matched-condition aggregation, measured counts and invalid comparison refusal."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from src.tasks.court_detection.ablation.evaluation import DENSE_METRICS
from src.tasks.court_detection.scaling.comparison import compare_evaluations

pytestmark = pytest.mark.unit


def _report(*, images: int = 8, training_seed: int = 42) -> dict[str, Any]:
    return {
        "schema": "court_data_scaling_evaluation_v1",
        "series": "A1",
        "global_step": 100,
        "config": {
            "model": {"dim": 768},
            "loss": {"pose": True},
            "training": {"step_schedule": {"max_steps": 100}},
            "run": {"init_weights": None},
            "data": {
                "batch_size": 8,
                "augmentation": {"long_side": 512},
                "processing": {"schema": "coverage"},
            },
        },
        "target_bundle": {"version": "coverage"},
        "parameters": {"transformer": 1},
        "precision": "bf16-mixed",
        "data_condition": {
            "data_version": f"condition-{images}",
            "generation_seed": 17,
            "training_seed": training_seed,
            "accepted_images": images,
            "scene_images": {"B00": images},
            "evaluation_registry_sha256": "fixed",
        },
        "splits": {
            "tennis_court_detector-val": {
                "source": "tennis_court_detector",
                "split": "val",
                "batch_size": 8,
                "count": 2,
                "ordered_sample_ids_sha256": "fixed-order",
                "pose_supervised": False,
                "metrics": dict.fromkeys(DENSE_METRICS, training_seed / 100),
            }
        },
    }


def test_comparison_retains_each_seed_and_measured_count(tmp_path: Path) -> None:
    inputs = []
    for index, report in enumerate(
        [_report(), _report(training_seed=43), _report(images=17)]
    ):
        path = tmp_path / f"evaluation-{index}.json"
        path.write_text(json.dumps(report))
        inputs.append(path)
    result = compare_evaluations(inputs, tmp_path / "comparison")
    receipt = json.loads(result.with_suffix(".json").read_text())
    assert [row["accepted_images"] for row in receipt["rows"]] == [8, 8, 17]
    point = next(curve for curve in receipt["curves"] if curve["images"] == 8)
    assert point["runs"] == 2 and point["mean"] == pytest.approx(0.425)
    assert point["std"] > 0
    assert {point["images"] for point in receipt["curves"]} == {8, 17}
    assert (result.parent / "scaling.png").is_file()
    assert (result.parent / "comparison.csv").is_file()
    assert receipt["evaluations"][0]["sha256"]


@pytest.mark.parametrize(
    "mutation",
    [
        "step",
        "order",
        "registry",
        "model",
        "preprocessing",
        "schedule",
        "metrics",
        "duplicate",
        "zero-count",
    ],
)
def test_comparison_rejects_incompatible_conditions(
    tmp_path: Path, mutation: str
) -> None:
    original = _report()
    changed = deepcopy(_report(images=16))
    if mutation == "step":
        changed["global_step"] = 101
    elif mutation == "order":
        changed["splits"]["tennis_court_detector-val"]["ordered_sample_ids_sha256"] = (
            "different"
        )
    elif mutation == "registry":
        changed["data_condition"]["evaluation_registry_sha256"] = "different"
    elif mutation == "model":
        changed["config"]["model"]["dim"] = 1024
    elif mutation == "preprocessing":
        changed["config"]["data"]["augmentation"]["long_side"] = 256
    elif mutation == "schedule":
        changed["config"]["training"]["step_schedule"]["max_steps"] = 200
    elif mutation == "metrics":
        changed["splits"]["tennis_court_detector-val"]["metrics"]["line_dice"] = float(
            "nan"
        )
    elif mutation == "duplicate":
        changed = deepcopy(original)
    else:
        changed["data_condition"]["accepted_images"] = 0
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path, report in zip(paths, [original, changed], strict=True):
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        compare_evaluations(paths, tmp_path / "invalid")


def test_val_and_test_are_separate_rows_and_curves(tmp_path: Path) -> None:
    validation = _report()
    testing = _report()
    testing["splits"]["tennis_court_detector-test"] = testing["splits"].pop(
        "tennis_court_detector-val"
    )
    testing["splits"]["tennis_court_detector-test"]["split"] = "test"
    paths = [tmp_path / "val.json", tmp_path / "test.json"]
    for path, report in zip(paths, [validation, testing], strict=True):
        path.write_text(json.dumps(report))
    output = compare_evaluations(paths, tmp_path / "combined")
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert {row["split"] for row in receipt["rows"]} == {"val", "test"}
    assert {curve["split"] for curve in receipt["curves"]} == {"val", "test"}
    assert {curve["runs"] for curve in receipt["curves"]} == {1}
