from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

_BUNDLE = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-court-assets-colab-l4-20261022"
)
_evaluate = runpy.run_path(str(_BUNDLE / "court_assets_acceptance.py"))["evaluate_comparison"]


@pytest.fixture
def evidence():
    def load(name):
        return json.loads((_BUNDLE / name).read_text())

    return (
        load("cross-gpu-comparison.json"),
        load("local-visibility.json"),
        load("colab-visibility.json"),
    )


def test_saved_cross_gpu_repeats_satisfy_teacher_and_rgb_contract(evidence):
    report, reference, candidate = evidence
    for name in ("cross-gpu-comparison.json", "cross-gpu-repeat-comparison.json"):
        report = json.loads((_BUNDLE / name).read_text())
        decision = _evaluate(report, reference, candidate)
        assert decision["accepted"]
        assert sum(scene["views"] for scene in decision["scenes"].values()) == 16
        assert sum(scene["points"] for scene in decision["scenes"].values()) == 392
        assert decision["scenes"]["B02"]["scene_psnr_db"] == pytest.approx(57.985742562)
        assert decision["scenes"]["B02"]["min_view_psnr_db"] == pytest.approx(55.858969822)


@pytest.mark.parametrize("field", ["uv", "depth_m", "physical_index", "renderer_visible"])
def test_any_teacher_difference_fails_even_when_rgb_passes(evidence, field):
    report, reference, candidate = evidence
    point = candidate["views"][0]["points"][0]
    point[field] = {
        "uv": [point["uv"][0] + 1e-9, point["uv"][1]],
        "depth_m": point["depth_m"] + 1e-9,
        "physical_index": 99,
        "renderer_visible": not point["renderer_visible"],
    }[field]
    decision = _evaluate(report, reference, candidate)
    assert not decision["accepted"]
    assert any("teacher inventory differs" in item for item in decision["failures"])


@pytest.mark.parametrize("scope,floor", [("scene", 57.0), ("view", 55.0)])
def test_rgb_floor_inclusive_and_not_hidden_by_other_views(evidence, scope, floor):
    report, reference, candidate = evidence
    scene = report["scenes"]["B02"]
    metric = scene["summary"]["rgb"] if scope == "scene" else scene["views"][0]["metrics"]["rgb"]
    metric.update(rmse=10 ** (-floor / 20), psnr_db=floor)
    assert _evaluate(report, reference, candidate)["accepted"]
    metric.update(rmse=10 ** (-(floor - 0.001) / 20), psnr_db=floor - 0.001)
    assert not _evaluate(report, reference, candidate)["accepted"]


def test_exact_rgb_null_psnr_passes_only_with_zero_rmse(evidence):
    report, reference, candidate = evidence
    metric = report["scenes"]["B00"]["views"][0]["metrics"]["rgb"]
    metric.update(rmse=0.0, psnr_db=None)
    assert _evaluate(report, reference, candidate)["accepted"]
    metric["rmse"] = 0.001
    with pytest.raises(ValueError, match="PSNR"):
        _evaluate(report, reference, candidate)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, True])
def test_invalid_rgb_evidence_is_rejected(evidence, value):
    report, reference, candidate = evidence
    report["scenes"]["B00"]["summary"]["rgb"]["rmse"] = value
    with pytest.raises(ValueError, match="RMSE"):
        _evaluate(report, reference, candidate)


@pytest.mark.parametrize("mutation", ["scene", "camera", "duplicate", "repeat", "schema"])
def test_incomplete_or_unsupported_comparison_is_rejected(evidence, mutation):
    report, reference, candidate = evidence
    if mutation == "scene":
        del report["scenes"]["B00"]
    elif mutation == "camera":
        report["scenes"]["B00"]["views"].pop()
    elif mutation == "duplicate":
        report["scenes"]["B00"]["views"].append(copy.deepcopy(report["scenes"]["B00"]["views"][0]))
    elif mutation == "repeat":
        report["candidate_repeat"] = "absent"
    else:
        report["schema"] = "court_render_comparison_v99"
    with pytest.raises(ValueError):
        _evaluate(report, reference, candidate)
