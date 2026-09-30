"""Regression tests for the frozen all-ten-comparisons reproduction rule."""

import copy
import runpy
from pathlib import Path
from typing import Any

import pytest

SCRIPT = (Path(__file__).resolve().parents[5]
          / "knowledge/runs/run-i935-seed-reproduction-r25-20260930/compare_seeds.py")


def gate_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    row = dict(frames=10, error_px_frames=10, nll_px_frames=10,
               median_error_px=1.0, p90_error_px=2.0, p95_error_px=3.0, mean_nll_px=4.0)
    ref = {f"meiji/{condition}/observed": row.copy() for condition in ("observed", "evidence_gap")}
    candidate = copy.deepcopy(ref)
    for item in candidate.values():
        for key in ("median_error_px", "p90_error_px", "p95_error_px", "mean_nll_px"):
            item[key] -= 0.1
    return {seed: copy.deepcopy(candidate) for seed in ("43", "44")}, ref


@pytest.mark.parametrize("seed", ["43", "44"])
@pytest.mark.parametrize("condition,metric", [
    ("observed", "median_error_px"), ("observed", "p90_error_px"),
    ("observed", "p95_error_px"), ("observed", "mean_nll_px"),
    ("evidence_gap", "mean_nll_px"),
])
def test_each_comparison_must_strictly_improve(seed: str, condition: str, metric: str) -> None:
    gate = runpy.run_path(str(SCRIPT))["gate"]
    seeds, reference = gate_inputs()
    assert gate(seeds, reference, reference)["status"] == "pass"
    key = f"meiji/{condition}/observed"
    seeds[seed][key][metric] = reference[key][metric]
    result = gate(seeds, reference, reference)
    assert result["status"] == "fail"
    assert result["passed"] == 9 and result["total"] == 10


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_metric_cannot_pass(invalid: float | None) -> None:
    gate = runpy.run_path(str(SCRIPT))["gate"]
    seeds, reference = gate_inputs()
    seeds["44"]["meiji/evidence_gap/observed"]["mean_nll_px"] = invalid
    assert gate(seeds, reference, reference)["status"] == "unconfirmed"


def test_missing_seed_or_incomplete_denominator_cannot_pass() -> None:
    gate = runpy.run_path(str(SCRIPT))["gate"]
    seeds, reference = gate_inputs()
    assert gate({"43": seeds["43"]}, reference, reference)["status"] == "unconfirmed"
    seeds["44"]["meiji/evidence_gap/observed"]["nll_px_frames"] = 9
    assert gate(seeds, reference, reference)["status"] == "unconfirmed"
