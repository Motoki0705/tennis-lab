"""The predeclared source/cache gate compares all components and rejects invalid data."""

import runpy
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (Path(__file__).resolve().parents[5]
          / "knowledge/runs/run-i935-pipeline-candidate-r24-20260930/check_pipeline.py")


def test_parity_checks_every_component_and_counts_outliers():
    compare = runpy.run_path(str(SCRIPT))["compare_arrays"]
    reference: dict[str, np.ndarray] = {"means": np.zeros((270, 4, 2), np.float32)}
    actual = {"means": reference["means"].copy()}
    actual["means"][-1, -1, -1] = 0.001
    report = compare(actual, reference, {"means": 5e-4})
    assert not report["passed"]
    assert report["fields"]["means"]["exceeded"] == 1
    assert report["fields"]["means"]["elements"] == 2160
    assert compare(reference, reference, {"means": 5e-4})["passed"]


@pytest.mark.parametrize("invalid", [np.array([np.nan]), np.array([np.inf]), np.zeros(2)])
def test_parity_rejects_nonfinite_values_and_shape_drift(invalid):
    compare = runpy.run_path(str(SCRIPT))["compare_arrays"]
    assert not compare({"means": invalid}, {"means": np.zeros(1)}, {"means": 5e-4})["passed"]
