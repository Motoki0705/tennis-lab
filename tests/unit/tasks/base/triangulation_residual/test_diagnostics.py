"""Paired aggregation must not hide the small-error population behind tails."""

import numpy as np
import pytest

from src.tasks.base.triangulation_residual.diagnostics import (
    paired_errors,
    prediction_diagnostics,
)


def test_tail_improvement_does_not_imply_population_improvement():
    target = np.zeros((1, 100, 1, 3))
    initial = target.copy()
    initial[..., 0] = 0.1
    initial[:, -1, :, 0] = 10.0
    predicted = initial.copy()
    predicted[:, :-1, :, 0] += 0.01
    predicted[:, -1, :, 0] = 1.0
    summary = paired_errors(initial, predicted, target, np.ones((1, 100), bool))
    assert summary["predicted_m"]["mean"] < summary["initial_m"]["mean"]
    assert summary["predicted_m"]["median"] > summary["initial_m"]["median"]
    assert summary["improved_fraction"] == 0.01
    assert summary["worsened_fraction"] == 0.99
    assert summary["largest_correction_one_percent"]["net_gain_share"] > 1.0


def test_padding_and_empty_strata_are_explicit():
    x = np.zeros((2, 3, 1, 3))
    y = x.copy()
    y[:, 0, 0, 0] = [1.0, 2.0]
    x[:, 1:] = np.nan
    y[:, 1:] = np.inf
    valid = np.array([[True, False, False]] * 2)
    payload = dict(
        initial_world=y,
        pred_world=x,
        target_world=x,
        frame_valid=valid,
        severity=np.zeros(2),
    )
    with np.errstate(invalid="ignore"):
        result = prediction_diagnostics(payload, "blcs")
    assert result["world"]["initial_m"]["mean"] == 1.5
    assert result["world"]["predicted_m"]["mean"] == 0.0
    assert result["strata"]["hard"]["initial_m"]["count"] == 0
    assert result["strata"]["hard"]["improved_fraction"] is None
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        prediction_diagnostics(payload, "blcs")
