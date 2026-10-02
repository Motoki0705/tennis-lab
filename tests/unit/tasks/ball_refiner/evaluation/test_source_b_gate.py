"""The source-video B gate uses pooled observed frames and the frozen inclusive limits."""

import math
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D

SCRIPT = (Path(__file__).resolve().parents[5]
          / "knowledge/runs/run-i935-source-b-gate-r26-20261001/evaluate_gate.py")


@pytest.fixture(scope="module")
def gate_module():
    return runpy.run_path(str(SCRIPT))


def test_frame_pooling_does_not_average_camera_quantiles(gate_module):
    rows = [{"error_px": np.zeros(100), "nll_px": np.zeros(100)},
            {"error_px": np.full(2, 100.), "nll_px": np.full(2, 100.)}]
    summary = gate_module["summarize"](rows)
    assert summary["median_error_px"] == summary["p90_error_px"] == 0
    assert summary["mean_nll_px"] == pytest.approx(200 / 102)


def test_p90_alone_blocks_switch_and_boundary_is_inclusive(gate_module):
    cached = [{"error_px": np.zeros(100), "nll_px": np.zeros(100)}]
    errors = np.r_[np.full(80, .5), np.full(20, 5.)]
    source = [{"error_px": errors, "nll_px": np.full(100, .1)}]
    check = gate_module["accuracy_gate"]
    assert check(cached, source)["passed"]
    source[0]["error_px"][-20:] += .0001
    result = check(cached, source)
    assert not result["passed"]
    assert result["tests"]["median_error_px"]["passed"]
    assert result["tests"]["mean_nll_px"]["passed"]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_gate_rejects_nonfinite_observed_predictions(gate_module, value):
    with pytest.raises(ValueError, match="Cannot drop"):
        gate_module["summarize"]([{"error_px": np.array([value]), "nll_px": np.zeros(1)}])


def test_gate_rejects_mismatched_camera_populations(gate_module):
    with pytest.raises(ValueError, match="same camera/frame"):
        gate_module["accuracy_gate"](
            [{"error_px": np.zeros(2), "nll_px": np.zeros(2)}],
            [{"error_px": np.zeros(1), "nll_px": np.zeros(1)}],
        )


def test_score_selects_heaviest_component_and_conditional_pixel_nll(gate_module):
    prediction = BallGMM2D(
        means=torch.tensor([[[[.25, .5], [.75, .5]], [[.5, .5], [.5, .5]]]], dtype=torch.float64),
        scale_tril=torch.eye(2, dtype=torch.float64).expand(1, 2, 2, 2, 2).clone(),
        mixture_logits=torch.tensor([[[0., math.log(3)], [0., 0.]]], dtype=torch.float64),
        presence_logits=torch.full((1, 2), -100., dtype=torch.float64),
    )
    result = gate_module["score"](prediction, np.array([[.75, .5], [np.nan, np.nan]]),
                                  np.array([True, False]), (101, 51))
    np.testing.assert_equal(result["error_px"], [0.])  # A mixture mean would have nonzero error.
    density_uv = (.75 + .25 * math.exp(-.5 * .5**2)) / (2 * math.pi)
    np.testing.assert_allclose(result["nll_px"], [-math.log(density_uv) + math.log(100 * 50)])
