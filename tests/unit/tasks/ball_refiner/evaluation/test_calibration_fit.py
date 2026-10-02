"""Analytical scale optimum and exclusion of every held-out camera/condition."""

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from src.tasks.ball_refiner.evaluation.calibration_fit import (
    FitBlock,
    cross_validate_clips,
    density_terms,
    fit_covariance_multiplier,
)
from src.tasks.ball_refiner.refiner_2d.calibration import CovarianceCalibration
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D


def block(group, condition, q, n=4):
    return FitBlock(group, condition, np.full((n, 1), q, dtype=np.float64), np.zeros((n, 1)))


def test_analytic_optimum_balances_groups_and_conditions_despite_frame_counts():
    # A one-component 2D Gaussian has s*=E[q]/2. Counts/cameras must not
    # outweigh clips or the two conditions: (2+6+10+14)/4/2 = 4.
    blocks = [block("a", "observed", 2, 100), block("a", "evidence_gap", 6, 3),
              block("b", "observed", 10, 8), block("b", "evidence_gap", 14, 20)]
    result = fit_covariance_multiplier(blocks, bounds=(.25, 64), grid_points=129)
    assert result.covariance_multiplier == pytest.approx(4, rel=1e-6)
    assert result.nll_uv_after < result.nll_uv_before


def test_loco_excludes_all_camera_blocks_and_both_conditions():
    blocks = [block(g, c, 8, camera + 1) for g in ("clip0", "clip1", "clip2")
              for c in ("observed", "evidence_gap") for camera in range(3)]
    _, before = cross_validate_clips(blocks, bounds=(.25, 64), grid_points=65)
    changed = [block(b.group, b.condition, 40, len(b.mahalanobis)) if b.group == "clip0" else b for b in blocks]
    full, after = cross_validate_clips(changed, bounds=(.25, 64), grid_points=65)
    assert before["clip0"] == after["clip0"]
    assert after["clip0"].groups == ("clip1", "clip2")
    assert full.covariance_multiplier > after["clip0"].covariance_multiplier
    assert after["clip1"].covariance_multiplier > before["clip1"].covariance_multiplier


def test_precomputed_mixture_likelihood_matches_full_gmm_after_scaling():
    torch.manual_seed(42)
    means = torch.rand((1, 12, 4, 2), dtype=torch.float64)
    tril = torch.tensor([[.1, 0], [.02, .05]], dtype=torch.float64).expand(1, 12, 4, 2, 2)
    logits = torch.randn((1, 12, 4), dtype=torch.float64)
    prediction = BallGMM2D(means, tril, logits, torch.zeros((1, 12), dtype=torch.float64))
    target = torch.rand((1, 12, 2), dtype=torch.float64)
    terms = density_terms(prediction, target, group="clip0", condition="observed")
    for scale in (.5, 1, 3.7):
        actual = CovarianceCalibration(scale, "a" * 64).apply(prediction).log_prob(target)[0].numpy()
        expected = logsumexp(terms.log_coefficient - np.log(scale) - .5 * terms.mahalanobis / scale, axis=-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_missing_condition_and_bound_optimum_are_explicit_errors():
    with pytest.raises(ValueError, match="both evidence"):
        fit_covariance_multiplier([block("a", "observed", 8)], bounds=(.25, 64), grid_points=9)
    with pytest.raises(ValueError, match="boundary"):
        fit_covariance_multiplier([block("a", c, 1000) for c in ("observed", "evidence_gap")],
                                  bounds=(.25, 64), grid_points=9)
    with pytest.raises(ValueError, match="three"):
        cross_validate_clips([block("a", c, 8) for c in ("observed", "evidence_gap")],
                             bounds=(.25, 64), grid_points=9)
