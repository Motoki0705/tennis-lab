"""HDR means density level sets of the full mixture, not mean ellipsoids."""

import numpy as np
import pytest
from scipy.stats import chi2

from src.tasks.ball_refiner.refiner_3d.scoring import score_mixture
from src.utils.geometry.probabilistic_triangulation import GaussianMixture3D


def test_single_gaussian_hdr_threshold_matches_chi_squared():
    mixture = GaussianMixture3D(np.zeros((1, 3)), np.eye(3)[None], np.ones(1))
    score = score_mixture(mixture, np.array([1., 0., 0.]), samples=131072, seed=93608)
    assert score["nll_nat"] == pytest.approx(.5 * (1 + 3 * np.log(2 * np.pi)))
    assert score["mean_error_m"] == 1.
    for mass in (50, 90, 95):
        expected = -.5 * (chi2.ppf(mass / 100, 3) + 3 * np.log(2 * np.pi))
        assert score[f"hdr{mass}_log_density_threshold"] == pytest.approx(expected, abs=.04)
        assert score[f"hdr{mass}"]


def test_multimodal_hdr_excludes_low_density_gap_at_mixture_mean():
    mixture = GaussianMixture3D(np.array([[-5., 0., 0.], [5., 0., 0.]]), np.tile(np.eye(3) * .1, (2, 1, 1)), np.full(2, .5))
    score = score_mixture(mixture, np.zeros(3), samples=4096, seed=3)
    assert score["mean_error_m"] == 0
    assert not score["hdr95"]
    assert score == score_mixture(mixture, np.zeros(3), samples=4096, seed=3)
