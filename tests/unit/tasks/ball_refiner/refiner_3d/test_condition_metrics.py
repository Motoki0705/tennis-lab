"""World-volume units, multi-mode HDR, and zero-weight component preservation."""
import numpy as np
import pytest
from scipy.stats import chi2

from src.tasks.ball_refiner.refiner_3d.condition_metrics import (
    LEVELS,
    mixture_hdr_metrics,
)
from src.utils.geometry.probabilistic_triangulation.distributions import (
    GaussianMixture3D,
)


def test_gaussian_hdr_matches_analytic_volume_and_nll() -> None:
    covariance = np.array([[4., .4, .2], [.4, 1., .1], [.2, .1, .25]])
    mixture = GaussianMixture3D(np.zeros((1, 3)), covariance[None], np.ones(1))
    result = mixture_hdr_metrics(mixture, np.zeros(3), samples=30000, seed=936)
    expected = 4 * np.pi / 3 * chi2.ppf(LEVELS, df=3)**1.5 * np.sqrt(np.linalg.det(covariance))
    np.testing.assert_allclose(result['volume_m3'], expected, rtol=.04)
    assert result['nll_nat'] == pytest.approx(-float(mixture.log_prob(np.zeros(3))))
    assert result['covered'].tolist() == [1., 1., 1.]
    assert (result['volume_mc_se_m3'] > 0).all()


def test_multimodal_volume_is_not_a_moment_ellipsoid() -> None:
    mixture = GaussianMixture3D(np.array([[-20., 0., 0.], [20., 0., 0.], [0., 0., 0.]]),
                                np.tile(np.eye(3), (3, 1, 1)), np.array([.5, .5, 0.]))
    result = mixture_hdr_metrics(mixture, np.zeros(3), samples=30000, seed=936)
    expected = 2 * 4 * np.pi / 3 * chi2.ppf(LEVELS, df=3)**1.5
    np.testing.assert_allclose(result['volume_m3'], expected, rtol=.04)
    assert result['covered'].tolist() == [0., 0., 0.]
    assert result['nll_nat'] == pytest.approx(-float(mixture.log_prob(np.zeros(3))))
    assert len(mixture.weights) == 3
