"""Fixed conditioning is deterministic and does not invent convergence evidence."""

import numpy as np
import pytest

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
)
from src.utils.geometry.probabilistic_triangulation.conditioning import (
    conditioning_config,
    triangulate_conditioning,
)
from src.utils.geometry.probabilistic_triangulation.solver import (
    HybridConfig,
    triangulate_hybrid,
)
from src.utils.geometry.triangulation import PinholeCamera


def test_fixed_budget_matches_the_compared_hybrid_and_keeps_all_products():
    cameras = (PinholeCamera("one", np.diag([10., 10., 1.]), np.eye(3), np.zeros(3)),)
    prior = GaussianPrior3D(np.array([0., 0., 2.]), np.diag([36., 144., 9.]))
    obs = CameraGMM(np.zeros((1, 4, 2)), np.tile(np.eye(2) * 100, (1, 4, 1, 1)), np.full((1, 4), .25), np.array([.8]))
    config = conditioning_config(dict(method="fixed_hybrid", initial_cells=8, levels=4, refine_cells=64, prior_sigmas=5.))
    laplace = LaplaceConfig(5, 100)
    result = triangulate_conditioning(obs, cameras, prior=prior, laplace=laplace, config=config)
    compared = triangulate_hybrid(obs, cameras, prior=prior, config=HybridConfig(laplace, config))
    assert not result.convergence_assessed and not result.converged
    assert result.rounds == 1 and result.history == ()
    assert not result.component_converged.any()
    assert len(result.posterior.distribution.weights) == 5
    for field in ("means", "covariance", "weights"):
        np.testing.assert_array_equal(getattr(result.posterior.distribution, field), getattr(compared.distribution, field))
    assert result.posterior.prior_only_probability == pytest.approx(.2)
    assert all(m.startswith("volume:") for m in result.posterior.component_methods[1:])


def test_unknown_conditioning_method_is_an_error():
    with pytest.raises(ValueError, match="Unknown integration method"):
        conditioning_config({"method": "auto"})


def test_fixed_budget_retains_known_linear_gaussian_moments():
    k = np.diag([10000., 10000., 1.])
    cameras = (PinholeCamera("one", k, np.eye(3), np.array([0., 0., 10000.])),)
    obs = CameraGMM(np.zeros((1, 1, 2)), np.eye(2)[None, None] * .25, np.ones((1, 1)), np.ones(1))
    config = conditioning_config(dict(method="fixed_hybrid", initial_cells=8, levels=4, refine_cells=64, prior_sigmas=5.))
    result = triangulate_conditioning(obs, cameras, prior=GaussianPrior3D(np.zeros(3), np.eye(3)), laplace=LaplaceConfig(1, 100), config=config)
    mean, covariance = result.posterior.distribution.moments()
    np.testing.assert_allclose(mean, 0., atol=1e-10)
    np.testing.assert_allclose(covariance, np.diag([.2, .2, 1.]), atol=1e-10)
