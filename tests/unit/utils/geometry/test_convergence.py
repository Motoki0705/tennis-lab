"""Numerical stopping is separate from retaining the full posterior."""

from dataclasses import replace

import numpy as np
import pytest

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
)
from src.utils.geometry.probabilistic_triangulation.convergence import (
    ConvergenceConfig,
    triangulate_converged,
)
from src.utils.geometry.probabilistic_triangulation.solver import (
    HybridConfig,
    triangulate_hybrid,
)
from src.utils.geometry.triangulation import PinholeCamera


def fixture():
    camera = PinholeCamera("a", np.diag([10., 10., 1.]), np.eye(3), np.zeros(3))
    obs = CameraGMM(np.zeros((1, 1, 2)), np.eye(2)[None, None] * 100, np.ones((1, 1)), np.array([.8]))
    prior = GaussianPrior3D(np.array([0., 0., -.5]), np.eye(3))
    config = ConvergenceConfig((16, 24, 32), (6, 7, 8), (512, 1024, 2048), 4., .1, .03, .02, .1)
    return obs, (camera,), prior, config


def test_checked_boundary_matches_independent_last_grid_and_reuses_laplace(monkeypatch):
    from src.utils.geometry.probabilistic_triangulation import solver
    obs, cameras, prior, config = fixture()
    calls = []
    original = solver.fit_component
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(solver, "fit_component", counted)
    result = triangulate_converged(obs, cameras, prior=prior, laplace=LaplaceConfig(2, 100), config=config)
    assert len(calls) == 2  # prior + boundary; neither is optimized again
    assert result.converged and result.component_converged.all()
    assert 2 <= result.rounds <= 3
    expected = triangulate_hybrid(obs, cameras, prior=prior, config=HybridConfig(LaplaceConfig(2, 100), config.budget(result.rounds - 1)))
    np.testing.assert_array_equal(result.posterior.distribution.means, expected.distribution.means)
    np.testing.assert_array_equal(result.posterior.distribution.covariance, expected.distribution.covariance)
    assert result.posterior.distribution.weights == pytest.approx([.2, .8])


def test_cap_retains_nonconverged_tiny_mass_component_without_selection():
    obs, cameras, prior, config = fixture()
    obs = replace(obs, presence=np.array([1.e-12]))
    config = replace(config, nll_tolerance_nat=1.e-12, log_evidence_tolerance_nat=1.e-12, mean_tolerance=1.e-12, covariance_relative_tolerance=1.e-12)
    result = triangulate_converged(obs, cameras, prior=prior, laplace=LaplaceConfig(2, 100), config=config)
    assert not result.converged and result.rounds == 3
    assert len(result.history) == 2
    assert result.component_converged.tolist() == [True, False]
    assert result.posterior.distribution.weights == pytest.approx([1 - 1.e-12, 1.e-12], abs=1.e-20)
    expected = triangulate_hybrid(obs, cameras, prior=prior, config=HybridConfig(LaplaceConfig(2, 100), config.budget(2)))
    np.testing.assert_array_equal(result.posterior.distribution.means, expected.distribution.means)


@pytest.mark.parametrize("change", [{"initial_cells": (16,)}, {"levels": (6, 6, 8)}, {"prior_sigmas": float("nan")}, {"mean_tolerance": 0.}])
def test_bad_refinement_contract_fails(change):
    with pytest.raises(ValueError):
        replace(fixture()[3], **change)
