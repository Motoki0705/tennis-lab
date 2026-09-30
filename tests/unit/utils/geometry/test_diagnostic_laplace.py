"""Approximation diagnostics must not remove components or change algorithms."""

import numpy as np
import pytest

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
    triangulate_gmm,
)
from src.utils.geometry.probabilistic_triangulation.optimization import (
    NonregularComponentError,
)
from src.utils.geometry.triangulation import PinholeCamera


@pytest.mark.parametrize("budget", [1, 100])
def test_positive_depth_approximation_keeps_broad_and_budget_limited_products(budget):
    cameras = tuple(PinholeCamera(str(i), np.diag([10., 10., 1.]), np.eye(3), np.array([float(i), 0., 0.])) for i in range(3))
    obs = CameraGMM(np.zeros((3, 4, 2)), np.tile(np.eye(2) * 100, (3, 4, 1, 1)),
                    np.full((3, 4), .25), np.array([.1, .8, .9]))
    prior = GaussianPrior3D(np.array([0., 0., 2.]), np.diag([36., 144., 9.]))
    with pytest.raises(NonregularComponentError):
        triangulate_gmm(obs, cameras, prior=prior, config=LaplaceConfig(125, budget))
    result = triangulate_gmm(obs, cameras, prior=prior, config=LaplaceConfig(125, budget, diagnose_nonregular=True))
    assert len(result.distribution.weights) == 125
    assert any(method.startswith("laplace_diagnostic:") for method in result.component_methods)
    for mask in np.unique(result.camera_subsets, axis=0):
        got = result.distribution.weights[(result.camera_subsets == mask).all(1)].sum()
        assert got == pytest.approx(np.prod(np.where(mask, obs.presence, 1 - obs.presence)))
    for point, active in zip(result.distribution.means, result.camera_subsets, strict=True):
        assert all(cameras[i].project(point)[1] for i in np.flatnonzero(active))
    if budget == 1:
        assert any(not d["optimizer_converged"] for d in result.component_optimization_diagnostics)
    np.linalg.cholesky(result.distribution.covariance)
    repeated = triangulate_gmm(obs, cameras, prior=prior, config=LaplaceConfig(125, budget, diagnose_nonregular=True))
    np.testing.assert_array_equal(repeated.distribution.means, result.distribution.means)


def test_diagnostic_policy_still_rejects_impossible_camera_intersection():
    cameras = (PinholeCamera("a", np.eye(3), np.eye(3), np.zeros(3)),
               PinholeCamera("b", np.eye(3), np.diag([1., -1., -1.]), np.zeros(3)))
    obs = CameraGMM(np.zeros((2, 1, 2)), np.tile(np.eye(2), (2, 1, 1, 1)), np.ones((2, 1)), np.ones(2))
    with pytest.raises(NonregularComponentError, match="no_feasible_initial_point"):
        triangulate_gmm(obs, cameras, prior=GaussianPrior3D(np.zeros(3), np.eye(3)), config=LaplaceConfig(1, 100, True))
