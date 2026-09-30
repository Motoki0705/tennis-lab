"""Independent limits for the research comparison baselines."""

import numpy as np
import pytest

from src.tasks.ball_refiner.refiner_3d.comparison import (
    VoxelConfig,
    triangulate_samples,
    triangulate_stratified_samples,
    triangulate_volume,
)
from src.utils.geometry.probabilistic_triangulation import CameraGMM, GaussianPrior3D
from src.utils.geometry.triangulation import PinholeCamera


def cameras():
    k = np.diag([10000.0, 10000.0, 1.0])
    return (
        PinholeCamera("xy", k, np.eye(3), np.array([0.0, 0.0, 10000.0])),
        PinholeCamera(
            "yz",
            k,
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            np.array([0.0, 0.0, 10000.0]),
        ),
    )


def test_voxel_density_normalizes_and_matches_analytic_gaussian():
    views = cameras()
    obs = CameraGMM(
        np.zeros((2, 1, 2)),
        np.tile(np.eye(2) * 0.25, (2, 1, 1, 1)),
        np.ones((2, 1)),
        np.ones(2),
    )
    result = triangulate_volume(
        obs,
        views,
        prior=GaussianPrior3D(np.zeros(3), np.eye(3)),
        config=VoxelConfig(12, 4, 512, 4.0),
    )
    density = result.components[0]
    assert density.weights.sum() == pytest.approx(1.0)
    # Cells cover the entire original box, including retained coarse tails.
    assert np.prod(density.widths, axis=1).sum() == pytest.approx(8**3)
    mean = density.weights @ density.centers
    second = np.einsum("n,ni,nj->ij", density.weights, density.centers, density.centers)
    second += np.diag(density.weights @ (density.widths**2 / 12))
    np.testing.assert_allclose(mean, 0, atol=0.001)
    np.testing.assert_allclose(second, np.diag([1 / 5, 1 / 9, 1 / 5]), atol=0.006)
    truth = np.array([0.1, 0.1, 0.1])
    analytic_logp = -0.5 * (
        np.sum(truth**2 * np.array([5, 9, 5])) + np.log(1 / 225) + 3 * np.log(2 * np.pi)
    )
    assert float(result.log_prob(truth)) == pytest.approx(analytic_logp, abs=0.15)
    samples = result.sample(1000, np.random.default_rng(3))
    assert np.isfinite(result.log_prob(samples)).all()
    assert float(result.log_prob(np.array([100.0, 0, 0]))) == -np.inf


def test_particle_linear_limit_has_known_mean_covariance():
    views = cameras()
    obs = CameraGMM(
        np.zeros((2, 1, 2)),
        np.tile(np.eye(2) * 0.25, (2, 1, 1, 1)),
        np.ones((2, 1)),
        np.ones(2),
    )
    result = triangulate_samples(
        obs,
        views,
        prior=GaussianPrior3D(np.zeros(3), np.eye(3)),
        samples=400,
        rng=np.random.default_rng(9),
        max_nfev=100,
    )
    # Test particles before KDE, whose documented bandwidth deliberately broadens.
    np.testing.assert_allclose(result.means.mean(0), 0, atol=0.08)
    np.testing.assert_allclose(
        np.cov(result.means.T), np.diag([1 / 5, 1 / 9, 1 / 5]), atol=0.045
    )
    assert np.isfinite(result.log_prob(np.zeros(3)))


def test_stratified_sampling_keeps_small_mass_products_and_exact_absence():
    views = cameras()
    obs = CameraGMM(np.zeros((2, 2, 2)), np.tile(np.eye(2) * .25, (2, 2, 1, 1)),
                    np.tile([1 - 1e-9, 1e-9], (2, 1)), np.array([.2, .7]))
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    kwargs = dict(prior=prior, samples_per_product=8, seed=93608, max_nfev=100)
    result = triangulate_stratified_samples(obs, views, **kwargs)
    assert len(result.distribution.weights) == 9
    assert (result.distribution.weights > 0).all()
    assert result.prior_only_probability == pytest.approx(.24)
    np.testing.assert_array_equal(result.distribution.covariance[0], prior.covariance)
    repeat = triangulate_stratified_samples(obs, views, **kwargs)
    np.testing.assert_array_equal(result.distribution.means, repeat.distribution.means)
    for active in np.unique(result.camera_subsets, axis=0):
        mass = result.distribution.weights[(result.camera_subsets == active).all(1)].sum()
        assert mass == pytest.approx(np.prod(np.where(active, obs.presence, 1 - obs.presence)))
