"""Known posterior moments, marginalization, nonlinear calibration and failures."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianMixture3D,
    GaussianPrior3D,
    LaplaceConfig,
    triangulate_gmm,
)
from src.utils.geometry.triangulation import PinholeCamera

CONFIG = LaplaceConfig(max_components=128, max_nfev=100)


def orthogonal_cameras(distance=10000.0):
    # At the origin these project (x,y) and (y,z), respectively, with unit scale.
    k = np.diag([distance, distance, 1.0])
    r = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    return (
        PinholeCamera("a", k, np.eye(3), np.array([0.0, 0.0, distance])),
        PinholeCamera("b", k, r, np.array([0.0, 0.0, distance])),
    )


def observations(cameras, point, sigma=0.2, presence=None):
    return CameraGMM(
        np.stack([c.project(point)[0] for c in cameras])[:, None],
        np.tile(np.eye(2) * sigma**2, (len(cameras), 1, 1, 1)),
        np.ones((len(cameras), 1)),
        np.ones(len(cameras)) if presence is None else presence,
    )


@pytest.mark.parametrize("diagnose_nonregular", [False, True])
def test_known_linear_limit_posterior_mean_and_covariance(diagnose_nonregular):
    cameras = orthogonal_cameras()
    prior = GaussianPrior3D(np.zeros(3), np.eye(3) * 4)
    obs = observations(cameras, np.zeros(3))
    obs.means_px[:, 0] = [[0.3, -0.1], [0.2, 0.5]]
    # Independent Gaussian conjugate posterior: x,z observed once, y twice.
    covariance = np.diag(1 / (np.array([25.0, 50.0, 25.0]) + 0.25))
    mean = covariance @ (25 * np.array([0.3, 0.1, 0.5]))
    result = triangulate_gmm(obs, cameras, prior=prior, config=replace(CONFIG, diagnose_nonregular=diagnose_nonregular))
    got_mean, got_cov = result.distribution.moments()
    np.testing.assert_allclose(got_mean, mean, atol=4e-5)
    np.testing.assert_allclose(got_cov, covariance, rtol=2e-4, atol=4e-6)


def test_prior_only_is_exact_and_does_not_invent_a_point():
    cameras = orthogonal_cameras()
    prior = GaussianPrior3D(np.array([1.0, -1.0, 2.0]), np.diag([4.0, 9.0, 16.0]))
    result = triangulate_gmm(
        observations(cameras, np.zeros(3), presence=np.zeros(2)),
        cameras,
        prior=prior,
        config=CONFIG,
    )
    mean, cov = result.distribution.moments()
    np.testing.assert_array_equal(mean, prior.mean)
    np.testing.assert_array_equal(cov, prior.covariance)
    assert result.prior_only_probability == 1
    assert not result.camera_subsets.any()


def test_presence_marginalization_preserves_every_subset_mass():
    cameras = orthogonal_cameras()
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    result = triangulate_gmm(
        observations(cameras, np.zeros(3), presence=np.array([0.2, 0.7])),
        cameras,
        prior=prior,
        config=CONFIG,
    )
    masses = {
        tuple(s): w
        for s, w in zip(result.camera_subsets, result.distribution.weights, strict=True)
    }
    assert masses == pytest.approx(
        {
            (False, False): 0.24,
            (False, True): 0.56,
            (True, False): 0.06,
            (True, True): 0.14,
        }
    )
    # One view retains prior depth variance, rather than a singular covariance.
    index = next(
        i for i, s in enumerate(result.camera_subsets) if list(s) == [True, False]
    )
    assert result.distribution.covariance[index, 2, 2] == pytest.approx(1.0)


def test_component_weights_include_evidence_and_laplace_volume():
    cameras = orthogonal_cameras()
    prior = GaussianPrior3D(np.zeros(3), np.eye(3) * 4)
    obs = CameraGMM(
        np.zeros((2, 2, 2)),
        np.tile(np.eye(2), (2, 2, 1, 1)) * np.array([0.04, 1.0])[None, :, None, None],
        np.tile([0.6, 0.4], (2, 1)),
        np.ones(2),
    )
    result = triangulate_gmm(obs, cameras, prior=prior, config=CONFIG)
    # Analytic evidence of the stacked linear observation, independent of solver.
    h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    expected = []
    for a in range(2):
        for b in range(2):
            noise = np.diag(
                [obs.covariance_px2[0, a, 0, 0]] * 2
                + [obs.covariance_px2[1, b, 0, 0]] * 2
            )
            expected.append(
                obs.weights[0, a]
                * obs.weights[1, b]
                / np.sqrt(np.linalg.det(h @ prior.covariance @ h.T + noise))
            )
    np.testing.assert_allclose(
        result.distribution.weights, np.array(expected) / sum(expected), rtol=1e-9
    )


def test_monte_carlo_known_posterior_calibration():
    rng = np.random.default_rng(936)
    cameras = orthogonal_cameras(distance=1000)
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    errors, predicted = [], []
    for _ in range(180):
        truth = rng.multivariate_normal(prior.mean, prior.covariance)
        obs = observations(cameras, truth, sigma=0.2)
        obs.means_px[:] += rng.normal(0, 0.2, obs.means_px.shape)
        result = triangulate_gmm(obs, cameras, prior=prior, config=CONFIG).distribution
        mean, cov = result.moments()
        errors.append(np.linalg.solve(np.linalg.cholesky(cov), mean - truth))
        predicted.append(cov)
    white = np.asarray(errors)
    np.testing.assert_allclose(white.mean(0), 0, atol=0.2)
    np.testing.assert_allclose(np.cov(white.T), np.eye(3), atol=0.24)
    assert 0.87 < np.mean(np.square(white).sum(1) <= 7.814727903) < 1.0


def test_mixture_moments_include_between_mode_uncertainty():
    mixture = GaussianMixture3D(
        np.array([[-2.0, 0, 0], [2.0, 0, 0]]),
        np.tile(np.eye(3), (2, 1, 1)),
        np.array([0.25, 0.75]),
    )
    mean, covariance = mixture.moments()
    np.testing.assert_allclose(mean, [1, 0, 0])
    np.testing.assert_allclose(covariance, np.diag([4, 1, 1]))
    points = mixture.sample(30000, np.random.default_rng(42))
    np.testing.assert_allclose(points.mean(0), mean, atol=0.035)
    np.testing.assert_allclose(np.cov(points.T), covariance, atol=0.06)
    assert np.isfinite(mixture.log_prob(points[:3])).all()


def test_invalid_covariance_camera_count_and_enumeration_fail():
    cameras = orthogonal_cameras()
    obs = observations(cameras, np.zeros(3))
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    with pytest.raises(ValueError, match="positive definite"):
        replace(obs, covariance_px2=np.zeros((2, 1, 2, 2)))
    with pytest.raises(ValueError, match="Distinct cameras"):
        triangulate_gmm(obs, (cameras[0], cameras[0]), prior=prior, config=CONFIG)
    with pytest.raises(ValueError, match="budget"):
        triangulate_gmm(
            replace(obs, presence=np.array([0.5, 0.5])),
            cameras,
            prior=prior,
            config=LaplaceConfig(1, 100),
        )
    with pytest.raises(RuntimeError, match="optimization failed"):
        triangulate_gmm(
            observations(cameras, np.ones(3)),
            cameras,
            prior=prior,
            config=LaplaceConfig(128, 1),
        )


def test_view_order_invariance_with_correlated_covariance():
    cameras = orthogonal_cameras()
    obs = observations(
        cameras, np.array([0.1, 0.4, 0.3]), presence=np.array([0.7, 0.9])
    )
    obs.covariance_px2[:, 0] = [[0.2, 0.08], [0.08, 0.1]]
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    a = triangulate_gmm(obs, cameras, prior=prior, config=CONFIG).distribution
    reverse = CameraGMM(
        obs.means_px[::-1],
        obs.covariance_px2[::-1],
        obs.weights[::-1],
        obs.presence[::-1],
    )
    b = triangulate_gmm(reverse, cameras[::-1], prior=prior, config=CONFIG).distribution
    np.testing.assert_allclose(
        a.log_prob(np.array([[0.0, 0.0, 0.0], [0.1, 0.4, 0.3]])),
        b.log_prob(np.array([[0.0, 0.0, 0.0], [0.1, 0.4, 0.3]])),
        atol=1e-9,
    )


def test_incompatible_modes_lose_weight_without_collapsing_valid_alternatives():
    cameras = orthogonal_cameras()
    prior = GaussianPrior3D(np.zeros(3), np.eye(3) * 4)
    means = np.array([[[0.0, -1.0], [0.0, 1.0]], [[-1.0, 0.0], [1.0, 0.0]]])
    obs = CameraGMM(
        means,
        np.tile(np.eye(2) * 0.01, (2, 2, 1, 1)),
        np.ones((2, 2)) * 0.5,
        np.ones(2),
    )
    result = triangulate_gmm(obs, cameras, prior=prior, config=CONFIG).distribution
    assert result.weights[1] < 1e-30
    assert result.weights[2] < 1e-30
    np.testing.assert_allclose(result.weights[[0, 3]], [0.5, 0.5], atol=1e-7)
    assert result.moments()[1][1, 1] > 0.99


def test_source_pixel_rescaling_preserves_3d_density():
    cameras = orthogonal_cameras()
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    obs = observations(
        cameras, np.array([0.3, 0.2, 0.1]), presence=np.array([0.8, 0.5])
    )
    before = triangulate_gmm(obs, cameras, prior=prior, config=CONFIG).distribution
    scale = np.array([2.0, 3.0])
    transformed = tuple(
        PinholeCamera(
            c.camera_id, np.diag([*scale, 1.0]) @ c.intrinsic, c.rotation, c.translation
        )
        for c in cameras
    )
    changed = CameraGMM(
        obs.means_px * scale,
        obs.covariance_px2 * scale[:, None] * scale[None, :],
        obs.weights,
        obs.presence,
    )
    after = triangulate_gmm(
        changed, transformed, prior=prior, config=CONFIG
    ).distribution
    np.testing.assert_allclose(before.weights, after.weights, atol=1e-12)
    np.testing.assert_allclose(before.means, after.means, atol=1e-10)
    np.testing.assert_allclose(before.covariance, after.covariance, atol=1e-10)


def test_float32_weight_roundoff_remains_sampleable():
    weights = np.array([0.7, 0.3], dtype=np.float32).astype(np.float64)
    mixture = GaussianMixture3D(
        np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), weights
    )
    assert np.isfinite(mixture.sample(8, np.random.default_rng(1))).all()
    observations = CameraGMM(
        np.zeros((1, 2, 2)), np.tile(np.eye(2), (1, 2, 1, 1)), weights[None], np.ones(1)
    )
    assert float(observations.weights.sum()) == pytest.approx(1.0, abs=1e-15)
    with pytest.raises(ValueError, match="sum to one"):
        replace(observations, weights=np.array([[0.6, 0.3]]))
