import numpy as np
import pytest
from src.utils.geometry.probabilistic_triangulation import GaussianPrior3D
from src.utils.geometry.triangulation import PinholeCamera


def test_ray_volume_boundary_limit_includes_jacobian_and_positive_depth():
    from scipy.integrate import quad

    from experimental_ray_volume import (
        RayVolumeConfig,
        integrate_ray_component,
    )

    focal = 10000.
    camera = PinholeCamera("a", np.diag([focal, focal, 1.]), np.eye(3), np.zeros(3))
    prior = GaussianPrior3D(np.array([0., 0., -.5]), np.eye(3))
    integrals = [quad(lambda d, order=order: d**order * np.exp(-.5*(d+.5)**2), 0, np.inf)[0] for order in (2, 3, 4)]
    expected_mean = integrals[1] / integrals[0]
    expected_variance = integrals[2] / integrals[0] - expected_mean**2
    mean, covariance, evidence = integrate_ray_component((camera,), np.zeros((1, 2)), np.eye(2)[None], prior, RayVolumeConfig(12, 96))
    np.testing.assert_allclose(mean, [0, 0, expected_mean], atol=.001)
    assert covariance[2, 2] == pytest.approx(expected_variance, abs=.002)
    assert evidence == pytest.approx(np.log(integrals[0]) - 1.5*np.log(2*np.pi) - 2*np.log(focal), abs=.0002)
    np.linalg.cholesky(covariance.astype(np.float32))


def test_ray_volume_linear_limit_and_source_resize():
    from experimental_ray_volume import (
        RayVolumeConfig,
        integrate_ray_component,
    )

    camera = PinholeCamera("a", np.diag([10000., 10000., 1.]), np.eye(3), np.array([0.,0.,10000.]))
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    means, cov = np.zeros((1,2)), np.eye(2)[None] * .25
    result = integrate_ray_component((camera,), means, cov, prior, RayVolumeConfig(16, 96))
    np.testing.assert_allclose(result[0], 0, atol=.001)
    np.testing.assert_allclose(result[1], np.diag([.2,.2,1.]), atol=.002)
    assert result[2] == pytest.approx(-np.log(2*np.pi*1.25), abs=.001)
    scale = np.array([2.,3.])
    resized = PinholeCamera("a", np.diag([*scale,1.]) @ camera.intrinsic, camera.rotation, camera.translation)
    changed = integrate_ray_component((resized,), means*scale, cov*scale[:,None]*scale[None,:], prior, RayVolumeConfig(16,96))
    np.testing.assert_allclose(result[0], changed[0], atol=1e-9)
    np.testing.assert_allclose(result[1], changed[1], atol=1e-9)
    assert changed[2] == pytest.approx(result[2] - np.log(6), abs=1e-9)
