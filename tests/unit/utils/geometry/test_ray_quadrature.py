"""Independent radial integrals, all-component retention and refinement flags."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
)
from src.utils.geometry.probabilistic_triangulation.convergence import (
    RayConvergenceConfig,
    convergence_config,
    triangulate_converged,
)
from src.utils.geometry.probabilistic_triangulation.ray import (
    RayProposal,
    single_view_moments,
)
from src.utils.geometry.triangulation import PinholeCamera


@pytest.mark.parametrize('sigma', [.01, .3])
def test_single_view_matches_independent_polar_integral(sigma):
    camera = PinholeCamera('a', np.eye(3), np.eye(3), np.zeros(3))
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    mean, cov, log_z = single_view_moments(camera, np.zeros(2), np.eye(2) * sigma**2, prior, 64)
    def expectation(power):
        return quad(lambda radius: radius / sigma**2 * np.exp(-radius**2 / (2*sigma**2)) * (1+radius**2)**(-power), 0, np.inf, epsabs=1e-12)[0]
    norm = expectation(1.5)
    expected_z = 2 * np.sqrt(2/np.pi) * expectation(2) / norm
    expected_zz = 3 * expectation(2.5) / norm
    expected_xx = 1.5 * (expectation(1.5)-expectation(2.5)) / norm
    np.testing.assert_allclose(mean, [0, 0, expected_z], atol=1e-9)
    np.testing.assert_allclose(cov, np.diag([expected_xx, expected_xx, expected_zz-expected_z**2]), atol=1e-9)
    assert log_z == pytest.approx(np.log(norm / (4*np.pi)), abs=1e-10)


def test_multiview_ray_integrates_product_and_cheirality():
    cameras = tuple(PinholeCamera(name, np.diag([10., 10., 1.]), np.eye(3), np.zeros(3)) for name in ('a', 'b'))
    prior = GaussianPrior3D(np.array([0., 0., -.5]), np.eye(3))
    means = np.zeros((2, 2))
    covariances = np.tile(np.eye(2)*4, (2, 1, 1))
    expected = single_view_moments(cameras[0], means[0], covariances[0]/2, prior, 64)
    proposal = RayProposal(cameras, means, covariances, prior)
    mean, covariance, log_z = proposal.integrate(64)
    np.testing.assert_allclose(mean, expected[0], atol=2e-5)
    np.testing.assert_allclose(covariance, expected[1], atol=2e-5)
    assert log_z == pytest.approx(expected[2]-np.log(16*np.pi), abs=2e-5)
    assert mean[2] > 0


def test_ray_checks_keep_tiny_components_and_cap():
    camera = PinholeCamera('a', np.eye(3), np.eye(3), np.zeros(3))
    prior = GaussianPrior3D(np.zeros(3), np.eye(3))
    obs = CameraGMM(np.zeros((1,1,2)), np.eye(2)[None,None]*.8, np.ones((1,1)), np.array([1e-12]))
    config = RayConvergenceConfig((12,20,32), .05, .05, .02, .05)
    result = triangulate_converged(obs, (camera,), prior=prior, laplace=LaplaceConfig(2,100), config=config)
    assert len(result.posterior.distribution.weights) == 2
    assert result.posterior.distribution.weights[1] == pytest.approx(1e-12, abs=1e-25)
    strict = replace(config, nll_tolerance_nat=1e-15, log_evidence_tolerance_nat=1e-15, mean_tolerance=1e-15, covariance_relative_tolerance=1e-15)
    checked = triangulate_converged(obs, (camera,), prior=prior, laplace=LaplaceConfig(2,100), config=strict)
    assert not checked.converged and checked.rounds == 3
    assert not checked.component_converged[1]
    assert checked.posterior.component_methods[1].startswith('ray:')
    np.testing.assert_array_equal(checked.posterior.distribution.means, result.posterior.distribution.means)


def test_unknown_integrator_and_bad_orders_are_explicit_errors():
    with pytest.raises(ValueError, match='Unknown'):
        convergence_config({'method':'typo'})
    with pytest.raises(ValueError, match='increasing'):
        RayConvergenceConfig((12,12,32), .05,.05,.02,.05)


def test_bounded_depth_gradient_matches_independent_finite_difference():
    cameras = (PinholeCamera('a', np.eye(3), np.eye(3), np.zeros(3)),
        PinholeCamera('b',np.eye(3),np.diag([-1.,1.,-1.]),np.array([.2,0.,2.])))
    proposal = RayProposal(cameras,np.zeros((2,2)),np.tile(np.eye(2)*.1,(2,1,1)),
        GaussianPrior3D(np.array([0.,0.,1.]),np.eye(3)))
    coordinates = proposal.mode + np.array([.2,-.4,.2])
    axes = np.eye(3)*1e-6
    numerical = np.array([(proposal.cost(coordinates+x)-proposal.cost(coordinates-x))/2e-6 for x in axes])
    _, analytic = proposal.cost_gradient(coordinates)
    np.testing.assert_allclose(analytic,numerical,rtol=2e-5,atol=2e-6)


def test_camera_chart_change_transports_the_pilot_mode():
    from pathlib import Path

    # Fixed synthetic smoke test-00000/frame252, source SHA
    # 5ea82ed055995148461e5c6ea9646f5f324d711cd15dcd113c404092da75baee.
    # No GT is used. Reusing an independently fitted mode after changing charts
    # caused a nonpositive Hessian for one retained product in this frame.
    with np.load(Path(__file__).parent/'fixtures/ray_camera_boundary.npz') as data:
        obs=CameraGMM(data['means'],data['covariance'],data['weights'],data['presence'])
        cameras=tuple(PinholeCamera(str(i),data['K'][i],data['R'][i],data['t'][i]) for i in range(3))
    result=triangulate_converged(obs,cameras,prior=GaussianPrior3D(np.array([0.,0.,2.]),np.diag([36.,144.,9.])),
        laplace=LaplaceConfig(64,100),config=RayConvergenceConfig((12,20,32,48,64),.05,.05,.02,.05))
    assert result.posterior.distribution.means.shape==(64,3)
    np.linalg.cholesky(result.posterior.distribution.covariance)
    for mask in np.unique(result.posterior.camera_subsets,axis=0):
        selected=(result.posterior.camera_subsets==mask).all(-1)
        assert result.posterior.distribution.weights[selected].sum()==pytest.approx(np.prod(np.where(mask,obs.presence,1-obs.presence)))
