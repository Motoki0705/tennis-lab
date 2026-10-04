"""Adaptive cubature checks against an independent analytic radial reference."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
)
from src.utils.geometry.probabilistic_triangulation.adaptive import (
    AdaptiveRayConfig,
    AdaptiveRayProposal,
)
from src.utils.geometry.probabilistic_triangulation.convergence import (
    AdaptiveRayConvergenceConfig,
    triangulate_converged,
)
from src.utils.geometry.probabilistic_triangulation.ray import (
    RayProposal,
    single_view_moments,
)
from src.utils.geometry.triangulation import PinholeCamera


def fixture():
    cameras = tuple(PinholeCamera(name, np.diag([10., 10., 1.]), np.eye(3), np.zeros(3)) for name in ('a', 'b'))
    prior = GaussianPrior3D(np.array([0., 0., -.5]), np.eye(3))
    means = np.zeros((2, 2))
    covariance = np.tile(np.eye(2) * 4, (2, 1, 1))
    return cameras, prior, means, covariance


def test_adaptive_positive_depth_moments_and_evidence():
    cameras, prior, means, covariance = fixture()
    expected = single_view_moments(cameras[0], means[0], covariance[0] / 2, prior, 64)
    proposal = AdaptiveRayProposal(RayProposal(cameras, means, covariance, prior, adaptive_metric=True))
    mean, cov, log_z = proposal.integrate(AdaptiveRayConfig(32, 1e-4, 2048))
    np.testing.assert_allclose(mean, expected[0], atol=2e-4)
    np.testing.assert_allclose(cov, expected[1], atol=2e-4)
    assert log_z == pytest.approx(expected[2] - np.log(16 * np.pi), abs=2e-4)
    assert proposal.diagnostic['embedded_converged']
    before = proposal.evaluations
    proposal.integrate(AdaptiveRayConfig(48, 5e-5, 4096))
    assert proposal.evaluations > before


def test_cap_and_tiny_components_remain_flagged():
    cameras, prior, means, covariance = fixture()
    observations = CameraGMM(means[:, None], covariance[:, None], np.ones((2, 1)), np.array([1e-12, 1e-12]))
    config = AdaptiveRayConvergenceConfig((12, 20, 32), 1e-14, 1e-14, 1e-14, 1e-14, (1e-6, 1e-7, 1e-8), (8, 16, 24))
    result = triangulate_converged(observations, cameras, prior=prior, laplace=LaplaceConfig(4, 100), config=config)
    assert len(result.component_converged) == 4
    assert not result.converged and result.rounds == 3
    assert not result.component_converged[-1]
    assert result.posterior.distribution.weights[-1] == pytest.approx(1e-24, abs=1e-35)
    assert result.posterior.component_methods[-1].startswith('adaptive_ray:')
    assert not result.posterior.component_integration_diagnostics[-1]['embedded_converged']
    with pytest.raises(ValueError, match='decrease'):
        replace(config, relative_errors=(.01, .02, .003))


def test_dataset_requires_finite_embedded_diagnostic_without_changing_outer_rule():
    from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import (
        _validate_integration,
    )

    rule = dict(method='adaptive_ray', orders=[12, 20, 32], nll_tolerance_nat=.05, log_evidence_tolerance_nat=.05, mean_tolerance=.02, covariance_relative_tolerance=.05,
                relative_errors=[.01, .003, .001], max_cells=[8, 16, 24])
    arrays = dict(integration_component_changes=np.zeros((1, 1, 3)), integration_component_converged=np.ones((1, 1), bool),
                  integration_nll_delta_nat=np.zeros(1), integration_converged=np.ones(1, bool), integration_rounds=np.array([3]), integration_component_embedded_error=np.array([[.1]]), integration_component_metric_codes=np.array([[2]],dtype=np.uint8))
    record = dict(frames=1, integration=dict(rule=rule, converged_frames=1, nonconverged_frames=0))
    _validate_integration(arrays, record, dict(degradation=dict(boundary_convergence=rule)), 1)
    arrays['integration_component_metric_codes'][:] = 3
    with pytest.raises(ValueError, match='chart'):
        _validate_integration(arrays, record, dict(degradation=dict(boundary_convergence=rule)), 1)
    arrays['integration_component_metric_codes'][:] = 2
    arrays['integration_component_embedded_error'][:] = float('nan')
    with pytest.raises(ValueError, match='embedded'):
        _validate_integration(arrays, record, dict(degradation=dict(boundary_convergence=rule)), 1)


def test_saddle_chart_is_explicit_and_does_not_repair_covariance():
    # Pre-registered dev-r5 train-00003/frame257/component123 (combination 3,3,2).
    # The baseline failed at the full-Hessian factorization; no GT is included.
    with np.load(Path(__file__).parent / 'fixtures/adaptive_ray_saddle.npz') as data:
        cameras = tuple(PinholeCamera(str(i), data['K'][i], data['R'][i], data['t'][i]) for i in range(3))
        ray = RayProposal(cameras, data['means'], data['covariance'], GaussianPrior3D(np.array([0., 0., 2.]), np.diag([36., 144., 9.])), adaptive_metric=True)
    assert not ray.metric_is_local_hessian
    proposal = AdaptiveRayProposal(ray)
    mean, covariance, evidence = proposal.integrate(AdaptiveRayConfig(32, .003, 256))
    np.linalg.cholesky(covariance)
    assert np.isfinite(evidence)
    assert all((camera.rotation @ mean + camera.translation)[2] > 0 for camera in cameras)
    assert not proposal.diagnostic['metric_is_local_hessian']
