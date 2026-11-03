from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.ball_gmm.heatmap import candidate_gaussians, fit_peak
from tests.support.ball3d_gmm import gmm_config, make_gmm_inputs


@pytest.mark.parametrize("covariance", [np.eye(2)*1.7, np.array([[2.5, .8], [.8, .9]])])
def test_joint_log_curvature_recovers_subpixel_and_rotated_covariance(covariance):
    y, x = np.mgrid[-2:3, -2:3]
    delta = np.stack((x-.23, y+.31), -1)
    patch = .83*np.exp(-.5*np.einsum("...i,ij,...j->...", delta, np.linalg.inv(covariance), delta))
    peak = fit_peak(patch, np.ones((5, 5), bool), config=gmm_config())
    assert peak is not None
    np.testing.assert_allclose(peak.mean_cells, [.23, -.31], atol=1e-12)
    np.testing.assert_allclose(peak.covariance_cells2, covariance, atol=1e-12)
    assert not peak.bounded


def test_pixel_covariance_grid_convention_and_unequal_candidate_counts():
    inputs, _ = make_gmm_inputs(missing=True)
    evidence = inputs.views[1].detections.evidence
    result = candidate_gaussians(evidence, gmm_config(covariance_scale=2.3))
    scale = np.array([319/160, 239/120])
    np.testing.assert_allclose(result.covariance_px2[0, 0], np.diag(np.array([1.3, .7])*np.square(scale)*2.3), atol=2e-6)
    np.testing.assert_allclose(result.means_px[0, 0], [122.222222222, 55.555555556], atol=2e-6)
    np.testing.assert_array_equal(result.indices[0], [0, -1, -1])
    np.testing.assert_array_equal(result.weights[0], [1, 0, 0])
    assert result.presence[2] == 0 and result.indices[2].max() == -1
    assert 0 < result.presence[0] < 1


def test_curvature_policy_rejects_flat_boundary_and_saddle_without_fallback():
    config = gmm_config()
    assert fit_peak(np.full((3, 3), .8), np.ones((3, 3), bool), config=config) is None
    y, x = np.mgrid[-1:2, -1:2]
    assert fit_peak(.3*np.exp((x*x-y*y)/4), np.ones((3, 3), bool), config=config) is None
    valid: NDArray[np.bool_] = np.ones((3, 3), bool)
    valid[:2] = False
    assert fit_peak(.8*np.exp(-(x*x+y*y)/2), valid, config=config) is None
    bounded = fit_peak(.8*np.exp(-(x*x+y*y)/200), np.ones((3, 3), bool), config=config)
    assert bounded is not None and bounded.bounded
    np.testing.assert_allclose(bounded.covariance_cells2, np.eye(2)*25, atol=1e-12)


@pytest.mark.parametrize("change", [dict(top_k=8), dict(covariance_scale=0), dict(max_components=True),
    dict(prior_std_m=(2, -1, 3)), dict(min_multiview_mass=0), dict(estimator="peak"),
    dict(temporal="beam"), dict(presence_bias=float("nan"))])
def test_invalid_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        replace(gmm_config(), **change)
