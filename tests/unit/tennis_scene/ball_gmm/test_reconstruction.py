from dataclasses import replace

import numpy as np
import pytest

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.reconstruction import (
    point_estimate,
    reconstruct_candidates,
)
from tests.support.ball3d_gmm import make_gmm_inputs


def test_asymmetric_cameras_recover_trajectory_and_preserve_missing_evidence():
    inputs, xyz = make_gmm_inputs()
    config = CandidateGMMConfig(presence_bias=8)
    trajectory = reconstruct_candidates(tuple(v.detections for v in inputs.views),
        tuple(v.camera for v in inputs.views), config=config)
    np.testing.assert_array_equal(trajectory.valid, [True, True, False, True, True])
    np.testing.assert_allclose(trajectory.map_positions[trajectory.valid], xyz[trajectory.valid], atol=.002)
    np.testing.assert_allclose(trajectory.mean_positions[trajectory.valid], xyz[trajectory.valid], atol=.002)
    assert (trajectory.diagnostics["multiview_probability"][trajectory.valid] > .99).all()
    for temporal in ("kalman", "rts"):
        points, covariance, used = point_estimate(trajectory, config=replace(config, temporal=temporal), fps=inputs.fps)
        np.testing.assert_array_equal(points[2], 0)
        np.testing.assert_array_equal(covariance[2], 0)
        np.testing.assert_array_equal(used[:, 2], False)


def test_map_and_mixture_mean_keep_distractor_uncertainty_and_view_order():
    inputs, xyz = make_gmm_inputs(frames=2, distractors=True, missing=False)
    detections, cameras = tuple(v.detections for v in inputs.views), tuple(v.camera for v in inputs.views)
    config = CandidateGMMConfig()
    a = reconstruct_candidates(detections, cameras, config=config)
    b = reconstruct_candidates(detections[::-1], cameras[::-1], config=config)
    assert a.valid.all()
    np.testing.assert_allclose(a.map_positions, xyz, atol=.002)
    np.testing.assert_allclose(a.mean_positions, b.mean_positions, atol=1e-8)
    np.testing.assert_allclose(a.mean_covariance, b.mean_covariance, atol=1e-8)
    assert np.max(np.abs(a.map_positions-a.mean_positions)) > .01
    assert np.trace(a.mean_covariance[0]) > np.trace(a.map_covariance[0])


def test_axis_mismatch_and_component_budget_fail_explicitly():
    inputs, _ = make_gmm_inputs(frames=2, missing=False)
    detections, cameras = tuple(v.detections for v in inputs.views), tuple(v.camera for v in inputs.views)
    with pytest.raises(ValueError, match="axes"):
        reconstruct_candidates(detections[::-1], cameras, config=CandidateGMMConfig())
    with pytest.raises(ValueError, match="budget"):
        reconstruct_candidates(detections, cameras, config=CandidateGMMConfig(max_components=1))


def test_prior_single_view_mass_and_optimizer_failure_never_become_observations(monkeypatch):
    inputs, _ = make_gmm_inputs(frames=2, missing=False)
    detections, cameras = tuple(v.detections for v in inputs.views), tuple(v.camera for v in inputs.views)
    result = reconstruct_candidates(detections, cameras, config=CandidateGMMConfig(presence_bias=-20))
    assert not result.valid.any() and not result.map_views.any()
    np.testing.assert_array_equal(result.map_positions, 0)

    def fail(*args, **kwargs):
        raise RuntimeError("injected numerical failure")

    monkeypatch.setattr("src.tennis_scene.ball_gmm.reconstruction.triangulate_gmm", fail)
    failed = reconstruct_candidates(detections, cameras, config=CandidateGMMConfig())
    assert not failed.valid.any()
    assert failed.diagnostics["solver_failures"] == {"0": "injected numerical failure", "1": "injected numerical failure"}
