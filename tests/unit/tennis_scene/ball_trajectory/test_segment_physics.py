from dataclasses import replace

import numpy as np
import pytest

from src.tennis_scene.ball_trajectory.fit import _heatmap_residual, fit_trajectory
from src.tennis_scene.ball_trajectory.physics import flight_states
from src.tennis_scene.ball_trajectory.segments import detect_segments
from tests.support.ball3d_segment_physics import (
    segment_physics_config,
    synthetic_flights,
)


@pytest.fixture(scope="module")
def flights():
    return synthetic_flights()


def test_gravity_and_quadratic_drag_use_metres_seconds_and_z_up():
    # Independent analytic solutions: ballistic vertical motion and horizontal
    # drag at t=0 (finite derivative), with unequal components and signed time.
    initial = np.tile([2.0, -3.0, 5.0, 4.0, -7.0, 8.0], (2, 1))
    times = np.array([0.3, -0.1])
    result = flight_states(initial, times, 0.0, gravity_mps2=9.81, steps=48)
    np.testing.assert_allclose(
        result[:, :3], [[3.2, -5.1, 6.95855], [1.6, -2.3, 4.15095]], atol=1e-10
    )
    tiny = flight_states(
        initial[:1], np.array([1e-6]), 0.023, gravity_mps2=9.81, steps=48
    )
    np.testing.assert_allclose(
        (tiny[0, 3:] - initial[0, 3:]) / 1e-6,
        [-1.04491914, 1.82860849, -11.89983827],
        atol=1e-4,
    )


def test_detect_bounce_and_hit_without_crossing_long_missing_run(flights):
    _, _, initial, fps, _, _, bounce = flights
    config = segment_physics_config()
    segments, events = detect_segments(initial.positions, initial.valid, fps, config)
    assert [(e.frame, e.kind) for e in events] == [(bounce, "bounce")]
    assert [(s.start_frame, s.stop_frame) for s in segments] == [
        (0, bounce),
        (bounce, 76),
    ]
    xyz = initial.positions.copy()
    xyz[55:] += (np.arange(21) / fps)[:, None] * [-22, -12, 2]
    valid = initial.valid.copy()
    valid[12:19] = False
    xyz[~valid] = 10000  # Invalid coordinates must not affect event detection.
    segments, events = detect_segments(xyz, valid, fps, config)
    assert any(e.frame == 55 and e.kind == "hit" for e in events)
    assert not any(s.start_frame < 12 and s.stop_frame > 19 for s in segments)


def test_joint_fit_recovers_signed_offsets_drag_bounce_and_camera_reordering(flights):
    cameras, evidence, initial, fps, drag, offsets, bounce = flights
    # Real asymmetric support identifies incorrectly restored camera masks and
    # residual metadata, not just the scalar offsets.
    support = initial.inliers.copy()
    support[0, 11] = support[1, 14] = support[2, 24] = False
    initial = replace(initial, inliers=support)
    fit = fit_trajectory(
        cameras, evidence, initial, fps=fps, config=segment_physics_config()
    )
    assert fit.diagnostics["status"] == "ok", fit.diagnostics
    np.testing.assert_allclose(
        fit.time_offsets_seconds * fps, offsets * fps, atol=0.008
    )
    assert fit.diagnostics["drag_per_m"] == pytest.approx(drag, abs=0.0002)
    assert fit.diagnostics["events"][0]["frame"] == bounce
    assert fit.diagnostics["events"][0]["vertical_restitution"] == pytest.approx(
        0.73, abs=0.002
    )
    np.testing.assert_allclose(fit.positions, initial.positions, atol=0.001)
    assert fit.diagnostics["offset_covariance_seconds2"].shape == (3, 3)
    order = [2, 0, 1]
    permuted = replace(
        initial,
        inliers=initial.inliers[order],
        reprojection_px=initial.reprojection_px[order],
    )
    reordered = fit_trajectory(
        tuple(cameras[i] for i in order),
        tuple(evidence[i] for i in order),
        permuted,
        fps=fps,
        config=segment_physics_config(),
    )
    # Exact equality tests canonical solver inputs; no tolerance increase.
    np.testing.assert_array_equal(
        reordered.time_offsets_seconds, fit.time_offsets_seconds[order]
    )
    np.testing.assert_array_equal(reordered.positions, fit.positions)
    np.testing.assert_array_equal(reordered.used_views, fit.used_views[order])
    np.testing.assert_array_equal(
        reordered.diagnostics["offset_covariance_seconds2"],
        fit.diagnostics["offset_covariance_seconds2"][np.ix_(order, order)],
    )
    for index, excluded_frame in enumerate([24, 11, 14]):
        diag = reordered.diagnostics
        assert (
            excluded_frame
            not in diag["sample_frames"][diag["sample_camera_indices"] == index]
        )
    assert reordered.diagnostics["camera_ids"] == ["far", "right", "near"]


def test_dense_heatmap_fit_is_independent_of_candidate_coordinates_and_keeps_gaps(
    flights,
):
    cameras, evidence, initial, fps, _, _, _ = flights
    valid = initial.valid.copy()
    valid[48] = False
    initial = replace(initial, valid=valid, inliers=initial.inliers & valid[None])
    config = segment_physics_config(observation="heatmap")
    fit = fit_trajectory(cameras, evidence, initial, fps=fps, config=config)
    assert fit.diagnostics["status"] == "ok", fit.diagnostics
    assert not fit.valid[48] and (fit.positions[48] == 0).all()
    assert (
        np.median(
            np.linalg.norm(
                fit.positions[fit.valid] - initial.positions[fit.valid], axis=1
            )
        )
        < 0.08
    )
    # Actual likelihood changes with the map even if candidates stay identical.
    uv = evidence[0].candidate_uv_px[:2, 0].astype(np.float64) + [2.0, -1.0]
    a = _heatmap_residual(evidence[0], np.arange(2), uv, 6.0)
    altered = replace(
        evidence[0],
        candidate_uv_px=evidence[0].candidate_uv_px + np.array([0.001, 0], np.float32),
    )
    np.testing.assert_array_equal(a, _heatmap_residual(altered, np.arange(2), uv, 6.0))
    assert np.all(a[:, 0] > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"offset_bound_frames": 1.0},
        {"gravity_mps2": float("nan")},
        {"max_nfev": True},
        {"min_segment_frames": 3},
        {"max_drag_per_m": 0.001},
        {"observation": "unknown"},
    ],
)
def test_invalid_solver_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        segment_physics_config(**kwargs)


def test_failed_optimizer_has_explicit_status_and_no_baseline_fallback(flights):
    cameras, evidence, initial, fps, *_ = flights
    fit = fit_trajectory(
        cameras, evidence, initial, fps=fps, config=segment_physics_config(max_nfev=1)
    )
    assert fit.diagnostics["status"] == "optimization_failed"
    assert not fit.valid.any() and not fit.positions.any() and not fit.used_views.any()
    with pytest.raises(ValueError, match="axes"):
        fit_trajectory(
            cameras,
            evidence,
            replace(initial, inliers=initial.inliers[:2]),
            fps=fps,
            config=segment_physics_config(),
        )
