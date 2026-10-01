from __future__ import annotations

import numpy as np
import pytest

from src.tasks.player_association.geometry.affinity import (
    GeometryAffinityConfig,
    distance_log_likelihood_ratio,
    geometry_score,
    rayleigh_scale,
)
from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    GroundDistance,
    ground_distance,
    ground_footpoints,
)
from src.tasks.player_association.geometry.region import (
    PlayRegionConfig,
    in_play_region,
)
from src.tasks.player_association.geometry.switches import (
    SwitchConfig,
    footpoint_jumps,
    switch_candidates,
)
from src.utils.geometry.triangulation import PinholeCamera


def _camera() -> PinholeCamera:
    center = np.array([0.0, -20.0, 6.0])
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera("c", np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]]), rotation, -rotation @ center)


def test_box_bottom_centre_lands_on_the_ground_point_and_truncated_boxes_are_invalid() -> None:
    camera = _camera()
    feet = np.array([[1.0, 2.0, 0.0], [-3.0, 5.0, 0.0]])
    uv = camera.project(feet)[0]
    boxes = np.array([[u - 20, v - 120, u + 20, v] for u, v in uv])
    points, valid = ground_footpoints(boxes, np.array([True, True]), camera, 1080, FootpointConfig())
    assert valid.all()
    np.testing.assert_allclose(points, feet[:, :2], atol=1e-6)
    boxes[1, 3] = 1079.0  # bottom at the image border: feet cut off
    _, valid = ground_footpoints(boxes, np.array([True, True]), camera, 1080, FootpointConfig(bottom_border_px=4))
    assert valid.tolist() == [True, False]
    _, valid = ground_footpoints(boxes, np.array([False, True]), camera, 1080, FootpointConfig(bottom_border_px=0))
    assert valid.tolist() == [False, True]


def test_ground_distance_uses_only_shared_valid_frames() -> None:
    a = np.array([[0.0, 0], [0, 0], [0, 0], [9, 9]])
    b = np.array([[1.0, 0], [3, 0], [5, 0], [0, 0]])
    result = ground_distance(a, np.array([True, True, True, False]), b, np.array([True, True, True, True]))
    assert result.shared_frames == 3 and result.median_m == 3.0
    empty = ground_distance(a, np.zeros(4, bool), b, np.ones(4, bool))
    assert empty.shared_frames == 0 and np.isnan(empty.median_m)


def test_distance_evidence_changes_sign_where_the_two_models_are_equally_likely() -> None:
    config = GeometryAffinityConfig(sigma_m=1., area_m2=500., full_evidence_s=1., max_abs_score=10.)
    crossing = np.sqrt(2 * np.log(500. / (2 * np.pi)))
    assert distance_log_likelihood_ratio(crossing, config) == pytest.approx(0., abs=1e-9)
    assert distance_log_likelihood_ratio(.5, config) > 0 > distance_log_likelihood_ratio(crossing + .5, config)
    assert distance_log_likelihood_ratio(30., config) == -10.  # clipped
    # A median over half of the full-evidence time counts half; no shared frame is no evidence.
    assert geometry_score(GroundDistance(.5, 15), 30., config) == pytest.approx(distance_log_likelihood_ratio(.5, config) / 2)
    assert geometry_score(GroundDistance(float("nan"), 0), 30., config) == 0.


def test_rayleigh_scale_recovers_the_noise_of_same_person_distances() -> None:
    rng = np.random.default_rng(0)
    offsets = rng.normal(scale=.7, size=(20000, 2))
    assert rayleigh_scale(np.linalg.norm(offsets, axis=1)) == pytest.approx(.7, rel=.02)
    with pytest.raises(ValueError):
        rayleigh_scale(np.array([]))


def test_a_footpoint_jump_is_a_switch_candidate_and_running_is_not() -> None:
    frames = 120
    running = np.column_stack((np.linspace(-4, 4, frames), np.full(frames, 12.)))  # 8 m in 4 s at 30 fps
    valid: np.ndarray = np.ones(frames, bool)
    config = SwitchConfig(window_s=.25, max_jump_m=3.)
    assert switch_candidates(running, valid, 30., config) == []
    jumped = running.copy()
    jumped[70:] += [0., 6.]
    assert switch_candidates(jumped, valid, 30., config) == [70]
    # Without enough valid footpoints on one side, no jump is measured.
    sparse = valid.copy()
    sparse[60:70] = False
    sparse[70:80:3] = False
    jumps = footpoint_jumps(jumped, sparse, config.window_frames(30.))
    assert np.isnan(jumps[70]) and not np.isnan(jumps[40])


def test_play_region_is_the_doubles_court_with_margins() -> None:
    region = PlayRegionConfig(sideline_margin_m=2.5, baseline_margin_m=5.)
    points = np.array([[0., 0.], [7.9, 16.8], [8.1, 0.], [0., -17.], [-6., -14.]])
    assert in_play_region(points, region).tolist() == [True, True, False, False, True]
