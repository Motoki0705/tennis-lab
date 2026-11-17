"""Known motion, missing directions and contradictory stripes, independent of H fitting."""

from dataclasses import replace

import numpy as np
import pytest

from src.tasks.court_detection.geometry.paint_observations import PaintLine
from src.tasks.court_detection.geometry.video_drift.motion import (
    DriftConfig,
    solve_offsets,
    summarize_clip,
)

WORLD = np.array([[-5.485, 11.885], [5.485, 11.885], [-5.485, -11.885], [5.485, -11.885],
                  [-4.115, 11.885], [-4.115, -11.885], [4.115, 11.885], [4.115, -11.885],
                  [-4.115, 6.4], [4.115, 6.4], [-4.115, -6.4], [4.115, -6.4], [0, 6.4], [0, -6.4]])
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
H = np.array([[75., 6., 960.], [2.25, -27., 540.], [.001, -.001, 1.]])


def project(h, xy=WORLD):
    p = np.c_[xy, np.ones(len(xy))] @ h.T
    return p[:, :2] / p[:, 2:]


def observations(transform, *, indices=range(9), count=32):
    result = []
    for index in indices:
        a, b = WORLD[np.asarray(EDGES[index])]
        points = project(transform @ H, a + np.linspace(.05, .95, count)[:, None] * (b - a))
        result.append(PaintLine(index, 'ok', points, None, count, .9))
    return tuple(result)


@pytest.mark.parametrize('shift,status', [(0, 'ok'), (5, 'warning'), (11, 'failed')])
def test_translation_known_without_replacing_shared_h(shift, status):
    original = H.copy()
    t = np.array([[1., 0, shift], [0, 1, 0], [0, 0, 1]])
    report = solve_offsets(H, observations(t), (1920, 1080), DriftConfig())
    assert report['status'] == status, report
    assert report['mean_displacement_px'] == pytest.approx(shift, abs=1e-4)
    assert report['max_displacement_px'] == pytest.approx(shift, abs=1e-4)
    assert report['max_visible_projection_std_px'] < 2
    np.testing.assert_array_equal(H, original)


def test_rotation_and_scale_propagate_covariance_to_visible_keypoints():
    angle = .004
    t = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    t[:2, :2] *= 1.008
    t[:2, 2] = [960, 540] - t[:2, :2] @ [960, 540]
    report = solve_offsets(H, observations(t), (1920, 1080), DriftConfig())
    expected = project(t @ H) - project(H)
    np.testing.assert_allclose(report['keypoint_displacements_px'], expected, atol=1e-4)
    assert report['mean_displacement_px'] == pytest.approx(np.linalg.norm(expected, axis=1).mean())
    assert min(report['keypoint_std_px']) > 0


def test_dense_pixels_do_not_manufacture_independent_constraints():
    a = solve_offsets(H, observations(np.eye(3), count=32), (1920, 1080), DriftConfig())
    b = solve_offsets(H, observations(np.eye(3), count=320), (1920, 1080), DriftConfig())
    np.testing.assert_allclose(a['parameter_covariance'], b['parameter_covariance'], atol=1e-5)


@pytest.mark.parametrize('indices', [(), (2, 3, 4, 5, 8), (0, 1, 6, 7)])
def test_missing_two_families_are_unobservable_not_safe(indices):
    report = solve_offsets(H, observations(np.eye(3), indices=indices), (1920, 1080), DriftConfig())
    assert report['status'] == 'failed' and report['reason'] == 'drift_unobservable'
    assert report['mean_displacement_px'] is None


def test_one_conflicting_line_cannot_be_silently_discarded():
    lines = list(observations(np.eye(3)))
    lines[0] = replace(lines[0], points=lines[0].points + [0, 15])
    report = solve_offsets(H, tuple(lines), (1920, 1080), DriftConfig())
    assert report['status'] == 'failed'
    assert report['geometry_reasons'] == ['non_similarity_or_conflicting_line']


def test_competing_parallel_stripes_are_not_chosen_by_zero_motion_bias():
    lines = observations(np.eye(3))
    ambiguous = tuple(replace(line, alternatives=(line.points + [8, 5],)) for line in lines)
    report = solve_offsets(H, ambiguous, (1920, 1080), DriftConfig())
    assert report['status'] == 'failed'
    assert report['geometry_reasons'] == ['ambiguous_line_offsets']


def test_non_similarity_motion_is_not_reported_safe():
    t = np.array([[1.03, .04, -40], [0, 1, 0], [0, 0, 1.]])
    report = solve_offsets(H, observations(t), (1920, 1080), DriftConfig())
    assert report['status'] == 'failed'


def test_worst_frame_cannot_be_hidden_by_four_static_frames():
    frames = []
    for frame, shift in enumerate([0, 0, 11, 0, 0]):
        report = solve_offsets(H, observations(np.array([[1., 0, shift], [0, 1, 0], [0, 0, 1]])), (1920, 1080), DriftConfig())
        frames.append({'frame_index': frame, **report})
    clip = summarize_clip(frames)
    assert clip['status'] == 'failed' and clip['worst_frame_index'] == 2
    assert not clip['temporal_pooling']
    frames[0].update(status='failed', reason='drift_unobservable')
    assert summarize_clip(frames)['reason'] == 'drift_unobservable'
