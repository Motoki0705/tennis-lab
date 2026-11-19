"""Decide camera parity from synchronized ground positions and explicit masks."""

from dataclasses import replace

import numpy as np
import pytest

from src.tasks.court_side.hypothesis import (
    AMBIGUOUS_MARGIN,
    DISCONNECTED_VIEWS,
    INSUFFICIENT_FRAMES,
    NO_CONSISTENT_HYPOTHESIS,
    CourtSideUndecided,
    PlayerCourtSideConfig,
    judge_side_evidence,
)
from src.tasks.court_side.player import PlayerFootpoints, collect_player_side_evidence

CONFIG = PlayerCourtSideConfig(match_distance_m=2.5, symmetry_distance_m=.5, min_motion_m=.15,
                               bottom_border_px=4., min_frames=3, max_cost=.7, min_support=.2, min_margin=.05)


def view(points, visible=None):
    xy = np.asarray(points, np.float64)
    return PlayerFootpoints(xy, np.ones(xy.shape[:-1], bool) if visible is None else np.asarray(visible, bool))


@pytest.mark.parametrize("turned", [False, True])
def test_asymmetric_sets_recover_the_half_turn_with_masks_and_unordered_points(turned):
    first = np.array([[[1. + .3 * t, -7.], [-4., 3.]] for t in range(5)])
    second = first[:, ::-1].copy()
    second[:, 0, :] = 999.  # Masked row must add no cost or symmetry.
    if turned:
        second *= -1
    evidence, frames = collect_player_side_evidence(
        ("b", "reference", "c"), "reference",
        (view(second, [[False, True]] * 5), view(first), view(first[:, ::-1])), CONFIG)
    result = judge_side_evidence(evidence, CONFIG)
    np.testing.assert_array_equal(frames, np.arange(5))
    assert result.view_half_turns == (turned, False, False)
    assert result.margin == pytest.approx(2 / 3)
    assert [(h.cost, h.support, h.frames) for h in result.hypotheses] == [
        (0., 1., 5), (pytest.approx(2 / 3), 0., 5), (pytest.approx(2 / 3), 0., 5), (pytest.approx(2 / 3), 0., 5),
    ]


def test_small_asymmetric_sets_report_cost_and_exclude_static_repeats():
    # One person in both views, with a 0.5 m cross-view offset.
    first = np.array([[[1., y]] for y in (-7., -7., -6., -6., -5.)])
    second = first + [0.5, 0.]
    evidence, frames = collect_player_side_evidence(("a", "b"), "a", (view(first), view(second)), CONFIG)
    result = judge_side_evidence(evidence, CONFIG)
    np.testing.assert_array_equal(frames, [0, 2, 4])
    assert evidence.pair_record() == {"a-b": 3}
    assert result.view_half_turns == (False, False)
    assert result.hypotheses[0].cost == pytest.approx(.04)
    assert result.hypotheses[0].support == 1. and result.margin == pytest.approx(.96)


def test_multiple_points_cannot_all_match_the_same_point():
    first = view([[[2. + .3 * t, -8.], [3. + .3 * t, -8.]] for t in range(3)])
    second = view([[[2.5 + .3 * t, -8.], [2.5 + .3 * t, -2.]] for t in range(3)])
    evidence, _ = collect_player_side_evidence(("a", "b"), "a", (first, second), CONFIG)
    with pytest.raises(CourtSideUndecided) as stopped:
        judge_side_evidence(evidence, CONFIG)
    assert stopped.value.reason == NO_CONSISTENT_HYPOTHESIS
    assert stopped.value.hypotheses[0].cost == pytest.approx(.52)
    assert stopped.value.hypotheses[0].support == 0.


def test_nearly_symmetric_sets_do_not_contribute_even_when_moving():
    symmetric = view([[[3. + .3 * t, -7.], [-2.8 - .3 * t, 7.1]] for t in range(4)])
    asymmetric = view([[[3. + .3 * t, -7.]] for t in range(4)])
    evidence, frames = collect_player_side_evidence(("a", "b"), "a", (symmetric, asymmetric), CONFIG)
    with pytest.raises(CourtSideUndecided) as stopped:
        judge_side_evidence(evidence, CONFIG)
    assert stopped.value.reason == INSUFFICIENT_FRAMES
    assert frames.size == 0 and stopped.value.hypotheses == () and stopped.value.margin is None
    assert stopped.value.pair_frames == {"a-b": 0}


def test_observations_only_compare_the_same_frame_and_require_all_cameras_connected():
    points = [[[1. + .3 * t, -7.]] for t in range(6)]
    a = view(points, [[True]] * 3 + [[False]] * 3)
    b = view(points, [[False]] * 3 + [[True]] * 3)
    evidence, _ = collect_player_side_evidence(("a", "b"), "a", (a, b), CONFIG)
    with pytest.raises(CourtSideUndecided) as stopped:
        judge_side_evidence(evidence, CONFIG)
    assert stopped.value.reason == INSUFFICIENT_FRAMES and stopped.value.pair_frames == {"a-b": 0}
    evidence, _ = collect_player_side_evidence(("a", "b", "c"), "a", (a, b, a), CONFIG)
    with pytest.raises(CourtSideUndecided) as stopped:
        judge_side_evidence(evidence, CONFIG)
    assert stopped.value.reason == DISCONNECTED_VIEWS and stopped.value.pair_frames == {"a-b": 0, "a-c": 3, "b-c": 0}
    assert len(stopped.value.hypotheses) == 4 and stopped.value.margin == 0.


def test_equal_cost_hypotheses_stop_even_with_good_absolute_support():
    a = view([[[1. + .3 * t, 0.]] for t in range(3)])
    b = view([[[0., 1. + .3 * t]] for t in range(3)])
    config = replace(CONFIG, match_distance_m=5.)
    evidence, _ = collect_player_side_evidence(("a", "b"), "a", (a, b), config)
    with pytest.raises(CourtSideUndecided) as stopped:
        judge_side_evidence(evidence, config)
    assert stopped.value.reason == AMBIGUOUS_MARGIN
    assert len(stopped.value.hypotheses) == 2 and stopped.value.margin == 0.
    assert all(score.support == 1. for score in stopped.value.hypotheses)
