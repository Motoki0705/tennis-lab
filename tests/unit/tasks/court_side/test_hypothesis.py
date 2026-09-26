"""The ball hypothesis test recovers known half-turns and stops explicitly otherwise."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.court_side.hypothesis import (
    AMBIGUOUS_MARGIN,
    DISCONNECTED_VIEWS,
    INSUFFICIENT_FRAMES,
    NO_CONSISTENT_HYPOTHESIS,
    CourtSideConfig,
    CourtSideUndecided,
    collect_side_evidence,
    decide_court_side,
    distinct_observation_frames,
    half_turn_hypotheses,
)
from src.utils.geometry.triangulation import PinholeCamera

CONFIG = CourtSideConfig(reprojection_px=20., min_motion_px=5., min_frames=8, max_cost=.5, min_support=.5, min_margin=.1)


def look_at(name: str, center: list[float]) -> PinholeCamera:
    position = np.asarray(center, np.float64)
    forward = np.array([0, 0, 1.]) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1.])
    right /= np.linalg.norm(right)
    rotation = np.stack((right, np.cross(forward, right), forward))
    return PinholeCamera(name, np.array([[1400., 0, 960], [0, 1400, 540], [0, 0, 1]]), rotation, -rotation @ position)


def local_calibrations(physical: tuple[PinholeCamera, ...]) -> tuple[tuple[PinholeCamera, ...], tuple[bool, ...]]:
    """Camera-local courts put every camera at -Y: cameras beyond the net are half-turned locally."""
    flipped = tuple(bool(c.center[1] > 0) for c in physical)
    return tuple(c.half_turned(f) for c, f in zip(physical, flipped, strict=True)), flipped


def rally(frames: int = 60) -> NDArray[np.float64]:
    t = np.linspace(0, 1, frames)
    return np.stack((3 * np.sin(4 * t), -10 + 20 * t, 1 + 2.5 * np.sin(np.pi * t)), -1)


def observe(cameras: tuple[PinholeCamera, ...], xyz: NDArray[np.float64]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    projected = [c.project(xyz) for c in cameras]
    return np.stack([uv for uv, _ in projected]).astype(np.float32), np.stack([front for _, front in projected])


PHYSICAL = (look_at("a", [-8, -18, 7]), look_at("b", [9, 19, 6]), look_at("c", [7, -17, 8]), look_at("d", [-6, 20, 9]))


@pytest.mark.parametrize("views", [2, 3, 4])
@pytest.mark.parametrize("reference", [0, 1])
def test_known_half_turns_are_recovered(views: int, reference: int) -> None:
    physical = PHYSICAL[:views]
    local, flipped = local_calibrations(physical)
    uv, visible = observe(physical, rally())
    decision = decide_court_side(local, physical[reference].camera_id, uv, visible, CONFIG)
    expected = tuple(f != flipped[reference] for f in flipped)
    assert decision.view_half_turns == expected
    assert len(decision.hypotheses) == 2 ** (views - 1)
    assert decision.hypotheses[0].view_half_turns == expected and decision.hypotheses[0].cost < .01
    assert decision.margin > .5 and decision.frames == 60
    assert {h.view_half_turns for h in decision.hypotheses} == set(half_turn_hypotheses(views, reference))


def test_a_ball_on_the_symmetry_axis_carries_no_side_information() -> None:
    physical = PHYSICAL[:3]
    local, _ = local_calibrations(physical)
    xyz = np.stack((np.zeros(30), np.zeros(30), np.linspace(1, 4, 30)), -1)
    uv, visible = observe(physical, xyz)
    with pytest.raises(CourtSideUndecided) as stopped:
        decide_court_side(local, "a", uv, visible, CONFIG)
    assert stopped.value.reason == AMBIGUOUS_MARGIN
    assert len(stopped.value.hypotheses) == 4 and max(h.cost for h in stopped.value.hypotheses) < .01


def test_too_few_or_disconnected_frames_stop_before_scoring() -> None:
    physical = PHYSICAL[:3]
    local, _ = local_calibrations(physical)
    uv, visible = observe(physical, rally())
    sparse = visible.copy()
    sparse[:, 7:] = False
    with pytest.raises(CourtSideUndecided) as stopped:
        decide_court_side(local, "a", uv, sparse, CONFIG)
    assert stopped.value.reason == INSUFFICIENT_FRAMES and stopped.value.frames == 7 and len(stopped.value.hypotheses) == 4
    split = visible.copy()
    split[:2, 30:] = False  # a and b share the first half; c alone sees the second half
    split[2, :30] = False
    with pytest.raises(CourtSideUndecided) as stopped:
        decide_court_side(local, "a", uv, split, CONFIG)
    assert stopped.value.reason == DISCONNECTED_VIEWS and stopped.value.pair_frames == {"a-b": 30, "a-c": 0, "b-c": 0}


def test_inconsistent_observations_are_rejected_with_every_hypothesis_scored() -> None:
    physical = PHYSICAL[:3]
    local, _ = local_calibrations(physical)
    uv, visible = observe(physical, rally())
    uv = np.random.default_rng(0).uniform([0, 0], [1920, 1080], uv.shape).astype(np.float32)
    with pytest.raises(CourtSideUndecided, match=NO_CONSISTENT_HYPOTHESIS) as stopped:
        decide_court_side(local, "a", uv, visible, CONFIG)
    assert len(stopped.value.hypotheses) == 4


def test_invalid_inputs_are_contract_errors() -> None:
    local, _ = local_calibrations(PHYSICAL[:3])
    uv, visible = observe(PHYSICAL[:3], rally())
    with pytest.raises(ValueError, match="reference"):
        decide_court_side(local, "z", uv, visible, CONFIG)
    with pytest.raises(ValueError, match="boolean"):
        decide_court_side(local, "a", uv, visible.astype(np.uint8), CONFIG)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        CourtSideConfig(reprojection_px=20., min_motion_px=5., min_frames=8, max_cost=1.5, min_support=.5, min_margin=.1)


def test_repeated_static_observations_count_once() -> None:
    uv = np.zeros((2, 6, 2))
    uv[:, 3:, 0] = [[0, 10, 11], [0, 12, 30]]  # both views move at frame 4; at frame 5 only view 1 moves enough
    visible: NDArray[np.bool_] = np.ones((2, 6), bool)
    visible[1, 2] = False  # a changed visibility pattern is new evidence
    assert distinct_observation_frames(uv, visible, 5.).tolist() == [True, False, True, True, True, True]
    keep = distinct_observation_frames(uv, np.ones((2, 6), bool), 5.)
    assert keep.tolist() == [True, False, False, False, True, True]
    assert distinct_observation_frames(uv, np.ones((2, 6), bool), 0.).all()


def test_a_static_false_pair_cannot_outvote_the_moving_ball() -> None:
    physical = PHYSICAL[:3]
    local, _ = local_calibrations(physical)
    uv, visible = observe(physical, rally(40))
    held = np.array([6.5, -14., 1.])  # a ball held behind the baseline, seen by a and b for 200 frames
    static = np.stack([c.project(held[None])[0][0] for c in physical[:2]])
    uv = np.concatenate((uv, np.repeat(static[:, None], 200, 1).astype(np.float32)[[0, 1, 0]]), 1)
    visible = np.concatenate((visible, np.repeat([[True], [True], [False]], 200, 1)), 1)
    evidence = collect_side_evidence(local, "a", uv, visible, CONFIG)
    assert evidence.frames == 41  # 40 ball frames and the static pair once
