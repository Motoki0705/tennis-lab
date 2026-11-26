"""Contact rules on trajectories with analytically known event frames."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.ball_events import detect_ball_events
from src.tennis_scene.ball_events.rules import PlayerEvidence

FPS = 60.0
CONTACT = 12


def _trajectory(
    contact: tuple[float, float, float],
    incoming: tuple[float, float, float],
    outgoing: tuple[float, float, float],
) -> NDArray[np.float32]:
    seconds = (np.arange(2 * CONTACT + 1) - CONTACT)[:, None] / FPS
    return np.asarray(
        contact + seconds * np.where(seconds < 0, incoming, outgoing), dtype=np.float32
    )


def test_bounce_needs_near_ground_minimum_and_upward_exit() -> None:
    bounce = _trajectory((0, -4, 0.03), (1, 6, -7), (1, 4, 4))
    valid: NDArray[np.bool_] = np.ones(len(bounce), bool)
    events = detect_ball_events(bounce, valid, FPS)
    assert [(e.kind, e.frame) for e in events] == [("bounce", CONTACT)]
    assert events[0].features["vertical_reversal_mps"] == pytest.approx(4)
    assert events[0].time_s == pytest.approx(0.2)
    assert events[0].flags == ()
    high = bounce + np.array([0, 0, 1], np.float32)
    rolling = _trajectory((0, -4, 0.03), (1, 6, -7), (1, 4, 0))
    assert detect_ball_events(high, valid, FPS) == []
    assert detect_ball_events(rolling, valid, FPS) == []


def test_short_ball_gap_preserves_frame_numbers_and_marks_inference() -> None:
    xyz = _trajectory((0, -4, 0.03), (1, 6, -7), (1, 4, 4))
    valid: NDArray[np.bool_] = np.ones(len(xyz), bool)
    observed = detect_ball_events(xyz, valid, FPS)[0]
    valid[CONTACT - 2 : CONTACT + 3] = False
    xyz[~valid] = 0.0
    [event] = detect_ball_events(xyz, valid, FPS)
    assert (event.kind, event.frame) == ("bounce", CONTACT)
    assert event.flags == ("spans_ball_gap",)
    assert event.features["gap_frames"] == 5
    assert event.features["left_frame"] == CONTACT - 3
    assert event.features["right_frame"] == CONTACT + 3
    assert event.score < observed.score

    valid[CONTACT - 10 : CONTACT + 11] = False
    assert detect_ball_events(xyz, valid, FPS) == []


@pytest.mark.parametrize(
    ("contact", "incoming", "outgoing", "response"),
    [
        ((0, -0.03, 0.65), (0, 16, -1), (0, -4, -1), "reflection"),
        ((0, 0.03, 0.65), (0, -16, -1), (0, -2, -1), "slowdown"),
        ((5.0, 0, 1.08), (1, 16, -1), (0, 4, 3), "tape_deflection"),
        ((0, 0, 0.93), (0, 12, 1), (0, -0.5, -4), "tape_deflection"),
    ],
)
def test_net_contact_detects_mesh_and_tape_responses(
    contact: tuple[float, float, float],
    incoming: tuple[float, float, float],
    outgoing: tuple[float, float, float],
    response: str,
) -> None:
    xyz = _trajectory(contact, incoming, outgoing)
    [event] = detect_ball_events(xyz, np.ones(len(xyz), bool), FPS)
    assert event.kind == "net"
    assert abs(event.frame - CONTACT) <= 1
    assert event.features["net_response"] == response


def test_net_clearance_and_off_net_discontinuity_do_not_count_as_contact() -> None:
    near_tape_crossing = _trajectory((0, 0, 0.97), (0, 16, -1), (0, 14, -1.5))
    high_crossing = _trajectory((0, 0, 1.40), (0, 16, -1), (0, 4, 3))
    outside_net = _trajectory((7, 0, 0.65), (0, 16, -1), (0, -4, -1))
    valid: NDArray[np.bool_] = np.ones(len(near_tape_crossing), bool)
    for xyz in (near_tape_crossing, high_crossing, outside_net):
        assert all(e.kind != "net" for e in detect_ball_events(xyz, valid, FPS))


@pytest.mark.parametrize(
    "incoming_y",
    [12, 20],
    ids=["crossed_before_impact_step", "crossed_within_impact_step"],
)
def test_volley_just_after_clearing_the_tape_is_a_shot(incoming_y: int) -> None:
    frames, strike = 31, 15
    time = (np.arange(frames) - strike)[:, None] / FPS
    xyz = np.asarray(
        [0, 0.26, 0.90] + time * np.where(time < 0, [0, incoming_y, -6], [5, 0.3, 3]),
        dtype=np.float32,
    )
    xyz[:, 2] -= 0.5 * 9.81 * time[:, 0] ** 2
    # The flights cross y=0 about 11 cm or 6 cm above the tape, then change
    # velocity only at the wrist on the far side.
    wrists: NDArray[np.float32] = np.zeros((1, frames, 3, 3), np.float32)
    wrists[0, strike, 1] = xyz[strike] + [0.1, 0, 0]
    wrist_valid: NDArray[np.bool_] = np.zeros((1, frames, 3), bool)
    wrist_valid[0, strike, 1] = True
    players = PlayerEvidence(wrists, wrist_valid, np.array([1], np.int32))

    [event] = detect_ball_events(xyz, np.ones(frames, bool), FPS, players=players)
    assert (event.kind, event.frame, event.flags) == ("shot", strike, ())
    assert event.features["player_track_id"] == 1
    assert event.features["player_distance_m"] == pytest.approx(0.1)


def test_toss_to_serve_uses_speed_gain_with_ball_only_evidence() -> None:
    toss = _trajectory((1, -12, 2.6), (0.05, 0.20, -2), (0, 30, -3))
    [event] = detect_ball_events(toss, np.ones(len(toss), bool), FPS)
    assert (event.kind, event.frame) == ("shot", CONTACT)
    assert event.flags == ("player_unavailable",)
    assert event.features["horizontal_speed_before_mps"] == pytest.approx(
        0.206, abs=0.001
    )


def test_zero_origin_is_an_observation_and_mask_is_mandatory() -> None:
    xyz = _trajectory((0, 0, 0), (0, 7, -5), (0, 6, 3))
    [event] = detect_ball_events(xyz, np.ones(len(xyz), bool), FPS)
    assert (event.kind, event.frame) == ("bounce", CONTACT)
    with pytest.raises(ValueError, match="boolean mask"):
        detect_ball_events(xyz, np.ones(len(xyz), np.int64), FPS)  # type: ignore[arg-type]
