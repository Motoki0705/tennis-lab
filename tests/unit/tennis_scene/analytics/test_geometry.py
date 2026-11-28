from __future__ import annotations

from collections.abc import Sequence
from math import atan2, hypot

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.analytics.geometry import (
    BallTrajectory,
    GeometryEvent,
    GeometryResult,
    MatchFormat,
    extract_geometry,
)
from src.tennis_scene.analytics.profiles import (
    SOFT_TENNIS_STANDARD,
    TENNIS_ITF_STANDARD,
)


def _shot(
    frame: int, subtype: str = "groundstroke_forehand", **kwargs
) -> GeometryEvent:
    return GeometryEvent(
        f"shot-{frame}", frame, "shot", subtype, **({"court_side": -1} | kwargs)
    )


def _bounce(frame: int, **kwargs) -> GeometryEvent:
    return GeometryEvent(f"bounce-{frame}", frame, "bounce", "bounce", **kwargs)


def _net(frame: int, subtype: str) -> GeometryEvent:
    return GeometryEvent(f"net-{frame}", frame, "net", subtype)


def _axis(
    y: Sequence[float],
    events: Sequence[GeometryEvent],
    *,
    x: float = 0.0,
    invalid: Sequence[int] = (),
    unlabelable: Sequence[int] = (),
    estimated: Sequence[int] = (),
) -> GeometryResult:
    track = np.stack((np.full(len(y), x), y, np.full(len(y), 0.9)), axis=1).astype(
        float
    )
    valid: NDArray[np.bool_] = np.ones(len(y), dtype=bool)
    coverage = np.ones_like(valid)
    modelled = np.zeros_like(valid)
    valid[list(invalid)] = False
    coverage[list(unlabelable)] = False
    modelled[list(estimated)] = True
    return extract_geometry(
        BallTrajectory(track, valid, 20.0, coverage, modelled),
        events,
        profile=SOFT_TENNIS_STANDARD,
        match_format="doubles",
    )


def test_piecewise_ballistic_truth_quantifies_fractional_positions_speeds_and_clearances() -> (
    None
):
    # The two velocities either side of contact are different, with a second
    # discontinuity at the analytically known court impact.
    fps, contact_frame, bounce_time = 240.0, 10.25, 0.75
    t = (np.arange(200) - contact_frame) / fps
    gravity, radius = 9.81, 0.033
    height = radius + gravity * bounce_time**2 / 2 - 1.25 * bounce_time
    contact = np.array([-2.0, -8.0, height])
    velocity_in, velocity_out = (
        np.array([1.3, -15.0, -3.1]),
        np.array([5.5, 20.0, 1.25]),
    )
    track = np.where(t[:, None] < 0, velocity_in, velocity_out) * t[:, None] + contact
    track[:, 2] -= gravity * t**2 / 2
    after_bounce = t > bounce_time
    dt = t[after_bounce] - bounce_time
    track[after_bounce, 2] = radius + 2.5 * dt - gravity * dt**2 / 2
    result = extract_geometry(
        BallTrajectory(track, np.ones(len(track), dtype=bool), fps),
        [_shot(10, fractional_frame=10.25), _bounce(190, fractional_frame=190.25)],
        profile=SOFT_TENNIS_STANDARD,
        match_format="doubles",
    )
    shot = result.shots[0]
    assert shot.contact_position is not None
    assert shot.contact_position.source_kind == "estimated"
    assert shot.contact_position.used_frames == (9, 10)
    assert shot.contact_time_s == pytest.approx(10.25 / fps, abs=1e-12)
    assert shot.contact_position.xyz == pytest.approx(contact, abs=3e-5)

    assert shot.incoming is not None and shot.outgoing is not None
    assert shot.incoming.frames == (9, 10)
    assert shot.outgoing.frames == (11, 12)
    assert abs(shot.incoming.speed_mps - hypot(*velocity_in)) < 0.012
    assert abs(shot.outgoing.speed_mps - hypot(*velocity_out)) < 0.012
    assert shot.outgoing_heading_rad == pytest.approx(atan2(5.5, 20), abs=1e-10)

    (crossing,) = (row for row in result.crossings if row.status == "crossed")
    z_net_ball = height + 1.25 * 0.4 - gravity * 0.4**2 / 2
    assert crossing.fractional_frame == pytest.approx(106.25, abs=1e-10)
    assert crossing.used_frames == (106, 107)
    assert crossing.time_to_net_s == pytest.approx(0.4, abs=1e-10)
    assert crossing.x_m == pytest.approx(0.2, abs=1e-10)
    assert crossing.z_center_m == pytest.approx(z_net_ball, abs=2e-5)
    assert crossing.net_clearance_center_m == pytest.approx(z_net_ball - 1.07, abs=2e-5)
    assert crossing.net_clearance_bottom_m == pytest.approx(
        z_net_ball - 1.103, abs=2e-5
    )

    landing = shot.landing
    assert landing.position is not None and landing.bounce_event is not None
    assert landing.position.source_kind == "estimated"
    assert landing.position.xyz == pytest.approx((2.125, 7.0, radius), abs=3e-5)
    assert landing.bounce_event.center / fps - shot.contact_time_s == pytest.approx(
        0.75
    )
    assert landing.depth_m == pytest.approx(4.885)
    assert landing.line_margin_m == pytest.approx(3.393)
    assert (landing.lane, landing.course, landing.result) == (
        "right",
        "crosscourt",
        "in",
    )


def test_speed_windows_exclude_contact_uncertainty_and_flag_ambiguous_height() -> None:
    track = np.zeros((15, 3))
    track[:, 1] = [-8, -7, -6, -5, -4, -3, -2, -1, 2, 4, 6, 7, 8, 9, 10]
    track[:, 2] = 0.6
    track[5, 2] = 0.45
    result = extract_geometry(
        BallTrajectory(track, np.ones(len(track), dtype=bool), 10.0),
        [_shot(6, frame_uncertainty=1), _bounce(13)],
        profile=SOFT_TENNIS_STANDARD,
        match_format="doubles",
    )
    shot = result.shots[0]
    assert shot.incoming is not None and shot.outgoing is not None
    assert (shot.incoming.frames, shot.incoming.speed_mps) == ((3, 4), 10.0)
    assert (shot.outgoing.frames, shot.outgoing.speed_mps) == ((8, 9), 20.0)
    assert shot.contact_height_m == 0.6
    assert shot.quality["contact"]["time_uncertainty_s"] == 0.1
    assert shot.quality["contact_height_bin_ambiguous"]
    assert "contact_height" not in shot.bins
    assert result.arcs[1].start_event == shot.shot_event
    assert result.arcs[1].sample_runs == ((8, 12),)
    assert result.crossings[1].status == "unknown_gap"


@pytest.mark.parametrize(
    ("tail", "reason", "crossing_status", "invalid", "unlabelable"),
    [
        ([_net(8, "net_fault")], "net_stopped", "no_crossing_confirmed", [], []),
        (
            [_shot(8, "volley_center", court_side=1)],
            "intercepted_before_bounce",
            "no_crossing_confirmed",
            [],
            [],
        ),
        ([_bounce(10)], "tracking_gap", "unknown_gap", [5], []),
        ([_bounce(10)], "tracking_gap", "unknown_gap", [], [5]),
        (
            [GeometryEvent("unknown", 5, "unknown"), _bounce(10)],
            "unknown_event",
            "unknown_gap",
            [],
            [],
        ),
        ([], "point_truncated", "unknown_gap", [], []),
    ],
)
def test_missing_landings_and_crossings_keep_reason_and_null_measurements(
    tail: list[GeometryEvent],
    reason: str,
    crossing_status: str,
    invalid: list[int],
    unlabelable: list[int],
) -> None:
    y = np.linspace(-10.0, -0.1, 12)
    if invalid or unlabelable or reason == "unknown_event":
        y[5:] = np.arange(7) + 1
        y[5] = 0.0  # Sign change would be fabricated across a masked or unknown frame.
    result = _axis(
        y.tolist(), [_shot(1), *tail], invalid=invalid, unlabelable=unlabelable
    )
    landing = result.shots[0].landing
    assert landing.missing_reason == reason
    assert landing.position is None and landing.bounce_event is None
    assert landing.depth_m is None and landing.line_margin_m is None
    crossing = result.crossings[1]
    assert crossing.status == crossing_status
    assert (
        crossing.fractional_frame
        is crossing.time_to_net_s
        is crossing.z_center_m
        is None
    )
    assert not any(c.is_primary for c in result.crossings)


def test_fractional_contacts_do_not_borrow_samples_from_other_arcs() -> None:
    events = [
        _shot(2),
        GeometryEvent("unknown", 4, "unknown", fractional_frame=4.25),
        _shot(4, fractional_frame=4.5),
        _bounce(9),
    ]
    result = _axis([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], events, invalid=[5])
    assert result.shots[0].outgoing is not None
    assert result.shots[0].outgoing.frames == (3, 4)
    assert result.shots[1].incoming is None
    assert (
        result.shots[1].outgoing is None
    )  # No jump over the frame closest to contact.
    assert result.shots[1].contact_position is None


def test_all_crossings_include_exact_plane_run_and_return_but_skip_touches() -> None:
    y = [-4, -3, -2, 0, 0, 2, 0, 3, -1, -2, 4, 5, 6, 7]
    result = _axis(y, [_shot(0), _bounce(12)], x=6.60, estimated=[8, 12])
    rows = [c for c in result.crossings if c.status == "crossed"]
    assert [c.fractional_frame for c in rows] == pytest.approx([3, 7.75, 9 + 1 / 3])
    assert [c.direction_y for c in rows] == [1, -1, 1]
    assert [c.is_primary for c in rows] == [True, False, False]
    assert rows[0].used_frames == (2, 3, 4, 5)
    assert rows[1].used_frames == (7, 8)
    assert rows[1].source_kind == "estimated"
    assert all(c.crossing_in_net_span is False for c in rows)
    assert result.shots[0].landing.position is not None
    assert result.shots[0].landing.position.source_kind == "estimated"


@pytest.mark.parametrize("subtype", ["net_in", "service_let"])
def test_net_contact_has_its_own_boundary_and_crossing_uses_shot_time(
    subtype: str,
) -> None:
    net = _net(6, subtype)
    bounce = _bounce(13, in_point=subtype != "service_let")
    result = _axis(
        [-8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0.5, 2, 3, 4, 5, 6],
        [_shot(1), net, bounce],
    )
    (crossed,) = (row for row in result.crossings if row.status == "crossed")
    assert result.arcs[crossed.arc_index].start_event == net
    assert crossed.fractional_frame == 8.5
    assert crossed.time_to_net_s == pytest.approx(7.5 / 20)
    assert crossed.net_clearance_bottom_m == pytest.approx(-0.203)
    landing = result.shots[0].landing
    if subtype == "service_let":
        assert landing.missing_reason == "let_after_bounce"
        assert landing.position is None
        assert (
            result.shots[0].quality["first_bounce_evidence"]["source_event_id"]
            == bounce.source_event_id
        )
    else:
        assert landing.position is not None and landing.result == "in"


def test_rotating_both_horizontal_axes_preserves_heading_lane_course_and_depth() -> (
    None
):
    n = 17
    base = np.stack(
        (np.linspace(-2.8, 4.0, n), np.linspace(-7.0, 14.0, n), np.ones(n)), axis=1
    )
    features = []
    for side in (-1, 1):
        track = base * [-side, -side, 1]
        result = extract_geometry(
            BallTrajectory(track, np.ones(n, dtype=bool), 30.0),
            [_shot(1, court_side=side), _bounce(15)],
            profile=SOFT_TENNIS_STANDARD,
            match_format="doubles",
        )
        features.append(result.shots[0])
        assert [c.direction_y for c in result.crossings if c.is_primary] == [-side]
    a, b = features
    assert a.analysis_contact_xyz == b.analysis_contact_xyz
    assert a.analysis_landing_xyz == b.analysis_landing_xyz
    assert a.outgoing_heading_rad == pytest.approx(b.outgoing_heading_rad)
    assert a.landing.lane == b.landing.lane == "right"
    assert a.landing.course == b.landing.course == "crosscourt"
    assert a.landing.result == b.landing.result == "out_long"
    assert a.landing.depth_m == pytest.approx(-0.8025)
    assert a.bins["landing_depth"] == "out-long"


@pytest.mark.parametrize(
    ("match_format", "subtype", "x", "y", "call", "margin"),
    [
        ("singles", "groundstroke_forehand", 4.9, 8.0, "out_wide", -0.752),
        ("doubles", "groundstroke_forehand", 4.9, 8.0, "in", 0.618),
        ("doubles", "groundstroke_forehand", 5.518, 8.0, "in", 0.0),
        (
            "doubles",
            "groundstroke_forehand",
            5.511,
            11.911,
            "out_both",
            0.033 - hypot(0.026, 0.026),
        ),
        ("doubles", "serve_underhand", -1.0, 7.0, "out_long", -0.567),
        ("doubles", "serve_underhand", 1.0, 5.0, "out_wide", -0.967),
    ],
)
def test_landing_uses_match_lines_service_diagonal_and_ball_disk(
    match_format: MatchFormat,
    subtype: str,
    x: float,
    y: float,
    call: str,
    margin: float,
) -> None:
    track = np.tile(np.array([2.0, -5.0, 1.0]), (12, 1))
    track[9:, :] = (x, y, 0.033)
    result = extract_geometry(
        BallTrajectory(track, np.ones(len(track), dtype=bool), 30.0),
        [_shot(1, subtype), _bounce(10)],
        profile=SOFT_TENNIS_STANDARD,
        match_format=match_format,
    )
    landing = result.shots[0].landing
    assert landing.result == call
    assert landing.line_margin_m == pytest.approx(margin, abs=1e-10)
    assert landing.depth_m == pytest.approx(11.885 - y)


def test_bounce_crossing_a_service_line_within_uncertainty_withholds_call() -> None:
    track = np.tile(np.array([2.0, -5.0, 1.0]), (13, 1))
    track[9:12] = [(-1.0, 6.3, 0.04), (-1.0, 6.41, 0.035), (-1.0, 6.5, 0.04)]
    result = extract_geometry(
        BallTrajectory(track, np.ones(len(track), dtype=bool), 30.0),
        [_shot(1, "serve_overhead"), _bounce(10, frame_uncertainty=1)],
        profile=SOFT_TENNIS_STANDARD,
        match_format="doubles",
    )
    shot = result.shots[0]
    assert shot.landing.result == "uncertain"
    assert shot.landing.line_margin_m == pytest.approx(0.023)
    assert shot.quality["line_call_candidates"] == ["in", "out_long"]


def test_self_court_bounce_preserves_evidence_and_withholds_landing() -> None:
    result = _axis([-8, -7, -6, -5, -4, -3, -2, -1], [_shot(1), _bounce(6)])
    shot = result.shots[0]
    assert shot.landing.missing_reason == "self_court_bounce"
    assert shot.landing.position is None and shot.landing.depth_m is None
    assert shot.quality["first_bounce_evidence"]["position_xyz_m"] == [0.0, -2.0, 0.9]


def test_a_volley_retains_an_observed_crossing_but_has_no_landing_for_incoming_shot() -> (
    None
):
    result = _axis(
        [-8, -7, -6, -4, -2, 2, 4, 6, 7, 5, 3, 1, -1, -4, -6],
        [_shot(1), _shot(8, "volley_center", court_side=1), _bounce(14)],
    )
    launch, volley = result.shots
    assert launch.landing.missing_reason == "intercepted_before_bounce"
    assert launch.landing.position is None
    assert volley.incoming is not None and volley.outgoing is not None
    assert volley.incoming.frames == (6, 7)
    assert volley.outgoing.frames == (9, 10)
    assert [row.fractional_frame for row in result.crossings if row.is_primary] == [
        4.5,
        11.5,
    ]


def test_a_new_point_cannot_supply_a_prior_shots_bounce_or_crossing() -> None:
    result = _axis(
        [-8, -7, -6, -5, -4, -2, 1, 3, 5, 7, 9, 10],
        [_shot(1, point_id=1, point_end_frame=4), _bounce(10, point_id=2)],
    )
    assert result.shots[0].landing.missing_reason == "point_truncated"
    assert result.crossings[1].status == "unknown_gap"
    assert result.crossings[1].fractional_frame is None
    assert result.arcs[1].shot_end_frame == 4


def test_itf_clearance_uses_point_on_profile_and_its_ball_radius() -> None:
    track = np.tile(np.array([3.1995, -1.0, 1.2]), (8, 1))
    track[4:, 1] = 1.0
    result = extract_geometry(
        BallTrajectory(track, np.ones(len(track), dtype=bool), 25.0),
        [_shot(1), _bounce(7)],
        profile=TENNIS_ITF_STANDARD,
        match_format="singles",
    )
    (crossing,) = (row for row in result.crossings if row.is_primary)
    assert crossing.net_clearance_center_m == pytest.approx(0.247)
    assert crossing.net_clearance_bottom_m == pytest.approx(0.2135)
