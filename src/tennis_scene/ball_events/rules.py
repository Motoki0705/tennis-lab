"""Detect contact candidates from observed court-space ball motion."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import hypot, isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.ball_events.observations import (
    PLAYER_LANDMARKS,
    landmark_distances,
    side_velocities,
)
from src.tennis_scene.ball_events.observations import (
    PlayerEvidence as PlayerEvidence,
)
from src.utils.rendering.trajectory_analysis import detect_bounces
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    NET_POST_OFFSET_X,
    net_height_at_x,
)

EventKind = Literal["shot", "bounce", "net"]
EVENT_KINDS: tuple[EventKind, ...] = ("shot", "bounce", "net")
FeatureValue = float | int | str | None | list[float]


@dataclass(frozen=True, slots=True)
class BallEvent:
    kind: EventKind
    frame: int
    time_s: float
    score: float
    features: dict[str, FeatureValue]
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    velocity_window_s: float = 0.05
    separation_s: float = 0.10
    contact_conflict_s: float = 0.04
    max_gap_s: float = 0.34
    max_gap_fit_error_m: float = 0.55
    max_below_ground_m: float = 0.25
    bounce_max_height_m: float = 0.27
    bounce_min_prominence_m: float = 0.012
    bounce_min_vertical_mps: float = 0.75
    shot_near_delta_mps: float = 7.0
    shot_ball_only_delta_mps: float = 7.5
    shot_min_outgoing_mps: float = 4.5
    shot_min_turn_deg: float = 40.0
    shot_min_speed_gain_mps: float = 6.5
    wrist_reach_m: float = 1.15
    root_reach_m: float = 1.85
    net_plane_tolerance_m: float = 0.36
    net_tape_tolerance_m: float = 0.13
    net_min_incoming_y_mps: float = 3.0
    net_max_y_retention: float = 0.55
    net_max_tape_speed_retention: float = 0.70
    net_min_tape_turn_deg: float = 15.0
    net_min_local_change_fraction: float = 1 / 3

    def __post_init__(self) -> None:
        values = (getattr(self, field.name) for field in fields(self))
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("Rule thresholds must be positive and finite")
        if (
            not 0 < self.net_max_y_retention < 1
            or not 0 < self.net_max_tape_speed_retention < 1
        ):
            raise ValueError("Net speed retention thresholds must be between 0 and 1")
        if self.shot_near_delta_mps > self.shot_ball_only_delta_mps:
            raise ValueError(
                "Shot delta with a nearby player must be at most the ball-only delta"
            )


DEFAULT_THRESHOLDS = RuleThresholds()


@dataclass(frozen=True, slots=True)
class _Motion:
    frame: int
    left: int
    right: int
    position: NDArray[np.float64]
    incoming: NDArray[np.float64]
    outgoing: NDArray[np.float64]
    fit_error_m: float = 0.0

    @property
    def gap_frames(self) -> int:
        return max(0, self.right - self.left - 1)


def _gap_motions(
    xyz: NDArray[np.float64],
    valid: NDArray[np.bool_],
    incoming: NDArray[np.float64],
    outgoing: NDArray[np.float64],
    fps: float,
    thresholds: RuleThresholds,
) -> list[_Motion]:
    observed = np.flatnonzero(valid)
    motions = []
    gaps = np.diff(observed) - 1
    for index in np.flatnonzero((gaps > 0) & (gaps <= thresholds.max_gap_s * fps)):
        left, right = observed[index : index + 2]
        v_in, v_out = incoming[left], outgoing[right]
        if not (np.isfinite(v_in).all() and np.isfinite(v_out).all()):
            continue
        dt = (right - left) / fps
        change = v_in - v_out
        # Closest approach of the two flight lines at the same time.
        denominator = float(change @ change)
        if denominator == 0:
            continue
        offset = float((xyz[right] - xyz[left] - dt * v_out) @ change / denominator)
        if not 0 <= offset <= dt:
            continue
        before = xyz[left] + offset * v_in
        after = xyz[right] + (offset - dt) * v_out
        error = float(np.linalg.norm(before - after))
        if error <= thresholds.max_gap_fit_error_m:
            motions.append(
                _Motion(
                    int(np.rint(left + offset * fps)),
                    int(left),
                    int(right),
                    (before + after) / 2,
                    v_in,
                    v_out,
                    error,
                )
            )
    return motions


def _event(
    kind: EventKind,
    motion: _Motion,
    fps: float,
    strength: float,
    features: dict[str, FeatureValue],
    flags: tuple[str, ...] = (),
) -> BallEvent:
    score = strength / (1 + strength)
    if motion.gap_frames:
        flags = (*flags, "spans_ball_gap")
        score *= 0.75
    return BallEvent(
        kind,
        motion.frame,
        motion.frame / fps,
        score,
        {
            "ball_position_m": motion.position.tolist(),
            "velocity_before_mps": motion.incoming.tolist(),
            "velocity_after_mps": motion.outgoing.tolist(),
            "left_frame": motion.left,
            "right_frame": motion.right,
            "gap_frames": motion.gap_frames,
            "gap_fit_error_m": motion.fit_error_m,
            **features,
        },
        flags,
    )


def _bounce(motion: _Motion, fps: float, rules: RuleThresholds) -> BallEvent | None:
    z = float(motion.position[2])
    reversal = float(min(-motion.incoming[2], motion.outgoing[2]))
    if z >= rules.bounce_max_height_m or reversal < rules.bounce_min_vertical_mps:
        return None
    return _event(
        "bounce",
        motion,
        fps,
        reversal / rules.bounce_min_vertical_mps,
        {"height_m": z, "vertical_reversal_mps": reversal},
    )


def _turn_deg(incoming: NDArray[np.float64], outgoing: NDArray[np.float64]) -> float:
    # A stationary vector has no turn.
    cosine_scale = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if cosine_scale == 0:
        return 0.0
    return float(
        np.degrees(np.arccos(np.clip((incoming @ outgoing) / cosine_scale, -1, 1)))
    )


def _net(
    motion: _Motion, xyz: NDArray[np.float64], fps: float, rules: RuleThresholds
) -> BallEvent | None:
    x, y, z = map(float, motion.position)
    if (
        abs(y) > rules.net_plane_tolerance_m
        or abs(x) > HALF_DOUBLES_WIDTH + NET_POST_OFFSET_X
    ):
        return None
    height = net_height_at_x(x)
    clearance = z - height
    vy_in, vy_out = float(motion.incoming[1]), float(motion.outgoing[1])
    approach_y = y - vy_in * max(1, motion.frame - motion.left) / fps
    if (
        clearance > rules.net_tape_tolerance_m
        or abs(vy_in) < rules.net_min_incoming_y_mps
        or approach_y * vy_in >= 0
    ):
        return None
    y_retention = abs(vy_out / vy_in)
    speed_retention = float(
        np.linalg.norm(motion.outgoing) / np.linalg.norm(motion.incoming)
    )
    turn = _turn_deg(motion.incoming, motion.outgoing)
    if clearance <= 0 and vy_in * vy_out < 0:
        response = "reflection"
    elif clearance <= 0 and y_retention <= rules.net_max_y_retention:
        response = "slowdown"
    elif (
        abs(clearance) <= rules.net_tape_tolerance_m
        and speed_retention <= rules.net_max_tape_speed_retention
        and turn >= rules.net_min_tape_turn_deg
    ):
        response = "tape_deflection"
    else:
        return None

    local_change_fraction = None
    far_side_reach_m = None
    if not motion.gap_frames:
        # An impulse can be split across adjacent samples. A portion of the
        # longer-window change must be present here, at the net candidate.
        frame = motion.frame
        change = motion.outgoing - motion.incoming
        departing_step = xyz[frame + 1] - xyz[frame]
        local_change = (departing_step - (xyz[frame] - xyz[frame - 1])) * fps
        local_change_fraction = float((local_change @ change) / (change @ change))
        if local_change_fraction < rules.net_min_local_change_fraction:
            return None
        if y * vy_in > 0:
            # On the far side, the ball could only have traveled for one
            # sample since contact. Allow half the plane tolerance for 3D error.
            departure_y = float(departing_step[1] * np.sign(vy_in))
            far_side_reach_m = rules.net_plane_tolerance_m / 2 + max(0.0, departure_y)
            if abs(y) > far_side_reach_m:
                return None

    impulse = abs(vy_out - vy_in) / rules.net_min_incoming_y_mps
    proximity = 1 + (1 - abs(y) / rules.net_plane_tolerance_m)
    return _event(
        "net",
        motion,
        fps,
        impulse * proximity,
        {
            "distance_to_net_plane_m": abs(y),
            "net_height_m": height,
            "net_clearance_m": clearance,
            "y_speed_retention": y_retention,
            "speed_retention": speed_retention,
            "turn_deg": turn,
            "local_change_fraction": local_change_fraction,
            "far_side_reach_m": far_side_reach_m,
            "net_response": response,
        },
    )


def _player_support(
    motion: _Motion, players: PlayerEvidence | None, rules: RuleThresholds
) -> tuple[dict[str, FeatureValue], str | None]:
    if players is None or not players.valid[:, motion.frame].any():
        return {}, "player_unavailable"
    radii = np.array((rules.root_reach_m, rules.wrist_reach_m, rules.wrist_reach_m))
    distances = landmark_distances(
        players.positions[:, motion.frame],
        players.valid[:, motion.frame],
        motion.position,
    )
    relative = distances / radii
    player, landmark = np.unravel_index(relative.argmin(), relative.shape)
    return {
        "player_index": int(player),
        "player_track_id": int(players.track_ids[player])
        if players.track_ids is not None
        else None,
        "player_landmark": PLAYER_LANDMARKS[landmark],
        "player_distance_m": float(distances[player, landmark]),
    }, ("player_far" if relative[player, landmark] > 1 else None)


def _shot(
    motion: _Motion, fps: float, players: PlayerEvidence | None, rules: RuleThresholds
) -> BallEvent | None:
    before, after = motion.incoming[:2], motion.outgoing[:2]
    delta = hypot(after[0] - before[0], after[1] - before[1])
    speed_after = hypot(*after)
    if delta < rules.shot_near_delta_mps or speed_after < rules.shot_min_outgoing_mps:
        return None
    speed_before = hypot(*before)
    turn = _turn_deg(before, after)
    if (
        turn < rules.shot_min_turn_deg
        and speed_after - speed_before < rules.shot_min_speed_gain_mps
    ):
        return None
    support, flag = _player_support(motion, players, rules)
    threshold = (
        rules.shot_near_delta_mps if flag is None else rules.shot_ball_only_delta_mps
    )
    if delta < threshold:
        return None
    return _event(
        "shot",
        motion,
        fps,
        delta / threshold,
        {
            "horizontal_delta_mps": delta,
            "horizontal_speed_before_mps": speed_before,
            "horizontal_speed_after_mps": speed_after,
            "horizontal_turn_deg": turn,
            **support,
        },
        (flag,) if flag else (),
    )


def _separated(events: list[BallEvent], min_separation: int) -> list[BallEvent]:
    accepted: list[BallEvent] = []
    for event in sorted(events, key=lambda e: (-e.score, e.frame)):
        if all(abs(event.frame - other.frame) >= min_separation for other in accepted):
            accepted.append(event)
    return accepted


def detect_ball_events(
    ball_3d: NDArray[np.float32],
    valid: NDArray[np.bool_],
    fps: float,
    *,
    players: PlayerEvidence | None = None,
    thresholds: RuleThresholds = DEFAULT_THRESHOLDS,
) -> list[BallEvent]:
    """Return time-ordered contact candidates on the input's zero-based timeline."""
    return analyze_ball_events(
        ball_3d, valid, fps, players=players, thresholds=thresholds
    )[0]


def analyze_ball_events(
    ball_3d: NDArray[np.float32],
    valid: NDArray[np.bool_],
    fps: float,
    *,
    players: PlayerEvidence | None = None,
    thresholds: RuleThresholds = DEFAULT_THRESHOLDS,
) -> tuple[list[BallEvent], NDArray[np.bool_]]:
    """Return candidates and frames where the rules evaluated supported motion."""
    xyz = np.asarray(ball_3d, dtype=np.float64)
    if (
        xyz.ndim != 2
        or xyz.shape[1] != 3
        or valid.shape != (len(xyz),)
        or valid.dtype != np.bool_
    ):
        raise ValueError(
            "Expected ball positions (T,3) and an explicit boolean mask (T,)"
        )
    if not isfinite(fps) or fps <= 0 or not np.isfinite(xyz[valid]).all():
        raise ValueError(
            "Require positive finite FPS and finite valid ball observations"
        )
    if players is not None:
        shape = (len(players.positions), len(xyz), 3)
        if (
            players.positions.shape != (*shape, 3)
            or players.valid.shape != shape
            or players.valid.dtype != np.bool_
            or not np.isfinite(players.positions[players.valid]).all()
            or (
                players.track_ids is not None and players.track_ids.shape != (shape[0],)
            )
        ):
            raise ValueError(
                "Invalid player landmark positions, validity mask, or track IDs"
            )
    coverage = np.zeros_like(valid)
    if np.count_nonzero(valid) < 3:
        return [], coverage

    incoming, outgoing = side_velocities(
        xyz, valid, fps, max(1, round(thresholds.velocity_window_s * fps))
    )
    minima = set(
        detect_bounces(
            np.where(valid[:, None], ball_3d, np.float32(np.nan)),
            max_bounce_height=thresholds.bounce_max_height_m,
            min_prominence=thresholds.bounce_min_prominence_m,
            min_separation=1,
        )
    )
    observed = np.flatnonzero(
        np.isfinite(incoming).all(axis=1) & np.isfinite(outgoing).all(axis=1)
    )
    motions = [
        _Motion(
            int(frame),
            int(frame),
            int(frame),
            xyz[frame],
            incoming[frame],
            outgoing[frame],
        )
        for frame in observed
    ]
    motions.extend(_gap_motions(xyz, valid, incoming, outgoing, fps, thresholds))
    candidates: dict[EventKind, list[BallEvent]] = {kind: [] for kind in EVENT_KINDS}
    for motion in motions:
        if motion.position[2] < -thresholds.max_below_ground_m:
            continue
        coverage[motion.frame] = True
        event = (
            _bounce(motion, fps, thresholds)
            if motion.gap_frames or motion.frame in minima
            else None
        )
        event = (
            event
            or _net(motion, xyz, fps, thresholds)
            or _shot(motion, fps, players, thresholds)
        )
        if event is not None:
            candidates[event.kind].append(event)

    separation = max(1, round(thresholds.separation_s * fps))
    contacts = _separated(candidates["bounce"], separation) + _separated(
        candidates["net"], separation
    )
    conflict = round(thresholds.contact_conflict_s * fps)
    shots = [
        shot
        for shot in candidates["shot"]
        if all(abs(shot.frame - contact.frame) > conflict for contact in contacts)
    ]
    events = sorted(
        contacts + _separated(shots, separation),
        key=lambda event: (event.frame, event.kind),
    )
    return events, coverage
