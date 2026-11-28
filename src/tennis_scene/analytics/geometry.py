"""Shot and flight geometry from a masked court-coordinate ball track."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import atan2, ceil, floor, hypot, isfinite
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.analytics.profiles import CourtNetProfile
from src.tennis_scene.ball_event_labels import (
    EVENT_CLASSES_V4,
    EVENT_STATUS,
    EVENT_SUBTYPES_BY_KIND_V4,
    EVENT_SUBTYPES_V4,
)

DEFINITION_VERSION = "s4_geometry.v2"
BIN_VERSION = "s4_bins.v2"
HEIGHT_BIN_WIDTH_M = 0.5
DEPTH_BIN_EDGES_M = (0.0, 2.0, 4.0)
DEPTH_BINS = ("out-long", "0-2m", "2-4m", "4m+")
LANES = ("left", "center", "right")
SERVE_SUBTYPES = ("serve_overhead", "serve_underhand")
STOP_SUBTYPES = ("net_fault", "service_let")

XYZ: TypeAlias = tuple[float, float, float]
GeometrySource = Literal["observed", "estimated"]
CrossingStatus = Literal[
    "crossed", "no_crossing_confirmed", "unknown_gap", "not_applicable"
]
MatchFormat = Literal["singles", "doubles"]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class GeometryEvent:
    source_event_id: str
    frame: int
    kind: str
    subtype: str | None = None
    frame_uncertainty: float | None = 0.0
    court_side: int = 0
    status: str | None = None
    fractional_frame: float | None = None
    in_point: bool = True
    point_id: int | None = None
    point_end_frame: int | None = None
    excluded_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.source_event_id.strip()
            or type(self.frame) is not int
            or self.frame < 0
        ):
            raise ValueError(
                "Events require a source ID and non-negative original frame"
            )
        if self.kind not in EVENT_CLASSES_V4:
            raise ValueError(f"Unsupported geometry event kind: {self.kind}")
        allowed = EVENT_SUBTYPES_BY_KIND_V4[EVENT_CLASSES_V4.index(self.kind)]
        if self.subtype is not None and self.subtype not in (
            EVENT_SUBTYPES_V4[value] for value in allowed
        ):
            raise ValueError(
                f"Event subtype {self.subtype!r} does not match {self.kind}"
            )
        if self.court_side not in (-1, 0, 1) or (
            self.kind != "shot" and self.court_side
        ):
            raise ValueError("A shot's court side must be -1, 0 or +1")
        if self.status is not None and self.status not in EVENT_STATUS:
            raise ValueError(f"Invalid event status: {self.status}")
        if self.excluded_reason is not None and (
            not self.excluded_reason.strip() or self.in_point
        ):
            raise ValueError(
                "An excluded event requires a reason and cannot be in point"
            )
        if self.frame_uncertainty is not None and (
            not isfinite(self.frame_uncertainty) or self.frame_uncertainty < 0
        ):
            raise ValueError("Event uncertainty must be finite and non-negative")
        if self.fractional_frame is not None and (
            not isfinite(self.fractional_frame)
            or abs(self.fractional_frame - self.frame) > 0.5
        ):
            raise ValueError("Fractional event time must round to the original frame")

    @property
    def center(self) -> float:
        return (
            float(self.frame)
            if self.fractional_frame is None
            else self.fractional_frame
        )

    @property
    def lower(self) -> float:
        return self.center - (self.frame_uncertainty or 0.0)

    @property
    def upper(self) -> float:
        return self.center + (self.frame_uncertainty or 0.0)


@dataclass(frozen=True, slots=True)
class BallTrajectory:
    positions: NDArray[np.floating[Any]]
    valid: BoolArray
    fps: float
    event_coverage: BoolArray | None = None
    estimated: BoolArray | None = None

    def __post_init__(self) -> None:
        if (
            self.positions.ndim != 2
            or self.positions.shape[1] != 3
            or len(self.positions) == 0
            or not np.issubdtype(self.positions.dtype, np.floating)
            or not isfinite(self.fps)
            or self.fps <= 0
        ):
            raise ValueError(
                "Expected a non-empty floating-point (T, 3) track and positive fps"
            )
        for mask in (self.valid, self.event_coverage, self.estimated):
            if mask is not None and (
                mask.shape != (len(self.positions),) or mask.dtype != np.bool_
            ):
                raise ValueError("Trajectory masks must be Boolean and have shape (T,)")
        if not np.isfinite(self.positions[self.valid]).all():
            raise ValueError("Valid ball samples must be finite")
        if self.estimated is not None and (self.estimated & ~self.valid).any():
            raise ValueError("Estimated samples must be valid")

    def source(self, frames: Sequence[int]) -> GeometrySource:
        if self.estimated is not None and self.estimated[list(frames)].any():
            return "estimated"
        return "observed"

    def xyz(self, frame: int) -> XYZ:
        x, y, z = self.positions[frame]
        return float(x), float(y), float(z)


@dataclass(frozen=True, slots=True)
class FlightArc:
    arc_index: int
    start_event: GeometryEvent | None
    end_event: GeometryEvent | None
    shot_event: GeometryEvent | None
    start_frame: int
    end_frame: int
    valid_frame_count: int
    total_frame_count: int
    source_kind: GeometrySource
    sample_bounds: tuple[int, int]
    sample_runs: tuple[tuple[int, int], ...]
    shot_end_frame: int
    shot_boundary_reason: str | None

    def quality_record(self) -> dict[str, Any]:
        return {
            "sample_bounds_inclusive": list(self.sample_bounds),
            "sample_runs_inclusive": [list(run) for run in self.sample_runs],
            "shot_evidence_end_frame": self.shot_end_frame,
            "shot_boundary_reason": self.shot_boundary_reason,
        }


@dataclass(frozen=True, slots=True)
class EventPosition:
    xyz: XYZ
    source_kind: GeometrySource
    used_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class OneSidedSpeed:
    velocity_mps: XYZ
    frames: tuple[int, int]
    source_kind: GeometrySource

    @property
    def speed_mps(self) -> float:
        return hypot(*self.velocity_mps)


@dataclass(frozen=True, slots=True)
class NetPlaneCrossing:
    arc_index: int
    crossing_index: int
    status: CrossingStatus
    source_kind: GeometrySource
    is_primary: bool = False
    fractional_frame: float | None = None
    time_s: float | None = None
    time_to_net_s: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    z_center_m: float | None = None
    direction_y: int | None = None
    crossing_in_net_span: bool | None = None
    net_height_m: float | None = None
    net_clearance_center_m: float | None = None
    net_clearance_bottom_m: float | None = None
    used_frames: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Landing:
    bounce_event: GeometryEvent | None = None
    position: EventPosition | None = None
    depth_m: float | None = None
    lane: str | None = None
    course: str | None = None
    line_margin_m: float | None = None
    result: str = "not_applicable"
    missing_reason: str | None = None
    depth_bin: str | None = None


@dataclass(frozen=True, slots=True)
class ShotGeometry:
    shot_event: GeometryEvent
    flight_arc_index: int
    contact_time_s: float
    contact_position: EventPosition | None
    analysis_contact_xyz: XYZ | None
    incoming: OneSidedSpeed | None
    outgoing: OneSidedSpeed | None
    outgoing_heading_rad: float | None
    landing: Landing
    analysis_landing_xyz: XYZ | None
    bins: dict[str, str]
    quality: dict[str, Any]

    @property
    def contact_height_m(self) -> float | None:
        return None if self.contact_position is None else self.contact_position.xyz[2]


@dataclass(frozen=True, slots=True)
class GeometryResult:
    arcs: tuple[FlightArc, ...]
    shots: tuple[ShotGeometry, ...]
    crossings: tuple[NetPlaneCrossing, ...]
    profile: CourtNetProfile


def analysis_xyz(xyz: XYZ, hitter_side: int) -> XYZ | None:
    """Rotate both horizontal axes so the opponent's court is on +Y."""
    x, y, z = xyz
    if hitter_side == -1:
        return xyz
    if hitter_side == 1:
        return -x, -y, z
    return None


def _runs(mask: BoolArray, start: int, end: int) -> tuple[tuple[int, int], ...]:
    if start > end:
        return ()
    edges = np.flatnonzero(np.diff(mask[start : end + 1], prepend=False, append=False))
    return tuple(
        (start + int(a), start + int(b) - 1)
        for a, b in zip(edges[::2], edges[1::2], strict=True)
    )


def _different_point(shot: GeometryEvent, event: GeometryEvent) -> bool:
    return shot.point_id != event.point_id and event.in_point


def _arcs(
    track: BallTrajectory,
    events: Sequence[GeometryEvent],
    coverage: BoolArray,
) -> tuple[FlightArc, ...]:
    sampled = track.valid & coverage
    for event in events:
        sampled[max(0, ceil(event.lower)) : floor(event.upper) + 1] = False

    final_frame = len(track.valid) - 1
    active: GeometryEvent | None = None
    after_let = False
    arcs: list[FlightArc] = []
    boundaries = (None, *events, None)
    for index, (start, end) in enumerate(pairwise(boundaries)):
        first = 0 if start is None else start.frame
        last = final_frame if end is None else end.frame
        lo = 0 if start is None else floor(start.upper) + 1
        hi = final_frame if end is None else ceil(end.lower) - 1
        runs = _runs(sampled, lo, hi)

        if start is not None:
            if start.kind == "shot":
                active = start if start.in_point else None
                after_let = False
            elif (
                not start.in_point
                or start.kind == "unknown"
                or start.subtype == "net_fault"
                or (after_let and start.kind == "bounce")
                or (active is not None and _different_point(active, start))
            ):
                active = None
            elif start.subtype == "service_let":
                after_let = True

        shot_end, boundary_reason = last, None
        if active is not None:
            if active.point_end_frame is not None and active.point_end_frame < last:
                shot_end = active.point_end_frame
                boundary_reason = "point_boundary"
            if (
                end is not None
                and _different_point(active, end)
                and boundary_reason is None
            ):
                # The interval between known points has no reliable shot attribution.
                shot_end, boundary_reason = first - 1, "point_boundary"
            gaps = np.flatnonzero(~coverage[first : shot_end + 1])
            if gaps.size:
                shot_end, boundary_reason = (
                    first + int(gaps[0]) - 1,
                    "event_coverage_gap",
                )
            if boundary_reason is None and end is not None and end.excluded_reason:
                boundary_reason = "excluded_contact"

        valid = track.valid[first : last + 1]
        arcs.append(
            FlightArc(
                index,
                start,
                end,
                active,
                first,
                last,
                int(valid.sum()),
                len(valid),
                track.source(range(first, last + 1)),
                (lo, hi),
                runs,
                shot_end,
                boundary_reason,
            )
        )
        if boundary_reason is not None:
            active = None
    return tuple(arcs)


def _position_at_event(
    track: BallTrajectory,
    event: GeometryEvent,
    coverage: BoolArray,
    before: FlightArc,
    after: FlightArc,
) -> EventPosition | None:
    if event.center == event.frame:
        if track.valid[event.frame] and coverage[event.frame]:
            return EventPosition(
                track.xyz(event.frame), track.source((event.frame,)), (event.frame,)
            )
        return None

    # Extrapolate by less than one frame using two samples on a single safe side.
    speed_windows = [
        window
        for window in (
            _adjacent_frames(before, outgoing=False),
            _adjacent_frames(after, outgoing=True),
        )
        if window is not None
    ]
    for frames in sorted(
        speed_windows, key=lambda pair: min(abs(f - event.center) for f in pair)
    ):
        if min(abs(frame - event.center) for frame in frames) < 1:
            left, right = (np.asarray(track.xyz(frame)) for frame in frames)
            x, y, z = left + (event.center - frames[0]) * (right - left)
            return EventPosition((float(x), float(y), float(z)), "estimated", frames)
    return None


def _adjacent_frames(arc: FlightArc, *, outgoing: bool) -> tuple[int, int] | None:
    if not arc.sample_runs:
        return None
    if outgoing:
        first, last = arc.sample_runs[0]
        if first == arc.sample_bounds[0] and first + 1 <= min(last, arc.shot_end_frame):
            return first, first + 1
    else:
        first, last = arc.sample_runs[-1]
        if last == arc.sample_bounds[1] and first < last:
            return last - 1, last
    return None


def _speed(
    track: BallTrajectory,
    arc: FlightArc,
    *,
    outgoing: bool,
) -> OneSidedSpeed | None:
    frames = _adjacent_frames(arc, outgoing=outgoing)
    if frames is None:
        return None
    start, end = (track.xyz(frame) for frame in frames)
    vx, vy, vz = ((b - a) * track.fps for a, b in zip(start, end, strict=True))
    return OneSidedSpeed((vx, vy, vz), frames, track.source(frames))


def _crossings_in_run(
    track: BallTrajectory,
    first: int,
    last: int,
) -> list[tuple[float, XYZ, int, tuple[int, ...]]]:
    """Report sign changes once, including exact-zero samples or a zero run."""
    ys = track.positions[first : last + 1, 1]
    nonzero = np.flatnonzero(ys) + first
    hits: list[tuple[float, XYZ, int, tuple[int, ...]]] = []
    for a, b in pairwise(nonzero.tolist()):
        xa, ya, za = track.xyz(a)
        xb, yb, zb = track.xyz(b)
        if (ya < 0) == (yb < 0):
            continue
        direction = 1 if yb > 0 else -1
        if b == a + 1:
            fraction = -ya / (yb - ya)
            xyz = (xa + fraction * (xb - xa), 0.0, za + fraction * (zb - za))
            frame = a + fraction
        else:
            # A zero run brackets a transition but does not resolve the instant
            # of departure. Use its entry time and preserve the whole bracket.
            frame = float(a + 1)
            xyz = track.xyz(a + 1)
        hits.append((frame, xyz, direction, tuple(range(a, b + 1))))
    return hits


def _crossings(
    track: BallTrajectory,
    arcs: Sequence[FlightArc],
    profile: CourtNetProfile,
) -> tuple[NetPlaneCrossing, ...]:
    rows: list[NetPlaneCrossing] = []
    primaries: set[str] = set()
    for arc in arcs:
        shot = arc.shot_event
        if shot is None:
            rows.append(
                NetPlaneCrossing(arc.arc_index, 0, "not_applicable", arc.source_kind)
            )
            continue

        hits = [
            hit
            for first, last in arc.sample_runs
            for hit in _crossings_in_run(track, first, min(last, arc.shot_end_frame))
        ]
        if not hits:
            # Completeness covers the nominal endpoints as well. A possible
            # transition in an excluded contact window is an unknown crossing.
            y = track.positions[arc.start_frame : arc.end_frame + 1, 1]
            status: CrossingStatus = "unknown_gap"
            if (
                arc.end_event is not None
                and arc.end_event.kind != "unknown"
                and arc.end_event.in_point
                and arc.shot_boundary_reason is None
                and arc.valid_frame_count == arc.total_frame_count
                and any(last > first for first, last in arc.sample_runs)
                and y[0] != 0
                and y[-1] != 0
                and not (np.any(y < 0) and np.any(y > 0))
            ):
                status = "no_crossing_confirmed"
            rows.append(NetPlaneCrossing(arc.arc_index, 0, status, arc.source_kind))
            continue

        for index, (frame, (x, y_pos, z), direction, used_frames) in enumerate(hits):
            primary = (
                direction == -shot.court_side and shot.source_event_id not in primaries
            )
            if primary:
                primaries.add(shot.source_event_id)
            height = profile.net_height_at_x(x)
            rows.append(
                NetPlaneCrossing(
                    arc.arc_index,
                    index,
                    "crossed",
                    track.source(used_frames),
                    is_primary=primary,
                    fractional_frame=frame,
                    time_s=frame / track.fps,
                    time_to_net_s=(frame - shot.center) / track.fps
                    if primary
                    else None,
                    x_m=x,
                    y_m=y_pos,
                    z_center_m=z,
                    direction_y=direction,
                    crossing_in_net_span=profile.crossing_in_net_span(x),
                    net_height_m=height,
                    net_clearance_center_m=z - height,
                    net_clearance_bottom_m=z - height - profile.ball_radius_m,
                    used_frames=used_frames,
                )
            )
    return tuple(rows)


def _time_window_positions(
    track: BallTrajectory,
    event: GeometryEvent,
    coverage: BoolArray,
    nominal: EventPosition | None,
) -> tuple[list[XYZ], bool]:
    lo, hi = max(0, ceil(event.lower)), min(len(track.valid) - 1, floor(event.upper))
    frames = range(lo, hi + 1)
    positions = [track.xyz(f) for f in frames if track.valid[f] and coverage[f]]
    complete = (
        event.frame_uncertainty is not None
        and event.lower >= 0
        and event.upper <= len(track.valid) - 1
        and len(positions) == len(frames)
        and nominal is not None
    )
    if nominal is not None and event.center != event.frame:
        positions.append(nominal.xyz)
    return positions, complete


def _one_x_side(xs: Sequence[float], radius: float) -> int:
    if xs and min(xs) > radius:
        return 1
    if xs and max(xs) < -radius:
        return -1
    return 0


def _line_call(
    x: float,
    y: float,
    left: float,
    right: float,
    far: float,
    radius: float,
) -> tuple[float, str]:
    wide, long = min(x - left, right - x), min(y, far - y)
    if wide >= 0 and long >= 0:
        return min(wide, long) + radius, "in"
    margin = radius - hypot(min(0.0, wide), min(0.0, long))
    if margin >= -1e-12:
        return max(margin, 0.0), "in"
    if wide < 0 and long < 0:
        return margin, "out_both"
    return margin, "out_wide" if wide < 0 else "out_long"


def _landing_event(
    track: BallTrajectory,
    events: Sequence[GeometryEvent],
    index: int,
    coverage: BoolArray,
) -> tuple[int | None, str | None]:
    shot = previous = events[index]
    stop = (
        len(track.valid) - 1 if shot.point_end_frame is None else shot.point_end_frame
    )
    let = False
    for target in range(index + 1, len(events)):
        event = events[target]
        if event.frame > stop or _different_point(shot, event):
            break
        if not coverage[previous.frame : event.frame + 1].all():
            return None, "tracking_gap"
        if event.kind == "unknown":
            return None, "unknown_event"
        if event.kind == "shot":
            if let:
                return None, "service_let"
            reason = (
                "point_truncated"
                if event.subtype in SERVE_SUBTYPES or not event.in_point
                else "intercepted_before_bounce"
            )
            return None, reason
        if event.subtype == "net_fault":
            return None, "service_let" if let else "net_stopped"
        if event.subtype == "service_let":
            let = True
        if event.kind == "bounce":
            if let:
                return target, "let_after_bounce"
            if not event.in_point:
                return target, "bounce_outside_point"
            following = events[target + 1] if target + 1 < len(events) else None
            if event.lower <= previous.upper or (
                following is not None
                and following.kind == "shot"
                and not _different_point(shot, following)
                and following.lower <= event.upper
            ):
                return target, "ambiguous_event_order"
            if not track.valid[shot.frame : event.frame + 1].all():
                return target, "tracking_gap"
            return target, None
        if not event.in_point:
            return None, "point_truncated"
        previous = event

    if let:
        return None, "service_let"
    if not (coverage[shot.frame : stop + 1] & track.valid[shot.frame : stop + 1]).all():
        return None, "tracking_gap"
    return None, "point_truncated"


def _height_bin(z: float) -> str | None:
    if z < 0:
        return None
    lo = HEIGHT_BIN_WIDTH_M * floor(z / HEIGHT_BIN_WIDTH_M)
    return f"{lo:.1f}-{lo + HEIGHT_BIN_WIDTH_M:.1f}m"


def _lane_bin(x: float, half_width: float) -> str:
    if x < -half_width / 3:
        return "left"
    return "right" if x > half_width / 3 else "center"


def _depth_bin(depth: float) -> str:
    return DEPTH_BINS[sum(depth >= edge for edge in DEPTH_BIN_EDGES_M)]


def _resolve_landing_bin(
    candidates: set[str],
    *,
    window_complete: bool,
    all_in_opponent_half: bool,
) -> tuple[str | None, dict[str, Any]]:
    if not all_in_opponent_half:
        reason = "opponent_half_ambiguous"
    elif len(candidates) > 1:
        reason = "boundary_spanned"
    elif not window_complete or not candidates:
        reason = "timing_window_incomplete"
    else:
        return next(iter(candidates)), {
            "candidates": sorted(candidates),
            "held_reason": None,
        }
    return None, {"candidates": sorted(candidates), "held_reason": reason}


def _landing(
    profile: CourtNetProfile,
    width: float,
    shot: GeometryEvent,
    bounce: GeometryEvent,
    position: EventPosition,
    reason: str | None,
    contact_x_side: int,
    landing_window: Sequence[XYZ],
    window_complete: bool,
) -> tuple[Landing, XYZ | None, dict[str, Any]]:
    normalized = analysis_xyz(position.xyz, shot.court_side)
    if reason is not None:
        return Landing(missing_reason=reason), None, {}
    if normalized is None:
        return (
            Landing(bounce, position, result="uncertain"),
            None,
            {"landing_context_reason": "hitter_side_unknown"},
        )
    x, y, _ = normalized
    if y <= 0:
        return Landing(missing_reason="self_court_bounce"), None, {}
    half = width / 2
    landing_positions = [
        p
        for xyz in landing_window
        if (p := analysis_xyz(xyz, shot.court_side)) is not None
    ]
    opponent_positions = [point for point in landing_positions if point[1] > 0]
    all_in_opponent_half = len(opponent_positions) == len(landing_positions)
    lane, lane_evidence = _resolve_landing_bin(
        {_lane_bin(px, half) for px, _, _ in opponent_positions},
        window_complete=window_complete,
        all_in_opponent_half=all_in_opponent_half,
    )
    depth_bin, depth_evidence = _resolve_landing_bin(
        {_depth_bin(profile.half_length - py) for _, py, _ in opponent_positions},
        window_complete=window_complete,
        all_in_opponent_half=all_in_opponent_half,
    )
    bin_evidence = {"landing_lane": lane_evidence, "landing_depth": depth_evidence}
    landing_x_side = (
        _one_x_side([p[0] for p in landing_positions], profile.ball_radius_m)
        if window_complete
        else 0
    )
    course = None
    if contact_x_side and landing_x_side:
        course = "crosscourt" if contact_x_side != landing_x_side else "down_the_line"
    depth = profile.half_length - y

    serve = shot.subtype in SERVE_SUBTYPES
    left, right, far = -half, half, profile.half_length
    if serve:
        if not contact_x_side:
            return (
                Landing(
                    bounce,
                    position,
                    depth,
                    lane,
                    course,
                    result="uncertain",
                    depth_bin=depth_bin,
                ),
                normalized,
                {
                    "landing_context_reason": "service_box_side_unknown",
                    "landing_bin_evidence": bin_evidence,
                },
            )
        half_singles = profile.singles_width_m / 2
        left, right = (
            (-half_singles, 0.0) if contact_x_side > 0 else (0.0, half_singles)
        )
        far = profile.service_line_m
    margin, nominal = _line_call(x, y, left, right, far, profile.ball_radius_m)
    candidates = sorted(
        {
            _line_call(px, py, left, right, far, profile.ball_radius_m)[1]
            if py > 0
            else "self_court"
            for px, py, _ in landing_positions
        }
    )
    result = nominal if window_complete and candidates == [nominal] else "uncertain"
    return (
        Landing(
            bounce, position, depth, lane, course, margin, result, depth_bin=depth_bin
        ),
        normalized,
        {
            "line_call_nominal": nominal,
            "line_call_candidates": candidates,
            "line_call_bounds_m": {
                "left": left,
                "right": right,
                "near": 0.0,
                "far": far,
            },
            "landing_bin_evidence": bin_evidence,
        },
    )


def _event_quality(
    event: GeometryEvent, fps: float, position: EventPosition | None
) -> dict[str, Any]:
    return {
        "source_event_id": event.source_event_id,
        "original_frame": event.frame,
        "fractional_frame": event.center,
        "time_s": event.center / fps,
        "frame_uncertainty": event.frame_uncertainty,
        "time_uncertainty_s": None
        if event.frame_uncertainty is None
        else event.frame_uncertainty / fps,
        "annotation_status": event.status,
        "position_xyz_m": None if position is None else list(position.xyz),
        "position_source_kind": None if position is None else position.source_kind,
        "used_frames": [] if position is None else list(position.used_frames),
    }


def _shot(
    track: BallTrajectory,
    events: Sequence[GeometryEvent],
    index: int,
    arcs: Sequence[FlightArc],
    coverage: BoolArray,
    profile: CourtNetProfile,
    match_format: MatchFormat,
) -> ShotGeometry:
    shot = events[index]
    before, after = arcs[index], arcs[index + 1]
    contact = _position_at_event(track, shot, coverage, before, after)
    contact_xyz = analysis_xyz(contact.xyz, shot.court_side) if contact else None
    contact_window, contact_complete = _time_window_positions(
        track, shot, coverage, contact
    )
    height_bins = {_height_bin(p[2]) for p in contact_window}
    height_ambiguous = (
        True if len(height_bins) > 1 else False if contact_complete else None
    )
    contact_x_side = (
        _one_x_side(
            [p[0] * -shot.court_side for p in contact_window], profile.ball_radius_m
        )
        if contact_complete and shot.court_side
        else 0
    )

    incoming_applicable = shot.subtype not in SERVE_SUBTYPES and (
        before.start_event is None
        or (
            before.start_event.in_point
            and before.start_event.subtype not in STOP_SUBTYPES
            and not _different_point(shot, before.start_event)
        )
    )
    incoming = _speed(track, before, outgoing=False) if incoming_applicable else None
    outgoing = _speed(track, after, outgoing=True)
    velocity = (
        analysis_xyz(outgoing.velocity_mps, shot.court_side) if outgoing else None
    )
    heading = (
        atan2(velocity[0], velocity[1])
        if velocity is not None and any(velocity[:2])
        else None
    )

    bounce_index, missing_reason = _landing_event(track, events, index, coverage)
    landing = Landing(missing_reason=missing_reason)
    landing_xyz: XYZ | None = None
    quality: dict[str, Any] = {
        "contact": _event_quality(shot, track.fps, contact),
        "hitter_court_side": shot.court_side,
        "contact_analysis_xyz_m": None if contact_xyz is None else list(contact_xyz),
        "contact_height_bin_ambiguous": height_ambiguous,
        "contact_height_bin_candidates": sorted(
            bin_id for bin_id in height_bins if bin_id is not None
        ),
        "contact_height_range_m": [
            min(p[2] for p in contact_window),
            max(p[2] for p in contact_window),
        ]
        if contact_window
        else None,
        "contact_window_fully_observed": contact_complete,
        "incoming_speed_applicable": incoming_applicable,
        "speed_source_kind": {
            "incoming": None if incoming is None else incoming.source_kind,
            "outgoing": None if outgoing is None else outgoing.source_kind,
        },
        "match_format": match_format,
        "landing_region": "service_box"
        if shot.subtype in SERVE_SUBTYPES
        else "opponent_half",
        "line_rule": "projected_ball_disk_overlaps_landing_region",
    }
    width = (
        profile.singles_width_m
        if match_format == "singles"
        else profile.doubles_width_m
    )
    if bounce_index is not None:
        bounce = events[bounce_index]
        bounce_position = _position_at_event(
            track, bounce, coverage, arcs[bounce_index], arcs[bounce_index + 1]
        )
        quality["first_bounce_evidence"] = _event_quality(
            bounce, track.fps, bounce_position
        )
        if bounce_position is None:
            landing = Landing(missing_reason=missing_reason or "tracking_gap")
        else:
            window, complete = _time_window_positions(
                track, bounce, coverage, bounce_position
            )
            landing, landing_xyz, landing_quality = _landing(
                profile,
                width,
                shot,
                bounce,
                bounce_position,
                missing_reason,
                contact_x_side,
                window,
                complete,
            )
            quality.update(landing_quality, landing_window_fully_observed=complete)
    if landing_xyz is not None and landing.missing_reason is None:
        quality["landing_analysis_xyz_m"] = list(landing_xyz)

    bins: dict[str, str] = {}
    if contact is not None and height_ambiguous is False:
        height_id = _height_bin(contact.xyz[2])
        if height_id is not None:
            bins["contact_height"] = height_id
    if landing.depth_bin is not None:
        bins["landing_depth"] = landing.depth_bin
    if landing.lane is not None:
        bins["landing_lane"] = landing.lane
    if landing.course is not None:
        bins["landing_course"] = landing.course
    return ShotGeometry(
        shot,
        after.arc_index,
        shot.center / track.fps,
        contact,
        contact_xyz,
        incoming,
        outgoing,
        heading,
        landing,
        landing_xyz,
        bins,
        quality,
    )


def extract_geometry(
    track: BallTrajectory,
    events: Sequence[GeometryEvent],
    *,
    profile: CourtNetProfile,
    match_format: MatchFormat,
) -> GeometryResult:
    """Extract geometry with all event and tracking guards applied before fitting."""
    if match_format not in ("singles", "doubles"):
        raise ValueError("Geometry requires a singles or doubles match format")
    if (
        len({e.source_event_id for e in events}) != len(events)
        or any(a.frame > b.frame or a.center > b.center for a, b in pairwise(events))
        or any(
            not 0 <= event.center <= len(track.valid) - 1
            or event.frame >= len(track.valid)
            or (
                event.point_end_frame is not None
                and not event.frame <= event.point_end_frame < len(track.valid)
            )
            for event in events
        )
    ):
        raise ValueError("Event IDs, ordering or bounds disagree with the trajectory")
    coverage = track.event_coverage
    if coverage is None:
        coverage = np.ones(track.valid.shape, dtype=np.bool_)
    arcs = _arcs(track, events, coverage)
    shots = tuple(
        _shot(track, events, index, arcs, coverage, profile, match_format)
        for index, event in enumerate(events)
        if event.kind == "shot" and event.in_point
    )
    return GeometryResult(arcs, shots, _crossings(track, arcs, profile), profile)
