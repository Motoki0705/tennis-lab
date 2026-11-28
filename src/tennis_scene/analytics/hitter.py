"""Attribute annotated shots to clip-local player tracks in court space."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import ceil, isfinite
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.ball_events.observations import (
    PLAYER_LANDMARKS,
    PlayerEvidence,
    landmark_distances,
    side_velocities,
)

RULE_REVISION = "court-hitter-v1"

Resolution = Literal["resolved", "ambiguous", "unresolved"]


@dataclass(frozen=True, slots=True)
class HitterThresholds:
    wrist_reach_m: float = 1.30
    root_reach_m: float = 1.85
    min_margin: float = 0.18
    net_zone_m: float = 0.70
    side_deadband_m: float = 0.15
    max_frame_uncertainty_s: float = 0.10
    search_radius_s: float = 0.035
    motion_window_s: float = 0.05

    def __post_init__(self) -> None:
        if any(not isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Hitter thresholds must be finite and positive")
        if self.min_margin >= 1 or self.side_deadband_m > self.net_zone_m:
            raise ValueError("Invalid hitter margin or court-side deadband")

    @property
    def rule_version(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"{RULE_REVISION}.{sha256(payload.encode()).hexdigest()[:12]}"


DEFAULT_THRESHOLDS = HitterThresholds()


@dataclass(frozen=True, slots=True)
class HitterScene:
    ball: NDArray[np.float32]
    ball_valid: NDArray[np.bool_]
    players: PlayerEvidence
    fps: float

    def __post_init__(self) -> None:
        player_count = len(self.players.positions)
        shape = (player_count, len(self.ball), len(PLAYER_LANDMARKS))
        ids = self.players.track_ids
        if (
            not isfinite(self.fps)
            or self.fps <= 0
            or self.ball.shape != (len(self.ball), 3)
            or self.ball_valid.shape != (len(self.ball),)
            or self.ball_valid.dtype != np.bool_
            or self.players.positions.shape != (*shape, 3)
            or self.players.valid.shape != shape
            or self.players.valid.dtype != np.bool_
            or ids is None
            or ids.shape != (player_count,)
            or not np.issubdtype(ids.dtype, np.integer)
            or len(set(ids.tolist())) != player_count
            or np.any(ids < 0)
            or not np.isfinite(self.ball[self.ball_valid]).all()
            or not np.isfinite(self.players.positions[self.players.valid]).all()
        ):
            raise ValueError(
                "Invalid hitter timeline, observations, masks or local IDs"
            )

    @classmethod
    def from_archive(cls, archive: np.lib.npyio.NpzFile) -> HitterScene:
        return cls(
            archive["ball_3d"],
            archive["ball_3d_valid"],
            PlayerEvidence.from_pose(
                archive["player_position"],
                archive["player_valid"],
                archive["player_kp_3d"],
                archive["player_kp_3d_vis"],
                archive["player_track_ids"],
            ),
            float(archive["fps"]),
        )


@dataclass(frozen=True, slots=True)
class ContactEvidence:
    frame: int
    frame_offset: int
    landmark: str
    distance_m: float
    relative_distance: float
    root_y_m: float | None
    player_side: int
    side_alignment: str
    ball_y_m: float
    velocity_before_mps: list[float] | None
    velocity_after_mps: list[float] | None
    velocity_change_xy_mps: float | None


@dataclass(frozen=True, slots=True)
class HitterCandidate:
    local_player_index: int
    local_player_id: int
    score: float | None
    contact: ContactEvidence | None
    valid_frames_by_landmark: dict[str, list[int]]


@dataclass(frozen=True, slots=True)
class ShotHitter:
    rule_version: str
    resolution_status: Resolution
    local_player_index: int | None
    local_player_id: int | None
    selected_landmark: str | None
    best_score: float | None
    margin_to_second: float | None
    rejection_reason: str | None
    candidates: tuple[HitterCandidate, ...]
    evidence_json: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["candidates_json"] = list(record.pop("candidates"))
        return record


@dataclass(frozen=True, slots=True)
class _ContactWindow:
    frames: NDArray[np.intp]
    ball_valid: NDArray[np.bool_]
    motion_observed_frames: list[int]
    distances: NDArray[np.float64]
    root_y: NDArray[np.float32]
    player_sides: NDArray[np.int8]
    incoming: NDArray[np.float64]
    outgoing: NDArray[np.float64]
    motion_change: NDArray[np.float64]


def _contact_window(
    scene: HitterScene, frame: int, uncertainty: int, rules: HitterThresholds
) -> _ContactWindow:
    radius = max(uncertainty, ceil(scene.fps * rules.search_radius_s))
    start, stop = max(0, frame - radius), min(len(scene.ball), frame + radius + 1)
    frames = np.arange(start, stop)
    window = max(1, round(scene.fps * rules.motion_window_s))
    motion_start = max(0, start - window)
    motion_stop = min(len(scene.ball), stop + window)
    motion_slice = slice(start - motion_start, stop - motion_start)
    motion_valid = scene.ball_valid[motion_start:motion_stop]
    incoming, outgoing = side_velocities(
        scene.ball[motion_start:motion_stop].astype(np.float64),
        motion_valid,
        scene.fps,
        window,
    )
    incoming, outgoing = incoming[motion_slice], outgoing[motion_slice]
    motion_change = np.linalg.norm((outgoing - incoming)[:, :2], axis=-1)

    ball_valid = scene.ball_valid[start:stop]
    players = scene.players
    landmarks = players.positions[:, start:stop]
    root_y = landmarks[:, :, 0, 1]
    root_side_valid = players.valid[:, start:stop, 0] & (
        np.abs(root_y) > rules.side_deadband_m
    )
    player_sides = np.where(root_side_valid, np.sign(root_y), 0).astype(np.int8)
    distances = landmark_distances(
        landmarks,
        players.valid[:, start:stop] & ball_valid[:, None],
        scene.ball[start:stop, None, :].astype(np.float64),
    )
    return _ContactWindow(
        frames,
        ball_valid,
        (np.flatnonzero(motion_valid) + motion_start).tolist(),
        distances,
        root_y,
        player_sides,
        incoming,
        outgoing,
        motion_change,
    )


def _player_candidate(
    scene: HitterScene,
    window: _ContactWindow,
    player: int,
    track_id: int,
    event_frame: int,
    court_side: int,
    rules: HitterThresholds,
) -> HitterCandidate:
    distances = window.distances[player]
    observed = np.isfinite(distances)
    valid_frames = {
        name: window.frames[observed[:, index]].tolist()
        for index, name in enumerate(PLAYER_LANDMARKS)
    }
    if not observed.any():
        return HitterCandidate(player, track_id, None, None, valid_frames)
    relative = distances / (
        rules.root_reach_m,
        rules.wrist_reach_m,
        rules.wrist_reach_m,
    )
    has_motion = np.isfinite(window.motion_change)
    motion_score = np.where(has_motion, np.minimum(window.motion_change / 12.0, 1), 0)
    offsets = np.abs(window.frames - event_frame)
    normalized_offset = (offsets / scene.fps) / rules.max_frame_uncertainty_s
    sides = window.player_sides[player]
    side_penalty = np.zeros(len(offsets))
    if court_side:
        side_penalty = np.where(
            sides == court_side, 0, np.where(sides == 0, 0.10, 0.25)
        )
    # Distance dominates. Contact motion and mask support distinguish comparable samples.
    scores = (
        0.85 / (1 + relative**2) * (0.80, 1.0, 1.0)
        + 0.07 * motion_score[:, None]
        + 0.04 * observed.mean(axis=0)
        - 0.07 * np.minimum(normalized_offset, 1)[:, None]
        - side_penalty[:, None]
    )
    sample, landmark = np.unravel_index(
        np.where(observed, scores, -np.inf).argmax(), scores.shape
    )
    frame = int(window.frames[sample])
    side = int(sides[sample])
    if not court_side:
        alignment = "unconstrained"
    elif side == court_side:
        alignment = "match"
    else:
        alignment = "unknown" if side == 0 else "mismatch"
    before, after = window.incoming[sample], window.outgoing[sample]
    return HitterCandidate(
        player,
        track_id,
        float(scores[sample, landmark]),
        ContactEvidence(
            frame,
            frame - event_frame,
            PLAYER_LANDMARKS[landmark],
            float(distances[sample, landmark]),
            float(relative[sample, landmark]),
            float(window.root_y[player, sample])
            if scene.players.valid[player, frame, 0]
            else None,
            side,
            alignment,
            float(scene.ball[frame, 1]),
            before.tolist() if np.isfinite(before).all() else None,
            after.tolist() if np.isfinite(after).all() else None,
            float(window.motion_change[sample]) if has_motion[sample] else None,
        ),
        valid_frames,
    )


def _decision(
    first: ContactEvidence | None,
    second: ContactEvidence | None,
    margin: float | None,
    ball_frames: list[int],
    event_frame: int,
    uncertainty_s: float,
    rules: HitterThresholds,
) -> tuple[Resolution, str | None]:
    if not ball_frames:
        return "unresolved", "ball_3d_invalid"
    if first is None:
        return "unresolved", "no_valid_player"
    if first.relative_distance > 1:
        return "unresolved", "too_far"
    if uncertainty_s > rules.max_frame_uncertainty_s:
        return "ambiguous", "high_time_uncertainty"
    if abs(first.ball_y_m) <= rules.net_zone_m and (
        first.player_side == 0
        or (
            second is not None
            and second.relative_distance <= 1
            and first.player_side != second.player_side
        )
    ):
        return "ambiguous", "net_area_ambiguous"
    if first.side_alignment == "mismatch":
        return "unresolved", "side_mismatch"
    if first.side_alignment == "unknown":
        return "unresolved", "player_side_unavailable"
    if first.landmark == "root":
        return "unresolved", "root_only_low_confidence"
    if len(ball_frames) == 1 and first.frame != event_frame:
        return "unresolved", "sparse_ball_observations"
    if margin is not None and margin < rules.min_margin and second is not None:
        if first.player_side != 0 and first.player_side == second.player_side:
            return "ambiguous", "same_side_competing_players"
        return "ambiguous", "competing_players"
    return "resolved", None


def estimate_hitter(
    scene: HitterScene,
    *,
    frame: int,
    frame_uncertainty: int,
    court_side: int,
    thresholds: HitterThresholds = DEFAULT_THRESHOLDS,
) -> ShotHitter:
    """Score every player with joint-valid ball and landmark samples."""
    if (
        not 0 <= frame < len(scene.ball)
        or frame_uncertainty < 0
        or court_side not in (-1, 0, 1)
    ):
        raise ValueError("Invalid shot frame, frame uncertainty or court side")
    track_ids = scene.players.track_ids
    if track_ids is None:
        raise ValueError("Hitter estimation requires clip-local track IDs")
    window = _contact_window(scene, frame, frame_uncertainty, thresholds)
    candidates = sorted(
        (
            _player_candidate(
                scene, window, index, int(track_id), frame, court_side, thresholds
            )
            for index, track_id in enumerate(track_ids)
        ),
        key=lambda candidate: (
            -candidate.score if candidate.score is not None else np.inf,
            candidate.local_player_index,
        ),
    )
    best = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    score = best.score if best else None
    comparison = runner_up.score if runner_up else None
    margin = (
        score - comparison if score is not None and comparison is not None else None
    )
    first = best.contact if best else None
    second = runner_up.contact if runner_up else None
    ball_frames = window.frames[window.ball_valid].tolist()
    uncertainty_s = frame_uncertainty / scene.fps
    status, reason = _decision(
        first, second, margin, ball_frames, frame, uncertainty_s, thresholds
    )
    selected = best if status == "resolved" else None
    return ShotHitter(
        thresholds.rule_version,
        status,
        selected.local_player_index if selected else None,
        selected.local_player_id if selected else None,
        first.landmark if selected and first else None,
        score,
        margin,
        reason,
        tuple(candidates),
        {
            "decision": reason or "supported_wrist_with_score_margin",
            "event_frame": frame,
            "frame_uncertainty": frame_uncertainty,
            "frame_uncertainty_s": uncertainty_s,
            "court_side": court_side,
            "fps": scene.fps,
            "window_frames": window.frames.tolist(),
            "ball_valid_frames": ball_frames,
            "motion_observed_frames": window.motion_observed_frames,
            "motion_window_frames": max(
                1, round(scene.fps * thresholds.motion_window_s)
            ),
            "best_contact": asdict(first) if first else None,
        },
    )
