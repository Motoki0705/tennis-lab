"""Describe first-stage fragments and assemble candidate player slots."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.multiview_pose.schema import FramePoseCandidates
from src.tennis_scene.multiview_tracking.association import Motion, Observations
from src.tennis_scene.multiview_tracking.schema import FragmentConfig, PlayerSelection
from src.tennis_scene.multiview_tracking.slot_assignment import (
    FragmentLink,
    SlotAssignment,
    assign_slots,
    reachable_fragments,
)
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES

_BONES = np.asarray(COCO17_BONE_LENGTH_EDGES, dtype=np.int64)
_MIN_SHARED_BONES = 3
_MIN_RESIDENCE_SHARE = 0.55


@dataclass(frozen=True)
class Endpoint:
    frame: int
    position_m: NDArray[np.float64]
    velocity_mps: NDArray[np.float64] | None


@dataclass(frozen=True)
class Fragment:
    index: int
    first: int  # Candidate indices on the original measured timeline.
    last: int
    start: Endpoint
    end: Endpoint
    bone_lengths_m: NDArray[np.float64]
    side: int
    residence_frames: int
    quality: float
    rank: float
    eligible: bool


@dataclass(frozen=True)
class FragmentJoin:
    before_fragment: int
    after_fragment: int
    last_observed_frame: int
    first_observed_frame: int
    gap_frames: int
    long_gap: bool
    confidence: float  # Bounded compatibility score.
    objective_margin: float  # Global optimum loss when this slot link is forbidden.
    ambiguous: bool
    position_residual_m: float
    velocity_difference_mps: float | None
    log_bone_difference: float
    shared_bones: int


@dataclass(frozen=True)
class PlayerSlot:
    side: int
    core_fragment: int
    track_id: int  # Output ID after assigning merged predecessors.
    fragments: tuple[int, ...]
    joins: tuple[FragmentJoin, ...]


@dataclass(frozen=True)
class FragmentIntegration:
    first_stage_track_count: int
    slots: tuple[PlayerSlot, ...]


def _endpoint(
    indices: NDArray[np.int64],
    frame: int,
    observations: Observations,
    motion: Motion,
    fps: float,
) -> Endpoint:
    valid = indices[motion.valid[indices]]
    points = observations.centres[indices]
    velocity = None
    if len(valid):
        velocity = np.median(motion.velocity[valid], axis=0)
        time_to_endpoint = (frame - observations.frames[indices]) / fps
        points = points + time_to_endpoint[:, None] * velocity
    return Endpoint(frame, np.median(points, axis=0), velocity)


def describe_fragments(
    candidates: FramePoseCandidates,
    observations: Observations,
    motion: Motion,
    ids: NDArray[np.int64],
    selection: PlayerSelection,
    fps: float,
    config: FragmentConfig,
) -> tuple[Fragment, ...]:
    order = np.argsort(ids, kind="stable")
    bounds = np.r_[0, np.cumsum(np.bincount(ids))]
    frames = observations.frames
    span = round(config.endpoint_span_seconds * fps)
    min_residence = max(2, ceil(config.min_residence_seconds * fps))
    fragments = []
    for index, decision in enumerate(selection.decisions):
        rows = order[bounds[index] : bounds[index + 1]]
        start, end = int(frames[rows[0]]), int(frames[rows[-1]])
        points = candidates.positions_m[rows].astype(np.float64)
        valid = candidates.joint_valid[rows]
        lengths = np.linalg.norm(
            points[:, _BONES[:, 0]] - points[:, _BONES[:, 1]], axis=-1
        )
        supported = valid[:, _BONES].all(axis=-1) & (lengths > 0)
        bones: NDArray[np.float64] = np.zeros(len(_BONES), np.float64)
        for bone in np.flatnonzero(supported.sum(axis=0) >= 3):
            bones[bone] = np.median(lengths[supported[:, bone], bone])
        quality = float(
            1 - observations.penalty[rows].mean() - 0.25 * (1 - valid.mean())
        )
        residence = max(decision.side_frames)
        eligible = (
            decision.side != 0
            and residence >= min_residence
            and residence / len(rows) >= _MIN_RESIDENCE_SHARE
            and np.count_nonzero(bones) >= _MIN_SHARED_BONES
        )
        fragments.append(
            Fragment(
                index,
                int(rows[0]),
                int(rows[-1]),
                _endpoint(
                    rows[frames[rows] <= start + span], start, observations, motion, fps
                ),
                _endpoint(
                    rows[frames[rows] >= end - span], end, observations, motion, fps
                ),
                bones,
                decision.side,
                residence,
                quality,
                decision.rank_score * (0.75 + 0.25 * quality),
                bool(eligible),
            )
        )
    return tuple(fragments)


def _cores(
    options: list[Fragment],
    selected: set[int],
    players_per_side: int,
    links: list[FragmentLink],
) -> tuple[int, ...]:
    preferred = sorted(
        (f for f in options if f.index in selected), key=lambda f: f.index
    )
    provisional = sorted(
        (f for f in options if f.index not in selected),
        key=lambda f: (-f.rank, f.index),
    )
    first, *others = [*preferred, *provisional]
    if players_per_side == 1:
        return (first.index,)

    # Another piece of an admissible path through the first core can join
    # that player even if both pieces qualify on their own.
    continuations = reachable_fragments(first.index, links)
    first_path_options = [f for f in options if f.index in continuations]
    for fragment in others:
        if fragment.index not in continuations and (
            fragment.index in selected
            or any(
                fragment.start.frame <= piece.end.frame
                and piece.start.frame <= fragment.end.frame
                for piece in first_path_options
            )
        ):
            return (first.index, fragment.index)
    return (first.index,)


def _link(
    before: Fragment, after: Fragment, fps: float, config: FragmentConfig
) -> FragmentLink | None:
    end, start = before.end, after.start
    gap_seconds = (start.frame - end.frame - 1) / fps
    if not 0 <= gap_seconds <= config.max_gap_seconds:
        return None
    common = (before.bone_lengths_m > 0) & (after.bone_lengths_m > 0)
    shared = int(common.sum())
    if shared < _MIN_SHARED_BONES:
        return None
    shape = float(
        np.median(
            np.abs(np.log(after.bone_lengths_m[common] / before.bone_lengths_m[common]))
        )
    )
    if shape > config.max_log_bone_difference:
        return None
    dt = (start.frame - end.frame) / fps
    shift = start.position_m - end.position_m
    distance = float(np.linalg.norm(shift))
    # The extrapolation horizon is short even for long gaps. Tennis players
    # can turn or brake during the missing interval.
    horizon = min(dt, 0.25)
    velocities = [
        velocity
        for velocity in (end.velocity_mps, start.velocity_mps)
        if velocity is not None
    ]
    predictions = [
        float(np.linalg.norm(shift - horizon * velocity)) for velocity in velocities
    ]
    position = (distance + sum(predictions)) / (1 + len(predictions))
    position_limit = min(5.0, 0.95 + 2.4 * dt)
    if position > position_limit:
        return None
    velocity_error = None
    velocity_cost = 0.035
    if end.velocity_mps is not None and start.velocity_mps is not None:
        velocity_error = float(np.linalg.norm(end.velocity_mps - start.velocity_mps))
        velocity_cost = 0.12 * exp(-dt / 0.60) * min(1.0, velocity_error / 5.0)
    cost = (
        0.12
        + 0.30 * (position / position_limit) ** 2
        + 0.22 * (shape / config.max_log_bone_difference) ** 2
        + 0.18 * gap_seconds / config.max_gap_seconds
        + velocity_cost
    )
    return FragmentLink(
        before.index, after.index, cost, position, velocity_error, shape, shared
    )


def player_slots(
    fragments: tuple[Fragment, ...],
    selection: PlayerSelection,
    fps: float,
    config: FragmentConfig,
) -> tuple[SlotAssignment, ...]:
    assignments: list[SlotAssignment] = []
    selected = set(selection.selected_track_ids)
    for side in (-1, 1):
        options = sorted(
            (f for f in fragments if f.side == side and (f.eligible or f.index in selected)),
            key=lambda f: (f.start.frame, f.index),
        )
        if not options:
            continue
        rewards = {
            f.index: (
                min(1.0, f.residence_frames / (0.35 * fps))
                + min(1.0, max(0.0, (f.residence_frames / fps - 0.35) / 0.8))
            )
            * (0.75 + 0.25 * f.quality)
            for f in options
        }
        links = []
        for i, before in enumerate(options):
            for after in options[i + 1 :]:
                if (
                    after.start.frame - before.end.frame - 1
                    > config.max_gap_seconds * fps
                ):
                    break
                link = _link(before, after, fps, config)
                if link is not None:
                    links.append(link)
        cores = _cores(options, selected, selection.players_per_side, links)
        assignments.extend(assign_slots(cores, rewards, links))
    return tuple(assignments)
