"""Versioned clip-level trajectories, lifecycle and selection diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

NODE = "multiview_tracking"
SCHEMA = "multiview_pose_tracks"
VERSION = 1


@dataclass(frozen=True)
class TrackingConfig:
    players_per_side: int
    window_seconds: float = 1.2
    commit_seconds: float = 0.5
    max_occlusion_seconds: float = 0.4
    motion_span_seconds: float = 0.16
    max_speed_mps: float = 8.0
    ambiguity_margin: float = 0.08

    def __post_init__(self) -> None:
        durations = (
            self.window_seconds,
            self.commit_seconds,
            self.max_occlusion_seconds,
            self.motion_span_seconds,
            self.max_speed_mps,
            self.ambiguity_margin,
        )
        if (
            type(self.players_per_side) is not int
            or self.players_per_side not in (1, 2)
            or not all(isfinite(value) and value > 0 for value in durations)
            or self.window_seconds - self.commit_seconds
            < self.max_occlusion_seconds + self.motion_span_seconds
            or self.ambiguity_margin >= 1
        ):
            raise ValueError(
                "Invalid multiview tracking settings or insufficient window overlap"
            )


class SampleEvent(IntEnum):
    BORN = 1
    OBSERVED = 2
    OCCLUDED = 3
    REAPPEARED = 4


class TrackEnd(IntEnum):
    DISAPPEARED = 1
    CLIP_BOUNDARY = 2


SelectionReason = Literal[
    "selected", "insufficient_residence", "no_unique_side", "side_capacity"
]


@dataclass(frozen=True)
class PlayerDecision:
    observed_frames: int
    court_frames: int
    side_frames: tuple[int, int]
    alley_frames: int
    side: int
    motion_m: float
    motion_pairs: int
    rank_score: float
    reason: SelectionReason


@dataclass(frozen=True)
class PlayerSelection:
    players_per_side: int
    minimum_residence_frames: int
    decisions: tuple[PlayerDecision, ...]  # Indexed by clip-local track ID.

    @property
    def selected_track_ids(self) -> tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.decisions) if d.reason == "selected")

    @property
    def missing_per_side(self) -> tuple[int, int]:
        negative = sum(d.side == -1 and d.reason == "selected" for d in self.decisions)
        positive = sum(d.side == 1 and d.reason == "selected" for d in self.decisions)
        return self.players_per_side - negative, self.players_per_side - positive


@dataclass(frozen=True)
class PoseTracks:
    camera_ids: tuple[str, ...]
    frame_offsets: NDArray[np.int64]  # (T+1,), samples are sorted by (frame, ID).
    track_ids: NDArray[np.int64]  # (S,), IDs are stable within this clip.
    candidate_indices: NDArray[np.int64]  # (S,), flat C candidate indices or -1.
    source_rows: NDArray[np.int64]  # (S,V), original supporting DINO rows or -1.
    positions_m: NDArray[np.float32]  # (S,17,3), reference court coordinates.
    measured_joints: NDArray[np.bool_]  # (S,17), unchanged from C.
    predicted_joints: NDArray[np.bool_]  # (S,17), interpolated at occluded frames.
    torso_m: NDArray[np.float32]  # (S,3), centre of the available 3D torso.
    velocity_mps: NDArray[np.float32]  # (S,3), local estimate from track observations.
    velocity_valid: NDArray[np.bool_]  # (S,), zero velocity can be valid.
    event: NDArray[np.uint8]  # (S,), SampleEvent.
    frame_competing: NDArray[np.bool_]  # (S,), C compared another row assignment.
    frame_gap: NDArray[np.float32]  # (S,), C's relative score gap, zero if absent.
    frame_ambiguous: NDArray[np.bool_]  # (S,), C's per-frame ambiguity.
    temporal_margin: NDArray[np.float32]  # (S,), alternative link loss or -1.
    id_ambiguous: NDArray[np.bool_]  # (S,), local temporal ambiguity.
    end_frames: NDArray[np.int64]  # (K,), exclusive end, indexed by track ID.
    end_reasons: NDArray[np.uint8]  # (K,), TrackEnd.
    selection: PlayerSelection

    @property
    def observed(self) -> NDArray[np.bool_]:
        return self.candidate_indices >= 0

    def __post_init__(self) -> None:
        count, views = len(self.track_ids), len(self.camera_ids)
        tracks = len(self.end_frames)
        offsets = self.frame_offsets
        if (
            not 2 <= views <= 5
            or len(set(self.camera_ids)) != views
            or any(not camera for camera in self.camera_ids)
            or offsets.ndim != 1
            or offsets.dtype != np.int64
            or len(offsets) < 2
            or offsets[0] != 0
            or offsets[-1] != count
            or (np.diff(offsets) < 0).any()
        ):
            raise ValueError("Invalid tracking timeline or camera IDs")
        for array, dtype, shape in (
            (self.track_ids, np.int64, (count,)),
            (self.candidate_indices, np.int64, (count,)),
            (self.source_rows, np.int64, (count, views)),
            (self.positions_m, np.float32, (count, 17, 3)),
            (self.measured_joints, np.bool_, (count, 17)),
            (self.predicted_joints, np.bool_, (count, 17)),
            (self.torso_m, np.float32, (count, 3)),
            (self.velocity_mps, np.float32, (count, 3)),
            (self.velocity_valid, np.bool_, (count,)),
            (self.event, np.uint8, (count,)),
            (self.frame_competing, np.bool_, (count,)),
            (self.frame_gap, np.float32, (count,)),
            (self.frame_ambiguous, np.bool_, (count,)),
            (self.temporal_margin, np.float32, (count,)),
            (self.id_ambiguous, np.bool_, (count,)),
            (self.end_frames, np.int64, (tracks,)),
            (self.end_reasons, np.uint8, (tracks,)),
        ):
            if (
                array.shape != shape
                or array.dtype != dtype
                or not np.isfinite(array).all()
            ):
                raise ValueError("Invalid track array shape, dtype or value")
        self._validate_observations()
        self._validate_lifetimes()
        observed_counts = np.bincount(self.track_ids[self.observed], minlength=tracks)
        if (
            self.selection.players_per_side not in (1, 2)
            or not 1 <= self.selection.minimum_residence_frames <= len(offsets) - 1
            or len(self.selection.decisions) != tracks
            or min(self.selection.missing_per_side) < 0
        ):
            raise ValueError("Invalid clip selection")
        for decision, observed in zip(
            self.selection.decisions, observed_counts, strict=True
        ):
            if (
                decision.observed_frames != observed
                or not 0 <= decision.alley_frames <= decision.court_frames <= observed
                or min(decision.side_frames) < 0
                or sum(decision.side_frames) > decision.court_frames
                or decision.side
                != int(np.sign(decision.side_frames[1] - decision.side_frames[0]))
                or not 0 <= decision.motion_pairs <= observed
                or not isfinite(decision.motion_m)
                or not isfinite(decision.rank_score)
                or min(decision.motion_m, decision.rank_score) < 0
                or (
                    decision.reason == "selected"
                    and (
                        decision.side == 0
                        or decision.court_frames
                        < self.selection.minimum_residence_frames
                    )
                )
            ):
                raise ValueError("Invalid selection evidence")

    def _validate_observations(self) -> None:
        observed = self.observed
        present = self.source_rows >= 0
        valid = self.measured_joints | self.predicted_joints
        if (
            (self.candidate_indices < -1).any()
            or not np.array_equal(
                np.sort(self.candidate_indices[observed]), np.arange(observed.sum())
            )
            or (self.source_rows < -1).any()
            or not np.array_equal(present.sum(axis=1) >= 2, observed)
            or present[~observed].any()
            or not np.array_equal(self.measured_joints.any(axis=1), observed)
            or self.predicted_joints[observed].any()
            or (self.positions_m[~valid] != 0).any()
            or (self.velocity_mps[~self.velocity_valid] != 0).any()
            or not np.array_equal(self.event == SampleEvent.OCCLUDED, ~observed)
            or (self.frame_competing & ~observed).any()
            or (self.frame_ambiguous & ~self.frame_competing).any()
            or (self.frame_gap[~self.frame_competing] != 0).any()
            or ((self.frame_gap < 0) | (self.frame_gap > 1)).any()
            or ((self.temporal_margin < 0) & (self.temporal_margin != -1)).any()
            or (self.id_ambiguous & (self.temporal_margin < 0)).any()
        ):
            raise ValueError("Measurement, prediction or ambiguity masks disagree")

    def _validate_lifetimes(self) -> None:
        ids = self.track_ids
        frames: NDArray[np.int64] = np.repeat(
            np.arange(len(self.frame_offsets) - 1), np.diff(self.frame_offsets)
        )
        if (
            not np.array_equal(np.unique(ids), np.arange(len(self.end_frames)))
            or ((np.diff(ids) <= 0) & (np.diff(frames) == 0)).any()
            or not np.isin(self.end_reasons, list(TrackEnd)).all()
        ):
            raise ValueError("Invalid track IDs, frame order or endings")
        if not len(ids):
            return
        ordered = np.argsort(ids, kind="stable")
        events = self.event[ordered]
        same_track = np.diff(ids[ordered]) == 0
        starts = np.r_[True, ~same_track]
        ends = np.r_[~same_track, True]
        back_from_gap = np.r_[False, same_track & (events[:-1] == SampleEvent.OCCLUDED)]
        expected = np.where(starts, SampleEvent.BORN, SampleEvent.OBSERVED)
        expected[back_from_gap] = SampleEvent.REAPPEARED
        expected[(events == SampleEvent.OCCLUDED) & ~starts & ~ends] = (
            SampleEvent.OCCLUDED
        )
        if (
            not np.array_equal(events, expected)
            or not np.array_equal(self.end_frames, frames[ordered[ends]] + 1)
            or (np.diff(frames[ordered])[same_track] != 1).any()
        ):
            raise ValueError("Track lifecycle disagrees with its sample timeline")
