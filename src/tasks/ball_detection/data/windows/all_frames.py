"""Common native-frame intervals, paired training packs and evaluation owners."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.annotation_states import AnnotationStates
from src.tasks.ball_detection.data.store import BallFrameStore, ClipRecord
from src.tasks.ball_detection.data.windows.hybrid import (
    AbsencePolicy,
    eligible_spans,
    negative_mask,
    reviewed_empty_evidence,
)
from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig

LABEL_KINDS = ("masked", "observed", "reviewed_empty", "out_of_frame")
EDGE_BINS = ("0", "1-7", "8-15", "16-31", "32-63", "64+")


@dataclass(frozen=True)
class AllFrameWindowConfig:
    window_length: int
    max_window_length: int = 128
    pack_length: int = 128
    pack_stride: int = 16
    frame_step: int = 1
    timestamp_gap_factor: float = 2.5

    def __post_init__(self) -> None:
        values = (self.window_length, self.max_window_length, self.pack_length, self.pack_stride, self.frame_step)
        if any(type(v) is not int for v in values):
            raise ValueError("All-frame lengths and strides must be integers")
        if (self.window_length < 2 or self.window_length % 2 or self.window_length > self.max_window_length
                or self.pack_length != self.max_window_length or self.pack_length % self.window_length):
            raise ValueError("Even T must divide the pack length, which equals the common maximum T")
        if self.frame_step != 1 or not 1 <= self.pack_stride <= self.pack_length:
            raise ValueError("All-frame windows require native frame_step=1 and 1 <= pack stride <= pack length")
        if not math.isfinite(self.timestamp_gap_factor) or self.timestamp_gap_factor <= 1:
            raise ValueError("Timestamp gap factor must exceed one")


def window_starts(start: int, stop: int, length: int, stride: int) -> list[int]:
    """Left-anchored grid plus a single real-frame tail; never pad or repeat."""
    if (any(type(v) is not int for v in (start, stop, length, stride))
            or start < 0 or stop < start or length < 1 or not 1 <= stride <= length):
        raise ValueError("Invalid interval, window length or stride")
    last = stop - length
    if last < start:
        return []
    starts = list(range(start, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def common_intervals(states: AnnotationStates, config: AllFrameWindowConfig) -> list[tuple[int, int]]:
    # Use the existing native-PTS/reference barriers without a play-density gate.
    spans = eligible_spans(states, PlayIntervalConfig(timestamp_gap_factor=config.timestamp_gap_factor))
    return [(a, b) for a, b in spans if b - a >= config.max_window_length]


def label_kinds(states: AnnotationStates, reviewed_empty: NDArray[np.bool_]) -> NDArray[np.uint8]:
    masks = [states.supervision]
    for kind in LABEL_KINDS[2:]:
        policy = AbsencePolicy((kind,), dict(play=1., absent=0., nonplay=0.))
        masks.append(negative_mask(states, policy, reviewed_empty))
    if (np.sum(masks, axis=0) > 1).any():
        raise ValueError("All-frame teacher kinds overlap")
    codes: NDArray[np.uint8] = np.zeros(len(states.count), dtype=np.uint8)
    for code, mask in enumerate(masks, 1):
        codes[mask] = code
    return codes


def verified_label_kinds(
    store: BallFrameStore, clip: ClipRecord, states: AnnotationStates,
) -> tuple[NDArray[np.uint8], dict[str, Any]]:
    """Freeze Chat empty-frame evidence only after checking the original notes."""
    empty, evidence = reviewed_empty_evidence(store, clip, states)
    codes = stored_label_kinds(store, clip, states, empty)
    return codes, evidence


def stored_label_kinds(
    store: BallFrameStore, clip: ClipRecord, states: AnnotationStates, empty: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    codes = label_kinds(states, empty)
    if store.teacher_masks is not None:
        rows = store.clip_rows(clip)
        # Adoption can only restrict normalized teachers. No excluded frame is revived.
        codes[store.teacher_masks.heatmap[rows] == 0] = 0
    return codes


def training_packs(
    intervals: list[tuple[int, int]], codes: NDArray[np.uint8], config: AllFrameWindowConfig,
    *, require_position: bool = True,
) -> list[dict[str, int]]:
    prefix = np.r_[0, np.cumsum(codes == 1 if require_position else codes != 0)]
    return [dict(start=start, interval_index=i) for i, (a, b) in enumerate(intervals)
            for start in window_starts(a, b, config.pack_length, config.pack_stride)
            if prefix[start + config.pack_length] > prefix[start]]


def child_starts(pack_start: int, config: AllFrameWindowConfig) -> list[int]:
    return list(range(pack_start, pack_start + config.pack_length, config.window_length))


def owner_starts(start: int, stop: int, length: int) -> NDArray[np.int64]:
    """Nearest input center, then earliest start; independent of labels/scores."""
    if length < 2 or length % 2:
        raise ValueError("Evaluation requires an even window length >= 2")
    owners: NDArray[np.int64] = np.full(stop - start, -1, dtype=np.int64)
    best: NDArray[np.int64] = np.full(stop - start, np.iinfo(np.int64).max, dtype=np.int64)
    for first in window_starts(start, stop, length, length // 2):
        distance = np.abs(2 * np.arange(first, first + length) - (2 * first + length - 1))
        section = slice(first - start, first - start + length)
        better = distance < best[section]
        best[section][better] = distance[better]
        owners[section][better] = first
    if (owners < 0).any():
        raise ValueError("Uncovered common evaluation frame")
    return owners


def evaluation_windows(intervals: list[tuple[int, int]], length: int) -> list[dict[str, Any]]:
    windows = []
    for i, (a, b) in enumerate(intervals):
        owners = owner_starts(a, b, length)
        # Centers are ordered, so each window owns one contiguous, nonempty range.
        boundaries = np.r_[0, np.flatnonzero(np.diff(owners)) + 1, len(owners)]
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            windows.append(dict(start=int(owners[left]), interval_index=i,
                                owner_range=[a + int(left), a + int(right)]))
    return windows


def encode_label_runs(codes: NDArray[np.uint8]) -> list[list[int]]:
    bounds = np.r_[0, np.flatnonzero(np.diff(codes.astype(np.int16))) + 1, len(codes)]
    return [[int(a), int(b), int(codes[a])] for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def decode_label_runs(runs: list[list[int]], frame_count: int) -> NDArray[np.uint8]:
    codes: NDArray[np.uint8] = np.zeros(frame_count, dtype=np.uint8)
    previous = 0
    for run in runs:
        if (not isinstance(run, list) or len(run) != 3 or any(type(v) is not int for v in run)
                or run[0] != previous or not previous < run[1] <= frame_count or not 0 <= run[2] < len(LABEL_KINDS)):
            raise ValueError("Frozen label-kind runs must partition the entire clip")
        a, b, code = run
        codes[a:b] = code
        previous = b
    if previous != frame_count or encode_label_runs(codes) != runs:
        raise ValueError("Frozen label-kind runs must be complete and canonical")
    return codes
