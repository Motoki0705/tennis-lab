"""Annotation-based play proposals; rejected frames are not semantic negatives.

This is a dataset selection policy, never an inference-time ball detector.
Short interior evidence gaps may be bridged; leading/trailing absence is not.
Hit/bounce segment markers do not by themselves end a rally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PlayIntervalConfig:
    window_length: int = 32
    window_stride: int = 16
    max_gap_seconds: float = 0.4
    min_presence_fraction: float = 0.5
    min_observed_frames: int = 8
    timestamp_gap_factor: float = 2.5

    def __post_init__(self) -> None:
        if not 1 <= self.window_stride <= self.window_length:
            raise ValueError("Require 1 <= stride <= window length")
        if not 1 <= self.min_observed_frames <= self.window_length:
            raise ValueError("Observed-frame requirement exceeds window length")
        if not np.isfinite(self.max_gap_seconds) or self.max_gap_seconds < 0:
            raise ValueError("Gap duration must be finite and nonnegative")
        if not 0 < self.min_presence_fraction <= 1:
            raise ValueError("Presence fraction must be in (0, 1]")
        if not np.isfinite(self.timestamp_gap_factor) or self.timestamp_gap_factor <= 1:
            raise ValueError("Timestamp gap factor must exceed one")


@dataclass(frozen=True)
class PlaySelection:
    intervals: tuple[tuple[int, int], ...]
    excluded: tuple[tuple[int, int], ...]
    window_starts: tuple[int, ...]
    bridged: NDArray[np.bool_]
    selected: NDArray[np.bool_]
    play: NDArray[np.bool_]


def mask_intervals(mask: NDArray[np.bool_]) -> tuple[tuple[int, int], ...]:
    """Maximal half-open runs; every frame occurs exactly once in a partition."""
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("Expected a one-dimensional boolean mask")
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return tuple((int(a), int(b)) for a, b in zip(
        np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True,
    ))


def infer_play_intervals(
    presence: NDArray[np.bool_],
    observed: NDArray[np.bool_],
    timestamps: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    config: PlayIntervalConfig,
) -> PlaySelection:
    """Bridge bounded gaps, then select real windows with enough ball evidence.

    Presence includes a single annotated observed/estimated/unresolved ball;
    only observed frames can supply coordinate supervision. Explicit absence,
    unknown and ambiguous frames supply neither. Bridging changes selection,
    never labels. Ineligible reference frames and timestamp jumps are barriers.
    Play proposals and supervised-window coverage are separate: an unresolved
    rally may be play even when no coordinate-training window can be used.
    """
    n = len(timestamps)
    if any(x.dtype != np.bool_ or x.shape != (n,) for x in (presence, observed, eligible)):
        raise ValueError("Evidence masks must be boolean vectors on one timeline")
    if timestamps.shape != (n,) or not np.isfinite(timestamps).all():
        raise ValueError("Expected finite timestamps")
    if (observed & ~presence).any():
        raise ValueError("Observed evidence must imply presence")
    if n > 1 and (np.diff(timestamps) <= 0).any():
        raise ValueError("Timestamps must increase strictly")
    filled = presence & eligible
    selected: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
    play: NDArray[np.bool_] = np.zeros(n, dtype=np.bool_)
    if n < config.window_length:
        return PlaySelection((), mask_intervals(~selected), (), filled & ~presence, selected, play)
    dt = float(np.median(np.diff(timestamps)))
    cuts = np.flatnonzero(np.diff(timestamps) > config.timestamp_gap_factor * dt) + 1
    boundaries = np.r_[0, cuts, n]
    starts: list[int] = []
    for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
        # Eligibility barriers (e.g. reference-only images) may not be crossed.
        for ea, eb in mask_intervals(eligible[left:right]):
            a, b = int(left + ea), int(left + eb)
            for ga, gb in mask_intervals(~filled[a:b]):
                ga, gb = ga + a, gb + a
                if ga == a or gb == b:
                    continue
                missing_duration = float(timestamps[gb] - timestamps[ga - 1] - dt)
                if missing_duration <= config.max_gap_seconds + 1e-9:
                    filled[ga:gb] = True
            for ra, rb in mask_intervals(filled[a:b]):
                ra, rb = ra + a, rb + a
                if rb - ra < config.window_length or presence[ra:rb].mean() < config.min_presence_fraction:
                    continue
                play[ra:rb] = True
                candidates = list(range(ra, rb - config.window_length + 1, config.window_stride))
                tail = rb - config.window_length
                if candidates[-1] != tail:
                    candidates.append(tail)
                for start in candidates:
                    stop = start + config.window_length
                    if (presence[start:stop].mean() >= config.min_presence_fraction
                            and observed[start:stop].sum() >= config.min_observed_frames):
                        starts.append(start)
                        selected[start:stop] = True
    return PlaySelection(
        mask_intervals(play), mask_intervals(~play), tuple(starts),
        filled & ~presence, selected, play,
    )
