"""Detect event boundaries from selected-view baseline 3D, without labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SegmentPhysicsConfig


@dataclass(frozen=True)
class FlightSegment:
    start_frame: int
    stop_frame: int


@dataclass(frozen=True)
class BallEvent:
    frame: int
    kind: str
    velocity_change_mps: float


def detect_segments(
    positions: NDArray[np.float32],
    valid: NDArray[np.bool_],
    fps: float,
    config: SegmentPhysicsConfig,
) -> tuple[tuple[FlightSegment, ...], tuple[BallEvent, ...]]:
    if (
        positions.shape != (len(valid), 3)
        or valid.dtype != bool
        or not np.isfinite(fps)
        or fps <= 0
        or not np.isfinite(positions[valid]).all()
    ):
        raise ValueError("Invalid initial trajectory/frame/fps axes")
    indices = np.flatnonzero(valid)
    groups = np.split(
        indices, np.flatnonzero(np.diff(indices) > config.max_gap_frames + 1) + 1
    )
    segments, events = [], []
    w = config.event_window_frames
    for group in groups:
        if len(group) < config.min_segment_frames:
            continue
        start, stop = int(group[0]), int(group[-1]) + 1
        candidates = []
        for f in range(start + w, stop - w):
            left = np.arange(f - w, f + 1)
            right = np.arange(f, f + w + 1)
            if not valid[np.concatenate((left, right))].all():
                continue
            # Independent side fits estimate velocity at the possible event.
            before = np.polynomial.polynomial.polyfit(
                (left - f) / fps, positions[left], 2
            )[1]
            after = np.polynomial.polynomial.polyfit(
                (right - f) / fps, positions[right], 2
            )[1]
            change = float(np.linalg.norm(after - before))
            bounce = (
                positions[f, 2] <= config.bounce_height_m
                and before[2] < -1
                and after[2] > 1
            )
            if change >= config.event_velocity_change_mps:
                candidates.append(BallEvent(f, "bounce" if bounce else "hit", change))
        selected: list[BallEvent] = []
        for event in sorted(
            candidates, key=lambda e: (-e.velocity_change_mps, e.frame)
        ):
            if min(event.frame - start, stop - event.frame) < config.min_segment_frames:
                continue
            if all(
                abs(event.frame - e.frame) >= config.min_segment_frames
                for e in selected
            ):
                selected.append(event)
        selected.sort(key=lambda e: e.frame)
        events.extend(selected)
        bounds = [start, *(e.frame for e in selected), stop]
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            # Long flights are numerically bounded; these cuts are not events.
            cuts: NDArray[np.int64] = np.linspace(
                lo,
                hi,
                int(np.ceil((hi - lo) / config.max_segment_frames)) + 1,
                dtype=int,
            )
            for a, b in zip(cuts[:-1], cuts[1:], strict=True):
                if valid[a:b].sum() >= config.min_segment_frames:
                    segments.append(FlightSegment(int(a), int(b)))
    return tuple(segments), tuple(sorted(events, key=lambda e: e.frame))
