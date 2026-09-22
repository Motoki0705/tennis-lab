"""Deterministic time-uniform sampling without duplicated target frames."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from fractions import Fraction

from .runtime.contracts import FrameRange
from .runtime.media import Timeline


def frame_boundaries(timeline: Timeline) -> list[Fraction]:
    if not timeline.pts:
        raise ValueError("cannot sample a video with no frames")
    return [timeline.boundary(i) for i in range(len(timeline.pts) + 1)]


def dense_target_ranges(
    timeline: Timeline, duration: float, context: float
) -> list[FrameRange]:
    boundaries = frame_boundaries(timeline)
    core = Fraction(str(duration)) - 2 * Fraction(str(context))
    if core <= 0:
        raise ValueError("duration must exceed twice the context")
    result: list[FrameRange] = []
    start = 0
    while start < len(timeline.pts):
        stop = max(start + 1, bisect_right(boundaries, boundaries[start] + core) - 1)
        result.append(FrameRange(start=start, stop=stop))
        start = stop
    return result


def select_target_ranges(
    timeline: Timeline,
    duration: float,
    context: float,
    max_clips: int | None,
    strategy: str,
) -> tuple[list[FrameRange], int]:
    """Cap clips by choosing windows at equal time-bin centers, not frame quantiles.

    If the complete partition already fits the cap, keep it, including its
    final short clip. Otherwise select one window per time bin; quantize to
    presentation boundaries and keep the target wholly inside its bin.
    """
    if strategy != "uniform_midpoints":
        raise ValueError(f"unsupported sampling strategy: {strategy}")
    if max_clips is not None and (type(max_clips) is not int or max_clips < 1):
        raise ValueError("max_clips must be a positive integer or null")
    candidates = dense_target_ranges(timeline, duration, context)
    if max_clips is None or len(candidates) <= max_clips:
        return candidates, len(candidates)
    boundaries = frame_boundaries(timeline)
    total = boundaries[-1]
    core = Fraction(str(duration)) - 2 * Fraction(str(context))
    selected: list[FrameRange] = []
    for slot in range(max_clips):
        left, right = total * slot / max_clips, total * (slot + 1) / max_clips
        center = (left + right) / 2
        window = min(core, right - left)
        start = bisect_left(boundaries, center - window / 2)
        stop = bisect_right(boundaries, min(right, boundaries[start] + window)) - 1
        if stop <= start or start >= len(timeline.pts):
            raise ValueError(
                "a uniform time bin contains no complete frame; reduce the sampling count"
            )
        selected.append(FrameRange(start=start, stop=stop))
    return selected, len(candidates)


def centered_subrange(timeline: Timeline, target: FrameRange) -> FrameRange:
    """Shrink a size-limited sampling slot, never turn one slot into two files."""
    if target.stop - target.start < 2:
        raise ValueError("cannot shrink a one-frame target")
    boundaries = frame_boundaries(timeline)
    center = (boundaries[target.start] + boundaries[target.stop]) / 2
    quarter = (boundaries[target.stop] - boundaries[target.start]) / 4
    start = bisect_left(boundaries, center - quarter, target.start, target.stop + 1)
    stop = bisect_right(boundaries, center + quarter, target.start, target.stop + 1) - 1
    if (
        stop <= start
        or stop - start >= target.stop - target.start
        or not boundaries[start] <= center < boundaries[stop]
    ):
        start = min(
            target.stop - 1, max(target.start, bisect_right(boundaries, center) - 1)
        )
        stop = start + 1
    return FrameRange(start=start, stop=stop)
