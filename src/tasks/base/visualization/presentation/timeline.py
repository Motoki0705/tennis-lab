"""Map presentation frames to named segments and learning steps."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Segment:
    """A positive video duration mapped to a nondecreasing learning interval."""

    name: str
    video_seconds: float
    step_start: int
    step_stop: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("segment name must be nonempty")
        if (
            isinstance(self.video_seconds, bool)
            or not math.isfinite(self.video_seconds)
            or self.video_seconds <= 0
        ):
            raise ValueError("video_seconds must be finite and positive")
        for step in (self.step_start, self.step_stop):
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError("learning steps must be nonnegative integers")
        if self.step_stop < self.step_start:
            raise ValueError("step_stop must be at least step_start")


class StepClock:
    """Quantize cumulative segment boundaries to the nearest frame (ties up).

    Each segment owns a half-open frame interval. Within it, the first and last
    frames have progress 0 and 1; a one-frame segment represents its endpoint.
    Step values may reset between segments. Durations that quantize to zero
    frames are rejected rather than silently removing a stage.
    """

    def __init__(self, segments: Sequence[Segment], fps: float) -> None:
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        self._segments = tuple(segments)
        if not self._segments:
            raise ValueError("at least one segment is required")
        boundaries = [0]
        seconds = Fraction(0)
        for segment in self._segments:
            seconds += Fraction(str(segment.video_seconds))
            boundary = math.floor(seconds * Fraction(str(fps)) + Fraction(1, 2))
            if boundary <= boundaries[-1]:
                raise ValueError(f"segment {segment.name!r} quantizes to zero frames")
            boundaries.append(boundary)
        self._boundaries = tuple(boundaries)

    def frame_count(self) -> int:
        """Return the total number of frames; valid indices start at zero."""
        return self._boundaries[-1]

    def at(self, frame: int) -> tuple[Segment, float, float]:
        """Return the segment, local progress in [0, 1], and linear step."""
        if isinstance(frame, bool) or not isinstance(frame, int):
            raise TypeError("frame must be an integer")
        if not 0 <= frame < self.frame_count():
            raise IndexError(f"frame must be in [0, {self.frame_count()})")
        index = bisect_right(self._boundaries, frame) - 1
        start, stop = self._boundaries[index : index + 2]
        progress = (frame - start) / (stop - start - 1) if stop - start > 1 else 1.0
        segment = self._segments[index]
        step = segment.step_start + progress * (segment.step_stop - segment.step_start)
        return segment, progress, step
