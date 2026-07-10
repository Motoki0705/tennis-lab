"""Global-timeline math shared by the clip studio GUI and exporter.

All functions use the project-wide sync convention
``local_time = global_time + offset_sec`` (see :mod:`.project`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


def source_frame_index(
    global_sec: float,
    *,
    offset_sec: float,
    fps: float,
    frame_count: int,
) -> int | None:
    """Map a global time to the nearest source frame index.

    Returns:
        The nearest frame index, or ``None`` when the global time falls
        outside the source's coverage.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if frame_count < 0:
        raise ValueError(f"frame_count must be non-negative, got {frame_count}")
    index = round((global_sec + offset_sec) * fps)
    if 0 <= index < frame_count:
        return index
    return None


def source_coverage_sec(*, offset_sec: float, duration_sec: float) -> tuple[float, float]:
    """Global interval ``[start, end]`` covered by a source."""
    if duration_sec < 0:
        raise ValueError(f"duration_sec must be non-negative, got {duration_sec}")
    return (-offset_sec, duration_sec - offset_sec)


def timeline_extent_sec(
    coverages: Iterable[tuple[float, float]],
) -> tuple[float, float]:
    """Union extent of all source coverages on the global timeline."""
    coverage_list = list(coverages)
    if not coverage_list:
        raise ValueError("coverages must contain at least one interval")
    return (
        min(start for start, _ in coverage_list),
        max(end for _, end in coverage_list),
    )


def format_timecode(seconds: float) -> str:
    """Format seconds as ``[-]H:MM:SS.mmm``."""
    sign = "-" if seconds < 0 else ""
    total_ms = round(abs(seconds) * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    hours, rem = divmod(total_s, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{sign}{hours}:{minutes:02d}:{secs:02d}.{ms:03d}"


@dataclass(frozen=True)
class TimelineGeometry:
    """Pixel <-> seconds mapping of the on-screen timeline strip.

    Attributes:
        x, y, width, height: Strip rectangle in canvas pixels.
        view_start_sec, view_end_sec: Visible global-time window.
    """

    x: int
    y: int
    width: int
    height: int
    view_start_sec: float
    view_end_sec: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"timeline rect must be positive, got {self.width}x{self.height}")
        if self.view_end_sec <= self.view_start_sec:
            raise ValueError(
                "view_end_sec must be greater than view_start_sec, "
                f"got [{self.view_start_sec}, {self.view_end_sec}]"
            )

    @property
    def view_span_sec(self) -> float:
        return self.view_end_sec - self.view_start_sec

    def sec_to_x(self, seconds: float) -> int:
        """Map global seconds to a canvas x pixel (not clamped)."""
        ratio = (seconds - self.view_start_sec) / self.view_span_sec
        return self.x + round(ratio * (self.width - 1))

    def x_to_sec(self, x_pixel: float) -> float:
        """Map a canvas x pixel to global seconds (not clamped)."""
        ratio = (x_pixel - self.x) / max(self.width - 1, 1)
        return self.view_start_sec + ratio * self.view_span_sec

    def contains(self, x_pixel: int, y_pixel: int) -> bool:
        return (
            self.x <= x_pixel < self.x + self.width
            and self.y <= y_pixel < self.y + self.height
        )


__all__ = [
    "TimelineGeometry",
    "format_timecode",
    "source_coverage_sec",
    "source_frame_index",
    "timeline_extent_sec",
]
