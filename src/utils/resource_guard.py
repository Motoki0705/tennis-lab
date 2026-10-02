"""Sampled host RAM guard with sustained-pressure detection and coarse telemetry."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import psutil  # type: ignore[import-untyped]

GIB = 1024**3


def available_ram_bytes() -> int:
    """Read Linux MemAvailable; missing/malformed telemetry is an error."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def process_tree_rss_bytes() -> int:
    """Sample current process and its living children, including compiler workers."""
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.NoSuchProcess:
            continue  # Child exited between discovery and measurement.
    return int(total)


@dataclass
class HostRAMGuard:
    """Trip below 3 GiB immediately, or after continuous samples below 6 GiB for 30s.

    Recovery to exactly the floor resets the timer. Time uses a monotonic clock.
    Timeline buckets retain every sampled minimum, maximum and process-tree peak;
    RSS sums can double-count shared pages, and peaks between samples are unknown.
    """

    floor_bytes: int = 6 * GIB
    emergency_bytes: int = 3 * GIB
    sustained_seconds: float = 30.0
    timeline_seconds: float = 10.0
    minimum_available_bytes: int | None = field(default=None, init=False)
    peak_process_tree_rss_bytes: int = field(default=0, init=False)
    samples: int = field(default=0, init=False)
    timeline: list[dict[str, float | int]] = field(default_factory=list, init=False)
    _below_since: float | None = field(default=None, init=False)
    _last_time: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.emergency_bytes < self.floor_bytes:
            raise ValueError("RAM thresholds require 0 < emergency < floor")
        if any(not math.isfinite(x) or x <= 0 for x in (self.sustained_seconds, self.timeline_seconds)):
            raise ValueError("RAM guard durations must be finite and positive")

    def sample(self, available_bytes: int, seconds: float, *, rss_bytes: int = 0) -> str | None:
        if (available_bytes < 0 or rss_bytes < 0 or not math.isfinite(seconds) or seconds < 0
                or (self._last_time is not None and seconds < self._last_time)):
            raise ValueError("Invalid/non-monotonic RAM sample")
        self._last_time = seconds
        self.samples += 1
        self.minimum_available_bytes = min(
            available_bytes, available_bytes if self.minimum_available_bytes is None else self.minimum_available_bytes,
        )
        self.peak_process_tree_rss_bytes = max(self.peak_process_tree_rss_bytes, rss_bytes)
        bucket = int(seconds // self.timeline_seconds)
        if not self.timeline or self.timeline[-1]["bucket"] != bucket:
            self.timeline.append({
                "bucket": bucket, "first_seconds": seconds, "last_seconds": seconds, "samples": 0,
                "min_available_bytes": available_bytes, "max_available_bytes": available_bytes,
                "peak_process_tree_rss_bytes": 0,
            })
        row = self.timeline[-1]
        row["last_seconds"] = seconds
        row["samples"] += 1
        row["min_available_bytes"] = min(row["min_available_bytes"], available_bytes)
        row["max_available_bytes"] = max(row["max_available_bytes"], available_bytes)
        row["peak_process_tree_rss_bytes"] = max(row["peak_process_tree_rss_bytes"], rss_bytes)
        if available_bytes < self.emergency_bytes:
            return f"RAM emergency: MemAvailable={available_bytes} < {self.emergency_bytes}"
        if available_bytes >= self.floor_bytes:
            self._below_since = None
        elif self._below_since is None:
            self._below_since = seconds
        elif seconds - self._below_since >= self.sustained_seconds:
            return f"RAM sustained breach: MemAvailable < {self.floor_bytes} for {seconds - self._below_since:.1f}s"
        return None

    def report(self) -> dict[str, object]:
        return {
            "floor_bytes": self.floor_bytes, "emergency_bytes": self.emergency_bytes,
            "sustained_seconds": self.sustained_seconds, "samples": self.samples,
            "minimum_available_bytes": self.minimum_available_bytes,
            "peak_process_tree_rss_bytes": self.peak_process_tree_rss_bytes,
            "timeline": self.timeline,
            "measurement": "sampled MemAvailable and process-tree RSS sum; shared pages may be counted twice",
        }
