"""Frame-index sampling helpers for decoded videos."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def parse_time_seconds(value: Any) -> float:
    """Parse seconds from a number or ``HH:MM:SS``-style string."""
    if isinstance(value, int | float):
        return float(value)
    parts = str(value).split(":")
    if len(parts) == 1:
        return float(parts[0])
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def sample_step_seconds(
    *,
    sample_mode: str,
    fps: float,
    interval_seconds: float,
    target_fps: float,
    every_n_frames: int,
) -> float:
    """Return frame sampling step in seconds for a named sampling mode."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if sample_mode == "interval_seconds":
        return float(interval_seconds)
    if sample_mode == "fps":
        return 1.0 / float(target_fps)
    if sample_mode == "every_n_frames":
        return float(every_n_frames) / fps
    raise ValueError(f"Unsupported sample_mode={sample_mode!r}.")


def sample_frame_indices_by_time_ranges(
    raw_ranges: Iterable[Mapping[str, Any]] | None,
    *,
    duration: float,
    fps: float,
    sample_mode: str,
    interval_seconds: float,
    target_fps: float,
    every_n_frames: int,
    max_frames: int | None = None,
) -> list[int]:
    """Sample unique frame indices from timestamp ranges."""
    ranges = [] if raw_ranges is None else list(raw_ranges)
    if not ranges:
        ranges = [{"start": 0.0, "end": duration}]
    step_sec = sample_step_seconds(
        sample_mode=sample_mode,
        fps=fps,
        interval_seconds=interval_seconds,
        target_fps=target_fps,
        every_n_frames=every_n_frames,
    )
    frame_indices: list[int] = []
    seen: set[int] = set()
    for time_range in ranges:
        expected_keys = {"start", "end"}
        actual_keys = set(time_range)
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            unknown = actual_keys - expected_keys
            raise ValueError(
                "time range keys must be exactly ['end', 'start']: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        start_sec = parse_time_seconds(time_range["start"])
        end_sec = parse_time_seconds(time_range["end"])
        timestamp = max(start_sec, 0.0)
        while timestamp <= max(end_sec, start_sec):
            frame_index = int(round(timestamp * fps))
            if frame_index not in seen:
                seen.add(frame_index)
                frame_indices.append(frame_index)
                if max_frames is not None and len(frame_indices) >= max_frames:
                    return frame_indices
            timestamp += step_sec
    return frame_indices


def sample_uniform_frame_indices(frame_count: int, sample_count: int) -> list[int]:
    """Sample up to ``sample_count`` unique indices uniformly over a video."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}.")
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}.")
    if sample_count >= frame_count:
        return list(range(frame_count))
    if sample_count == 1:
        return [frame_count // 2]

    last_index = frame_count - 1
    return [round(i * last_index / (sample_count - 1)) for i in range(sample_count)]


__all__ = [
    "parse_time_seconds",
    "sample_frame_indices_by_time_ranges",
    "sample_step_seconds",
    "sample_uniform_frame_indices",
]
