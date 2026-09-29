"""Rational resampling and retained simulator events, independent of I/O."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyResult


@dataclass(frozen=True)
class Event:
    kind: str
    native_frame: int
    seconds: float
    frame: int


def resample(
    positions: NDArray[np.floating], *, native_hz: int, numerator: int,
    denominator: int, max_frames: int,
) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("Need at least two native XYZ samples")
    if min(native_hz, numerator, denominator, max_frames) <= 0 or not np.isfinite(positions).all():
        raise ValueError("Invalid sampling contract")
    # Integer arithmetic decides support; the last sample never extrapolates.
    count = min(max_frames, (len(positions) - 1) * numerator // (native_hz * denominator) + 1)
    timestamps = np.arange(count, dtype=np.float64) * denominator / numerator
    native_times = np.arange(len(positions), dtype=np.float64) / native_hz
    sampled = np.column_stack([np.interp(timestamps, native_times, positions[:, axis]) for axis in range(3)])
    return timestamps, sampled.astype(np.float32)


def retained_events(result: RallyResult, timestamps: NDArray[np.float64]) -> list[Event]:
    if result.fps_out != result.sim_fps:
        raise ValueError("Simulator events must use native-rate frame indices")
    events: list[Event] = []
    for i, shot in enumerate(result.shot_events):
        stop = result.shot_events[i + 1].t_start if i + 1 < len(result.shot_events) else len(result.trajectory_sim)
        candidates = [
            ("hit", shot.t_start),
            ("bounce", shot.t_bounce1), ("bounce", shot.t_bounce2),
            ("bounce", shot.t_bounce3), ("net", shot.t_net),
        ]
        for kind, native_frame in candidates:
            seconds = native_frame / result.sim_fps
            # Future bounces in an interrupted shot are NOT rally events.
            if not shot.t_start <= native_frame < stop or seconds > timestamps[-1]:
                continue
            frame = int(np.abs(timestamps - seconds).argmin())
            events.append(Event(kind, native_frame, seconds, frame))
    return sorted(events, key=lambda event: (event.native_frame, event.kind))


def event_masks(
    count: int, events: list[Event], *, radius: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
    if count < 1 or radius < 1:
        raise ValueError("Need positive count and event exclusion radius")
    labels: NDArray[np.bool_] = np.zeros((count, 2), dtype=bool)  # independent hit/bounce labels
    event_region: NDArray[np.bool_] = np.zeros(count, dtype=bool)
    excluded: NDArray[np.bool_] = np.zeros(count, dtype=bool)
    for event in events:
        if not 0 <= event.frame < count:
            raise ValueError("Event is outside stored support")
        region = slice(max(0, event.frame - radius), min(count, event.frame + radius + 1))
        excluded[region] = True  # net crossing/collision also excluded conservatively
        if event.kind in ("hit", "bounce"):
            labels[event.frame, int(event.kind == "bounce")] = True
            event_region[region] = True
        elif event.kind != "net":
            raise ValueError(f"Unknown event type: {event.kind}")
    free_flight = ~excluded
    free_flight[[0, -1]] = False
    return labels, event_region, free_flight
