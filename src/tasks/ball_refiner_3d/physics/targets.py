"""Physics supervision derived from a rally's ``ball_physics.v1`` record."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.physics.units import (
    SURFACES,
    Field,
    State,
    encode_field,
    encode_state,
)
from src.utils.physics.ball.record import BallPhysicsRecord


@dataclass(frozen=True)
class FlightClock:
    """Integration constants shared by every rally of a dataset."""

    gravity: float
    dt: float
    substeps: int

    @classmethod
    def of(cls, records: list[BallPhysicsRecord]) -> FlightClock:
        clocks = {(r.gravity, r.dt, r.stride) for r in records}
        if len(clocks) != 1:
            raise ValueError(f"Rallies disagree on gravity/dt/stride: {sorted(clocks)}")
        gravity, dt, substeps = clocks.pop()
        return cls(float(gravity), float(dt), int(substeps))


@dataclass(frozen=True)
class PhysicsTargets:
    """Per-frame segment labels and states, and the rally field, in network units.

    ``state[t]`` is the flight state at frame ``t``; integrating it reproduces
    the rest of its segment, so every frame can open a window-local segment.
    """

    segment: NDArray[np.int64]  # (T,)
    state: NDArray[np.float32]  # (T,9)
    field: NDArray[np.float32]  # (4,)
    surface: int


def physics_targets(
    record: BallPhysicsRecord, positions: NDArray[np.floating]
) -> PhysicsTargets:
    if positions.shape != (record.frames, 3):
        raise ValueError("Positions must match the record frames")
    state = encode_state(
        State(
            torch.as_tensor(positions, dtype=torch.float32),
            torch.as_tensor(record.velocity, dtype=torch.float32),
            torch.as_tensor(record.frame_spin(), dtype=torch.float32),
        )
    )
    field = encode_field(
        Field(
            torch.as_tensor(record.wind, dtype=torch.float32),
            torch.tensor(record.k_drag, dtype=torch.float32),
            torch.tensor(record.k_magnus, dtype=torch.float32),
        )
    )
    return PhysicsTargets(
        record.frame_segment(),
        state.numpy(),
        field.numpy(),
        SURFACES.index(record.surface),
    )


def local_segments(segment: NDArray[np.integer]) -> NDArray[np.int64]:
    """Relabel a contiguous slice of segment labels from 0, in order."""
    if len(segment) == 0 or np.any(np.diff(segment) < 0):
        raise ValueError("Segment labels must be nonempty and nondecreasing")
    opening: NDArray[np.bool_] = np.ones(len(segment), dtype=bool)
    opening[1:] = segment[1:] != segment[:-1]
    labels: NDArray[np.int64] = np.cumsum(opening) - 1
    return labels


def segments_from_events(
    event_frames: NDArray[np.integer], frames: int
) -> NDArray[np.int64]:
    """Segment labels when flights start at frame 0 and at each event frame."""
    opening: NDArray[np.bool_] = np.zeros(frames, dtype=bool)
    selected = np.asarray(event_frames, dtype=np.int64)
    if np.any((selected < 0) | (selected >= frames)):
        raise ValueError("Event frames must lie inside the clip")
    opening[selected] = True
    opening[0] = True
    labels: NDArray[np.int64] = np.cumsum(opening) - 1
    return labels
