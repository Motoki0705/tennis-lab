"""Physics ground truth of one simulated ball trajectory (``ball_physics.v1``).

A record stores what the trajectory cannot reveal by itself and nothing that
can be derived from it:

- field parameters of the rally: gravity, k_drag, k_magnus, wind, surface;
- the simulator velocity at every output frame;
- every event that changed the state discontinuously, with the simulation step
  whose recorded state is the post-event state, and the state before/after it.

Spin is constant between events, so per-frame spin, flight segments and event
frames are derived.  Output frame ``k`` is simulation step ``k * stride``.
A segment starts at the first output frame at or after an event and ends before
the first frame at or after the next event; integrating its first frame state
reproduces every frame of the segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from src.utils.physics.ball.dynamics import BallField
from src.utils.physics.ball.integrate import integrate_flight
from src.utils.physics.ball.surfaces import SURFACE_NAMES

RECORD_SCHEMA = "ball_physics.v1"
# ``toss`` starts a serve toss; ``shot`` is a racket hit.  Codes are stored.
EVENT_KINDS: tuple[str, ...] = ("toss", "shot", "bounce", "net", "fence")
# Event-bit flags used by per-frame event masks.
EVENT_BITS = {"shot": 1, "bounce": 2, "net": 4, "fence": 8}
_VECTOR_FIELDS = (
    "event_position",
    "event_velocity_before",
    "event_velocity_after",
    "event_spin_before",
    "event_spin_after",
)


@dataclass(frozen=True)
class FlightSegment:
    """Frames ``[start_frame, end_frame)`` of free flight after ``event``."""

    start_frame: int
    end_frame: int
    event: int

    @property
    def frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class BallPhysicsRecord:
    gravity: float
    k_drag: float
    k_magnus: float
    wind: NDArray[np.float64]
    surface: str
    sim_fps: int
    output_fps: int
    # Simulation step actually used (configs round 1/sim_fps).
    dt: float
    velocity: NDArray[np.float32]
    event_kind: NDArray[np.int8]
    event_step: NDArray[np.int64]
    event_position: NDArray[np.float32]
    event_velocity_before: NDArray[np.float32]
    event_velocity_after: NDArray[np.float32]
    event_spin_before: NDArray[np.float32]
    event_spin_after: NDArray[np.float32]
    _segments: tuple[FlightSegment, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.surface not in SURFACE_NAMES:
            raise ValueError(f"Unknown surface {self.surface!r}")
        if self.sim_fps <= 0 or self.output_fps <= 0 or self.sim_fps % self.output_fps:
            raise ValueError("Simulation FPS must be a positive multiple of output FPS")
        if not np.isclose(self.dt, 1 / self.sim_fps, rtol=1e-5, atol=0.0):
            raise ValueError("dt must match the simulation FPS")
        if self.wind.shape != (3,) or self.velocity.ndim != 2:
            raise ValueError("Wind must be (3,) and velocity (T,3)")
        if self.velocity.shape[1:] != (3,) or len(self.velocity) < 1:
            raise ValueError("Velocity must be a nonempty (T,3) array")
        events = len(self.event_step)
        if events < 1 or self.event_kind.shape != (events,):
            raise ValueError("A record needs aligned events, starting at step 0")
        if any(getattr(self, name).shape != (events, 3) for name in _VECTOR_FIELDS):
            raise ValueError("Event states must be (E,3)")
        if self.event_step[0] != 0 or (np.diff(self.event_step) < 0).any():
            raise ValueError("Event steps must start at 0 and be nondecreasing")
        if self.event_step[-1] >= len(self.velocity) * self.stride:
            raise ValueError("Every event must precede the last output frame")
        if ((self.event_kind < 0) | (self.event_kind >= len(EVENT_KINDS))).any():
            raise ValueError("Unknown event kind code")
        if self.k_drag < 0 or self.k_magnus < 0 or not self.gravity > 0:
            raise ValueError("Invalid force coefficients")
        values = [self.velocity, self.wind, *(getattr(self, n) for n in _VECTOR_FIELDS)]
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("Physics record values must be finite")
        object.__setattr__(self, "_segments", self._build_segments())

    @property
    def stride(self) -> int:
        return self.sim_fps // self.output_fps

    @property
    def frames(self) -> int:
        return len(self.velocity)

    def event_frames(self) -> NDArray[np.int64]:
        """First output frame at or after each event (may equal ``frames``)."""
        return -(-self.event_step // self.stride)

    def _build_segments(self) -> tuple[FlightSegment, ...]:
        starts = self.event_frames()
        segments = []
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else self.frames
            # Later events at the same first frame own it: the recorded frame
            # state is the state after all of them.
            if end > start:
                segments.append(FlightSegment(int(start), int(end), index))
        return tuple(segments)

    @property
    def segments(self) -> tuple[FlightSegment, ...]:
        return self._segments

    def frame_segment(self) -> NDArray[np.int64]:
        """Index into ``segments`` for every output frame."""
        result: NDArray[np.int64] = np.empty(self.frames, np.int64)
        for index, segment in enumerate(self._segments):
            result[segment.start_frame : segment.end_frame] = index
        return result

    def frame_spin(self) -> NDArray[np.float32]:
        """Spin ``(T,3)``: constant within a segment, set by its opening event."""
        spin: NDArray[np.float32] = np.empty((self.frames, 3), np.float32)
        for segment in self._segments:
            spin[segment.start_frame : segment.end_frame] = self.event_spin_after[
                segment.event
            ]
        return spin

    def event_mask(self) -> NDArray[np.uint8]:
        """Per-frame :data:`EVENT_BITS` at each event's first frame (toss excluded)."""
        mask: NDArray[np.uint8] = np.zeros(self.frames, np.uint8)
        for code, frame in zip(self.event_kind, self.event_frames(), strict=True):
            kind = EVENT_KINDS[int(code)]
            if kind in EVENT_BITS and frame < self.frames:
                mask[frame] |= EVENT_BITS[kind]
        return mask

    def ball_field(self) -> BallField:
        """Float32 field tensors, as used by the simulator."""
        return BallField(
            gravity=self.gravity,
            k_drag=torch.tensor(self.k_drag),
            k_magnus=torch.tensor(self.k_magnus),
            wind=torch.tensor(self.wind, dtype=torch.float32),
        )

    def truncate(self, frames: int) -> BallPhysicsRecord:
        """The record of the first ``frames`` output frames."""
        if not 1 <= frames <= self.frames:
            raise ValueError("Truncation must keep 1..frames frames")
        keep = self.event_step < frames * self.stride
        arrays = {name: getattr(self, name)[keep] for name in _VECTOR_FIELDS}
        return BallPhysicsRecord(
            gravity=self.gravity,
            k_drag=self.k_drag,
            k_magnus=self.k_magnus,
            wind=self.wind,
            surface=self.surface,
            sim_fps=self.sim_fps,
            output_fps=self.output_fps,
            dt=self.dt,
            velocity=self.velocity[:frames],
            event_kind=self.event_kind[keep],
            event_step=self.event_step[keep],
            **arrays,
        )

    def integrate_segments(
        self, positions: NDArray[np.floating]
    ) -> NDArray[np.float32]:
        """Re-simulate every segment from its first frame of ``positions``."""
        if positions.shape != (self.frames, 3):
            raise ValueError("Positions must be (T,3)")
        field_ = self.ball_field()
        spin = self.frame_spin()
        result: NDArray[np.float32] = np.empty((self.frames, 3), np.float32)
        for segment in self._segments:
            start = segment.start_frame
            integrated, _ = integrate_flight(
                torch.tensor(positions[start], dtype=torch.float32),
                torch.tensor(self.velocity[start]),
                torch.tensor(spin[start]),
                field_,
                dt=self.dt,
                substeps=self.stride,
                frames=segment.frames,
            )
            result[start : segment.end_frame] = integrated.numpy()
        return result

    def verify(self, positions: NDArray[np.floating], tolerance_m: float) -> float:
        """Max re-simulation error; raise when it exceeds ``tolerance_m``."""
        error = float(
            np.abs(self.integrate_segments(positions) - positions).max(initial=0.0)
        )
        if not error <= tolerance_m:
            raise ValueError(
                f"Physics record does not reproduce the trajectory: {error:.3e} m"
            )
        return error

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "physics_schema": np.array(RECORD_SCHEMA),
            "physics_gravity": np.array(self.gravity, np.float64),
            "physics_k_drag": np.array(self.k_drag, np.float64),
            "physics_k_magnus": np.array(self.k_magnus, np.float64),
            "physics_wind": self.wind.astype(np.float64),
            "physics_surface": np.array(self.surface),
            "physics_sim_fps": np.array(self.sim_fps, np.int64),
            "physics_output_fps": np.array(self.output_fps, np.int64),
            "physics_dt": np.array(self.dt, np.float64),
            "physics_velocity": self.velocity.astype(np.float32),
            "physics_event_kind": self.event_kind.astype(np.int8),
            "physics_event_step": self.event_step.astype(np.int64),
        }
        for name in _VECTOR_FIELDS:
            arrays[f"physics_{name}"] = getattr(self, name).astype(np.float32)
        return arrays

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray]) -> BallPhysicsRecord:
        names = {key for key in arrays if key.startswith("physics_")}
        expected = set(record_keys())
        if names != expected:
            raise ValueError(
                f"Physics record keys mismatch: missing {sorted(expected - names)}, "
                f"unexpected {sorted(names - expected)}"
            )
        if str(arrays["physics_schema"]) != RECORD_SCHEMA:
            raise ValueError(f"Unsupported physics record {arrays['physics_schema']}")
        typed = {
            "physics_velocity": np.float32,
            "physics_event_kind": np.int8,
            "physics_event_step": np.int64,
            **{f"physics_{name}": np.float32 for name in _VECTOR_FIELDS},
        }
        for key, dtype in typed.items():
            if arrays[key].dtype != dtype:
                raise ValueError(f"{key} must be {np.dtype(dtype)}")
        return cls(
            gravity=float(arrays["physics_gravity"]),
            k_drag=float(arrays["physics_k_drag"]),
            k_magnus=float(arrays["physics_k_magnus"]),
            wind=arrays["physics_wind"].astype(np.float64),
            surface=str(arrays["physics_surface"]),
            sim_fps=int(arrays["physics_sim_fps"]),
            output_fps=int(arrays["physics_output_fps"]),
            dt=float(arrays["physics_dt"]),
            velocity=arrays["physics_velocity"],
            event_kind=arrays["physics_event_kind"],
            event_step=arrays["physics_event_step"],
            **{name: arrays[f"physics_{name}"] for name in _VECTOR_FIELDS},
        )


def record_keys() -> tuple[str, ...]:
    """Exact array keys written by :meth:`BallPhysicsRecord.to_arrays`."""
    return (
        "physics_schema",
        "physics_gravity",
        "physics_k_drag",
        "physics_k_magnus",
        "physics_wind",
        "physics_surface",
        "physics_sim_fps",
        "physics_output_fps",
        "physics_dt",
        "physics_velocity",
        "physics_event_kind",
        "physics_event_step",
        *(f"physics_{name}" for name in _VECTOR_FIELDS),
    )
