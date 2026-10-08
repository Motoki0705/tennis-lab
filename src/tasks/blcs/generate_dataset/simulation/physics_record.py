"""Convert a simulated rally into the shared ``ball_physics.v1`` record."""

from __future__ import annotations

import numpy as np
import torch

from src.tasks.blcs.generate_dataset.simulation.ball_physics import PhysicsConfig
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyResult
from src.utils.physics.ball.record import EVENT_KINDS, BallPhysicsRecord

# Re-simulating a recorded segment repeats the simulator's float32 operations
# bit for bit; allow only rounding noise.
REPRODUCTION_TOLERANCE_M = 1e-6


def build_physics_record(
    result: RallyResult, physics: PhysicsConfig, frames: int | None = None
) -> BallPhysicsRecord:
    """Record of the rally's first ``frames`` output frames, verified.

    ``physics`` must be the sampled (deterministic) config the rally used.
    """
    if physics.surface_choices is not None or physics.k_drag_range is not None:
        raise ValueError("Record the sampled physics config, not the base config")
    if not np.isclose(physics.dt, 1 / result.sim_fps):
        raise ValueError("Physics dt must match the rally simulation FPS")
    stride = result.sim_fps // result.fps_out
    available = len(result.trajectory)
    frames = available if frames is None else frames
    if not 1 <= frames <= available:
        raise ValueError("Requested frames exceed the rally")
    events = [e for e in result.physics_events if e.step < frames * stride]
    if not events or events[0].step != 0:
        raise ValueError("A rally must open with an event at simulation step 0")

    def stack(name: str) -> np.ndarray:
        stacked: np.ndarray = torch.stack([getattr(e, name) for e in events]).numpy()
        return stacked

    record = BallPhysicsRecord(
        gravity=physics.gravity,
        k_drag=physics.k_drag if physics.use_drag else 0.0,
        k_magnus=physics.k_magnus if physics.use_magnus else 0.0,
        wind=np.asarray(physics.wind, np.float64),
        surface=physics.surface,
        sim_fps=result.sim_fps,
        output_fps=result.fps_out,
        dt=physics.dt,
        velocity=result.velocities[:frames].numpy().astype(np.float32),
        event_kind=np.array([EVENT_KINDS.index(e.kind) for e in events], np.int8),
        event_step=np.array([e.step for e in events], np.int64),
        event_position=stack("position").astype(np.float32),
        event_velocity_before=stack("velocity_before").astype(np.float32),
        event_velocity_after=stack("velocity_after").astype(np.float32),
        event_spin_before=stack("spin_before").astype(np.float32),
        event_spin_after=stack("spin_after").astype(np.float32),
    )
    record.verify(result.trajectory[:frames].numpy(), REPRODUCTION_TOLERANCE_M)
    return record
