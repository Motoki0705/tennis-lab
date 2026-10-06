"""Exact ``ball_physics.v1`` records for tests: one or more simulated flights."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.physics.ball import BallField, semi_implicit_euler_step
from src.utils.physics.ball.record import EVENT_KINDS, BallPhysicsRecord


def simulated_record(
    frames: int,
    *,
    output_fps: int = 30,
    sim_fps: int = 120,
    hits: tuple[int, ...] = (),
    hit_kinds: tuple[str, ...] | None = None,
    k_drag: float = 0.01,
    k_magnus: float = 0.001,
    wind: tuple[float, float, float] = (1.0, -0.5, 0.0),
    surface: str = "hard",
) -> tuple[np.ndarray, BallPhysicsRecord]:
    """Positions ``(frames,3)`` and their record.

    The ball is hit at simulation step 0 and at every step in ``hits``; each hit
    resets the velocity and spin, so the record has one segment per hit.
    ``hit_kinds`` labels the events in ``hits`` (default: all ``shot``).
    """
    kinds = ("shot", *(hit_kinds or ("shot",) * len(hits)))
    if len(kinds) != len(hits) + 1:
        raise ValueError("hit_kinds must label every hit")
    stride = sim_fps // output_fps
    dt = 1.0 / sim_fps
    field = BallField(
        9.8,
        torch.tensor(k_drag),
        torch.tensor(k_magnus),
        torch.tensor(wind, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(len(hits) + frames)
    position = torch.tensor([0.0, -10.0, 1.0])
    velocity = torch.zeros(3)
    spin = torch.zeros(3)
    positions, velocities = [], []
    events: list[tuple[int, list[torch.Tensor]]] = []
    for step in range(frames * stride):
        if step:
            position, velocity = semi_implicit_euler_step(
                position, velocity, spin, field, dt
            )
        if step == 0 or step in hits:
            # A hit replaces the state of the sample it happens at.
            before_v, before_w = velocity, spin
            velocity = torch.tensor([0.0, 20.0, 6.0]) + torch.randn(
                3, generator=generator
            )
            spin = torch.randn(3, generator=generator) * 100
            events.append((step, [position, before_v, velocity, before_w, spin]))
        if step % stride == 0:
            positions.append(position)
            velocities.append(velocity)
    record = BallPhysicsRecord(
        gravity=9.8,
        k_drag=k_drag,
        k_magnus=k_magnus,
        wind=np.asarray(wind, np.float64),
        surface=surface,
        sim_fps=sim_fps,
        output_fps=output_fps,
        dt=dt,
        velocity=torch.stack(velocities).numpy(),
        event_kind=np.array([EVENT_KINDS.index(kind) for kind in kinds], np.int8),
        event_step=np.array([step for step, _ in events], np.int64),
        event_position=torch.stack([s[0] for _, s in events]).numpy(),
        event_velocity_before=torch.stack([s[1] for _, s in events]).numpy(),
        event_velocity_after=torch.stack([s[2] for _, s in events]).numpy(),
        event_spin_before=torch.stack([s[3] for _, s in events]).numpy(),
        event_spin_after=torch.stack([s[4] for _, s in events]).numpy(),
    )
    return torch.stack(positions).numpy(), record
