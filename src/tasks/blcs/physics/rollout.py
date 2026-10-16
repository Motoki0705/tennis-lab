"""Differentiable short flights and optional learned-parameter reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.blcs.physics.targets import PhysicsTargets
from src.utils.physics.ball.dynamics import BallField
from src.utils.physics.ball.integrate import integrate_flight


@dataclass(frozen=True)
class RolloutResidual:
    position: Tensor
    velocity: Tensor
    sample: Tensor
    start: Tensor
    horizon: Tensor
    duration_s: Tensor
    simulation_steps: Tensor
    possible: int


def rollout_residuals(
    position: Tensor,
    velocity: Tensor,
    valid: Tensor,
    physics: PhysicsTargets,
    horizons: tuple[int, ...],
    *,
    parameters: Tensor | None = None,
) -> RolloutResidual:
    """Compare integrated starts with predicted endpoints; neither is detached.

    State is in m and m/s. Optional parameters are per-frame
    [k_drag, wind_xyz, effective_magnus_xyz], with effective_magnus=k_magnus*spin.
    Both modes use the shared simulator, never an output-FPS Euler substitute.
    """
    p_errors, v_errors, samples, starts, hops, times, steps = [], [], [], [], [], [], []
    possible = 0
    for horizon in horizons:
        support = physics.support(valid, horizon)
        possible += int(
            (
                valid[:, : support.shape[1]]
                & valid[:, horizon : horizon + support.shape[1]]
            ).sum()
        )
        for b in range(len(position)):
            index = torch.where(support[b])[0]
            if not len(index):
                continue
            raw = physics.field[b]
            dt, _, _, stride = physics.clock[b].tolist()
            if parameters is None:
                field = BallField(
                    float(raw[0]),
                    raw[1].to(position),
                    raw[2].to(position),
                    raw[3:].to(position),
                )
                spin = physics.spin[b, index].to(position)
            else:
                values = parameters[b, index]
                field = BallField(
                    float(raw[0]),
                    values[:, 0],
                    torch.ones_like(values[:, 0]),
                    values[:, 1:4],
                )
                spin = values[:, 4:7]
            p, v = integrate_flight(
                position[b, index],
                velocity[b, index],
                spin,
                field,
                dt=dt,
                substeps=int(stride),
                frames=horizon + 1,
            )
            p_errors.append(p[:, -1] - position[b, index + horizon])
            v_errors.append(v[:, -1] - velocity[b, index + horizon])
            samples.append(torch.full_like(index, b))
            starts.append(index)
            hops.append(torch.full_like(index, horizon))
            times.append(position.new_full((len(index),), dt * stride * horizon))
            steps.append(torch.full_like(index, int(stride) * horizon))
    empty = position.reshape(-1, 3)[:0]
    empty_index = torch.empty(0, device=position.device, dtype=torch.long)
    return RolloutResidual(
        torch.cat(p_errors) if p_errors else empty,
        torch.cat(v_errors) if v_errors else velocity.reshape(-1, 3)[:0],
        torch.cat(samples) if samples else empty_index,
        torch.cat(starts) if starts else empty_index,
        torch.cat(hops) if hops else empty_index,
        torch.cat(times) if times else position.new_empty(0),
        torch.cat(steps) if steps else empty_index,
        possible,
    )


def target_parameters(physics: PhysicsTargets) -> Tensor:
    """Small identifiable flight representation; no separate spin/Magnus heads."""
    b, t = physics.segment.shape
    field = physics.field.to(physics.spin)
    return torch.cat(
        (
            field[:, None, 1:2].expand(b, t, 1),
            field[:, None, 3:6].expand(b, t, 3),
            field[:, None, 2:3] * physics.spin,
        ),
        dim=-1,
    )


def reconstruct_segments(
    position: Tensor,
    velocity: Tensor,
    parameters: Tensor,
    event: Tensor,
    valid: Tensor,
    clock: Tensor,
    *,
    gravity: float,
) -> tuple[Tensor, Tensor]:
    """Reconstruct using reset events and observable clock, without GT physics.

    Callers explicitly supply either predicted events (deployment) or GT segment
    starts (oracle evaluation). Each segment starts at its direct predicted state.
    """
    p_rows, v_rows = [], []
    for b in range(len(position)):
        p, v = position[b].clone(), velocity[b].clone()
        real = torch.where(valid[b])[0]
        if not len(real):
            p_rows.append(p)
            v_rows.append(v)
            continue
        boundaries = torch.where(event[b].bool() & valid[b])[0].tolist()
        boundaries = sorted(set([int(real[0]), *boundaries, int(real[-1]) + 1]))
        dt, _, _, stride = clock[b].tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            if not bool(valid[b, start:end].all()):
                raise ValueError("Segment reconstruction cannot cross padding gaps")
            values = parameters[b, start]
            field = BallField(
                gravity, values[0], torch.ones_like(values[0]), values[1:4]
            )
            out_p, out_v = integrate_flight(
                position[b, start],
                velocity[b, start],
                values[4:7],
                field,
                dt=dt,
                substeps=int(stride),
                frames=end - start,
            )
            p[start:end], v[start:end] = out_p, out_v
        p_rows.append(p)
        v_rows.append(v)
    return torch.stack(p_rows), torch.stack(v_rows)
