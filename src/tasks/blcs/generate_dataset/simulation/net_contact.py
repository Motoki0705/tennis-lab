"""Opt-in net contact proposals for synthetic rallies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import Tensor

from src.tasks.blcs.generate_dataset.simulation.ball_physics import (
    BallPhysics,
    BallState,
)
from src.utils.schema.court import HALF_DOUBLES_WIDTH, net_height_at_x


class NetContact(StrEnum):
    CROSSED = "crossed"
    DROPPED = "dropped"


@dataclass(frozen=True)
class NetContactPlan:
    shot_index: int
    outcome: NetContact


class NetContactProposalRejected(RuntimeError):
    """The sampled flight could not be aimed at the requested net height."""


def swept_net_contact(
    previous: Tensor, current: Tensor, physics: BallPhysics
) -> Tensor | None:
    """First mesh or cord contact along a simulation step.

    The top of the mesh follows the court's piecewise linear net height. In
    each half of the net, the cord becomes a circle in scaled Y/Z coordinates,
    so a segment-circle intersection also detects steps spanning the slab.
    """
    half_t = max(float(physics.config.net_half_thickness), 1e-4)
    if (
        min(previous[1].item(), current[1].item()) > half_t
        or max(previous[1].item(), current[1].item()) < -half_t
    ):
        return None
    x, y, z = previous.tolist()
    dx, dy, dz = (current - previous).tolist()
    start, end = 0.0, 1.0
    for origin, delta, limit in ((x, dx, HALF_DOUBLES_WIDTH), (y, dy, half_t)):
        if delta == 0:
            if abs(origin) > limit:
                return None
            continue
        entry, exit_ = sorted(((-limit - origin) / delta, (limit - origin) / delta))
        start, end = max(start, entry), min(end, exit_)
    if start > end:
        return None

    radius = max(float(physics.config.net_cord_radius), 1e-4)

    def scaled_point(t: float) -> tuple[float, float]:
        height_above_mesh = z + t * dz - (net_height_at_x(x + t * dx) - radius)
        return (y + t * dy) / half_t, height_above_mesh / radius

    cuts = [start, end]
    if dx != 0 and start < -x / dx < end:
        cuts.insert(1, -x / dx)
    for left, right in zip(cuts, cuts[1:], strict=False):
        y0, z0 = scaled_point(left)
        c = y0 * y0 + z0 * z0 - 1
        if z0 <= 0 or c <= 0:
            return previous.lerp(current, left)
        y1, z1 = scaled_point(right)
        uy, uz = y1 - y0, z1 - z0
        a = uy * uy + uz * uz
        b = y0 * uy + z0 * uz
        discriminant = b * b - a * c
        if a > 0 and b < 0 and discriminant >= 0:
            fraction = c / (-b + math.sqrt(discriminant))
            if fraction <= 1:
                return previous.lerp(current, left + fraction * (right - left))
    return None


def sample_net_velocity(
    position: Tensor, spin: Tensor, physics: BallPhysics, outcome: NetContact
) -> Tensor:
    """Aim at the net plane under the sampled drag, Magnus force and wind."""
    x, y, z = position.tolist()
    if abs(y) <= physics.config.net_half_thickness:
        raise NetContactProposalRejected("The shot starts within the net slab")
    speed = 15 + 20 * torch.rand(1).item()
    vx = (2 * torch.rand(1).item() - 1) * min(4.0, abs(y))
    vy = math.copysign(speed, -y)
    duration = abs(y / vy)
    depth = (
        physics.config.net_cord_radius * (0.01 + 0.06 * torch.rand(1).item())
        if outcome is NetContact.CROSSED
        else 0.12 + 0.58 * torch.rand(1).item()
    )
    height = net_height_at_x(x + vx * duration) - depth
    vz = (height - z) / duration + 0.5 * physics.config.gravity * duration
    previous_error: float | None = None
    previous_vz = vz
    # The first update uses the projectile derivative; subsequent updates use
    # the secant of the full-physics height error.
    for _ in range(8):
        velocity = position.new_tensor((vx, vy, vz))
        state = BallState(position, velocity, spin)
        for step in range(1, math.ceil(2.0 / physics.config.dt) + 1):
            before = state.position
            state = physics.step(state)
            if state.position[1].item() * y <= 0:
                fraction = before[1].item() / (before[1] - state.position[1]).item()
                crossing_x, _, crossing_z = before.lerp(
                    state.position, fraction
                ).tolist()
                error = crossing_z - (net_height_at_x(crossing_x) - depth)
                if abs(error) <= min(0.0001, depth / 4):
                    return velocity
                duration = (step - 1 + fraction) * physics.config.dt
                break
        else:
            break
        slope = (
            (error - previous_error) / (vz - previous_vz)
            if previous_error is not None and abs(vz - previous_vz) > 1e-8
            else duration
        )
        previous_vz, previous_error = vz, error
        vz -= error / (slope if slope > 0 else duration)
    raise NetContactProposalRejected("Could not aim the sampled shot at the net")
