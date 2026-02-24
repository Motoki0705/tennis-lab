"""Derived metrics for simulator visualization.

These are not required for training, but are useful for interactive exploration.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.utils.schema.court import HALF_DOUBLES_WIDTH, NET_HEIGHT_CENTER, NET_HEIGHT_POST


def apex_height_m(positions: Tensor) -> float:
    """Maximum Z in meters."""
    if positions.numel() == 0:
        return 0.0
    return float(positions[:, 2].max().item())


def time_to_bounce1_s(t_bounce1: int, fps_out: int) -> float | None:
    if t_bounce1 < 0:
        return None
    return float(t_bounce1) / float(fps_out)


def net_clearance_m(trajectory_sim: Tensor) -> float | None:
    """Approximate net clearance at y=0 crossing.

    Returns:
        z_at_net - net_height_at_x, or None if we never cross the net.
    """
    if trajectory_sim.numel() == 0:
        return None

    y = trajectory_sim[:, 1]
    sign = torch.sign(y)
    # Find first index where sign changes or hits 0.
    # We skip the first point since we look at segments (i-1 -> i).
    for i in range(1, len(y)):
        if sign[i - 1].item() == 0.0:
            # Already at y=0; use this point.
            x_at = float(trajectory_sim[i - 1, 0].item())
            z_at = float(trajectory_sim[i - 1, 2].item())
            return z_at - _net_height_at_x(x_at)
        if sign[i].item() == 0.0 or sign[i - 1].item() != sign[i].item():
            p0 = trajectory_sim[i - 1]
            p1 = trajectory_sim[i]
            y0 = float(p0[1].item())
            y1 = float(p1[1].item())
            # Linear interpolation to y=0
            t = y0 / (y0 - y1 + 1e-8)
            x_at = float((p0[0] + t * (p1[0] - p0[0])).item())
            z_at = float((p0[2] + t * (p1[2] - p0[2])).item())
            return z_at - _net_height_at_x(x_at)

    return None


def _net_height_at_x(x: float) -> float:
    x_ratio = min(1.0, abs(x) / float(HALF_DOUBLES_WIDTH))
    return float(NET_HEIGHT_CENTER + x_ratio * (NET_HEIGHT_POST - NET_HEIGHT_CENTER))
