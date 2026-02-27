"""Soft-label utilities for event detection tasks."""

from __future__ import annotations

import torch
from torch import Tensor


def gaussian_soft_labels(
    length: int,
    event_indices: list[int],
    sigma: float,
    device: torch.device | None = None,
) -> Tensor:
    """Create soft labels with Gaussian peaks at given frame indices.

    Each event index produces a Gaussian centred at that frame. Overlapping
    peaks are combined via element-wise max so that the output stays in
    ``[0, 1]``.

    Args:
        length: Sequence length *T*.
        event_indices: 0-based frame indices where events occur.
        sigma: Standard deviation in frames.
        device: Output device. Defaults to CPU.

    Returns:
        Soft label tensor of shape ``(T,)``.
    """
    if device is None:
        device = torch.device("cpu")

    if length <= 0:
        return torch.zeros((0,), device=device)
    if not event_indices:
        return torch.zeros((length,), device=device)

    t = torch.arange(length, device=device, dtype=torch.float32)
    out = torch.zeros((length,), device=device, dtype=torch.float32)
    denom = 2.0 * float(sigma) * float(sigma)
    for idx in event_indices:
        if 0 <= idx < length:
            out = torch.maximum(out, torch.exp(-((t - float(idx)) ** 2) / denom))
    return out
