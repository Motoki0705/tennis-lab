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


def extract_event_indices(meta: dict, key: str) -> list[int]:
    """Extract non-negative integer event frame indices from scene metadata.

    Reads ``meta["shots"]`` (a list of dicts) and collects ``int(shot[key])``
    for each entry where the value is >= 0.

    Args:
        meta: Scene metadata dictionary.
        key: Key to look up within each shot dict (e.g. ``"t_start"``).

    Returns:
        Sorted list of event frame indices.
    """
    shots = meta.get("shots", []) or []
    indices: list[int] = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        t = int(s.get(key, -1))
        if t >= 0:
            indices.append(t)
    return indices
