"""Deterministic linear GAN transition, in caller-selected epochs or updates."""

from __future__ import annotations


def gan_weight_at(index: int, *, start: int, warmup: int, target: float) -> float:
    """Zero-based index: wait `start` units, then reach target in `warmup` units."""
    if min(index, start, target) < 0 or warmup < 1:
        raise ValueError("Require nonnegative index/start/target and positive warmup")
    return target * min(max(index - start + 1, 0) / warmup, 1.0)
