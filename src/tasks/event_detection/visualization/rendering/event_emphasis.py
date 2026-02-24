"""Event emphasis helpers for animation coloring."""

from __future__ import annotations

import numpy as np


def build_event_impact(
    *,
    num_frames: int,
    event_indices: list[int],
    radius_frames: int,
    sigma_frames: float,
) -> np.ndarray:
    """Build per-frame Gaussian impact around predicted event indices."""
    impact = np.zeros((num_frames,), dtype=np.float32)
    if num_frames <= 0 or not event_indices:
        return impact

    radius = max(0, int(radius_frames))
    sigma = max(float(sigma_frames), 1e-6)
    denom = 2.0 * sigma * sigma

    for idx in event_indices:
        i = int(idx)
        if i < 0 or i >= num_frames:
            continue
        start = max(0, i - radius)
        end = min(num_frames, i + radius + 1)
        t = np.arange(start, end, dtype=np.float32)
        local = np.exp(-((t - float(i)) ** 2) / denom).astype(np.float32)
        impact[start:end] = np.maximum(impact[start:end], local)

    return impact


def mix_color(base_rgb: np.ndarray, event_rgb: np.ndarray, alpha: float) -> tuple[float, float, float]:
    """Linearly blend RGB colors with alpha in [0, 1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    mixed = (1.0 - a) * base_rgb + a * event_rgb
    return float(mixed[0]), float(mixed[1]), float(mixed[2])

