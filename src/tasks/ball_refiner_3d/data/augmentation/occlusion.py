"""Event-centered but asymmetric contiguous gaps and rare isolated drops."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.configuration.data import CorruptionConfig
from src.tasks.ball_refiner_3d.data.schema import BoolArray


def sample_occlusion(
    visible: BoolArray,
    events: NDArray[np.integer],
    config: CorruptionConfig,
    event_rng: np.random.Generator,
    isolated_rng: np.random.Generator,
) -> tuple[BoolArray, BoolArray, NDArray[np.int64]]:
    frames = visible.shape[1]
    indices = np.flatnonzero(events)
    selected = event_rng.random(len(indices)) < config.event_probability
    left = event_rng.integers(config.gap_min, config.gap_max + 1, len(indices))
    # Sample right uniformly among the other widths: midpoint is never the event.
    right = event_rng.integers(config.gap_min, config.gap_max, len(indices))
    right += right >= left
    blocked = np.zeros(frames, dtype=bool)
    intervals = []
    for event, before, after in zip(
        indices[selected], left[selected], right[selected], strict=True
    ):
        start, end = max(0, int(event - before)), min(frames, int(event + after + 1))
        blocked[start:end] = True
        intervals.append((int(event), start, end))
    isolated = (
        (isolated_rng.random(visible.shape) < config.isolated_probability)
        & visible
        & ~blocked[None]
    )
    missing = ~visible | blocked[None] | isolated
    return missing, isolated, np.asarray(intervals, dtype=np.int64).reshape(-1, 3)
