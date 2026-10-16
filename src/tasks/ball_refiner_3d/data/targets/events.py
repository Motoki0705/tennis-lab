"""Full-rally Gaussian event targets; no input features."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


def gaussian_event_target(
    events: np.ndarray, sigma_frames: float
) -> NDArray[np.float32]:
    """Peak 1 at shot/bounce; max of full-rally Gaussian tails for overlaps."""
    if events.ndim != 1 or not math.isfinite(sigma_frames) or sigma_frames <= 0:
        raise ValueError("Require a 1D event sequence and positive finite sigma")
    result: NDArray[np.float32] = np.zeros(len(events), np.float32)
    time: NDArray[np.float64] = np.arange(len(events), dtype=np.float64)
    for index in np.flatnonzero(events):
        result = np.maximum(
            result,
            np.exp(-0.5 * ((time - index) / sigma_frames) ** 2).astype(np.float32),
        )
    return result
