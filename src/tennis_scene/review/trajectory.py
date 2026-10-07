"""Keep the source frame axis when plotting masked trajectories."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def masked_trajectory(points: NDArray, valid: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Return plot coordinates with NaN separators, including isolated samples.

    A valid coordinate at the origin remains valid. Invalid coordinates never
    appear and consecutive valid samples on opposite sides of a gap cannot
    form a line segment. The input arrays are never modified.
    """
    if valid.ndim != 1 or valid.dtype != np.bool_ or points.shape[0] != len(valid):
        raise ValueError("Trajectory requires a boolean mask on its source frame axis")
    result = np.array(points, dtype=np.float64, copy=True)
    result[~valid] = np.nan
    return result


def mask_intervals(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return half-open intervals of true values without compressing the axis."""
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("Intervals require a one-dimensional boolean mask")
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True)]
