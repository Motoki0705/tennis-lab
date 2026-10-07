"""Deterministic interpolation baseline on the same model input."""

from __future__ import annotations

import numpy as np


def linear_baseline(coordinates: np.ndarray, missing: np.ndarray) -> np.ndarray:
    result = coordinates.copy()
    for view in range(len(result)):
        known = np.flatnonzero(~missing[view])
        # Explicit all-missing baseline is zero in physical coordinates.
        for axis in range(result.shape[-1]):
            result[view, :, axis] = (
                np.interp(
                    np.arange(result.shape[1]), known, coordinates[view, known, axis]
                )
                if len(known)
                else 0
            )
    return result
