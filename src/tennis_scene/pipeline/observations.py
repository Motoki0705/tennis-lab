"""Explicit conversion from model observations to the scene contract."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pose_visibility_from_heatmap_peaks(
    peaks: NDArray[np.float32],
) -> tuple[NDArray[np.float32], dict[str, str | float | int]]:
    """Bound finite ViTPose peaks; preserve the conversion evidence in metadata.

    Heatmap peaks are not probabilities and can exceed one. Raw GVHMR stage
    artifacts retain these peaks; scene consumers receive visibility in [0, 1].
    """
    if not peaks.size or not np.isfinite(peaks).all():
        raise ValueError("ViTPose heatmap peaks must be nonempty and finite")
    audit: dict[str, str | float | int] = {
        "source": "vitpose_heatmap_peak",
        "conversion": "clip_0_1",
        "raw_min": float(peaks.min()),
        "raw_max": float(peaks.max()),
        "saturated_below_zero_count": int(np.count_nonzero(peaks < 0)),
        "saturated_above_one_count": int(np.count_nonzero(peaks > 1)),
    }
    return np.clip(peaks, 0, 1).astype(np.float32), audit
