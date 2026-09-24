"""One synchronized full-clip input timeline for PLCS person models."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.errors import ReconstructionUnavailable


def association_frame_indices(
    frames: int, fps: float, *, target_fps: float = 30.0, max_frames: int = 1024
) -> NDArray[np.int64]:
    if frames < 1 or not math.isfinite(fps) or fps < 28 or target_fps != 30.0 or max_frames < 1:
        raise ValueError("Association requires nonempty video at >=28fps and a 30fps target")
    rate = min(target_fps, fps)
    grid: NDArray[np.float64] = np.arange(math.ceil(frames * rate / fps) + 1, dtype=np.float64)
    indices = np.unique(np.rint(grid * fps / rate).astype(np.int64))
    indices = indices[indices < frames]
    if len(indices) > max_frames:
        raise ReconstructionUnavailable("clip_too_long", f"Full clip has {len(indices)} association frames; maximum is {max_frames}")
    return indices
