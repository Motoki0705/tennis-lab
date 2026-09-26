"""The shared ~30 fps full-clip frame grid used by geometry and body-view selection."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.tennis_scene.pipeline.errors import ReconstructionUnavailable

TARGET_FPS = 30.0
MIN_SOURCE_FPS = 28.0


def sampled_frame_indices(frames: int, fps: float, *, max_frames: int) -> NDArray[np.int64]:
    """Source frames nearest a ~30 fps grid over the whole clip, never windowed.

    Clips at or below 30 fps keep every frame. A clip whose grid exceeds
    ``max_frames`` is rejected instead of being truncated or subsampled further.
    """
    if frames < 1 or not math.isfinite(fps) or fps < MIN_SOURCE_FPS or max_frames < 1:
        raise ValueError(f"Frame sampling requires a nonempty video at >= {MIN_SOURCE_FPS} fps")
    rate = min(TARGET_FPS, fps)
    grid: NDArray[np.float64] = np.arange(math.ceil(frames * rate / fps) + 1, dtype=np.float64)
    indices = np.unique(np.rint(grid * fps / rate).astype(np.int64))
    indices = indices[indices < frames]
    if len(indices) > max_frames:
        raise ReconstructionUnavailable("clip_too_long", f"Full clip has {len(indices)} sampled frames; maximum is {max_frames}")
    return indices
