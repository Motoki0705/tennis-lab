"""Camera selection utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def select_camera(
    camera_mode: str | int,
    num_cameras: int,
    rng: np.random.Generator | None = None,
) -> int:
    """Select a camera index using a consistent policy.

    Args:
        camera_mode: "random", integer index, or numeric string.
        num_cameras: Number of available cameras.
        rng: Optional numpy RNG for reproducibility.

    Returns:
        Selected camera index (clamped to available range).
    """
    if num_cameras <= 0:
        raise ValueError(f"num_cameras must be positive, got {num_cameras}.")
    if camera_mode == "random":
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, num_cameras))
    if isinstance(camera_mode, int):
        return min(int(camera_mode), num_cameras - 1)
    if isinstance(camera_mode, str) and camera_mode.isdigit():
        return min(int(camera_mode), num_cameras - 1)
    return 0
