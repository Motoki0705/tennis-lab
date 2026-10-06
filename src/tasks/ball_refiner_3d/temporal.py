"""Real-frame temporal windows and deterministic centre ownership."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def window_starts(frames: int, length: int, stride: int) -> tuple[int, ...]:
    """Backfill the tail with real frames; a short clip is an explicit error."""
    if length < 1 or not 1 <= stride <= length or frames < length:
        raise ValueError("Windows require frames >= length >= stride >= 1; no padding")
    starts = list(range(0, frames - length + 1, stride))
    if starts[-1] != frames - length:
        starts.append(frames - length)
    return tuple(starts)


def window_owners(frames: int, starts: Sequence[int], length: int) -> NDArray[np.int64]:
    """One prediction per frame: nearest centre, ties resolved by earlier start."""
    owners: NDArray[np.int64] = np.full(frames, -1, dtype=np.int64)
    distances = np.full(frames, np.inf)
    if not starts or list(starts) != sorted(set(starts)):
        raise ValueError("Window starts must be nonempty, sorted and unique")
    for index, start in enumerate(starts):
        if start < 0 or start + length > frames:
            raise ValueError("Window exceeds the source timeline")
        rows = np.arange(start, start + length)
        distance = np.abs(rows - (start + (length - 1) / 2))
        take = distance < distances[rows]
        owners[rows[take]] = index
        distances[rows[take]] = distance[take]
    if (owners < 0).any():
        raise ValueError("Window policy leaves source frames uncovered")
    return owners


