"""Explicit, bounded linear interpolation across short invalid runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GapFillResult:
    """Filled time series plus the exact frames that were synthesized.

    ``values`` equals the input on originally valid frames, holds linear
    interpolations on ``filled_frames`` and NaN on every frame that stays
    invalid (``valid`` is False there).
    """

    values: NDArray[np.float32]
    valid: NDArray[np.bool_]
    filled_frames: tuple[int, ...]
    unfilled_frames: tuple[int, ...]


def fill_short_gaps(
    values: NDArray[np.floating],
    valid: NDArray[np.bool_],
    *,
    max_gap_frames: int,
) -> GapFillResult:
    """Interpolate interior invalid runs of at most ``max_gap_frames`` frames.

    Only runs bounded by valid frames on both sides are filled; leading,
    trailing and longer runs stay invalid. ``max_gap_frames=0`` disables
    filling. Values on invalid input frames are ignored.
    """
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be >= 0.")
    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if mask.ndim != 1 or data.shape[:1] != mask.shape:
        raise ValueError(
            f"valid must have shape (T,) matching values {data.shape}, got {mask.shape}."
        )
    if not np.isfinite(data[mask]).all():
        raise ValueError("values must be finite on every valid frame.")

    out = data.copy()
    out[~mask] = np.nan
    out_valid = mask.copy()
    filled: list[int] = []
    edges = np.flatnonzero(np.diff(np.r_[1, mask.astype(np.int8), 1]))
    for start, stop in edges.reshape(-1, 2):
        length = int(stop - start)
        if start == 0 or stop == mask.size or length > max_gap_frames:
            continue
        before, after = data[start - 1], data[stop]
        weights = np.arange(1, length + 1, dtype=np.float32) / (length + 1)
        weights = weights.reshape((length,) + (1,) * (data.ndim - 1))
        out[start:stop] = before + (after - before) * weights
        out_valid[start:stop] = True
        filled.extend(range(int(start), int(stop)))
    return GapFillResult(
        values=out,
        valid=out_valid,
        filled_frames=tuple(filled),
        unfilled_frames=tuple(int(i) for i in np.flatnonzero(~out_valid)),
    )


__all__ = ["GapFillResult", "fill_short_gaps"]
