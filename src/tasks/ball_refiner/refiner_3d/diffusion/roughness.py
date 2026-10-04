"""CPU decomposition of saved trajectories without changing their predictions."""
from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import Array


def window_owners(windows: list[dict[str, int]], length: int) -> Array:
    """Recover exact source-window ownership; reject gaps and overlapping claims."""
    owners: Array = np.full(length, -1, dtype=np.int64)
    stop = 0
    for index, window in enumerate(windows):
        start, end = window['owned_start'], window['owned_stop']
        if start != stop or not 0 <= start < end <= length:
            raise ValueError('Window ownership must cover each frame exactly once')
        owners[start:end] = index
        stop = end
    if stop != length:
        raise ValueError('Window ownership is incomplete')
    return owners


def stencil_masks(free: Array, owners: Array, order: int) -> dict[str, Array]:
    """A seam stencil touches multiple owners; free requires every input frame."""
    if free.ndim != 1 or free.dtype != np.bool_ or owners.shape != free.shape:
        raise ValueError('Need matching boolean free-flight and owner vectors')
    if order not in (2, 3) or len(free) <= order:
        raise ValueError('Need a second/third derivative with sufficient frames')
    width = len(free) - order
    support = np.logical_and.reduce([free[i:i + width] for i in range(order + 1)])
    inside = np.logical_and.reduce([owners[i:i + width] == owners[:width] for i in range(1, order + 1)])
    return {'all': np.ones(width, dtype=bool), 'free': support,
            'seam_all': ~inside, 'inside_all': inside,
            'seam_free': support & ~inside, 'inside_free': support & inside}


def magnitude_summary(vectors: Array) -> dict[str, Any]:
    magnitude = np.linalg.norm(vectors, axis=-1).ravel()
    if not len(magnitude):
        return {'count': 0, 'mean': None, 'p50': None, 'p95': None, 'rms': None, 'squared_sum': 0.}
    return {'count': len(magnitude), 'mean': float(magnitude.mean()),
            'p50': float(np.quantile(magnitude, .5)), 'p95': float(np.quantile(magnitude, .95)),
            'rms': float(np.sqrt(np.square(magnitude).mean())), 'squared_sum': float(np.square(magnitude).sum())}


def local_error_components(error: Array, free: Array, owners: Array, dt: float) -> dict[str, Array]:
    """Five-frame error trend and residual, strictly within free-flight windows.

    No padding, cross-window smoothing, or concatenation of disjoint frames.
    Second derivatives require all seven original support frames to be free.
    This is a diagnostic decomposition, never a replacement prediction.
    """
    if error.ndim != 2 or error.shape != (len(free), 3) or owners.shape != free.shape:
        raise ValueError('Need matching T,3 errors and T masks/owners')
    if len(error) < 7 or not np.isfinite(error).all() or not np.isfinite(dt) or dt <= 0:
        raise ValueError('Need finite errors, positive dt and >=7 frames')
    trend = sum(error[i:len(error) - 4 + i] for i in range(5)) / 5
    residual = error[2:-2] - trend
    masks = {}
    for width in (5, 7):
        length = len(error) - width + 1
        masks[width] = np.logical_and.reduce([free[i:i + length] & (owners[i:i + length] == owners[:length]) for i in range(width)])
    return {'trend_position': trend[masks[5]], 'residual_position': residual[masks[5]],
            'error_acceleration': (np.diff(error[2:-2], n=2, axis=0) / dt**2)[masks[7]],
            'trend_acceleration': (np.diff(trend, n=2, axis=0) / dt**2)[masks[7]],
            'residual_acceleration': (np.diff(residual, n=2, axis=0) / dt**2)[masks[7]]}
