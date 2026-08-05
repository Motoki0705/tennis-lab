"""Overlapped temporal-window inference.

Sequence models trained on bounded clip lengths (e.g. PLCS/BLCS with
``seq_len_range`` up to 256) degrade when run far beyond that range in one
shot (RoPE time-axis extrapolation). These helpers split a long sequence into
overlapping windows that stay inside the training distribution and blend the
per-window predictions back into one sequence.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def window_slices(
    total_len: int,
    window_len: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Split ``[0, total_len)`` into overlapping ``[start, end)`` windows.

    Windows advance by ``window_len - overlap``; the final window is aligned
    to the sequence end so every frame is covered by at least one window and
    no window exceeds ``window_len``.

    Args:
        total_len: Sequence length to cover (> 0).
        window_len: Maximum window length (> 0).
        overlap: Frames shared by consecutive windows (0 <= overlap < window_len).

    Returns:
        List of ``(start, end)`` pairs, in increasing ``start`` order.
    """
    if total_len <= 0:
        raise ValueError(f"total_len must be positive, got {total_len}")
    if window_len <= 0:
        raise ValueError(f"window_len must be positive, got {window_len}")
    if not 0 <= overlap < window_len:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < window_len, "
            f"got overlap={overlap}, window_len={window_len}"
        )
    if total_len <= window_len:
        return [(0, total_len)]

    stride = window_len - overlap
    starts = list(range(0, total_len - window_len, stride))
    starts.append(total_len - window_len)
    return [(start, start + window_len) for start in starts]


def blend_windows(
    chunks: Sequence[tuple[int, NDArray[np.floating]]],
    total_len: int,
) -> NDArray[np.floating]:
    """Blend per-window predictions into one sequence along axis 0.

    Overlapping regions are combined with triangular (center-peaked) weights,
    so frames near a window edge — where temporally contextual models are
    least reliable — defer to the neighbouring window's interior.

    Args:
        chunks: ``(start, values)`` pairs where ``values`` has the window's
            time length on axis 0. Together the chunks must cover every frame
            in ``[0, total_len)``.
        total_len: Length of the blended output on axis 0.

    Returns:
        Array of shape ``(total_len, *values.shape[1:])``.
    """
    if not chunks:
        raise ValueError("chunks must not be empty")

    trailing_shape = chunks[0][1].shape[1:]
    accum = np.zeros((total_len, *trailing_shape), dtype=np.float64)
    weight_sum: NDArray[np.float64] = np.zeros(
        (total_len,) + (1,) * len(trailing_shape), dtype=np.float64
    )

    for start, values in chunks:
        win_len = values.shape[0]
        if start < 0 or start + win_len > total_len:
            raise ValueError(
                f"chunk [{start}, {start + win_len}) exceeds [0, {total_len})"
            )
        if values.shape[1:] != trailing_shape:
            raise ValueError(
                f"chunk trailing shape {values.shape[1:]} does not match "
                f"first chunk {trailing_shape}"
            )
        idx: NDArray[np.float64] = np.arange(win_len, dtype=np.float64)
        weights = np.minimum(idx + 1.0, win_len - idx)
        weights = weights.reshape((win_len,) + (1,) * len(trailing_shape))
        accum[start : start + win_len] += values.astype(np.float64) * weights
        weight_sum[start : start + win_len] += weights

    if not np.all(weight_sum > 0):
        uncovered = int(np.count_nonzero(weight_sum <= 0))
        raise ValueError(f"chunks leave {uncovered} frame(s) uncovered")
    return accum / weight_sum
