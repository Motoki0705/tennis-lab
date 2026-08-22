"""Temporal window planning for SLCS clip samples.

Pure logic: given a clip length, produce fixed-size windows with explicit
right-padding, and align sparse DINOv3 token samples to a window by their
recorded frame indices (never by interpolation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.utils.video.windows import build_window_starts


@dataclass(frozen=True)
class WindowPlan:
    """One fixed-size window over a clip timeline.

    Attributes:
        start: First clip frame covered by the window.
        length: Number of real frames (``<= window_size``).
        window_size: Fixed tensor length; ``window_size - length`` frames at
            the end are padding and must be masked out.
    """

    start: int
    length: int
    window_size: int

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}.")
        if not 0 < self.length <= self.window_size:
            raise ValueError(
                f"length must be in (0, window_size={self.window_size}], got {self.length}."
            )
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}.")

    @property
    def pad(self) -> int:
        return self.window_size - self.length

    def padding_mask(self) -> NDArray[np.bool_]:
        """Return the canonical window mask (``True`` for padded slots)."""
        mask: NDArray[np.bool_] = np.ones(self.window_size, dtype=np.bool_)
        mask[: self.length] = False
        return mask

    def frame_indices(self) -> NDArray[np.int64]:
        """Absolute clip frame index per window slot; padded slots repeat the last real frame."""
        idx: NDArray[np.int64] = np.arange(
            self.start, self.start + self.window_size, dtype=np.int64
        )
        idx[self.length :] = self.start + self.length - 1
        return idx


def plan_windows(num_frames: int, *, window_size: int, stride: int) -> list[WindowPlan]:
    """Plan sliding windows covering every frame of a clip.

    Clips shorter than ``window_size`` yield a single padded window. For longer
    clips the final window is anchored at the clip tail (no padding), matching
    :func:`src.utils.video.windows.build_window_starts`.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}.")
    if num_frames < window_size:
        return [WindowPlan(start=0, length=num_frames, window_size=window_size)]
    starts = build_window_starts(
        frame_count=num_frames, sequence_length=window_size, stride=stride
    )
    return [
        WindowPlan(start=start, length=window_size, window_size=window_size)
        for start in starts
    ]


def select_window_tokens(
    frame_idx: NDArray[np.int64], plan: WindowPlan
) -> NDArray[np.int64]:
    """Indices into a token sample axis whose frames fall inside the window.

    Only real (non-padded) window frames are matched; the result may be empty
    when no token sample lies inside the window.
    """
    if frame_idx.ndim != 1:
        raise ValueError(f"frame_idx must be 1-D, got shape {frame_idx.shape}.")
    inside = (frame_idx >= plan.start) & (frame_idx < plan.start + plan.length)
    return np.nonzero(inside)[0].astype(np.int64)


__all__ = ["WindowPlan", "plan_windows", "select_window_tokens"]
