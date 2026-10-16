"""Identity-switch candidates inside one camera-local track.

A tracker can move a track from one person to another (``video_001/clip_001``
cam0 t7 moves from a spectator behind the fence to the far player). On the
court plane that is a jump of the footpoint: the median position over the
``window_s`` before a frame differs from the median over the ``window_s``
after it by more than a person covers in that time. Every step of
that jump above ``max_jump_m`` is a candidate cut (the middle of each run of
frames above the threshold).

Candidates are deliberately liberal. The association links the pieces again
when the other cameras show one person (a cut costs nothing then), so a missed
switch is worse than a spurious cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SwitchConfig:
    window_s: float
    max_jump_m: float

    def __post_init__(self) -> None:
        if not (self.window_s > 0 and self.max_jump_m > 0):
            raise ValueError(f"Invalid switch config: {self}")

    def window_frames(self, fps: float) -> int:
        if fps <= 0:
            raise ValueError("fps must be positive")
        return max(2, round(self.window_s * fps))


def footpoint_jumps(points: NDArray[np.float64], valid: NDArray[np.bool_], window: int) -> NDArray[np.float64]:
    """``(T,)`` distance between the median footpoints of ``[t - window, t)`` and ``[t, t + window)``.

    ``nan`` where either window holds fewer than half its frames with a valid footpoint.
    """
    frames = len(valid)
    if points.shape != (frames, 2) or valid.shape != (frames,):
        raise ValueError("Footpoints must be (T, 2) aligned to validity (T,)")
    jumps = np.full(frames, np.nan)
    minimum = (window + 1) // 2
    counts = np.concatenate(([0], np.cumsum(valid)))
    for frame in range(1, frames):
        before, after = slice(max(0, frame - window), frame), slice(frame, min(frames, frame + window))
        if counts[before.stop] - counts[before.start] < minimum or counts[after.stop] - counts[after.start] < minimum:
            continue
        median_before = np.median(points[before][valid[before]], axis=0)
        median_after = np.median(points[after][valid[after]], axis=0)
        jumps[frame] = float(np.linalg.norm(median_after - median_before))
    return jumps


def switch_candidates(points: NDArray[np.float64], valid: NDArray[np.bool_], fps: float, config: SwitchConfig) -> list[int]:
    """Frames that start a new segment, at least one window apart.

    A step in the footpoint keeps the windowed jump above the threshold for
    about one window around the step, so each run of consecutive frames above
    the threshold is cut at its middle.
    """
    window = config.window_frames(fps)
    above = np.nan_to_num(footpoint_jumps(points, valid, window), nan=0.) > config.max_jump_m
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(np.int8), [0]))))
    cuts: list[int] = []
    for start, end in zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True):
        cut = (start + end) // 2
        if not cuts or cut - cuts[-1] >= window:
            cuts.append(cut)
    return cuts
