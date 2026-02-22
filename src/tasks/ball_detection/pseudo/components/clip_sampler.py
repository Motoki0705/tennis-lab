"""Clip sampler for contiguous detection-active windows."""

from __future__ import annotations

from src.ball_detection.data.type import ClipWindow


class ClipSampler:
    """Extract temporal windows from visibility sequence."""

    def __init__(self, min_length: int = 16, max_gap: int = 4) -> None:
        self.min_length = int(min_length)
        self.max_gap = int(max_gap)

    def sample(self, visibility: list[bool]) -> list[ClipWindow]:
        windows: list[ClipWindow] = []
        start = None
        gap = 0
        for i, vis in enumerate(visibility):
            if vis:
                if start is None:
                    start = i
                gap = 0
            else:
                if start is not None:
                    gap += 1
                    if gap > self.max_gap:
                        end = i - gap
                        if end - start + 1 >= self.min_length:
                            windows.append(ClipWindow(start=start, end=end))
                        start = None
                        gap = 0
        if start is not None:
            end = len(visibility) - 1
            if end - start + 1 >= self.min_length:
                windows.append(ClipWindow(start=start, end=end))
        return windows
