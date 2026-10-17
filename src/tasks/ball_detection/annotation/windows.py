"""Dense real-frame coverage and deterministic overlap ownership, without padding."""

from __future__ import annotations


def window_starts(count: int, stride: int) -> tuple[int, ...]:
    if count < 32:
        raise ValueError(
            "MDD inference requires at least 32 real frames; padding is forbidden"
        )
    if not 1 <= stride <= 32:
        raise ValueError("MDD stride must be between 1 and 32 real frames")
    starts = list(range(0, count - 31, stride))
    if starts[-1] != count - 32:
        starts.append(count - 32)
    return tuple(starts)


def window_owners(count: int, stride: int) -> tuple[int, ...]:
    starts = window_starts(count, stride)
    owners: list[tuple[float, int] | None] = [None] * count
    for start in starts:
        for index in range(start, start + 32):
            candidate = (abs(index - start - 15.5), start)
            previous = owners[index]
            if previous is None or candidate < previous:
                owners[index] = candidate
    if any(owner is None for owner in owners):
        raise ValueError("inference windows leave a real frame uncovered")
    return tuple(owner[1] for owner in owners if owner is not None)
