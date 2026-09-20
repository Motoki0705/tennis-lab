"""Stable inclusive sampling; splitting uses the original full image order."""

from __future__ import annotations


def sample_indices(first: int, last: int, count: int) -> list[int]:
    if first < 0 or count < 2 or last - first + 1 < count:
        raise ValueError(
            "Need at least two distinct samples in a valid inclusive interval"
        )
    return [first + round(i * (last - first) / (count - 1)) for i in range(count)]


def extend_indices(existing: list[int], count: int) -> list[int]:
    """Keep every existing view and bisect the largest gap; ties use earlier gaps."""
    if (
        len(existing) < 2
        or existing != sorted(set(existing))
        or existing[0] < 0
        or not len(existing) <= count <= existing[-1] - existing[0] + 1
    ):
        raise ValueError("Need sorted unique views and a feasible expanded count")
    selected = list(existing)
    while len(selected) < count:
        left, right = max(
            zip(selected[:-1], selected[1:], strict=True),
            key=lambda pair: (pair[1] - pair[0], -pair[0]),
        )
        selected.append((left + right) // 2)
        selected.sort()
    return selected
