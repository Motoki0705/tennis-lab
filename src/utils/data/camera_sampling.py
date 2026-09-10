"""Explicit, ordered camera candidate sets for generation and sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast


def camera_candidate_indices(
    value: object, *, capacity: int | None = None
) -> tuple[int, ...] | None:
    """Parse an opt-in candidate set; None explicitly selects the full rig."""
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            "camera candidate indices must be a sequence of integers or null."
        )
    indices = tuple(value)
    if not indices or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError(
            "camera candidate indices must be nonempty nonnegative integers."
        )
    if len(set(indices)) != len(indices):
        raise ValueError("camera candidate indices must be unique.")
    if capacity is not None and any(i >= capacity for i in indices):
        raise ValueError(f"camera candidate index is outside capacity {capacity}.")
    return cast(tuple[int, ...], indices)
