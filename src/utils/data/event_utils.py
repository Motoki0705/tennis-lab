"""Event frame extraction utilities for shot/bounce metadata."""

from __future__ import annotations

import torch
from torch import Tensor


def extract_event_indices(meta: dict, key: str) -> list[int]:
    """Extract non-negative integer event frame indices from scene metadata.

    Reads ``meta["shots"]`` (a list of dicts) and collects ``int(shot[key])``
    for each entry where the value is >= 0.

    Args:
        meta: Scene metadata dictionary.
        key: Key to look up within each shot dict (e.g. ``"t_start"``).

    Returns:
        Sorted list of event frame indices.
    """
    shots = meta.get("shots", []) or []
    indices: list[int] = []
    for s in shots:
        if not isinstance(s, dict):
            continue
        t = int(s.get(key, -1))
        if t >= 0:
            indices.append(t)
    return sorted(indices)


def _filter_frames(frames: list[int], length: int, *, offset: int = 0) -> Tensor:
    """Adjust and filter frame indices to a target window.

    Args:
        frames: Raw frame indices (absolute, before offset adjustment).
        length: Target sequence length after slicing/cropping.
        offset: Starting frame offset of the slice within the original sequence.

    Returns:
        Sorted, deduplicated tensor of valid frame indices (``dtype=torch.long``).
    """
    if length <= 0:
        return torch.empty(0, dtype=torch.long)
    adjusted: list[int] = []
    for t in frames:
        t_adj = int(t) - int(offset)
        if 0 <= t_adj < length:
            adjusted.append(t_adj)
    if not adjusted:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(set(adjusted)), dtype=torch.long)


def extract_event_frames(
    meta: dict,
    length: int,
    *,
    offset: int = 0,
) -> dict[str, Tensor]:
    """Extract bounce and shot frame indices from scene metadata.

    Reads ``meta["shots"]`` and collects ``t_bounce1``/``t_bounce2`` as bounce
    events and ``t_start`` as shot events.  Frame indices are adjusted by
    *offset* and clipped to ``[0, length)``.

    Args:
        meta: Decoded metadata dictionary.
        length: Target sequence length (after slicing/cropping).
        offset: Starting frame offset of the slice within the original sequence.

    Returns:
        Dictionary with ``"bounce"`` and ``"shot"`` tensors of valid frame
        indices (``dtype=torch.long``).
    """
    if not isinstance(meta, dict):
        return {
            "bounce": torch.empty(0, dtype=torch.long),
            "shot": torch.empty(0, dtype=torch.long),
        }

    bounce_raw = (
        extract_event_indices(meta, "t_bounce1")
        + extract_event_indices(meta, "t_bounce2")
    )
    shot_raw = extract_event_indices(meta, "t_start")

    return {
        "bounce": _filter_frames(bounce_raw, length, offset=offset),
        "shot": _filter_frames(shot_raw, length, offset=offset),
    }
