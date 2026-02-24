"""Event frame extraction for trajectory completion."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from src.tasks.blcs.data.meta_codec import parse_blcs_scene_meta, parse_rally_scene_meta


def _filter_frames(frames: list[int], length: int, *, offset: int = 0) -> Tensor:
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


def _coerce_frame(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_from_shots(shots: Any) -> tuple[list[int], list[int]]:
    bounce_frames: list[int] = []
    shot_frames: list[int] = []
    if not isinstance(shots, list):
        return shot_frames, bounce_frames
    for shot in shots:
        if not isinstance(shot, Mapping):
            continue
        t_start = _coerce_frame(shot.get("t_start"))
        if t_start is not None and t_start >= 0:
            shot_frames.append(t_start)
        t_b1 = _coerce_frame(shot.get("t_bounce1"))
        t_b2 = _coerce_frame(shot.get("t_bounce2"))
        if t_b1 is not None and t_b1 >= 0:
            bounce_frames.append(t_b1)
        if t_b2 is not None and t_b2 >= 0:
            bounce_frames.append(t_b2)
    return shot_frames, bounce_frames


def extract_event_frames(meta: Mapping, length: int, *, offset: int = 0) -> dict[str, Tensor]:
    """Extract bounce/shot frame indices from BLCS metadata.

    Args:
        meta: Decoded metadata dictionary (or mapping-like object).
        length: Target sequence length (after slicing/cropping).
        offset: Starting frame offset of the slice within the original sequence.
    """
    meta_dict = dict(meta) if isinstance(meta, Mapping) else {}

    rally = parse_rally_scene_meta(meta_dict)
    if rally is not None:
        shot_frames, bounce_frames = _collect_from_shots(rally.shots)
        return {
            "bounce": _filter_frames(bounce_frames, length, offset=offset),
            "shot": _filter_frames(shot_frames, length, offset=offset),
        }

    if "shots" in meta_dict:
        shot_frames, bounce_frames = _collect_from_shots(meta_dict.get("shots"))
        if shot_frames or bounce_frames:
            return {
                "bounce": _filter_frames(bounce_frames, length, offset=offset),
                "shot": _filter_frames(shot_frames, length, offset=offset),
            }

    single = parse_blcs_scene_meta(meta_dict)
    if single is not None:
        bounce_frames: list[int] = []
        shot_frames: list[int] = []
        if single.t_bounce1 >= 0:
            bounce_frames.append(int(single.t_bounce1))
        if single.t_bounce2 >= 0:
            bounce_frames.append(int(single.t_bounce2))
        shot_frames.append(0)
        return {
            "bounce": _filter_frames(bounce_frames, length, offset=offset),
            "shot": _filter_frames(shot_frames, length, offset=offset),
        }

    return {
        "bounce": torch.empty(0, dtype=torch.long),
        "shot": torch.empty(0, dtype=torch.long),
    }
