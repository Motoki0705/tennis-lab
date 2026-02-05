"""Event frame extraction for trajectory completion."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor

from src.blcs.data.meta_codec import parse_blcs_scene_meta, parse_rally_scene_meta


def _filter_frames(frames: list[int], length: int) -> Tensor:
    valid = [t for t in frames if 0 <= t < length]
    if not valid:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(set(valid)), dtype=torch.long)


def _extract_from_shots(meta_dict: Mapping, length: int) -> dict[str, Tensor] | None:
    shots = meta_dict.get("shots")
    if not isinstance(shots, list):
        return None
    bounce_frames: list[int] = []
    shot_frames: list[int] = []
    for shot in shots:
        if not isinstance(shot, Mapping):
            continue
        t_start = int(shot.get("t_start", -1))
        if t_start >= 0:
            shot_frames.append(t_start)
        t_b1 = int(shot.get("t_bounce1", -1))
        t_b2 = int(shot.get("t_bounce2", -1))
        if t_b1 >= 0:
            bounce_frames.append(t_b1)
        if t_b2 >= 0:
            bounce_frames.append(t_b2)
    return {
        "bounce": _filter_frames(bounce_frames, length),
        "shot": _filter_frames(shot_frames, length),
    }


def extract_event_frames(meta: Mapping, length: int) -> dict[str, Tensor]:
    """Extract bounce/shot frame indices from BLCS metadata."""
    meta_dict = dict(meta) if isinstance(meta, Mapping) else {}

    bounce_frames: list[int] = []
    shot_frames: list[int] = []

    rally = parse_rally_scene_meta(meta_dict)
    if rally is not None:
        for shot in rally.shots:
            t_start = int(shot.get("t_start", -1))
            if t_start >= 0:
                shot_frames.append(t_start)
            t_b1 = int(shot.get("t_bounce1", -1))
            t_b2 = int(shot.get("t_bounce2", -1))
            if t_b1 >= 0:
                bounce_frames.append(t_b1)
            if t_b2 >= 0:
                bounce_frames.append(t_b2)
        return {
            "bounce": _filter_frames(bounce_frames, length),
            "shot": _filter_frames(shot_frames, length),
        }

    fallback = _extract_from_shots(meta_dict, length)
    if fallback is not None:
        return fallback

    single = parse_blcs_scene_meta(meta_dict)
    if single is not None:
        if single.t_bounce1 >= 0:
            bounce_frames.append(int(single.t_bounce1))
        if single.t_bounce2 >= 0:
            bounce_frames.append(int(single.t_bounce2))
        shot_frames.append(0)
        return {
            "bounce": _filter_frames(bounce_frames, length),
            "shot": _filter_frames(shot_frames, length),
        }

    for key in ("t_bounce1", "t_bounce2"):
        t = int(meta_dict.get(key, -1))
        if t >= 0:
            bounce_frames.append(t)
    if bounce_frames:
        shot_frames.append(0)
        return {
            "bounce": _filter_frames(bounce_frames, length),
            "shot": _filter_frames(shot_frames, length),
        }

    return {
        "bounce": torch.empty(0, dtype=torch.long),
        "shot": torch.empty(0, dtype=torch.long),
    }
