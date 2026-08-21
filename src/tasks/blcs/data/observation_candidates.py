"""Fixed-width lifecycle packing for BLCS observation candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment


@dataclass(frozen=True, slots=True)
class PackedObservationCandidates:
    """All-view synchronized observation candidates with an exact slot width."""

    uv: Tensor
    vis: Tensor
    gt_index: Tensor


def pack_observation_candidates(
    *,
    ball_uv: Tensor,
    ball_vis: Tensor,
    physical_presence: Tensor,
    num_slots: int,
    min_reuse_gap_frames: int,
    randomize_slots: bool,
    generator: torch.Generator | None = None,
) -> PackedObservationCandidates:
    """Pack physical observations ``[V,T,P]`` into exact ``[V,T,Q]`` streams."""
    if ball_uv.ndim != 4 or ball_uv.shape[-1] != 2:
        raise ValueError("ball_uv must have shape (V,T,P,2).")
    if ball_vis.shape != ball_uv.shape[:-1]:
        raise ValueError("ball_vis must match ball_uv without the UV axis.")
    if physical_presence.shape != ball_uv.shape[1:3]:
        raise ValueError("physical_presence must match ball_uv's (T,P) axes.")
    if ball_vis.dtype != torch.bool or physical_presence.dtype != torch.bool:
        raise TypeError("ball_vis and physical_presence must have dtype bool.")
    if ball_uv.device != ball_vis.device or ball_uv.device != physical_presence.device:
        raise ValueError("candidate packing tensors must share one device.")
    if bool((ball_vis & ~physical_presence.unsqueeze(0)).any()):
        raise ValueError("ball_vis cannot be true for an absent physical object.")

    assignment = build_fixed_lifecycle_assignment(
        physical_presence,
        num_slots=num_slots,
        min_reuse_gap_frames=min_reuse_gap_frames,
        randomize_slots=randomize_slots,
        generator=generator,
    )
    packed_uv = assignment.pack_tensor(
        ball_uv.permute(1, 2, 0, 3),
        physical_presence,
    ).permute(2, 0, 1, 3)
    packed_visible = assignment.pack_tensor(
        ball_vis.permute(1, 2, 0),
        physical_presence,
    ).permute(2, 0, 1)
    gt_index = assignment.target_instance_id.unsqueeze(0).expand(
        ball_uv.shape[0], -1, -1
    )
    return PackedObservationCandidates(
        uv=packed_uv,
        vis=packed_visible,
        gt_index=gt_index,
    )


__all__ = [
    "PackedObservationCandidates",
    "pack_observation_candidates",
]
