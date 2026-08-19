"""Fixed-width lifecycle packing for BLCS observation candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.tasks.base.data.lifecycle_slots import (
    LifecycleSlotAssignment,
    pack_lifecycle_slots,
)


@dataclass(frozen=True, slots=True)
class PackedObservationCandidates:
    """All-view synchronized observation candidates with an exact slot width."""

    uv: Tensor
    visible: Tensor
    candidate_mask: Tensor
    gt_index: Tensor


def build_fixed_lifecycle_assignment(
    physical_presence: Tensor,
    *,
    num_slots: int,
    min_reuse_gap_frames: int,
    randomize_slots: bool,
    generator: torch.Generator | None = None,
) -> LifecycleSlotAssignment:
    """Build one exact-width assignment using the worker-seeded Torch RNG.

    Lifecycle coloring is deterministic first.  Training-only randomization then
    permutes the slot labels with one independent ``torch.randperm`` draw.  A
    caller obtains independent target and observation assignments by calling
    this function twice.
    """
    assignment = pack_lifecycle_slots(
        physical_presence,
        num_slots=num_slots,
        min_reuse_gap_frames=min_reuse_gap_frames,
        randomize_slots=False,
    )
    if not randomize_slots:
        return assignment

    permutation = torch.randperm(
        num_slots,
        device=physical_presence.device,
        generator=generator,
    )
    track_to_slot = assignment.track_to_slot.clone()
    assigned = track_to_slot >= 0
    track_to_slot[assigned] = permutation[track_to_slot[assigned]]
    return LifecycleSlotAssignment(
        track_to_slot=track_to_slot,
        target_presence=assignment.target_presence[:, permutation.argsort()],
        target_instance_id=assignment.target_instance_id[:, permutation.argsort()],
    )


def pack_observation_candidates(
    *,
    ball_uv: Tensor,
    ball_visible: Tensor,
    physical_presence: Tensor,
    num_slots: int,
    min_reuse_gap_frames: int,
    randomize_slots: bool,
    generator: torch.Generator | None = None,
) -> PackedObservationCandidates:
    """Pack physical observations ``[V,T,P]`` into exact ``[V,T,Q]`` streams."""
    if ball_uv.ndim != 4 or ball_uv.shape[-1] != 2:
        raise ValueError("ball_uv must have shape (V,T,P,2).")
    if ball_visible.shape != ball_uv.shape[:-1]:
        raise ValueError("ball_visible must match ball_uv without the UV axis.")
    if physical_presence.shape != ball_uv.shape[1:3]:
        raise ValueError("physical_presence must match ball_uv's (T,P) axes.")
    if ball_visible.dtype != torch.bool or physical_presence.dtype != torch.bool:
        raise TypeError("ball_visible and physical_presence must have dtype bool.")
    if ball_uv.device != ball_visible.device or ball_uv.device != physical_presence.device:
        raise ValueError("candidate packing tensors must share one device.")
    if bool((ball_visible & ~physical_presence.unsqueeze(0)).any()):
        raise ValueError("ball_visible cannot be true for an absent physical object.")

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
        ball_visible.permute(1, 2, 0),
        physical_presence,
    ).permute(2, 0, 1)
    candidate_mask = assignment.target_presence.unsqueeze(0).expand(
        ball_uv.shape[0], -1, -1
    )
    gt_index = assignment.target_instance_id.unsqueeze(0).expand(
        ball_uv.shape[0], -1, -1
    )
    packed_uv = packed_uv.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
    packed_visible = packed_visible & candidate_mask
    return PackedObservationCandidates(
        uv=packed_uv,
        visible=packed_visible,
        candidate_mask=candidate_mask,
        gt_index=gt_index,
    )


__all__ = [
    "PackedObservationCandidates",
    "build_fixed_lifecycle_assignment",
    "pack_observation_candidates",
]
