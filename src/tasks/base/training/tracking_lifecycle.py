"""Shared transition-aware presence supervision for lifecycle tracking."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def lifecycle_transition_mask(
    target_presence: Tensor,
    valid_frames: Tensor,
    *,
    radius: int,
) -> Tensor:
    """Mark birth/death neighborhoods without crossing padded boundaries."""
    if target_presence.shape != valid_frames.shape:
        raise ValueError("target_presence and valid_frames must have equal shape.")
    if radius < 0:
        raise ValueError("transition radius must be non-negative.")
    squeeze_slot = target_presence.ndim == 1
    if squeeze_slot:
        target_presence = target_presence.unsqueeze(-1)
        valid_frames = valid_frames.unsqueeze(-1)
    transition = torch.zeros_like(target_presence, dtype=torch.bool)
    if target_presence.shape[-2] < 2:
        return transition.squeeze(-1) if squeeze_slot else transition
    changes = (
        (target_presence[..., 1:, :] != target_presence[..., :-1, :])
        & valid_frames[..., 1:, :]
        & valid_frames[..., :-1, :]
    )
    boundaries = torch.zeros_like(target_presence, dtype=torch.bool)
    boundaries[..., 1:, :] = changes
    transition |= boundaries
    for offset in range(1, radius + 1):
        transition[..., :-offset, :] |= boundaries[..., offset:, :]
        transition[..., offset:, :] |= boundaries[..., :-offset, :]
    transition &= valid_frames
    return transition.squeeze(-1) if squeeze_slot else transition


def weighted_presence_bce_with_logits(
    logits: Tensor,
    target_presence: Tensor,
    valid_frames: Tensor,
    *,
    inactive_weight: float,
    active_weight: float,
    transition_weight: float,
    transition_radius: int,
) -> Tensor:
    """Compute valid-frame BCE with stronger lifecycle-boundary supervision."""
    if logits.shape != target_presence.shape or logits.shape != valid_frames.shape:
        raise ValueError("logits, target_presence, and valid_frames must match.")
    if min(inactive_weight, active_weight, transition_weight) < 0:
        raise ValueError("Presence weights must be non-negative.")
    target = target_presence.to(dtype=logits.dtype)
    weights = torch.where(
        target_presence,
        torch.as_tensor(active_weight, dtype=logits.dtype, device=logits.device),
        torch.as_tensor(inactive_weight, dtype=logits.dtype, device=logits.device),
    )
    transition = lifecycle_transition_mask(
        target_presence,
        valid_frames,
        radius=transition_radius,
    )
    weights = torch.where(
        transition,
        torch.as_tensor(transition_weight, dtype=logits.dtype, device=logits.device),
        weights,
    )
    weights = weights * valid_frames.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = ["lifecycle_transition_mask", "weighted_presence_bce_with_logits"]
