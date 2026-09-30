"""Candidate-relative means with explicit free branches for missing evidence."""

from __future__ import annotations

import torch
from torch import Tensor


def candidate_mean_logits(
    mean_logits: Tensor, features: Tensor, valid: Tensor, *, count: int, max_offset_uv: float,
) -> Tensor:
    """Anchor ``count`` components; leave the remaining components unrestricted.

    Sort by score descending, then y/x ascending for permutation-invariant ties.
    This internal component assignment never changes detector ranking/selection.
    Invalid slots (including artificial gaps) use their learned absolute means.
    The output is logits so the existing float32 MDN decoder stays unchanged.
    """
    # Anchoring is computed in float32 even when the trunk uses autocast.
    coords, score = features[..., :2].float(), features[..., 2].float()
    order = torch.argsort(coords[..., 0], dim=-1, stable=True)
    order = order.gather(-1, torch.argsort(coords[..., 1].gather(-1, order), dim=-1, stable=True))
    score = score.masked_fill(~valid, -torch.inf)
    order = order.gather(-1, torch.argsort(score.gather(-1, order), dim=-1, descending=True, stable=True))
    chosen = order[..., :count]
    anchor = coords.gather(-2, chosen[..., None].expand(*chosen.shape, 2))
    has_anchor = valid.gather(-1, chosen)
    raw = mean_logits.float()
    anchored = anchor + max_offset_uv * raw[..., :count, :].tanh()
    # Clipping imposes the existing [0,1] mean contract. eps only makes logit finite.
    anchored = torch.logit(anchored.clamp(1e-6, 1 - 1e-6))
    head = torch.where(has_anchor[..., None], anchored, raw[..., :count, :])
    return torch.cat((head, raw[..., count:, :]), dim=-2)
