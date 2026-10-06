"""Soft two-class cross entropy for event probabilities."""

from __future__ import annotations

import torch
from torch import Tensor


def event_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Soft CE with class targets [1-p, p]; no temporal softmax or label threshold."""
    labels = torch.stack((1 - target, target), dim=-1)
    return -(labels * logits.log_softmax(dim=-1)).sum(dim=-1).mean()
