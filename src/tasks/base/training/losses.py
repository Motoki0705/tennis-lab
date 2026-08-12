"""Shared loss functions reused across tasks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalBCEWithLogitsLoss(nn.Module):
    """Focal-modulated binary cross-entropy operating on raw logits.

    Consolidates the previously duplicated focal losses in ``ball_detection``
    (``BallDetectionFocalLoss``) and ``court_detection``
    (``FocalBCEWithLogitsLoss``). The mathematical form is identical to both:

    ``loss = mean( (1 - p_t) ** gamma * BCE(logits, targets) )``

    Parameters
    ----------
    gamma:
        Focusing parameter. Must be non-negative.
    positive_weight:
        Finite positive multiplier for the BCE positive term. A value of 1.0
        preserves unweighted BCE, including for soft targets.
    """

    def __init__(self, *, gamma: float, positive_weight: float) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("focal loss gamma must be non-negative.")
        if not math.isfinite(positive_weight) or positive_weight <= 0.0:
            raise ValueError("focal loss positive_weight must be finite and positive.")
        self.gamma = float(gamma)
        self.positive_weight = float(positive_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=logits.new_tensor(self.positive_weight),
            reduction="none",
        )
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        return ((1.0 - p_t) ** self.gamma * bce).mean()


def validate_focal_bce_inputs(logits: Tensor, targets: Tensor) -> None:
    """Validate focal-loss tensors at the task training boundary."""
    if logits.shape != targets.shape:
        raise ValueError(
            "FocalBCEWithLogitsLoss expects logits and targets with the same "
            f"shape, got {tuple(logits.shape)} vs {tuple(targets.shape)}."
        )


__all__ = ["FocalBCEWithLogitsLoss", "validate_focal_bce_inputs"]
