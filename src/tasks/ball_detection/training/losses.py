"""Loss functions for supervised ball detection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BallDetectionFocalLoss(nn.Module):
    """Focal BCE-with-logits loss for dense ball heatmaps."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = config or {}
        self.gamma = float(config.get("gamma", 2.0))
        if self.gamma < 0:
            raise ValueError("loss.gamma must be non-negative.")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal BCE over the full heatmap tensor."""
        if logits.shape != targets.shape:
            raise ValueError(
                "BallDetectionFocalLoss expects logits and targets with the same shape, "
                f"got {tuple(logits.shape)} vs {tuple(targets.shape)}."
            )
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        loss = ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


__all__ = ["BallDetectionFocalLoss"]
