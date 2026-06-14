"""Loss functions for court detection training.

The focal heatmap loss is provided by the shared foundation as
:class:`src.tasks.base.training.losses.FocalBCEWithLogitsLoss`; it is
re-exported here so existing ``court_detection.training`` references keep
working unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.base.training.losses import (
    FocalBCEWithLogitsLoss as FocalBCEWithLogitsLoss,
)


class DiceLoss(nn.Module):
    """Per-class Dice loss averaged over classes (for segmentation)."""

    def __init__(self, num_classes: int, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Parameters
        ----------
        logits:
            ``[B, C, H, W]`` raw logits.
        targets:
            ``[B, H, W]`` int64 labels.
        """
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BinaryDiceLoss(nn.Module):
    """Dice loss for binary segmentation logits."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
