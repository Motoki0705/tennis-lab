"""Loss functions for court keypoint detection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CourtKeypointLoss(nn.Module):
    """Combined loss for court keypoint detection.

    Includes:
        - Heatmap loss (MSE or Focal)
        - Visibility loss (BCE)

    Args:
        heatmap_config: Heatmap loss configuration.
        visibility_config: Visibility loss configuration.
    """

    def __init__(
        self,
        heatmap_config: dict[str, Any] | None = None,
        visibility_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        heatmap_config = heatmap_config or {}
        visibility_config = visibility_config or {}

        self.heatmap_type = heatmap_config.get("type", "mse")
        self.heatmap_weight = heatmap_config.get("weight", 1.0)

        self.visibility_type = visibility_config.get("type", "bce")
        self.visibility_weight = visibility_config.get("weight", 0.1)

    def forward(
        self,
        pred_heatmaps: Tensor,
        target_heatmaps: Tensor,
        pred_visibility: Tensor,
        target_visibility: Tensor,
    ) -> dict[str, Tensor]:
        """Compute combined loss.

        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W).
            target_heatmaps: Target heatmaps (B, K, H, W).
            pred_visibility: Predicted visibility logits (B, K).
            target_visibility: Target visibility (B, K).

        Returns:
            Dictionary with 'total', 'heatmap', and 'visibility' losses.
        """
        # Heatmap loss
        if self.heatmap_type == "mse":
            heatmap_loss = F.mse_loss(pred_heatmaps, target_heatmaps)
        elif self.heatmap_type == "focal":
            heatmap_loss = self._focal_loss(pred_heatmaps, target_heatmaps)
        else:
            heatmap_loss = F.mse_loss(pred_heatmaps, target_heatmaps)

        # Visibility loss
        visibility_loss = F.binary_cross_entropy_with_logits(
            pred_visibility, target_visibility
        )

        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss
            + self.visibility_weight * visibility_loss
        )

        return {
            "total": total_loss,
            "heatmap": heatmap_loss,
            "visibility": visibility_loss,
        }

    def _focal_loss(
        self,
        pred: Tensor,
        target: Tensor,
        alpha: float = 2.0,
        beta: float = 4.0,
    ) -> Tensor:
        """Focal loss for heatmap regression.

        Args:
            pred: Predicted heatmaps.
            target: Target heatmaps.
            alpha: Focusing parameter for positives.
            beta: Focusing parameter for negatives.

        Returns:
            Focal loss value.
        """
        pred = torch.sigmoid(pred)

        pos_mask = target.ge(0.01).float()
        neg_mask = target.lt(0.01).float()

        pos_loss = -torch.pow(1 - pred, alpha) * torch.log(pred + 1e-8) * pos_mask
        neg_loss = (
            -torch.pow(1 - target, beta)
            * torch.pow(pred, alpha)
            * torch.log(1 - pred + 1e-8)
            * neg_mask
        )

        num_pos = pos_mask.sum()
        if num_pos == 0:
            return neg_loss.sum()

        return (pos_loss.sum() + neg_loss.sum()) / num_pos
