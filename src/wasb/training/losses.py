"""Loss functions for WASB tennis training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LossWeights:
    """Weights for WASB heatmap and optional visibility losses."""

    heatmap: float = 1.0


class WASBLoss(nn.Module):
    """Binary cross-entropy heatmap loss for WASB tennis models."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        pred_heatmaps: Tensor,
        target_heatmaps: Tensor,
        visibility: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute heatmap BCE with optional frame masking.

        Args:
            pred_heatmaps: Predicted heatmaps (B, T, H, W).
            target_heatmaps: Target heatmaps (B, T, H, W).
            visibility: Optional visibility mask (B, T) where >0 is valid.

        Returns:
            Dict with individual losses and total.
        """
        heatmap_loss = heatmap_bce(pred_heatmaps, target_heatmaps, visibility)
        total = self.weights.heatmap * heatmap_loss

        return {
            "total": total,
            "heatmap": heatmap_loss,
        }


def heatmap_bce(
    pred_heatmaps: Tensor,
    target_heatmaps: Tensor,
    visibility: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Frame-wise BCE between predicted and target heatmaps."""
    loss = F.binary_cross_entropy_with_logits(
        pred_heatmaps, target_heatmaps, reduction="none"
    )

    if visibility is None:
        return loss.mean()

    vis_mask = (visibility > 0).to(dtype=pred_heatmaps.dtype, device=pred_heatmaps.device)
    vis_mask = vis_mask.view(pred_heatmaps.shape[0], pred_heatmaps.shape[1], 1, 1)
    masked = loss * vis_mask
    denom = vis_mask.sum() * pred_heatmaps.shape[-2] * pred_heatmaps.shape[-1]
    return masked.sum() / (denom + eps)
