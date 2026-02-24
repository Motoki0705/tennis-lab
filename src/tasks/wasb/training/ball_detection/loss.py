"""Loss functions for WASB ball detection training.

Adds an optional temporal-consistency loss that encourages the predicted heatmap
peak to move smoothly over time (velocity or acceleration regularization) using
a differentiable soft-argmax style coordinate extraction.

Notes:
- `pred_heatmaps` are treated as logits for BCEWithLogits (as in your original code).
- The temporal loss converts logits -> probability map internally.
- If a loss weight is 0.0, that loss is NOT computed (skipped).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LossWeights:
    """Weights for WASB losses.

    If a weight is 0.0, the corresponding loss is skipped entirely.
    """

    bce: float = 1.0
    mse: float = 1.0
    temporal: float = 0.0


@dataclass(frozen=True)
class TemporalPeakLossConfig:
    """Configuration for temporal peak consistency loss.

    Attributes:
        order: 1 => velocity smoothness, 2 => acceleration smoothness (recommended).
        robust: If True, use SmoothL1 (Huber) for stability against outliers.
        prob_mode:
            - "sigmoid_norm": sigmoid(logits) then normalize spatially to sum to 1.
              Works well with BCE-with-logits training.
            - "spatial_softmax": softmax(beta * logits) across H*W.
        beta: Temperature scale for "spatial_softmax" (ignored for "sigmoid_norm").
        detach_conf: If True, prevents the model from "cheating" by lowering confidence
                     to reduce the temporal penalty.
    """

    order: int = 2
    robust: bool = True
    prob_mode: str = "sigmoid_norm"
    beta: float = 25.0
    detach_conf: bool = True


class WASBLoss(nn.Module):
    """Composite heatmap loss for WASB tennis models (BCE + MSE + optional temporal)."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        temporal_cfg: TemporalPeakLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.temporal_cfg = temporal_cfg or TemporalPeakLossConfig()

    def forward(
        self,
        pred_heatmaps: Tensor,
        target_heatmaps: Tensor,
        visibility: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses with optional frame masking.

        Args:
            pred_heatmaps: Predicted heatmaps (logits) of shape (B, T, H, W).
            target_heatmaps: Target heatmaps of shape (B, T, H, W).
            visibility: Optional visibility mask (B, T) where >0 is valid.

        Returns:
            Dict with individual losses and total.
        """
        zero = pred_heatmaps.new_zeros(())  # scalar 0 on the right device/dtype
        total = zero

        # BCE
        if self.weights.bce != 0.0:
            bce_loss = heatmap_bce(pred_heatmaps, target_heatmaps, visibility)
            total = total + (self.weights.bce * bce_loss)
        else:
            bce_loss = zero  # skipped

        # MSE
        if self.weights.mse != 0.0:
            mse_loss = heatmap_mse(pred_heatmaps, target_heatmaps, visibility)
            total = total + (self.weights.mse * mse_loss)
        else:
            mse_loss = zero  # skipped

        # Temporal consistency (peak smoothness)
        if self.weights.temporal != 0.0:
            temporal_loss = peak_temporal_consistency_loss(
                pred_logits=pred_heatmaps,
                visibility=visibility,
                cfg=self.temporal_cfg,
            )
            total = total + (self.weights.temporal * temporal_loss)
        else:
            temporal_loss = zero  # skipped

        return {
            "total": total,
            "bce": bce_loss,
            "mse": mse_loss,
            "temporal": temporal_loss,
        }


def heatmap_bce(
    pred_heatmaps: Tensor,
    target_heatmaps: Tensor,
    visibility: Tensor | None = None,
    eps: float = 1e-8,
    logit_clip: float | None = 20.0,
) -> Tensor:
    """Frame-wise BCE between predicted and target heatmaps (logits vs targets)."""
    logits = pred_heatmaps
    if logit_clip is not None and logit_clip > 0:
        # Prevent extreme logits that can cause saturated gradients/unstable stats.
        logits = torch.clamp(pred_heatmaps, -logit_clip, logit_clip)

    loss = F.binary_cross_entropy_with_logits(logits, target_heatmaps, reduction="none")

    if visibility is None:
        return loss.mean()

    vis_mask = (visibility > 0).to(dtype=pred_heatmaps.dtype, device=pred_heatmaps.device)
    vis_mask = vis_mask.view(pred_heatmaps.shape[0], pred_heatmaps.shape[1], 1, 1)
    masked = loss * vis_mask
    denom = vis_mask.sum() * pred_heatmaps.shape[-2] * pred_heatmaps.shape[-1]
    return masked.sum() / (denom + eps)


def heatmap_mse(
    pred_heatmaps: Tensor,
    target_heatmaps: Tensor,
    visibility: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Frame-wise mean squared error between predicted and target heatmaps."""
    loss = F.mse_loss(pred_heatmaps, target_heatmaps, reduction="none")

    if visibility is None:
        return loss.mean()

    vis_mask = (visibility > 0).to(dtype=pred_heatmaps.dtype, device=pred_heatmaps.device)
    vis_mask = vis_mask.view(pred_heatmaps.shape[0], pred_heatmaps.shape[1], 1, 1)
    masked = loss * vis_mask
    denom = vis_mask.sum() * pred_heatmaps.shape[-2] * pred_heatmaps.shape[-1]
    return masked.sum() / (denom + eps)


def peak_temporal_consistency_loss(
    pred_logits: Tensor,  # (B, T, H, W) logits
    visibility: Tensor | None,
    cfg: TemporalPeakLossConfig,
    eps: float = 1e-8,
) -> Tensor:
    """Temporal smoothness loss on the predicted heatmap peak coordinates.

    - Extracts (x_t, y_t) via a differentiable soft-argmax.
    - Penalizes either velocity (order=1) or acceleration (order=2).

    Returns:
        Scalar tensor loss.
    """
    B, T, H, W = pred_logits.shape
    if cfg.order == 2 and T < 3:
        return pred_logits.new_zeros(())
    if cfg.order == 1 and T < 2:
        return pred_logits.new_zeros(())

    coords, conf = _heatmap_to_coords(
        pred_logits=pred_logits,
        prob_mode=cfg.prob_mode,
        beta=cfg.beta,
        eps=eps,
    )  # coords: (B,T,2), conf: (B,T)

    if cfg.order == 1:
        # d_t = p_{t+1} - p_t
        d = coords[:, 1:] - coords[:, :-1]  # (B, T-1, 2)
        w = conf[:, 1:] * conf[:, :-1]      # (B, T-1)
        if visibility is not None:
            v = (visibility > 0).to(dtype=coords.dtype, device=coords.device)
            w = w * (v[:, 1:] * v[:, :-1])

    elif cfg.order == 2:
        # d_t = p_{t+1} - 2*p_t + p_{t-1}  (discrete acceleration)
        d = coords[:, 2:] - 2.0 * coords[:, 1:-1] + coords[:, :-2]  # (B, T-2, 2)
        w = conf[:, 2:] * conf[:, 1:-1] * conf[:, :-2]              # (B, T-2)
        if visibility is not None:
            v = (visibility > 0).to(dtype=coords.dtype, device=coords.device)
            w = w * (v[:, 2:] * v[:, 1:-1] * v[:, :-2])
    else:
        raise ValueError(f"Unsupported cfg.order={cfg.order} (expected 1 or 2).")

    if cfg.detach_conf:
        w = w.detach()

    if cfg.robust:
        per = F.smooth_l1_loss(d, torch.zeros_like(d), reduction="none").sum(dim=-1)  # (B, *,)
    else:
        per = (d * d).sum(dim=-1)  # (B, *,)

    num = (per * w).sum()
    den = w.sum().clamp_min(eps)

    # If everything is masked, return 0 (avoid noisy NaNs).
    if torch.isfinite(den).item() and den.item() <= eps:
        return pred_logits.new_zeros(())

    return num / den


def _heatmap_to_coords(
    pred_logits: Tensor,  # (B,T,H,W)
    prob_mode: str,
    beta: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Convert heatmap logits to expected peak coordinates (soft-argmax).

    Returns:
        coords: (B, T, 2) where last dim is (x, y) in pixel coordinates.
        conf:   (B, T) confidence proxy (max probability mass).
    """
    B, T, H, W = pred_logits.shape
    dtype = pred_logits.dtype
    device = pred_logits.device

    x_grid = torch.arange(W, device=device, dtype=dtype)
    y_grid = torch.arange(H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")  # (H,W)

    flat = pred_logits.view(B * T, H * W)

    if prob_mode == "spatial_softmax":
        p = F.softmax(flat * beta, dim=-1).view(B * T, H, W)
    elif prob_mode == "sigmoid_norm":
        p = torch.sigmoid(flat).view(B * T, H, W)
        p = p / (p.sum(dim=(1, 2), keepdim=True) + eps)
    else:
        raise ValueError(f"Unknown prob_mode={prob_mode!r} (expected 'sigmoid_norm' or 'spatial_softmax').")

    conf = p.amax(dim=(1, 2)).view(B, T)

    x = (p * xx).sum(dim=(1, 2))
    y = (p * yy).sum(dim=(1, 2))
    coords = torch.stack([x, y], dim=-1).view(B, T, 2)
    return coords, conf
