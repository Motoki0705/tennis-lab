"""Utilities for heatmap-based supervision and decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def build_target_heatmaps(
    target_xy: Tensor,
    target_vis: Tensor,
    *,
    heatmap_hw: tuple[int, int],
    sigma: float,
) -> Tensor:
    """Build per-frame Gaussian heatmaps from normalized xy targets."""
    if target_xy.dim() != 3 or target_xy.shape[-1] != 2:
        raise ValueError(f"target_xy must have shape [B, T, 2], got {tuple(target_xy.shape)}")
    if target_vis.shape[:2] != target_xy.shape[:2]:
        raise ValueError(
            "target_vis must match target_xy first two dims: "
            f"{tuple(target_vis.shape)} vs {tuple(target_xy.shape)}"
        )

    h, w = int(heatmap_hw[0]), int(heatmap_hw[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"heatmap_hw must be positive, got {(h, w)}")

    device = target_xy.device
    dtype = target_xy.dtype

    x = torch.clamp(target_xy[..., 0], 0.0, 1.0) * max(w - 1, 1)
    y = torch.clamp(target_xy[..., 1], 0.0, 1.0) * max(h - 1, 1)
    visible = (target_vis > 0).to(dtype=dtype)

    if sigma <= 0:
        heatmaps = torch.zeros((*target_xy.shape[:2], h, w), dtype=dtype, device=device)
        xi = torch.round(x).to(dtype=torch.long).clamp_(0, w - 1)
        yi = torch.round(y).to(dtype=torch.long).clamp_(0, h - 1)
        batch_idx = torch.arange(target_xy.shape[0], device=device).unsqueeze(1)
        time_idx = torch.arange(target_xy.shape[1], device=device).unsqueeze(0)
        heatmaps[batch_idx, time_idx, yi, xi] = visible
        return heatmaps

    ys = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)
    x = x.unsqueeze(-1).unsqueeze(-1)
    y = y.unsqueeze(-1).unsqueeze(-1)
    gauss = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2.0 * sigma * sigma))
    return gauss * visible.unsqueeze(-1).unsqueeze(-1)


def weighted_heatmap_bce_loss(
    logits: Tensor,
    target_heatmaps: Tensor,
    *,
    frame_weight: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute weighted BCE-with-logits loss for heatmap targets."""
    if logits.shape != target_heatmaps.shape:
        raise ValueError(
            "logits and target_heatmaps must share shape, got "
            f"{tuple(logits.shape)} and {tuple(target_heatmaps.shape)}"
        )

    loss = F.binary_cross_entropy_with_logits(logits, target_heatmaps, reduction="none")

    if frame_weight is not None:
        if frame_weight.shape != logits.shape[:2]:
            raise ValueError(
                "frame_weight must match [B, T], got "
                f"{tuple(frame_weight.shape)} for logits {tuple(logits.shape)}"
            )
        loss = loss * frame_weight.unsqueeze(-1).unsqueeze(-1)

    if valid_mask is None:
        return loss.mean()

    if valid_mask.shape != logits.shape[:2]:
        raise ValueError(
            "valid_mask must match [B, T], got "
            f"{tuple(valid_mask.shape)} for logits {tuple(logits.shape)}"
        )

    mask = valid_mask.to(dtype=loss.dtype).unsqueeze(-1).unsqueeze(-1)
    loss = loss * mask
    denom = mask.sum() * float(logits.shape[-2] * logits.shape[-1])
    return loss.sum() / denom.clamp_min(1.0)


def tracknet_weighted_bce_with_logits_loss(
    logits: Tensor,
    target_heatmaps: Tensor,
    *,
    frame_weight: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """TrackNet-style weighted BCE using probabilities from logits.

    Loss per pixel:
        - ((1 - p)^2 * y * log(p) + p^2 * (1 - y) * log(1 - p))
    where p = sigmoid(logits).
    """
    if logits.shape != target_heatmaps.shape:
        raise ValueError(
            "logits and target_heatmaps must share shape, got "
            f"{tuple(logits.shape)} and {tuple(target_heatmaps.shape)}"
        )

    prob = torch.sigmoid(logits)
    eps = 1e-7
    loss = -(
        ((1.0 - prob) ** 2) * target_heatmaps * torch.log(torch.clamp(prob, min=eps, max=1.0))
        + (prob**2) * (1.0 - target_heatmaps) * torch.log(
            torch.clamp(1.0 - prob, min=eps, max=1.0)
        )
    )

    if frame_weight is not None:
        if frame_weight.shape != logits.shape[:2]:
            raise ValueError(
                "frame_weight must match [B, T], got "
                f"{tuple(frame_weight.shape)} for logits {tuple(logits.shape)}"
            )
        loss = loss * frame_weight.unsqueeze(-1).unsqueeze(-1)

    if valid_mask is None:
        return loss.mean()

    if valid_mask.shape != logits.shape[:2]:
        raise ValueError(
            "valid_mask must match [B, T], got "
            f"{tuple(valid_mask.shape)} for logits {tuple(logits.shape)}"
        )

    mask = valid_mask.to(dtype=loss.dtype).unsqueeze(-1).unsqueeze(-1)
    loss = loss * mask
    denom = mask.sum() * float(logits.shape[-2] * logits.shape[-1])
    return loss.sum() / denom.clamp_min(1.0)


def decode_heatmap_logits(heatmap_logits: Tensor) -> tuple[Tensor, Tensor]:
    """Decode xy and visibility logits from heatmap logits."""
    squeeze_time = False
    if heatmap_logits.dim() == 3:
        heatmap_logits = heatmap_logits.unsqueeze(1)
        squeeze_time = True
    if heatmap_logits.dim() != 4:
        raise ValueError(
            "heatmap_logits must have shape [B, T, H, W] or [B, H, W], "
            f"got {tuple(heatmap_logits.shape)}"
        )

    bsz, seq_len, h, w = heatmap_logits.shape
    flat = heatmap_logits.view(bsz, seq_len, -1)
    vis_logit, indices = torch.max(flat, dim=-1)

    xs = (indices % w).to(dtype=heatmap_logits.dtype)
    ys = (indices // w).to(dtype=heatmap_logits.dtype)

    xy = torch.stack(
        [
            xs / max(w - 1, 1),
            ys / max(h - 1, 1),
        ],
        dim=-1,
    )

    if squeeze_time:
        return xy[:, 0], vis_logit[:, 0]
    return xy, vis_logit
