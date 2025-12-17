"""Loss utilities for WASB trajectory completion training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TrajectoryLossWeights:
    block: float = 1.0
    sparse: float = 1.0
    noise: float = 1.0


def masked_mean(loss: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=loss.dtype, device=loss.device)
    denom = mask.sum()
    if denom <= 0:
        return torch.zeros((), dtype=loss.dtype, device=loss.device)
    return (loss * mask).sum() / (denom + 1e-8)


def trajectory_losses(
    *,
    mse_per_frame: Tensor,
    loss_mask_block: Tensor,
    loss_mask_sparse: Tensor,
    loss_mask_noise: Tensor,
    weights: TrajectoryLossWeights,
) -> dict[str, Tensor]:
    loss_block = masked_mean(mse_per_frame, loss_mask_block)
    loss_sparse = masked_mean(mse_per_frame, loss_mask_sparse)
    loss_noise = masked_mean(mse_per_frame, loss_mask_noise)
    total = (
        float(weights.block) * loss_block
        + float(weights.sparse) * loss_sparse
        + float(weights.noise) * loss_noise
    )
    return {
        "total": total,
        "block": loss_block,
        "sparse": loss_sparse,
        "noise": loss_noise,
    }

