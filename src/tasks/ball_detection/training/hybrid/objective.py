"""Native heatmap and query-presence losses on the same reviewed frame mask."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd.hybrid import MDDHybridOutput
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from src.tasks.ball_detection.training.hybrid.configuration import validate_loss


def hybrid_objective(output: MDDHybridOutput, batch: dict[str, Any], *,
                     loss_config: dict[str, float]) -> tuple[Tensor, dict[str, Tensor]]:
    validate_loss(loss_config)
    positive = batch["position_valid"].to(output.presence_logits.device)
    valid = batch["heatmap_valid"].to(output.presence_logits.device)
    if positive.dtype != torch.bool or valid.dtype != torch.bool or positive.shape != output.presence_logits.shape:
        raise ValueError("Hybrid targets must be boolean B,T masks aligned with presence logits")
    if valid.shape != positive.shape or bool((positive & ~valid).any()):
        raise ValueError("Position supervision must imply heatmap/presence supervision")
    heatmap, heatmap_frames = heatmap_objective(output.heatmap_logits, batch,
        sigma_ratio=loss_config["sigma_ratio"], gamma=loss_config["focal_gamma"])
    presence_frames = F.binary_cross_entropy_with_logits(output.presence_logits.float(), positive.float(), reduction="none")
    presence = torch.where(valid, presence_frames, 0.).sum() / valid.sum()
    total = loss_config["heatmap_weight"] * heatmap + loss_config["presence_weight"] * presence
    if not bool(torch.isfinite(total)):
        raise ValueError("Nonfinite hybrid objective")
    return total, dict(heatmap=heatmap, presence=presence, heatmap_frames=heatmap_frames, presence_frames=presence_frames)
