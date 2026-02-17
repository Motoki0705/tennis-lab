"""Lightning module for sequence pretrain and self-train stages."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.ball_detection.models import build_model
from src.ball_detection.training.losses import event_aware_weight, visibility_bce_loss, weighted_xy_loss


class BallDetectionLightningModule(BaseLightningModule):
    """Trainable sequence module with event-aware sample weighting."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = build_model(config)

        train_cfg = config.get("training", {})
        self.xy_weight = float(train_cfg.get("xy_weight", 1.0))
        self.vis_weight = float(train_cfg.get("vis_weight", 0.5))
        self.event_boost = float(train_cfg.get("event_boost", 0.0))

    def forward(self, frames: Tensor, frame_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.model(frames, frame_mask=frame_mask)

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        out = self.model(batch["frames"], frame_mask=batch.get("frame_mask"))
        pred_xy = out["xy"]
        pred_vis_logit = out["visibility_logit"]

        target_xy = batch["target_xy"]
        target_vis = batch["target_vis"]
        frame_mask = batch.get("frame_mask", torch.ones_like(target_vis))

        base_weight = batch.get("target_weight", torch.ones_like(target_vis))
        event_mask = batch.get("event_mask")
        w = event_aware_weight(base_weight, event_mask, self.event_boost)

        xy_valid = (frame_mask > 0) & (target_vis > 0)
        vis_valid = frame_mask > 0

        loss_xy = weighted_xy_loss(
            pred_xy,
            target_xy,
            weight=w,
            valid_mask=xy_valid,
        ) * self.xy_weight
        loss_vis = visibility_bce_loss(
            pred_vis_logit,
            target_vis,
            weight=w,
            valid_mask=vis_valid,
        ) * self.vis_weight
        loss = loss_xy + loss_vis

        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/loss_xy", loss_xy, prog_bar=False)
        self.log(f"{stage}/loss_vis", loss_vis, prog_bar=False)
        self.log(f"{stage}/event_boost", torch.tensor(self.event_boost), prog_bar=False)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        _ = batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        self._shared_step(batch, "val")
