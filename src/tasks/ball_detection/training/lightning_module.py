"""Lightning module for sequence pretrain and self-train stages."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from src.ball_detection.models.heatmap_utils import (
    build_target_heatmaps,
    decode_heatmap_logits,
    tracknet_weighted_bce_with_logits_loss,
    weighted_heatmap_bce_loss,
)
from src.base.training.lightning_module import BaseLightningModule
from src.ball_detection.models import build_model
from src.ball_detection.training.losses import event_aware_weight, visibility_bce_loss, weighted_xy_loss


class BallDetectionLightningModule(BaseLightningModule):
    """Trainable sequence module with event-aware sample weighting."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.model = build_model(config)

        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        self.xy_weight = float(train_cfg.get("xy_weight", 1.0))
        self.vis_weight = float(train_cfg.get("vis_weight", 0.5))
        self.heatmap_weight = float(train_cfg.get("heatmap_weight", 1.0))
        self.heatmap_sigma = float(train_cfg.get("heatmap_sigma", 2.5))
        self.heatmap_loss_type = str(train_cfg.get("heatmap_loss_type", "bce")).lower()
        self.event_boost = float(train_cfg.get("event_boost", 0.0))
        self.acc_threshold_px = float(train_cfg.get("acc_threshold_px", 4.0))
        self.acc_image_w = int(train_cfg.get("acc_image_w", data_cfg.get("image_w", 512)))
        self.acc_image_h = int(train_cfg.get("acc_image_h", data_cfg.get("image_h", 288)))

    def forward(self, frames: Tensor, frame_mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.model(frames, frame_mask=frame_mask)

    def _compute_distance_accuracy(
        self,
        pred_xy: Tensor,
        target_xy: Tensor,
        target_vis: Tensor,
        frame_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        valid = (frame_mask > 0) & (target_vis > 0)

        dx = (pred_xy[..., 0] - target_xy[..., 0]) * max(self.acc_image_w - 1, 1)
        dy = (pred_xy[..., 1] - target_xy[..., 1]) * max(self.acc_image_h - 1, 1)
        dist = torch.sqrt(dx * dx + dy * dy)

        correct = (dist <= self.acc_threshold_px) & valid
        correct_count = correct.sum(dtype=torch.float32)
        total_count = valid.sum(dtype=torch.float32)
        acc = correct_count / total_count.clamp_min(1.0)
        return acc, correct_count, total_count

    def _log_accuracy(self, stage: str, acc: Tensor, total_count: Tensor) -> None:
        is_train = stage == "train"
        weight = max(int(total_count.detach().item()), 1)
        self.log(
            f"{stage}/acc",
            acc,
            prog_bar=not is_train,
            on_step=is_train,
            on_epoch=True,
            batch_size=weight,
        )

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> Tensor:
        out = self.model(batch["frames"], frame_mask=batch.get("frame_mask"))

        if "heatmap_logits" in out:
            heatmap_logits = out["heatmap_logits"]
            if heatmap_logits.dim() == 3:
                heatmap_logits = heatmap_logits.unsqueeze(1)

            target_xy = batch["target_xy"]
            target_vis = batch["target_vis"]
            frame_mask = batch.get("frame_mask", torch.ones_like(target_vis))

            if heatmap_logits.shape[1] != target_xy.shape[1]:
                min_len = min(int(heatmap_logits.shape[1]), int(target_xy.shape[1]))
                heatmap_logits = heatmap_logits[:, :min_len]
                target_xy = target_xy[:, :min_len]
                target_vis = target_vis[:, :min_len]
                frame_mask = frame_mask[:, :min_len]

            base_weight = batch.get("target_weight", torch.ones_like(target_vis))
            base_weight = base_weight[:, : target_xy.shape[1]]
            event_mask = batch.get("event_mask")
            if event_mask is not None:
                event_mask = event_mask[:, : target_xy.shape[1]]
            w = event_aware_weight(base_weight, event_mask, self.event_boost)

            target_heatmaps = build_target_heatmaps(
                target_xy,
                target_vis,
                heatmap_hw=(int(heatmap_logits.shape[-2]), int(heatmap_logits.shape[-1])),
                sigma=self.heatmap_sigma,
            )
            valid = frame_mask > 0
            if self.heatmap_loss_type == "tracknet_wbce":
                loss_heatmap = tracknet_weighted_bce_with_logits_loss(
                    heatmap_logits,
                    target_heatmaps,
                    frame_weight=w,
                    valid_mask=valid,
                )
            else:
                loss_heatmap = weighted_heatmap_bce_loss(
                    heatmap_logits,
                    target_heatmaps,
                    frame_weight=w,
                    valid_mask=valid,
                )
            loss_heatmap = loss_heatmap * self.heatmap_weight

            pred_xy, pred_vis_logit = decode_heatmap_logits(heatmap_logits)
            xy_valid = (frame_mask > 0) & (target_vis > 0)
            vis_valid = frame_mask > 0
            acc, _, total_count = self._compute_distance_accuracy(
                pred_xy,
                target_xy,
                target_vis,
                frame_mask,
            )

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

            loss = loss_heatmap + loss_xy + loss_vis
            self.log(f"{stage}/loss", loss, prog_bar=True)
            self.log(f"{stage}/loss_heatmap", loss_heatmap, prog_bar=False)
            self.log(f"{stage}/loss_xy", loss_xy, prog_bar=False)
            self.log(f"{stage}/loss_vis", loss_vis, prog_bar=False)
            self.log(f"{stage}/event_boost", torch.tensor(self.event_boost), prog_bar=False)
            self._log_accuracy(stage, acc, total_count)
            return loss

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
        acc, _, total_count = self._compute_distance_accuracy(
            pred_xy,
            target_xy,
            target_vis,
            frame_mask,
        )

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
        self._log_accuracy(stage, acc, total_count)
        return loss

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        _ = batch_idx
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        _ = batch_idx
        self._shared_step(batch, "test")
