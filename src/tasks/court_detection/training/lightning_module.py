"""PyTorch Lightning module for court detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.base.training.qualitative_callback import save_image_to_tensorboard
from src.tasks.court_detection.models import build_court_detection_model
from src.tasks.court_detection.training.losses import (
    BinaryDiceLoss,
    DiceLoss,
    FocalBCEWithLogitsLoss,
)
from src.tasks.court_detection.training.metrics import CourtDetectionMetrics
from src.tasks.court_detection.training.visualization import (
    save_kp_vis,
    save_line_vis,
    save_seg_vis,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CourtDetectionLightningModule(BaseLightningModule):
    """Unified Lightning module for court detection tasks.

    Supports three tasks via ``config.data.task``:

    * ``seg`` — Court cell segmentation (CE + Dice, 7 classes).
    * ``kp``  — Court keypoint heatmap regression (Focal BCE, 14 channels).
    * ``line`` — Court white-line segmentation (BCE + Dice, 1 channel).

    Inherits optimizer/scheduler logic from
    :class:`~src.tasks.base.training.lightning_module.BaseLightningModule`.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)
        self.save_hyperparameters()

        data_cfg = self.config.get("data", {})
        loss_cfg = self.config.get("loss", {})
        self.task = str(data_cfg.get("task", "seg"))

        # Model
        self.model = build_court_detection_model(self.config)

        # Task-specific loss
        if self.task == "seg":
            seg_cfg = loss_cfg.get("seg", {})
            num_classes = int(data_cfg.get("num_classes", 7))
            self.ce_weight = float(seg_cfg.get("ce_weight", 1.0))
            self.dice_weight = float(seg_cfg.get("dice_weight", 1.0))
            self.ce_loss_fn = nn.CrossEntropyLoss()
            self.dice_loss_fn = DiceLoss(num_classes=num_classes)
        elif self.task == "kp":
            kp_cfg = loss_cfg.get("kp", {})
            self.loss_fn = FocalBCEWithLogitsLoss(
                gamma=float(kp_cfg.get("focal_gamma", 2.0)),
            )
        elif self.task == "line":
            line_cfg = loss_cfg.get("line", {})
            self.bce_weight = float(line_cfg.get("bce_weight", 1.0))
            self.dice_weight = float(line_cfg.get("dice_weight", 1.0))
            pos_weight = torch.tensor([float(line_cfg.get("pos_weight", 8.0))])
            self.bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.dice_loss_fn = BinaryDiceLoss()

        # Metrics
        self.train_metrics = CourtDetectionMetrics(self.task, data_cfg)
        self.val_metrics = CourtDetectionMetrics(self.task, data_cfg)

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def _compute_loss(self, logits: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Compute task-specific loss."""
        if self.task == "seg":
            masks = batch["mask"]
            loss_ce = self.ce_loss_fn(logits, masks)
            loss_dice = self.dice_loss_fn(logits, masks)
            return self.ce_weight * loss_ce + self.dice_weight * loss_dice
        elif self.task == "kp":
            heatmaps = batch["heatmap"]
            return self.loss_fn(logits, heatmaps)
        else:  # line
            masks = batch["mask"]
            loss_bce = self.bce_loss_fn(logits, masks)
            loss_dice = self.dice_loss_fn(logits, masks)
            return self.bce_weight * loss_bce + self.dice_weight * loss_dice

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str,
    ) -> dict[str, Tensor]:
        """Shared computation for train/val steps."""
        images = batch["image"]
        logits = self.model(images)
        loss = self._compute_loss(logits, batch)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)
        return {"loss": loss, "logits": logits}

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        outputs = self._shared_step(batch, "train")
        self.train_metrics.update(outputs["logits"], batch)
        return outputs["loss"]

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        outputs = self._shared_step(batch, "val")
        self.val_metrics.update(outputs["logits"], batch)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=(name in ("miou", "mean_dist", "dice")))
        self.val_metrics.reset()

    # ------------------------------------------------------------------
    # Qualitative validation logging
    # ------------------------------------------------------------------

    def render_qualitative_samples(
        self,
        batches: list[dict[str, Any]],
        outputs: list[dict[str, Any]],
        artifact_dir: Path,
        tb_writer: Any | None,
        global_step: int,
        epoch: int,
    ) -> None:
        """Render court detection panels using existing visualization helpers."""
        import cv2

        device = next(self.parameters()).device

        for batch_idx, batch in enumerate(batches):
            images = batch["image"].to(device)

            with torch.no_grad():
                logits = self.model(images).cpu()  # (B, C, H, W)

            # Render first sample in each batch
            img_tensor = batch["image"][0]  # (3, H, W)
            pred_logits_sample = logits[0]  # (C, H, W)

            path = artifact_dir / f"court_batch{batch_idx:02d}.png"

            if self.task == "seg":
                gt = batch["mask"][0]  # (H, W) long
                save_seg_vis(img_tensor, gt, pred_logits_sample, path)
            elif self.task == "kp":
                gt = batch["heatmap"][0]  # (K, H, W)
                save_kp_vis(img_tensor, gt, pred_logits_sample, path)
            elif self.task == "line":
                gt = batch["mask"][0]  # (1, H, W)
                save_line_vis(img_tensor, gt, pred_logits_sample, path)

            # Log to TensorBoard
            panel = cv2.imread(str(path))
            if panel is not None:
                save_image_to_tensorboard(
                    tb_writer,
                    f"qualitative/court_detection/{self.task}/batch{batch_idx:02d}",
                    panel,
                    global_step,
                )
