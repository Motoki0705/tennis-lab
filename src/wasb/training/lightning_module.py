"""PyTorch Lightning module for WASB tennis training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.wasb.training.losses import LossWeights, WASBLoss
from src.wasb.training.metrics import WASBMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class WASBLightningModule(pl.LightningModule):
    """Lightning module wrapping a WASB-style ball localization model."""

    def __init__(
        self,
        config: DictConfig | dict | None = None,
        model: nn.Module | None = None,
        steps_per_epoch: int | None = None,
        io_handlers: tuple[Callable[[Tensor], Tensor], Callable[[Any], Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        if model is None:
            raise ValueError("model must be provided to WASBLightningModule")

        self.config = config or {}
        self.model = model

        if io_handlers is None:
            raise ValueError(
                "io_handlers (prepare_frames, extract_heatmaps) must be provided to WASBLightningModule"
            )
        self.prepare_frames, self.extract_heatmaps = io_handlers

        train_cfg = self.config.get("training", {})
        heatmap_weight = train_cfg.get(
            "heatmap_loss_weight", train_cfg.get("coord_loss_weight", 1.0)
        )
        loss_weights = LossWeights(
            heatmap=heatmap_weight,
        )
        self.loss_fn = WASBLoss(weights=loss_weights)

        metrics_cfg = self.config.get("metrics", {})
        acc_thresh = metrics_cfg.get("accuracy_thresh_px", 5.0)
        self.train_metrics = WASBMetrics(accuracy_thresh_px=acc_thresh)
        self.val_metrics = WASBMetrics(accuracy_thresh_px=acc_thresh)
        self.test_metrics = WASBMetrics(accuracy_thresh_px=acc_thresh)

        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.max_epochs = train_cfg.get("max_epochs", 50)
        self.min_lr = train_cfg.get("min_lr", 1e-6)
        self.steps_per_epoch = steps_per_epoch

    def forward(self, frames: Tensor) -> dict[str, Tensor] | Tensor:
        """Forward pass delegating to the underlying model."""
        return self.model(frames)

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        frames: Tensor = batch["frames"]
        frames_input = self.prepare_frames(frames)

        outputs = self(frames_input)
        pred_heatmaps = self.extract_heatmaps(outputs)

        target_heatmaps: Tensor = batch["target_heatmaps"].to(pred_heatmaps.device)
        visibility: Tensor | None = batch.get("visibility")

        if pred_heatmaps.shape != target_heatmaps.shape:
            raise ValueError(
                f"Prediction shape {tuple(pred_heatmaps.shape)} "
                f"does not match target heatmaps {tuple(target_heatmaps.shape)}"
            )

        losses = self.loss_fn(
            pred_heatmaps=pred_heatmaps,
            target_heatmaps=target_heatmaps,
            visibility=visibility,
        )

        h, w = frames.shape[-2:]
        metrics = self._metrics_for_stage(stage).update(
            pred_heatmaps=pred_heatmaps,
            target_coords_norm=batch["targets_norm"],
            visibility=visibility,
            image_hw=(h, w),
        )

        return losses["total"], {**metrics, **{f"loss_{k}": v.item() for k, v in losses.items()}}

    def _metrics_for_stage(self, stage: str) -> WASBMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/rmse_px", metrics["rmse_px"], prog_bar=True)
        self.log("train/accuracy", metrics["accuracy"], prog_bar=True)
        self.log("train/pred_min", metrics["pred_min"], prog_bar=True)
        self.log("train/pred_max", metrics["pred_max"], prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset(self.device)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse_px", metrics["rmse_px"], prog_bar=True)
        self.log("val/accuracy", metrics["accuracy"], prog_bar=True)
        self.log("val/pred_min", metrics["pred_min"], prog_bar=True)
        self.log("val/pred_max", metrics["pred_max"], prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset(self.device)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/rmse_px", metrics["rmse_px"])
        self.log("test/accuracy", metrics["accuracy"])
        self.log("test/pred_min", metrics["pred_min"], prog_bar=True)
        self.log("test/pred_max", metrics["pred_max"], prog_bar=True)

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset(self.device)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = 1000

        total_steps = steps_per_epoch * max(self.max_epochs, 1)
        if total_steps <= self.warmup_steps + 1:
            return {"optimizer": optimizer}

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - self.warmup_steps, 1),
            eta_min=self.min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
