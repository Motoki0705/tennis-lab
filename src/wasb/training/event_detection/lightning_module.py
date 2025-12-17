"""Lightning module for trajectory event detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.wasb.models.event_detection import TrajectoryEventTransformer
from src.wasb.training.event_detection.loss import (
    EventDetectionLossConfig,
    event_detection_loss,
)
from src.wasb.training.event_detection.metrics import event_metrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class EventDetectionLightningModule(pl.LightningModule):
    """Lightning wrapper for per-frame event detection from trajectories."""

    def __init__(
        self,
        config: DictConfig | dict | None = None,
        *,
        steps_per_epoch: int | None = None,
        class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["steps_per_epoch", "class_weights"])
        self.config = config or {}
        self.steps_per_epoch = steps_per_epoch

        train_cfg = self.config.get("training", {})
        model_cfg = self.config.get("model", {})

        self.learning_rate = float(train_cfg.get("learning_rate", 1e-3))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.warmup_steps = int(train_cfg.get("warmup_steps", 1000))
        self.max_epochs = int(train_cfg.get("max_epochs", 50))
        self.min_lr = float(train_cfg.get("min_lr", 1e-6))

        loss_cfg = self.config.get("loss", {})
        self.loss_cfg = EventDetectionLossConfig(
            ignore_index=int(loss_cfg.get("ignore_index", train_cfg.get("ignore_index", -100))),
            label_smoothing=float(loss_cfg.get("label_smoothing", train_cfg.get("label_smoothing", 0.0))),
            event_boost=float(loss_cfg.get("event_boost", train_cfg.get("event_boost", 1.0))),
            background_weight_scale=float(
                loss_cfg.get(
                    "background_weight_scale",
                    train_cfg.get("background_weight_scale", 1.0),
                )
            ),
        )

        cfg_class_weights = train_cfg.get("class_weights")
        cfg_class_weights = loss_cfg.get("class_weights", cfg_class_weights)
        if class_weights is None and cfg_class_weights is not None:
            class_weights = torch.tensor(list(cfg_class_weights), dtype=torch.float32)
        if class_weights is not None:
            if class_weights.numel() != 3:
                raise ValueError("class_weights must have 3 elements for classes [0, 1, 2]")
        self.register_buffer("_class_weights", class_weights, persistent=False)

        self.model = TrajectoryEventTransformer(
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 4)),
            num_heads=int(model_cfg.get("num_heads", 4)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", 256)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 128)),
            num_classes=int(model_cfg.get("num_classes", 3)),
            max_len=int(model_cfg.get("max_len", 512)),
            positional_encoding=str(model_cfg.get("positional_encoding", "sin")),
        )

    def forward(self, xy_norm: Tensor, *, key_padding_mask: Tensor | None = None) -> Tensor:
        """Perform a forward pass through the transformer."""
        return self.model(xy_norm, key_padding_mask=key_padding_mask)

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> tuple[Tensor, dict[str, Tensor]]:
        xy_norm: Tensor = batch["xy_norm"].to(self.device)
        visibility: Tensor = batch["visibility"].to(self.device)
        target_status: Tensor = batch["target_status"].to(self.device)

        key_padding_mask = visibility <= 0
        logits = self(xy_norm, key_padding_mask=key_padding_mask)
        loss = event_detection_loss(
            logits=logits,
            target=target_status,
            cfg=self.loss_cfg,
            class_weights=self._class_weights,
        )

        pred = logits.argmax(dim=-1)
        metrics = event_metrics(
            pred=pred,
            target=target_status,
            ignore_index=int(self.loss_cfg.ignore_index),
        )
        metrics["loss"] = loss.detach()
        return loss, metrics

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Run a single training step and log metrics."""
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", metrics["acc"], prog_bar=True)
        self.log("train/event_f1", metrics["event_f1"], prog_bar=True)
        self.log("train/shot_f1", metrics["shot_f1"])
        self.log("train/bounce_f1", metrics["bounce_f1"])
        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Run a single validation step and log metrics."""
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", metrics["acc"], prog_bar=True)
        self.log("val/event_f1", metrics["event_f1"], prog_bar=True)
        self.log("val/shot_f1", metrics["shot_f1"])
        self.log("val/bounce_f1", metrics["bounce_f1"])

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Run a single test step and log metrics."""
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/acc", metrics["acc"])
        self.log("test/event_f1", metrics["event_f1"])
        self.log("test/shot_f1", metrics["shot_f1"])
        self.log("test/bounce_f1", metrics["bounce_f1"])

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        steps_per_epoch = self.steps_per_epoch or 1000
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
