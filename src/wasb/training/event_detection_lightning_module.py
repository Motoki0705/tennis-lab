"""PyTorch Lightning module for trajectory event detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.wasb.models.event_detection import TrajectoryEventTransformer

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _safe_div(num: Tensor, denom: Tensor) -> Tensor:
    return num / (denom.clamp(min=1.0))


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

        self.ignore_index = int(train_cfg.get("ignore_index", -100))
        self.label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
        self.event_boost = float(train_cfg.get("event_boost", 1.0))
        self.background_weight_scale = float(train_cfg.get("background_weight_scale", 1.0))

        cfg_class_weights = train_cfg.get("class_weights")
        if class_weights is None and cfg_class_weights is not None:
            class_weights = torch.tensor(list(cfg_class_weights), dtype=torch.float32)
        if class_weights is not None:
            if class_weights.numel() != 3:
                raise ValueError("class_weights must have 3 elements for classes [0, 1, 2]")
            if self.background_weight_scale != 1.0:
                class_weights = class_weights.clone()
                class_weights[0] = class_weights[0] * float(self.background_weight_scale)
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

    def _loss(self, logits: Tensor, target: Tensor) -> Tensor:
        b, t, c = logits.shape
        logits_flat = logits.reshape(b * t, c)
        target_flat = target.reshape(b * t)
        valid = target_flat != self.ignore_index

        loss_flat = F.cross_entropy(
            logits_flat,
            target_flat,
            weight=self._class_weights,
            ignore_index=self.ignore_index,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.event_boost != 1.0:
            is_event = (target_flat == 1) | (target_flat == 2)
            boost = torch.ones_like(loss_flat)
            boost[is_event] = float(self.event_boost)
            loss_flat = loss_flat * boost

        if not valid.any():
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        return (loss_flat[valid]).mean()

    @staticmethod
    def _event_metrics(pred: Tensor, target: Tensor, ignore_index: int) -> dict[str, Tensor]:
        valid = target != ignore_index
        if not valid.any():
            z = torch.zeros((), device=target.device, dtype=torch.float32)
            return {
                "acc": z,
                "event_f1": z,
                "shot_f1": z,
                "bounce_f1": z,
                "event_recall": z,
                "event_precision": z,
            }

        pred = pred[valid]
        target = target[valid]
        event_mask = (target == 1) | (target == 2)
        if event_mask.any():
            acc = (pred[event_mask] == target[event_mask]).to(torch.float32).mean()
        else:
            acc = torch.zeros((), device=target.device, dtype=torch.float32)

        pred_event = pred > 0
        target_event = target > 0
        tp = (pred_event & target_event).sum().to(torch.float32)
        fp = (pred_event & ~target_event).sum().to(torch.float32)
        fn = (~pred_event & target_event).sum().to(torch.float32)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        event_f1 = _safe_div(2 * precision * recall, precision + recall)

        def f1_for_class(cls: int) -> Tensor:
            pred_c = pred == cls
            tgt_c = target == cls
            tp_c = (pred_c & tgt_c).sum().to(torch.float32)
            fp_c = (pred_c & ~tgt_c).sum().to(torch.float32)
            fn_c = (~pred_c & tgt_c).sum().to(torch.float32)
            p_c = _safe_div(tp_c, tp_c + fp_c)
            r_c = _safe_div(tp_c, tp_c + fn_c)
            return _safe_div(2 * p_c * r_c, p_c + r_c)

        shot_f1 = f1_for_class(1)
        bounce_f1 = f1_for_class(2)

        return {
            "acc": acc,
            "event_f1": event_f1,
            "shot_f1": shot_f1,
            "bounce_f1": bounce_f1,
            "event_recall": recall,
            "event_precision": precision,
        }

    def _shared_step(self, batch: dict[str, Tensor], stage: str) -> tuple[Tensor, dict[str, Tensor]]:
        xy_norm: Tensor = batch["xy_norm"].to(self.device)
        visibility: Tensor = batch["visibility"].to(self.device)
        target_status: Tensor = batch["target_status"].to(self.device)

        key_padding_mask = visibility <= 0
        logits = self(xy_norm, key_padding_mask=key_padding_mask)
        loss = self._loss(logits, target_status)

        pred = logits.argmax(dim=-1)
        metrics = self._event_metrics(pred, target_status, self.ignore_index)
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
