"""PyTorch Lightning module for WASB ball detection training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.wasb.models import build_model
from src.wasb.training.ball_detection.loss import LossWeights, WASBLoss
from src.wasb.training.ball_detection.metrics import WASBMetrics

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
        self.save_hyperparameters(ignore=["model", "io_handlers"])

        # When training, the caller usually provides both `model` and
        # `io_handlers`. When loading from a checkpoint via
        # `load_from_checkpoint`, these may be omitted and we need to
        # reconstruct them from the saved config instead.

        self.config = config or {}

        if (model is None or io_handlers is None) and config is not None:
            built_model, built_handlers = build_model(self.config)
            if model is None:
                model = built_model
            if io_handlers is None:
                io_handlers = built_handlers

        if model is None or io_handlers is None:
            raise ValueError(
                "model and io_handlers (prepare_frames, extract_heatmaps) must be provided to WASBLightningModule, "
                "either directly or via a valid config."
            )

        self.model = model
        self.prepare_frames, self.extract_heatmaps = io_handlers

        train_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("loss", {})
        bce_weight = loss_cfg.get("bce_weight", train_cfg.get("bce_weight", 1.0))
        mse_weight = loss_cfg.get("mse_weight", train_cfg.get("mse_weight", 1.0))
        loss_weights = LossWeights(
            bce=bce_weight,
            mse=mse_weight,
        )
        self.loss_fn = WASBLoss(weights=loss_weights)

        self.use_metrics = bool(train_cfg.get("use_metrics", True))
        if self.use_metrics:
            metrics_cfg = self.config.get("metrics", {})
            acc_thresh = metrics_cfg.get("accuracy_thresh_px", 5.0)
            self.train_metrics = WASBMetrics(accuracy_thresh_px=acc_thresh)
            self.val_metrics = WASBMetrics(accuracy_thresh_px=acc_thresh)
            self.test_metrics = WASBMetrics(accuracy_thresh_px=acc_thresh)
        else:
            self.train_metrics = None
            self.val_metrics = None
            self.test_metrics = None

        self.learning_rate = train_cfg.get("learning_rate", 1e-3)
        self.backbone_learning_rate = train_cfg.get("backbone_learning_rate", 1e-5)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.max_epochs = train_cfg.get("max_epochs", 50)
        self.min_lr = train_cfg.get("min_lr", 1e-6)
        self.steps_per_epoch = steps_per_epoch
        self.freeze_backbone_epochs = int(train_cfg.get("freeze_backbone_epochs", 0) or 0)
        self._backbone_frozen = False

    def forward(self, frames: Tensor) -> dict[str, Tensor] | Tensor:
        """Forward pass delegating to the underlying model."""
        return self.model(frames)

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        input_key = self.config.get("training", {}).get("input_key", "frames")
        frames: Tensor = batch[input_key]
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

        metrics: dict[str, float] = {}
        if self.use_metrics:
            if frames.dim() >= 4 and frames.shape[-3] == 3:
                h, w = frames.shape[-2:]
            else:
                h, w = target_heatmaps.shape[-2:]
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
        if self.use_metrics:
            self.log("train/rmse_px", metrics["rmse_px"], prog_bar=True)
            self.log("train/accuracy", metrics["accuracy"], prog_bar=True)
            self.log("train/pred_min", metrics["pred_min"], prog_bar=True)
            self.log("train/pred_max", metrics["pred_max"], prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if not self.use_metrics or self.train_metrics is None:
            return
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset(self.device)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        if self.use_metrics:
            self.log("val/rmse_px", metrics["rmse_px"], prog_bar=True)
            self.log("val/accuracy", metrics["accuracy"], prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        if not self.use_metrics or self.val_metrics is None:
            return
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset(self.device)

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        if self.use_metrics:
            self.log("test/rmse_px", metrics["rmse_px"])
            self.log("test/accuracy", metrics["accuracy"])

    def on_test_epoch_end(self) -> None:
        if not self.use_metrics or self.test_metrics is None:
            return
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset(self.device)

    def on_fit_start(self) -> None:
        if (
            self.freeze_backbone_epochs > 0
            and hasattr(self.model, "freeze_backbone")
            and not self._backbone_frozen
        ):
            self.model.freeze_backbone()
            self._backbone_frozen = True
            print("Backbone Frozen")
    def on_train_epoch_start(self) -> None:
        if (
            self.freeze_backbone_epochs > 0
            and self._backbone_frozen
            and self.current_epoch >= self.freeze_backbone_epochs
            and hasattr(self.model, "unfreeze_backbone")
        ):
            self.model.unfreeze_backbone()
            self._backbone_frozen = False
            print("Backbone Unfrozen")

    def _build_warmup_cosine_lambda(
        self,
        *,
        total_steps: int,
        warmup_steps: int,
        start_step: int,
        base_lr: float,
    ) -> Callable[[int], float]:
        start_factor = 0.01
        active_steps = max(total_steps - start_step, 1)
        adjusted_warmup = max(min(warmup_steps, active_steps - 1), 0)
        cosine_steps = max(active_steps - adjusted_warmup, 1)
        min_factor = 0.0 if base_lr == 0 else min(self.min_lr / base_lr, 1.0)

        def lr_lambda(current_step: int) -> float:
            if current_step < start_step:
                return 0.0

            step = current_step - start_step

            if adjusted_warmup > 0 and step < adjusted_warmup:
                progress = step / float(adjusted_warmup)
                return start_factor + (1.0 - start_factor) * progress

            cosine_step = min(step - adjusted_warmup, cosine_steps)
            cosine_progress = cosine_step / float(cosine_steps)
            return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * cosine_progress))

        return lr_lambda

    def configure_optimizers(self) -> dict[str, Any]:
        backbone_params: list[nn.Parameter] = []
        if hasattr(self.model, "backbone") and isinstance(self.model.backbone, nn.Module):
            backbone_params = list(self.model.backbone.parameters())
        backbone_param_ids = {id(p) for p in backbone_params}
        non_backbone_params = [p for p in self.model.parameters() if id(p) not in backbone_param_ids]

        param_groups: list[dict[str, Any]] = []
        lr_lambdas: list[Callable[[int], float]] = []

        if non_backbone_params:
            param_groups.append(
                {"params": non_backbone_params, "lr": self.learning_rate, "weight_decay": self.weight_decay}
            )
        if backbone_params:
            param_groups.append(
                {
                    "params": backbone_params,
                    "lr": self.backbone_learning_rate,
                    "weight_decay": self.weight_decay,
                }
            )

        optimizer = AdamW(
            param_groups if param_groups else self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        steps_per_epoch = self.steps_per_epoch
        if steps_per_epoch is None:
            steps_per_epoch = 1000

        total_steps = steps_per_epoch * max(self.max_epochs, 1)
        freeze_steps = steps_per_epoch * max(self.freeze_backbone_epochs, 0)

        if non_backbone_params:
            lr_lambdas.append(
                self._build_warmup_cosine_lambda(
                    total_steps=total_steps,
                    warmup_steps=self.warmup_steps,
                    start_step=0,
                    base_lr=self.learning_rate,
                )
            )
        if backbone_params:
            lr_lambdas.append(
                self._build_warmup_cosine_lambda(
                    total_steps=total_steps,
                    warmup_steps=self.warmup_steps,
                    start_step=freeze_steps,
                    base_lr=self.backbone_learning_rate,
                )
            )

        if not lr_lambdas:
            return {"optimizer": optimizer}

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas if len(lr_lambdas) > 1 else lr_lambdas[0])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
