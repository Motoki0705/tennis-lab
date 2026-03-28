"""Unified PyTorch Lightning module for PLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor, nn

from src.tasks.base.training.lightning_module import BaseLightningModule
from src.tasks.plcs.models import build_plcs_model
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.metrics import PLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSLightningModule(BaseLightningModule):
    """Lightning module for unified PLCS I/O training."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)

        self.model: nn.Module = build_plcs_model(self.config)

        loss_cfg = PLCSLossConfig.from_dict(dict(self.config.loss))
        self.loss_fn = PLCSLoss(config=loss_cfg)

        metrics_cfg = self.config.metrics
        self.train_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.position_threshold_m),
            angle_threshold_deg=float(metrics_cfg.angle_threshold_deg),
        )
        self.val_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.position_threshold_m),
            angle_threshold_deg=float(metrics_cfg.angle_threshold_deg),
        )
        self.test_metrics = PLCSMetrics(
            position_threshold_m=float(metrics_cfg.position_threshold_m),
            angle_threshold_deg=float(metrics_cfg.angle_threshold_deg),
        )

    def _forward_from_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(
            human_kp=batch["human_kp"],
            court_kp=batch["court_kp"],
            human_vis=batch.get("human_vis"),
            human_mask=batch.get("human_mask"),
            court_vis=batch.get("court_vis"),
        )

    def _select_metrics(self, stage: str) -> PLCSMetrics:
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        outputs = self._forward_from_batch(batch)
        human_mask = batch.get("human_mask")

        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            human_mask=human_mask,
        )

        metrics = self._select_metrics(stage).update(
            outputs["position"],
            outputs["rotation"],
            batch["position"],
            batch["rotation"],
            human_mask=human_mask,
        )

        return losses["total"], {
            **metrics,
            **{f"loss_{k}": float(v.item()) for k, v in losses.items()},
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_error_m", metrics["position_error_m"], prog_bar=True)
        self.log("train/ang_error_deg", metrics["angular_error_deg"], prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos_error_m", metrics["position_error_m"], prog_bar=True)
        self.log("val/ang_error_deg", metrics["angular_error_deg"], prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/pos_error_m", metrics["position_error_m"])
        self.log("test/ang_error_deg", metrics["angular_error_deg"])

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()
