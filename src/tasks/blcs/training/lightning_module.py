"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.blcs.data.types import BLCSBatch, BLCSMultiViewBatch
from src.blcs.models import build_blcs_model
from src.blcs.training.losses import BLCSLoss
from src.blcs.training.metrics import BLCSMetrics


if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSLightningModule(BaseLightningModule):
    """Lightning module for BLCS models.

    This module supports both single-view and multiview BLCS training.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        self.model = build_blcs_model(self.config)

        train_cfg = self.config.get("training", {})
        self.loss_fn = BLCSLoss(
            position_weight=train_cfg.get("position_loss_weight", 1.0),
            velocity_weight=train_cfg.get("velocity_loss_weight", 0.1),
            smoothness_weight=train_cfg.get("smoothness_loss_weight", 0.05),
        )

        metrics_cfg = self.config.get("metrics", {})
        self.train_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.get("position_threshold_m", 0.3),
            endpoint_threshold_m=metrics_cfg.get("endpoint_threshold_m", 0.5),
        )
        self.val_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.get("position_threshold_m", 0.3),
            endpoint_threshold_m=metrics_cfg.get("endpoint_threshold_m", 0.5),
        )
        self.test_metrics = BLCSMetrics(
            position_threshold_m=metrics_cfg.get("position_threshold_m", 0.3),
            endpoint_threshold_m=metrics_cfg.get("endpoint_threshold_m", 0.5),
        )

    def _forward_from_batch(self, batch: BLCSBatch | BLCSMultiViewBatch) -> dict[str, Tensor]:
        """Forward model from a batch."""
        return self.model(
            ball_uv=batch["ball_uv"],
            court_kp=batch["court_kp"],
            ball_vis=batch.get("ball_vis"),
            ball_mask=batch.get("ball_mask"),
            court_vis=batch.get("court_vis"),
        )

    def _normalize_loss_mask(self, batch: BLCSBatch | BLCSMultiViewBatch) -> Tensor | None:
        """Normalize loss/metric mask to shape (B, T)."""
        ball_mask = batch.get("ball_mask")
        if ball_mask is None:
            return None
        if ball_mask.ndim == 2:
            return ball_mask
        if ball_mask.ndim == 3:
            return (ball_mask > 0).any(dim=1)
        raise ValueError(
            f"ball_mask must have 2 or 3 dims, got shape {tuple(ball_mask.shape)}"
        )

    def _select_metrics(self, stage: str) -> BLCSMetrics:
        """Return metrics object for the current stage."""
        if stage == "train":
            return self.train_metrics
        if stage == "val":
            return self.val_metrics
        return self.test_metrics

    def _shared_step(
        self, batch: BLCSBatch | BLCSMultiViewBatch, stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        """Shared step for training, validation, and test."""
        outputs = self._forward_from_batch(batch)
        mask = self._normalize_loss_mask(batch)

        losses = self.loss_fn(
            pred_position=outputs["position"],
            target_position=batch["position_3d"],
            mask=mask,
        )

        metrics = self._select_metrics(stage).update(
            outputs["position"],
            batch["position_3d"],
            mask,
        )

        return losses["total"], {
            **metrics,
            **{f"loss_{k}": v.item() for k, v in losses.items()},
        }

    def training_step(self, batch: BLCSBatch | BLCSMultiViewBatch, batch_idx: int) -> Tensor:
        """Training step."""
        loss, metrics = self._shared_step(batch, "train")
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_error_m", metrics.get("position_error_m", 0), prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch."""
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: BLCSBatch | BLCSMultiViewBatch, batch_idx: int) -> None:
        """Validation step."""
        loss, metrics = self._shared_step(batch, "val")
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos_error_m", metrics.get("position_error_m", 0), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Called at end of validation epoch."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: BLCSBatch | BLCSMultiViewBatch, batch_idx: int) -> None:
        """Test step."""
        loss, metrics = self._shared_step(batch, "test")
        self.log("test/loss", loss)
        self.log("test/pos_error_m", metrics.get("position_error_m", 0))

    def on_test_epoch_end(self) -> None:
        """Called at end of test epoch."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()
