"""PyTorch Lightning module for BLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        self.model_name = str(self.config.data.output_mode)

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

    def _forward_single(
        self,
        *,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_vis: Tensor | None,
        ball_mask: Tensor | None,
        court_vis: Tensor | None,
    ) -> dict[str, Tensor]:
        """Forward pass for single-view model."""
        return self.model(
            ball_uv,
            court_kp,
            ball_vis=ball_vis,
            ball_mask=ball_mask,
            court_vis=court_vis,
        )

    def _forward_multiview(
        self,
        *,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None,
        court_vis: Tensor | None,
        num_views: Tensor | None,
        seq_len: Tensor | None,
        camera_params: list[dict[str, Any]] | None,
    ) -> dict[str, Tensor]:
        """Forward pass for multiview model."""
        return self.model(
            ball_uv=ball_uv,
            court_kp=court_kp,
            ball_mask=ball_mask,
            court_vis=court_vis,
            num_views=num_views,
            seq_len=seq_len,
            camera_params=camera_params,
        )

    def _forward_from_batch(self, batch: BLCSBatch | BLCSMultiViewBatch) -> dict[str, Tensor]:
        """Forward model using a training batch."""
        if self.model_name == "multiview":
            mv_batch = batch
            return self._forward_multiview(
                ball_uv=mv_batch["ball_uv"],
                court_kp=mv_batch["court_kp"],
                ball_mask=mv_batch.get("ball_mask"),
                court_vis=mv_batch.get("court_vis"),
                num_views=mv_batch.get("num_views"),
                seq_len=mv_batch.get("seq_len"),
                camera_params=mv_batch.get("camera_params"),
            )
        elif self.model_name == "single":
            single_batch = batch
            return self._forward_single(
                ball_uv=single_batch["ball_uv"],
                court_kp=single_batch["court_kp"],
                ball_vis=single_batch.get("ball_vis"),
                ball_mask=single_batch.get("ball_mask"),
                court_vis=single_batch.get("court_vis"),
            )
        else:
            raise ValueError(
                f"Unsupported model_name='{self.model_name}'. "
                "Supported: ['single', 'multiview']"
            )

    def _build_loss_mask(self, batch: BLCSBatch | BLCSMultiViewBatch) -> Tensor | None:
        """Build sequence mask used for loss and metrics."""
        if self.model_name == "multiview":
            seq_len = batch.get("seq_len")
            if seq_len is None:
                return None
            max_len = batch["position_3d"].shape[1]
            return torch.arange(max_len, device=seq_len.device).unsqueeze(0) < seq_len.unsqueeze(1)
        elif self.model_name == "single":
            return batch.get("ball_mask")
        else:
            raise ValueError(
                f"Unsupported model_name='{self.model_name}'. "
                "Supported: ['single', 'multiview']"
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
        mask = self._build_loss_mask(batch)

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
