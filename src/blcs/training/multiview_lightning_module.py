"""PyTorch Lightning module for multi-view BLCS training."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.blcs.models.blcs_multiview_model import BLCSMultiViewModel
from src.blcs.training.losses import BLCSLoss
from src.blcs.training.metrics import BLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BLCSMultiViewLightningModule(BaseLightningModule):
    """Lightning module for training multi-view BLCS models.

    Similar to BLCSLightningModule but handles multi-view inputs
    where ball trajectory observations come from multiple cameras.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        # Build multi-view model
        self.model = BLCSMultiViewModel.from_config(self.config)

        # Loss function (same as single-view)
        train_cfg = self.config.get("training", {})
        self.loss_fn = BLCSLoss(
            position_weight=train_cfg.get("position_loss_weight", 1.0),
            velocity_weight=train_cfg.get("velocity_loss_weight", 0.1),
            smoothness_weight=train_cfg.get("smoothness_loss_weight", 0.05),
        )

        # Metrics
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

    def forward(
        self,
        ball_uv: Tensor,
        court_kp: Tensor,
        ball_mask: Tensor | None = None,
        court_vis: Tensor | None = None,
        num_views: Tensor | None = None,
        seq_len: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            ball_uv: Ball 2D trajectories from multiple views, shape (B, N, T, 2).
            court_kp: Court keypoints from multiple views, shape (B, N, 20, 2).
            ball_mask: Ball visibility masks, shape (B, N, T).
            court_vis: Court visibility masks, shape (B, N, 20).
            num_views: Number of valid views per sample, shape (B,).
            seq_len: Sequence lengths, shape (B,).
            camera_params: Camera parameters per view.

        Returns:
            dict: Model outputs.

        """
        return typing.cast(
            dict[str, Any],
            self.model(
                ball_uv,
                court_kp,
                ball_mask,
                court_vis,
                num_views,
                seq_len,
                camera_params,
            ),
        )

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        """Shared step for training and validation.

        Args:
            batch: Batch dictionary from dataloader.
            stage: One of 'train', 'val', 'test'.

        Returns:
            tuple: (loss tensor, metrics dict).

        """
        # Forward pass with multi-view inputs
        # Tensor format: (B, N, T, ...) for alternating attention architecture
        outputs: dict[str, Any] = self.model(
            ball_uv=batch["ball_uv"],  # (B, N, T, 2)
            court_kp=batch["court_kp"],  # (B, N, T, 20, 2)
            ball_mask=batch.get("ball_mask"),  # (B, N, T)
            court_vis=batch.get("court_vis"),  # (B, N, T, 20)
            num_views=batch.get("num_views"),
            seq_len=batch.get("seq_len"),
            camera_params=batch.get("camera_params"),
        )

        # Create sequence mask for loss computation
        seq_len = batch.get("seq_len")
        if seq_len is not None:
            max_len = batch["position_3d"].shape[1]
            seq_mask = torch.arange(max_len, device=seq_len.device).unsqueeze(
                0
            ) < seq_len.unsqueeze(1)
        else:
            seq_mask = None

        # Compute loss
        losses = self.loss_fn(
            pred_position=outputs["position"],
            target_position=batch["position_3d"],
            mask=seq_mask,
        )

        # Update metrics
        if stage == "train":
            metrics = self.train_metrics.update(
                outputs["position"],
                batch["position_3d"],
                seq_mask,
            )
        elif stage == "val":
            metrics = self.val_metrics.update(
                outputs["position"],
                batch["position_3d"],
                seq_mask,
            )
        else:
            metrics = self.test_metrics.update(
                outputs["position"],
                batch["position_3d"],
                seq_mask,
            )

        return losses["total"], {
            **metrics,
            **{f"loss_{k}": v.item() for k, v in losses.items()},
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step.

        Args:
            batch: Batch from dataloader.
            batch_idx: Batch index.

        Returns:
            Tensor: Training loss.

        """
        loss, metrics = self._shared_step(batch, "train")

        # Log metrics
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/pos_error_m", metrics.get("position_error_m", 0), prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Called at end of training epoch."""
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/epoch_{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Batch from dataloader.
            batch_idx: Batch index.

        """
        loss, metrics = self._shared_step(batch, "val")

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/pos_error_m", metrics.get("position_error_m", 0), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Called at end of validation epoch."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/epoch_{name}", value)
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Test step.

        Args:
            batch: Batch from dataloader.
            batch_idx: Batch index.

        """
        loss, metrics = self._shared_step(batch, "test")

        self.log("test/loss", loss)
        self.log("test/pos_error_m", metrics.get("position_error_m", 0))

    def on_test_epoch_end(self) -> None:
        """Called at end of test epoch."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()

    # configure_optimizers inherited from BaseLightningModule
