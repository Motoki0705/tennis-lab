"""PyTorch Lightning module for multi-view PLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.training.losses import PLCSLoss
from src.plcs.training.metrics import PLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSMultiViewLightningModule(pl.LightningModule):
    """Lightning module for training multi-view PLCS models.

    Similar to PLCSLightningModule but handles multi-view inputs
    where observations come from multiple cameras simultaneously.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config or {}

        # Build multi-view model
        self.model = PLCSMultiViewModel.from_config(self.config)

        # Loss function (same as single-view)
        train_cfg = self.config.get("training", {})
        self.loss_fn = PLCSLoss(
            position_weight=train_cfg.get("position_loss_weight", 1.0),
            rotation_weight=train_cfg.get("rotation_loss_weight", 1.0),
        )

        # Metrics
        self.train_metrics = PLCSMetrics()
        self.val_metrics = PLCSMetrics()
        self.test_metrics = PLCSMetrics()

        # Training parameters
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.min_lr = train_cfg.get("min_lr", 1e-6)

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
        num_views: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Human keypoints from multiple views, shape (B, N, 17, 2).
            court_kp: Court keypoints from multiple views, shape (B, N, 20, 2).
            human_vis: Human visibility mask, shape (B, N, 17).
            court_vis: Court visibility mask, shape (B, N, 20).
            num_views: Number of valid views per sample, shape (B,).
            camera_params: Camera parameters per view.

        Returns:
            dict: Model outputs.

        """
        return self.model(
            human_kp, court_kp, human_vis, court_vis, num_views, camera_params
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
        outputs = self.model(
            human_kp=batch["human_kp"],  # (B, N, 17, 2)
            court_kp=batch["court_kp"],  # (B, N, 20, 2)
            human_vis=batch.get("human_vis"),
            court_vis=batch.get("court_vis"),
            num_views=batch.get("num_views"),
            camera_params=batch.get("camera_params"),
        )

        # Compute loss
        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
        )

        # Update metrics
        if stage == "train":
            metrics = self.train_metrics.update(
                outputs["position"],
                outputs["rotation"],
                batch["position"],
                batch["rotation"],
            )
        elif stage == "val":
            metrics = self.val_metrics.update(
                outputs["position"],
                outputs["rotation"],
                batch["position"],
                batch["rotation"],
            )
        else:
            metrics = self.test_metrics.update(
                outputs["position"],
                outputs["rotation"],
                batch["position"],
                batch["rotation"],
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
        self.log("train/pos_error_m", metrics["position_error_m"], prog_bar=True)
        self.log("train/ang_error_deg", metrics["angular_error_deg"], prog_bar=True)

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
        self.log("val/pos_error_m", metrics["position_error_m"], prog_bar=True)
        self.log("val/ang_error_deg", metrics["angular_error_deg"], prog_bar=True)

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
        self.log("test/pos_error_m", metrics["position_error_m"])
        self.log("test/ang_error_deg", metrics["angular_error_deg"])

    def on_test_epoch_end(self) -> None:
        """Called at end of test epoch."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            dict: Optimizer and scheduler configuration.

        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Warmup + Cosine annealing
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        # Estimate total steps
        data_cfg = self.config.get("data", {})
        num_samples = data_cfg.get("num_scenes_per_epoch", 10000)
        batch_size = data_cfg.get("batch_size", 64)
        steps_per_epoch = num_samples // batch_size
        total_steps = steps_per_epoch * self.max_epochs

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
