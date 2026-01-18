"""PyTorch Lightning module for court keypoint detection."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.court_detection.models.court_keypoint_model import CourtKeypointModel
from src.court_detection.training.losses import CourtKeypointLoss
from src.court_detection.training.metrics import CourtKeypointMetrics


class CourtKeypointLightningModule(pl.LightningModule):
    """Lightning module for training court keypoint detection.

    Args:
        model_config: Model configuration dict.
        training_config: Training configuration dict.
        loss_config: Loss configuration dict.
    """

    def __init__(
        self,
        model_config: dict[str, Any],
        training_config: dict[str, Any],
        loss_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model_config = model_config
        self.training_config = training_config
        self.loss_config = loss_config

        # Build model
        self.model = CourtKeypointModel(model_config)

        # Build loss
        self.loss_fn = CourtKeypointLoss(
            heatmap_config=loss_config.get("heatmap", {}),
            visibility_config=loss_config.get("visibility", {}),
        )

        # Build metrics
        input_size = tuple(model_config.get("input_size", [256, 256]))
        self.train_metrics = CourtKeypointMetrics(image_size=input_size)
        self.val_metrics = CourtKeypointMetrics(image_size=input_size)
        self.test_metrics = CourtKeypointMetrics(image_size=input_size)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass."""
        return self.model(x)

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        """Shared step for train/val/test."""
        images = batch["image"]
        target_heatmaps = batch["heatmaps"]
        target_keypoints = batch["keypoints"]
        target_visibility = batch["visibility"]

        # Forward pass
        outputs = self.model(images)

        # Compute loss
        losses = self.loss_fn(
            pred_heatmaps=outputs["heatmaps"],
            target_heatmaps=target_heatmaps,
            pred_visibility=outputs["visibility"],
            target_visibility=target_visibility,
        )

        # Log losses
        self.log(f"{stage}/loss", losses["total"], prog_bar=True)
        self.log(f"{stage}/loss_heatmap", losses["heatmap"])
        self.log(f"{stage}/loss_visibility", losses["visibility"])

        return {
            "loss": losses["total"],
            "pred_keypoints": outputs["keypoints"],
            "pred_visibility": torch.sigmoid(outputs["visibility"]),
            "target_keypoints": target_keypoints,
            "target_visibility": target_visibility,
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        outputs = self._shared_step(batch, "train")

        # Update metrics
        self.train_metrics.update(
            outputs["pred_keypoints"],
            outputs["target_keypoints"],
            outputs["pred_visibility"],
            outputs["target_visibility"],
        )

        return outputs["loss"]

    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        metrics = self.train_metrics.compute()
        for name, value in metrics.items():
            self.log(f"train/{name}", value)
        self.train_metrics.reset()

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Validation step."""
        outputs = self._shared_step(batch, "val")

        # Update metrics
        self.val_metrics.update(
            outputs["pred_keypoints"],
            outputs["target_keypoints"],
            outputs["pred_visibility"],
            outputs["target_visibility"],
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=(name == "pck"))
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Test step."""
        outputs = self._shared_step(batch, "test")

        # Update metrics
        self.test_metrics.update(
            outputs["pred_keypoints"],
            outputs["target_keypoints"],
            outputs["pred_visibility"],
            outputs["target_visibility"],
        )

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""
        lr = self.training_config.get("learning_rate", 1e-4)
        weight_decay = self.training_config.get("weight_decay", 1e-4)
        max_epochs = self.training_config.get("max_epochs", 100)
        warmup_epochs = self.training_config.get("warmup_epochs", 5)
        min_lr = self.training_config.get("scheduler", {}).get("min_lr", 1e-6)

        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(self.training_config.get("optimizer", {}).get("betas", [0.9, 0.999])),
        )

        # Scheduler with warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
