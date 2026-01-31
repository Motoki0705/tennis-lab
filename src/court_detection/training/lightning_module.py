"""PyTorch Lightning module for court keypoint detection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.court_detection.models.court_keypoint_model import CourtKeypointModel
from src.court_detection.training.losses import CourtKeypointLoss
from src.court_detection.training.metrics import CourtKeypointMetrics


class CourtKeypointLightningModule(BaseLightningModule):
    """Lightning module for training court keypoint detection.

    Inherits from BaseLightningModule but overrides configure_optimizers
    to use epoch-based warmup (matching original behavior).

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
        data_config: dict[str, Any] | None = None,
    ) -> None:
        # BaseLightningModule expects a config with 'training' key
        # but we maintain backward compatibility with direct dicts
        base_config = {
            "training": training_config,
            "data": data_config or {},
        }
        super().__init__(config=base_config)
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
        pred_heatmaps = outputs["heatmaps"]
        if pred_heatmaps.shape[-2:] != target_heatmaps.shape[-2:]:
            pred_heatmaps = F.interpolate(
                pred_heatmaps,
                size=target_heatmaps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        losses = self.loss_fn(
            pred_heatmaps=pred_heatmaps,
            target_heatmaps=target_heatmaps,
            pred_visibility=outputs["visibility"],
            target_visibility=target_visibility,
        )

        # Log losses
        self.log(f"{stage}/loss", losses["total"], prog_bar=True)
        self.log(f"{stage}/loss_heatmap", losses["heatmap"])
        self.log(f"{stage}/loss_visibility", losses["visibility"])

        pred_keypoints = self._heatmaps_to_coords(pred_heatmaps)

        return {
            "loss": losses["total"],
            "pred_keypoints": pred_keypoints,
            "pred_visibility": torch.sigmoid(outputs["visibility"]),
            "target_keypoints": target_keypoints,
            "target_visibility": target_visibility,
        }

    @staticmethod
    def _heatmaps_to_coords(heatmaps: Tensor) -> Tensor:
        """Convert heatmaps to keypoint coordinates using soft-argmax.

        Args:
            heatmaps: Heatmaps of shape (B, K, H, W).

        Returns:
            Coordinates of shape (B, K, 2) in normalized [0, 1] range.
        """
        bsz, num_kp, height, width = heatmaps.shape
        device = heatmaps.device

        heatmaps_flat = heatmaps.view(bsz, num_kp, -1)
        probs = F.softmax(heatmaps_flat, dim=-1)

        y_coords = torch.linspace(0, 1, height, device=device)
        x_coords = torch.linspace(0, 1, width, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
        xx_flat = xx.reshape(-1)
        yy_flat = yy.reshape(-1)

        x = (probs * xx_flat.view(1, 1, -1)).sum(dim=-1)
        y = (probs * yy_flat.view(1, 1, -1)).sum(dim=-1)

        return torch.stack([x, y], dim=-1)

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

    # configure_optimizers inherited from BaseLightningModule
