"""PyTorch Lightning module for ball detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from src.tasks.ball_detection.data.utils.input_adapter import to_model_input
from src.tasks.ball_detection.models import build_ball_detection_model
from src.tasks.ball_detection.training.losses import BallDetectionFocalLoss
from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.tasks.base.training.lightning_module import BaseLightningModule

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BallDetectionLightningModule(BaseLightningModule):
    """Lightning module for training ball detection.

    Inherits optimizer/scheduler logic from
    :class:`~src.tasks.base.training.lightning_module.BaseLightningModule`.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__(config)
        self.save_hyperparameters()

        loss_cfg = self.config.get("loss", {})
        metrics_cfg = self.config.get("metrics", {})

        self.model = build_ball_detection_model(self.config)

        self.loss_fn = BallDetectionFocalLoss(loss_cfg)

        self.train_metrics = BallDetectionMetrics(
            peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
            ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
        )
        self.val_metrics = BallDetectionMetrics(
            peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
            ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
        )
        self.test_metrics = BallDetectionMetrics(
            peak_threshold=float(metrics_cfg.get("peak_threshold", 0.5)),
            ball_distance_threshold=float(metrics_cfg.get("ball_distance_threshold", 4.0)),
        )

    def forward(self, images: Tensor) -> Tensor:
        """Forward pass through the model.

        Args:
            images: Input tensor of shape ``(B, C, T, H, W)``.

        Returns:
            Logits of shape ``(B, 1, T, Hh, Wh)``.
        """
        return self.model(images)

    def _shared_step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        """Shared computation for train/val/test steps."""
        images = batch["images"]
        target_heatmaps = batch["heatmaps"]

        model_cfg = self.config.get("model", {})
        model_input = to_model_input(images, model_cfg)

        logits = self.model(model_input)

        # Squeeze channel dim: (B, 1, T, Hh, Wh) -> (B, T, Hh, Wh)
        logits = logits.squeeze(1)

        # Interpolate if model output size != target heatmap size
        if logits.shape[-2:] != target_heatmaps.shape[-2:]:
            b, t = logits.shape[:2]
            logits_flat = logits.reshape(b * t, 1, *logits.shape[-2:])
            logits_flat = F.interpolate(
                logits_flat,
                size=target_heatmaps.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            logits = logits_flat.reshape(b, t, *target_heatmaps.shape[-2:])

        loss = self.loss_fn(logits, target_heatmaps)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True)

        pred_heatmaps = torch.sigmoid(logits)

        return {
            "loss": loss,
            "pred_heatmaps": pred_heatmaps,
            "target_coords": batch["coords"],
            "target_visibility": batch["visibility"],
            "original_size": batch["original_size"],
        }

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        outputs = self._shared_step(batch, "train")
        self.train_metrics.update(
            outputs["pred_heatmaps"],
            outputs["target_coords"],
            outputs["target_visibility"],
            outputs["original_size"],
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
        self.val_metrics.update(
            outputs["pred_heatmaps"],
            outputs["target_coords"],
            outputs["target_visibility"],
            outputs["original_size"],
        )

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(f"val/{name}", value, prog_bar=(name == "f1"))
        self.val_metrics.reset()

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """Test step."""
        outputs = self._shared_step(batch, "test")
        self.test_metrics.update(
            outputs["pred_heatmaps"],
            outputs["target_coords"],
            outputs["target_visibility"],
            outputs["original_size"],
        )

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()
