"""PyTorch Lightning module for multi-view PLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.plcs.models.plcs_multiview_model import PLCSMultiViewModel
from src.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.plcs.training.metrics import PLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSMultiViewLightningModule(BaseLightningModule):
    """Lightning module for training multi-view PLCS models.

    This module supports both frame-based and sequence-based multi-view inputs:
    - Frame-based: observations from multiple cameras for a single frame.
    - Sequence-based: observations from multiple cameras over a temporal sequence.

    Uses camera-time ordering throughout: (B, N, T, ...) where N=cameras, T=time.

    The data module determines which type of input is provided based on
    config.data.mode ('multiview' or 'multiview_sequence').
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        # Build multi-view model
        self.model = PLCSMultiViewModel.from_config(self.config)

        # Loss function (config-based)
        loss_cfg_dict = self.config.get("loss", {})
        if loss_cfg_dict:
            loss_cfg = PLCSLossConfig.from_dict(dict(loss_cfg_dict))
        else:
            # Legacy fallback: read from training config
            train_cfg = self.config.get("training", {})
            loss_cfg = PLCSLossConfig(
                position_weight=train_cfg.get("position_loss_weight", 1.0),
                rotation_weight=train_cfg.get("rotation_loss_weight", 1.0),
            )
        self.loss_fn = PLCSLoss(config=loss_cfg)

        # Metrics (only for val and test)
        self.val_metrics = PLCSMetrics()
        self.test_metrics = PLCSMetrics()

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
        view_mask: Tensor | None = None,
        seq_mask: Tensor | None = None,
        camera_params: list[list[dict]] | None = None,
    ) -> dict[str, Any]:
        """Forward pass.

        Args:
            human_kp: Human keypoints from multiple views.
                - Frame-based: shape (B, N, 17, 2)
                - Sequence-based: shape (B, N, T, 17, 2)
            court_kp: Court keypoints from multiple views.
                - Frame-based: shape (B, N, 20, 2)
                - Sequence-based: shape (B, N, T, 20, 2)
            human_vis: Human visibility mask.
                - Frame-based: shape (B, N, 17)
                - Sequence-based: shape (B, N, T, 17)
            court_vis: Court visibility mask.
                - Frame-based: shape (B, N, 20)
                - Sequence-based: shape (B, N, T, 20)
            view_mask: Valid view mask, shape (B, N), True = non-padding.
            seq_mask: Valid frame mask, shape (B, T), True = non-padding.
            camera_params: Camera parameters per view.

        Returns:
            dict: Model outputs.

        """
        outputs: dict[str, Any] = self.model(
            human_kp, court_kp, human_vis, court_vis, view_mask, seq_mask, camera_params
        )
        return outputs

    def _shared_step(
        self, batch: dict[str, Tensor], stage: str
    ) -> tuple[Tensor, dict[str, float]]:
        """Shared step for training and validation.

        Args:
            batch: Batch dictionary from dataloader.
                - Frame-based: human_kp/court_kp shape (B, N, K, 2)
                - Sequence-based: human_kp/court_kp shape (B, N, T, K, 2)
            stage: One of 'train', 'val', 'test'.

        Returns:
            tuple: (loss tensor, metrics dict).

        """
        # Forward pass with multi-view inputs (frame or sequence)
        outputs = self.model(
            human_kp=batch["human_kp"],
            court_kp=batch["court_kp"],
            human_vis=batch.get("human_vis"),
            court_vis=batch.get("court_vis"),
            view_mask=batch.get("view_mask"),
            seq_mask=batch.get("seq_mask"),
            camera_params=batch.get("camera_params"),
        )

        # Compute loss
        losses = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            seq_mask=batch.get("seq_mask"),
        )

        # Update metrics (only for val and test)
        if stage == "val":
            metrics = self.val_metrics.update(
                outputs["position"],
                outputs["rotation"],
                batch["position"],
                batch["rotation"],
            )
        elif stage == "test":
            metrics = self.test_metrics.update(
                outputs["position"],
                outputs["rotation"],
                batch["position"],
                batch["rotation"],
            )
        else:
            # For train, return empty metrics dict
            metrics = {}

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

        # Log losses only (no metrics for train)
        self.log("train/loss", loss, prog_bar=True)
        for k in ["position", "rotation", "temporal"]:
            loss_key = f"loss_{k}"
            if loss_key in metrics:
                self.log(f"train/{loss_key}", metrics[loss_key], prog_bar=False)

        return loss

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
        for k in ["position", "rotation", "temporal"]:
            loss_key = f"loss_{k}"
            if loss_key in metrics:
                self.log(f"val/{loss_key}", metrics[loss_key], prog_bar=False)

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
        for k in ["position", "rotation", "temporal"]:
            loss_key = f"loss_{k}"
            if loss_key in metrics:
                self.log(f"test/{loss_key}", metrics[loss_key])

    def on_test_epoch_end(self) -> None:
        """Called at end of test epoch."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(f"test/{name}", value)
        self.test_metrics.reset()

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)

    config = {
        "training": {
            "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-5,
            "warmup_steps": 5,
            "max_epochs": 1,
            "min_lr": 1.0e-6,
        },
        "data": {
            "num_scenes_per_epoch": 64,
            "batch_size": 16,
        },
    }

    module = PLCSMultiViewLightningModule(config)
    optim_config = module.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    assert scheduler is not None
