"""PyTorch Lightning module for PLCS training."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from torch import Tensor

from src.base.training.lightning_module import BaseLightningModule
from src.plcs.models.plcs_model import PLCSModel
from src.plcs.training.losses import PLCSLoss
from src.plcs.training.metrics import PLCSMetrics

if TYPE_CHECKING:
    from omegaconf import DictConfig


class PLCSLightningModule(BaseLightningModule):
    """Lightning module for training PLCS models.

    Handles training, validation, and test loops with configurable
    optimizers, schedulers, and logging.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        """Initialize the Lightning module.

        Args:
            config: Configuration dictionary with model and training parameters.

        """
        super().__init__(config)

        # Build model
        self.model: PLCSModel = PLCSModel.from_config(self.config)

        # Loss function
        train_cfg = self.config.get("training", {})
        self.loss_fn = PLCSLoss(
            position_weight=train_cfg.get("position_loss_weight", 1.0),
            rotation_weight=train_cfg.get("rotation_loss_weight", 1.0),
        )

        # Metrics
        self.train_metrics = PLCSMetrics()
        self.val_metrics = PLCSMetrics()
        self.test_metrics = PLCSMetrics()

    def forward(
        self,
        human_kp: Tensor,
        court_kp: Tensor,
        human_vis: Tensor | None = None,
        court_vis: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            human_kp: Human keypoints.
            court_kp: Court keypoints.
            human_vis: Human visibility mask.
            court_vis: Court visibility mask.

        Returns:
            dict: Model outputs.

        """
        return cast(dict[str, Tensor], self.model(human_kp, court_kp, human_vis, court_vis))

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
        # Forward pass
        outputs = self.model(
            human_kp=batch["human_kp"],
            court_kp=batch["court_kp"],
            human_vis=batch.get("human_vis"),
            court_vis=batch.get("court_vis"),
        )

        # Compute loss
        losses: dict[str, Tensor] = self.loss_fn(
            pred_position=outputs["position"],
            pred_rotation=outputs["rotation"],
            target_position=batch["position"],
            target_rotation=batch["rotation"],
            seq_mask=batch.get("seq_mask"),
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

    module = PLCSLightningModule(config)
    optim_config = module.configure_optimizers()
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    assert scheduler is not None
