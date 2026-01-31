"""Shared Lightning training utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseLightningModule(pl.LightningModule):
    """Base Lightning module with shared optimizer/scheduler logic.

    This class expects training settings under `config.training` and optional
    dataset sizing under `config.data`.
    """

    def __init__(self, config: DictConfig | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.config = config or {}

        train_cfg = self.config.get("training", {})
        self.learning_rate = train_cfg.get("learning_rate", 1e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-5)
        self.warmup_steps = train_cfg.get("warmup_steps", 1000)
        self.max_epochs = train_cfg.get("max_epochs", 100)
        self.min_lr = train_cfg.get("min_lr", 1e-6)

    def _estimate_total_steps(self) -> int:
        estimated_steps = None
        if getattr(self, "_trainer", None) is not None:
            estimated_steps = getattr(self._trainer, "estimated_stepping_batches", None)
        if estimated_steps is not None:
            return int(estimated_steps)

        data_cfg = self.config.get("data", {})
        num_samples = data_cfg.get("num_scenes_per_epoch", 10000)
        batch_size = data_cfg.get("batch_size", 64)
        steps_per_epoch = max(num_samples // batch_size, 1)
        return steps_per_epoch * self.max_epochs

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

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.warmup_steps,
        )

        total_steps = self._estimate_total_steps()
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
