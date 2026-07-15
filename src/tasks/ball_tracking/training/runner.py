"""Config-driven training runner for synthetic multi-ball tracking."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.ball_tracking.data import BallTrackingDataModule
from src.tasks.ball_tracking.training.lightning_module import (
    BallTrackingLightningModule,
)
from src.tasks.base.training.runner import BaseTrainingRunner


class BallTrackingTrainingRunner(BaseTrainingRunner):
    """Wire the task-specific synthetic data and Lightning module."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return BallTrackingDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        del datamodule, steps_per_epoch
        return BallTrackingLightningModule(config)
