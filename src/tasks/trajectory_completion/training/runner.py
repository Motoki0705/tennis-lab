"""Trajectory completion training runner using BaseTrainingRunner."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.trajectory_completion.data.datamodule import TrajectoryCompletionDataModule
from src.tasks.trajectory_completion.training.lightning_module import (
    TrajectoryCompletionLightningModule,
)


class TrajectoryCompletionTrainingRunner(BaseTrainingRunner):
    """Training runner for trajectory completion model."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build trajectory completion data module."""
        return TrajectoryCompletionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build trajectory completion lightning module."""
        return TrajectoryCompletionLightningModule(config)
