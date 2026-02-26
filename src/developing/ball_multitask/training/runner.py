"""Training runner for ball multi-task models."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.developing.ball_multitask.data.datamodule import BallMultitaskDataModule
from src.developing.ball_multitask.training.lightning_module import BallMultitaskLightningModule


class BallMultitaskTrainingRunner(BaseTrainingRunner):
    """Training runner for multi-task models."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return BallMultitaskDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        _ = datamodule, steps_per_epoch
        return BallMultitaskLightningModule(config)
