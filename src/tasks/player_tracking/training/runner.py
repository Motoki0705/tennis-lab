"""Config-driven runner for the synthetic multi-person baseline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.player_tracking.data import PlayerTrackingDataModule
from src.tasks.player_tracking.training.lightning_module import (
    PlayerTrackingLightningModule,
)


class PlayerTrackingTrainingRunner(BaseTrainingRunner):
    """Wire the player-tracking synthetic data and Lightning module."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return PlayerTrackingDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        del datamodule, steps_per_epoch
        return PlayerTrackingLightningModule(config)
