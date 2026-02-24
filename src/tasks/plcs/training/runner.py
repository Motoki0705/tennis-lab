"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return PLCSDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return PLCSLightningModule(config)
