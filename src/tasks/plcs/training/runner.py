"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

import pytorch_lightning as pl
from omegaconf import DictConfig

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def build_datamodule(self, config: DictConfig) -> pl.LightningDataModule:
        return PLCSDataModule(config)

    def build_lightning_module(
        self,
        config: DictConfig,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return PLCSLightningModule(config)
