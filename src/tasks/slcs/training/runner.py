"""SLCS training runner (thin BaseTrainingRunner specialization)."""

from __future__ import annotations

from collections.abc import Sized
from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.slcs.configuration import SLCSTrainingRuntimeConfig
from src.tasks.slcs.data.datamodule import SLCSDataModule
from src.tasks.slcs.training.lightning_module import SLCSLightningModule


class SLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for SLCS."""

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        if not isinstance(config, DictConfig):
            raise TypeError(
                "SLCS training requires a composed DictConfig boundary input."
            )
        return SLCSTrainingRuntimeConfig.from_config(config)

    @staticmethod
    def _typed(config: Any) -> SLCSTrainingRuntimeConfig:
        if not isinstance(config, DictConfig):
            raise TypeError(
                "SLCS training requires a composed DictConfig boundary input."
            )
        return SLCSTrainingRuntimeConfig.from_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return SLCSDataModule(self._typed(config).data)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        module = SLCSLightningModule(self._typed(config))
        if steps_per_epoch is not None:
            module.steps_per_epoch = steps_per_epoch
        return module

    def resolve_steps_per_epoch(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        train_loader: Any | None,
    ) -> int:
        """Use the real window loader length for the step-based LR schedule."""
        if train_loader is None:
            datamodule.setup(stage="fit")
            train_loader = datamodule.train_dataloader()
        if not isinstance(train_loader, Sized):
            raise TypeError("SLCS train loader must expose a finite batch count.")
        batches = len(train_loader)
        accumulate = int(
            self._typed(config).training.trainer.accumulate_grad_batches
        )
        return max((batches + accumulate - 1) // accumulate, 1)


__all__ = ["SLCSTrainingRunner"]
