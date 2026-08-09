"""Training runner for composable Court detection."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)


class CourtDetectionTrainingRunner(BaseTrainingRunner):
    """Construct the model only after the DataModule resolves its target bundle."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return CourtDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        if not isinstance(datamodule, CourtDetectionDataModule):
            raise TypeError(
                "Court training requires CourtDetectionDataModule to resolve "
                "the target bundle before model construction."
            )
        module = CourtDetectionLightningModule(
            config,
            target_bundle=datamodule.target_bundle_spec,
        )
        module.steps_per_epoch = steps_per_epoch
        return module

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        return CourtTrainingConfig.from_config(config).shared


__all__ = ["CourtDetectionTrainingRunner"]
