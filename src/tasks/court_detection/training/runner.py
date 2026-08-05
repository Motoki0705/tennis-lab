"""Training runner for court detection."""

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
    """Training runner for court detection.

    Overrides datamodule/model construction for court detection.
    """

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return CourtDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return CourtDetectionLightningModule(config)

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        """Validate shared and court-specific contracts before side effects."""
        return CourtTrainingConfig.from_config(config).shared
