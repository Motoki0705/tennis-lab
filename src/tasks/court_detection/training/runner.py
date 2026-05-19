"""Training runner for court detection."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
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

    def skip_test(self, config: Any) -> bool:
        """Court detection currently has fit/validation support only."""
        return True
