"""Training runner for court keypoint detection."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.base.training.runner import BaseTrainingRunner
from src.court_detection.data.datamodule import CourtKeypointDataModule
from src.court_detection.training.lightning_module import CourtKeypointLightningModule


class CourtDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for court keypoint detection.

    Overrides datamodule/model construction.
    """

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build CourtKeypointDataModule from config."""
        return CourtKeypointDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build CourtKeypointLightningModule from config."""
        return CourtKeypointLightningModule(config)
