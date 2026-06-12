"""Training runner for ball detection."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.ball_detection.data import build_ball_detection_datamodule
from src.tasks.ball_detection.training.lightning_module import (
    BallDetectionLightningModule,
)
from src.tasks.base.training.runner import BaseTrainingRunner


class BallDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for ball detection.

    Overrides datamodule/model construction for ball detection.
    """

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build the configured ball detection DataModule."""
        return build_ball_detection_datamodule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build BallDetectionLightningModule from config."""
        return BallDetectionLightningModule(config)
