"""Event detection training runner using BaseTrainingRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.event_detection.data.datamodule import EventDetectionDataModule
from src.tasks.event_detection.training.lightning_module import EventDetectionLightningModule

if TYPE_CHECKING:
    from omegaconf import DictConfig


class EventDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for event detection models.

    Supports both UV and 3D trajectory input types (determined by config).
    """

    def build_datamodule(self, config: DictConfig) -> pl.LightningDataModule:
        """Build the EventDetectionDataModule."""
        return EventDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: DictConfig,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build the EventDetectionLightningModule."""
        _ = datamodule, steps_per_epoch
        return EventDetectionLightningModule(config)
