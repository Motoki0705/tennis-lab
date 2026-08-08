"""BLCS training runner using BaseTrainingRunner."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.blcs.configuration import validate_training_boundary
from src.tasks.blcs.model_io.training import (
    BLCSTrainingComposition,
    compose_blcs_training,
)


class BLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for BLCS models.

    Data loading is unified and adapted to model profile by data collate.
    """

    def __init__(self) -> None:
        self._composition: BLCSTrainingComposition | None = None

    def _runtime(self, config: Any) -> BLCSTrainingComposition:
        if self._composition is None:
            self._composition = compose_blcs_training(config)
        return self._composition

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build unified BLCS data module."""
        return self._runtime(config).datamodule

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build BLCS lightning module."""
        return self._runtime(config).lightning_module

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        """Validate shared and BLCS-specific contracts before runner side effects."""
        validate_training_boundary(config)
        runtime: TrainingRuntimeConfig = super().validate_runtime_config(config)
        return runtime
