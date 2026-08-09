"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def prepare_config(self, config: Any) -> None:
        PLCSTrainingConfig.from_config(config)
        super().prepare_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return build_plcs_datamodule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return build_plcs_lightning_module(config)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        extras: list[Any] = super().callbacks_extra(config, datamodule, logger)

        runtime = PLCSTrainingConfig.from_config(config)
        if runtime.data.backend != "chunked":
            return extras

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        extras.append(ChunkRotationCallback())
        return extras
