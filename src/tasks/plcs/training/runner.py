"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.utils import prepare_generation_config


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def prepare_config(self, config: Any) -> None:
        backend = str(config.get("data", {}).get("backend", "default"))
        if backend == "chunked":
            prepare_generation_config(config, resolve=False)
        super().prepare_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        backend = str(config.get("data", {}).get("backend", "default"))
        if backend == "chunked":
            from src.tasks.plcs.data.chunked_datamodule import ChunkedPLCSDataModule

            return ChunkedPLCSDataModule(config)
        if backend == "default":
            return PLCSDataModule(config)
        raise ValueError(
            f"Unsupported data.backend='{backend}'. Supported: ['default', 'chunked']"
        )

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        return PLCSLightningModule(config)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        extras = super().callbacks_extra(config, datamodule, logger)

        if str(config.get("data", {}).get("backend", "default")) != "chunked":
            return extras

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        extras.append(ChunkRotationCallback())
        return extras
