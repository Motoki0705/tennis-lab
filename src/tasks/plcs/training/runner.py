"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.training.lightning_module import PLCSLightningModule
from src.tasks.plcs.training.tracking_lightning_module import (
    PLCSTrackingLightningModule,
)


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def prepare_config(self, config: Any) -> None:
        PLCSTrainingConfig.from_config(config)
        super().prepare_config(config)

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        runtime = PLCSTrainingConfig.from_config(config)
        backend = runtime.data.backend
        tracking = runtime.model.name == "plcs_track_query"
        if tracking:
            from src.tasks.plcs.data.tracking_datamodule import (
                ChunkedPLCSTrackingDataModule,
                PLCSTrackingDataModule,
            )

            if backend == "default":
                return PLCSTrackingDataModule(config)
            if backend == "chunked":
                return ChunkedPLCSTrackingDataModule(config)
            raise ValueError(
                f"Unsupported tracking data.backend='{backend}'. "
                "Supported: ['default', 'chunked']"
            )
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
        runtime = PLCSTrainingConfig.from_config(config)
        if runtime.model.name == "plcs_track_query":
            return PLCSTrackingLightningModule(config)
        return PLCSLightningModule(config)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        extras = super().callbacks_extra(config, datamodule, logger)

        runtime = PLCSTrainingConfig.from_config(config)
        if runtime.data.backend != "chunked":
            return extras

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        extras.append(ChunkRotationCallback())
        return extras
