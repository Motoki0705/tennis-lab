"""PLCS training runner using a unified data/model I/O pipeline."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.plcs.artifact_paths import validate_plcs_training_output_occupancy
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import validate_plcs_checkpoint_normalization
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)


class PLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for PLCS."""

    def prepare_config(self, config: Any) -> None:
        runtime = PLCSTrainingConfig.from_config(config)
        validate_plcs_training_output_occupancy(
            runtime.shared.run.output_dir,
            normalization=runtime.court_coordinate_normalization.contract,
        )
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

    def maybe_load_init_weights(
        self,
        config: Any,
        lightning_module: pl.LightningModule,
    ) -> None:
        """Validate normalization metadata before weight-only initialization."""
        init_path = config.run.init_weights
        if init_path is not None:
            checkpoint = torch.load(
                init_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid PLCS init_weights checkpoint: {init_path}.")
            runtime = PLCSTrainingConfig.from_config(lightning_module.config)
            validate_plcs_checkpoint_normalization(
                checkpoint,
                runtime.court_coordinate_normalization.contract,
            )
        super().maybe_load_init_weights(config, lightning_module)

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
