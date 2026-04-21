"""Trajectory completion training runner using BaseTrainingRunner."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.blcs.generate_dataset.config import (
    build_default_generator_config,
)
from src.tasks.trajectory_completion.data.datamodule import (
    TrajectoryCompletionDataModule,
)
from src.tasks.trajectory_completion.training.lightning_module import (
    TrajectoryCompletionLightningModule,
)


class TrajectoryCompletionTrainingRunner(BaseTrainingRunner):
    """Training runner for trajectory completion model."""

    def __init__(self, *, generator_config: Any | None = None) -> None:
        self.generator_config = generator_config

    def _resolve_generator_config(self) -> Any:
        if self.generator_config is not None:
            return self.generator_config
        return build_default_generator_config()

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build trajectory completion data module."""
        backend = str(config.get("data", {}).get("backend", "default"))
        if backend == "chunked":
            from src.tasks.trajectory_completion.data.chunked_datamodule import (
                ChunkedTrajectoryCompletionDataModule,
            )

            return ChunkedTrajectoryCompletionDataModule(
                config,
                generator_config=self._resolve_generator_config(),
            )
        if backend == "default":
            return TrajectoryCompletionDataModule(config)
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
        """Build trajectory completion lightning module."""
        return TrajectoryCompletionLightningModule(config)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        if str(config.get("data", {}).get("backend", "default")) != "chunked":
            return []

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        return [ChunkRotationCallback()]
