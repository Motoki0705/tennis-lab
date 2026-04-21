"""Event detection training runner using BaseTrainingRunner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.blcs.generate_dataset.config import (
    build_default_generator_config,
)
from src.tasks.event_detection.data.datamodule import EventDetectionDataModule
from src.tasks.event_detection.training.lightning_module import (
    EventDetectionLightningModule,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig


class EventDetectionTrainingRunner(BaseTrainingRunner):
    """Training runner for event detection models.

    Supports both UV and 3D trajectory input types (determined by config).
    """

    def __init__(self, *, generator_config: GeneratorConfig | None = None) -> None:
        self.generator_config = generator_config

    def _resolve_generator_config(self) -> GeneratorConfig:
        if self.generator_config is not None:
            return self.generator_config
        return build_default_generator_config()

    def build_datamodule(self, config: DictConfig) -> pl.LightningDataModule:
        """Build the EventDetectionDataModule."""
        backend = str(config.get("data", {}).get("backend", "default"))
        if backend == "chunked":
            from src.tasks.event_detection.data.chunked_datamodule import (
                ChunkedEventDetectionDataModule,
            )

            return ChunkedEventDetectionDataModule(
                config,
                generator_config=self._resolve_generator_config(),
            )
        if backend == "default":
            return EventDetectionDataModule(config)
        raise ValueError(
            f"Unsupported data.backend='{backend}'. Supported: ['default', 'chunked']"
        )

    def build_lightning_module(
        self,
        config: DictConfig,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        """Build the EventDetectionLightningModule."""
        return EventDetectionLightningModule(config)

    def callbacks_extra(
        self,
        config: DictConfig,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        if str(config.get("data", {}).get("backend", "default")) != "chunked":
            return []

        from src.tasks.base.training.chunk_rotation_callback import (
            ChunkRotationCallback,
        )

        return [ChunkRotationCallback()]
