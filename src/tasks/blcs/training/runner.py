"""BLCS training runner using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.blcs.data.datamodule import BLCSDataModule
from src.tasks.blcs.training.lightning_module import BLCSLightningModule

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig


class BLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for BLCS models.

    Data loading is unified and adapted to model profile by data collate.
    """

    def __init__(self, *, generator_config: GeneratorConfig | None = None) -> None:
        self.generator_config = generator_config

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        """Build unified BLCS data module."""
        backend = str(config.get("data", {}).get("backend", "npz"))
        if backend == "chunked":
            from src.tasks.blcs.data.chunked_datamodule import ChunkedBLCSDataModule

            if self.generator_config is None:
                raise RuntimeError(
                    "generator_config is required for data.backend=chunked. "
                    "Use the train_chunked entrypoint."
                )
            return ChunkedBLCSDataModule(
                config, generator_config=self.generator_config,
            )
        elif backend == "default":
            return BLCSDataModule(config)
        else:
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
        """Build BLCS lightning module."""
        return BLCSLightningModule(config)

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        """Add chunk rotation callback when using chunked backend."""
        extras = super().callbacks_extra(config, datamodule, logger)
        from src.tasks.blcs.data.chunked_datamodule import ChunkedBLCSDataModule

        if isinstance(datamodule, ChunkedBLCSDataModule):
            from src.tasks.base.training.chunk_rotation_callback import (
                ChunkRotationCallback,
            )

            extras.append(ChunkRotationCallback())
        return extras

    def dry_run_postprocess(self, batch: Any, output_dir: Path) -> None:
        """Log model parameters after dry run batch loading."""
        # Model parameter logging is handled in build_lightning_module
        pass