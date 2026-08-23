"""BLCS training runner using BaseTrainingRunner."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.blcs.configuration import (
    parse_court_coordinate_normalization,
    validate_training_boundary,
)
from src.tasks.blcs.model_io.checkpoints import validate_checkpoint_path
from src.tasks.blcs.model_io.training import (
    BLCSTrainingComposition,
    compose_blcs_training,
)

if TYPE_CHECKING:
    from src.tasks.blcs.generate_dataset.scene_generator import GeneratorConfig
    from src.utils.schema.court_normalization import CourtCoordinateNormalization


class BLCSTrainingRunner(BaseTrainingRunner):
    """Training runner for BLCS models.

    Data loading is unified and adapted to model profile by data collate.
    """

    def __init__(self, *, generator_config: GeneratorConfig | None = None) -> None:
        self.generator_config = generator_config
        self._composition: BLCSTrainingComposition | None = None
        self._court_coordinate_normalization: CourtCoordinateNormalization | None = (
            None
        )

    def _runtime(self, config: Any) -> BLCSTrainingComposition:
        if self._composition is None:
            self._composition = compose_blcs_training(
                config,
                generator_config=self.generator_config,
            )
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

    def callbacks_extra(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        logger: TensorBoardLogger,
    ) -> list[Any]:
        """Add chunk rotation callback when using chunked backend."""
        extras = super().callbacks_extra(config, datamodule, logger)
        from src.tasks.base.data.chunked_datamodule import BaseChunkedDataModule

        if isinstance(datamodule, BaseChunkedDataModule):
            from src.tasks.base.training.chunk_rotation_callback import (
                ChunkRotationCallback,
            )

            extras.append(ChunkRotationCallback())
        return list(extras)

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        """Validate shared and BLCS-specific contracts before runner side effects."""
        validate_training_boundary(config)
        self._court_coordinate_normalization = parse_court_coordinate_normalization(
            config
        )
        runtime: TrainingRuntimeConfig = super().validate_runtime_config(config)
        return runtime

    def resolve_resume(
        self,
        config: TrainingRuntimeConfig,
        output_dir: Path,
    ) -> str | None:
        """Reject normalization-incompatible full-state resumes before fit."""
        path: str | None = super().resolve_resume(config, output_dir)
        resume_path = config.run.resume
        if resume_path is not None:
            normalization = self._require_normalization()
            validate_checkpoint_path(resume_path, normalization)
        return path

    def maybe_load_init_weights(
        self,
        config: TrainingRuntimeConfig,
        lightning_module: pl.LightningModule,
    ) -> None:
        """Reject normalization-incompatible fine-tune weights before loading."""
        if config.run.init_weights is not None:
            validate_checkpoint_path(
                config.run.init_weights,
                self._require_normalization(),
            )
        super().maybe_load_init_weights(config, lightning_module)

    def _require_normalization(self) -> CourtCoordinateNormalization:
        if self._court_coordinate_normalization is None:
            raise RuntimeError(
                "BLCS normalization must be validated before checkpoint access."
            )
        return self._court_coordinate_normalization
