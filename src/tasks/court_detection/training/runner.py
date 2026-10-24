"""Training runner for fixed-ratio mixed Court source batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.court_detection.configuration import (
    CourtTrainingConfig,
    SyntheticCourtDatasetVersionSourceConfig,
    SyntheticCourtStoresSourceConfig,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.mixed import (
    CourtMixedDataConfig,
)
from src.tasks.court_detection.scaling.schedule import has_step_schedule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
from src.utils.artifact_store import ArtifactStore
from src.utils.configuration import SemanticConfigurationError


def resolve_training_config(
    config: object,
) -> tuple[DictConfig, CourtMixedDataConfig]:
    """Validate the complete configuration and resolve the two input sources."""
    if not isinstance(config, DictConfig):
        raise TypeError("Mixed Court training requires a Hydra DictConfig.")
    run_node = config.get("run")
    if not isinstance(run_node, DictConfig) or OmegaConf.is_missing(
        run_node, "output_dir"
    ):
        raise SemanticConfigurationError(
            "Mixed Court training requires an explicit variant-specific run.output_dir."
        )
    runtime = CourtTrainingConfig.from_config(config)
    if isinstance(
        runtime.data.source,
        (SyntheticCourtStoresSourceConfig, SyntheticCourtDatasetVersionSourceConfig),
    ) != has_step_schedule(config):
        raise SemanticConfigurationError(
            "Explicit all-train sources and step_schedule must be selected together."
        )
    mixed = CourtMixedDataConfig.from_mapping(config.get("mixed"), runtime=runtime)
    return config, mixed


def validate_train_boundary(config: DictConfig) -> None:
    resolve_training_config(config)


class CourtDetectionTrainingRunner(BaseTrainingRunner):
    """Run the standard Court model against a mixed-source DataModule."""

    def __init__(self) -> None:
        super().__init__()
        self._mixed_config: CourtMixedDataConfig | None = None

    def run(self, config: Any) -> None:
        standard, mixed = resolve_training_config(config)
        self._mixed_config = mixed
        super().run(standard)

    def _require_mixed_config(self) -> CourtMixedDataConfig:
        if self._mixed_config is None:
            raise RuntimeError("Mixed Court source configuration is unresolved.")
        return self._mixed_config

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return CourtDetectionDataModule(
            config,
            mixed_config=self._require_mixed_config(),
        )

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        if not isinstance(datamodule, CourtDetectionDataModule):
            raise TypeError("Mixed Court training requires CourtDetectionDataModule.")
        CourtTrainingConfig.from_config(config)
        module = CourtDetectionLightningModule(
            config,
            target_bundle=datamodule.target_bundle_spec,
        )
        module.steps_per_epoch = steps_per_epoch
        if has_step_schedule(config):
            from src.tasks.court_detection.scaling.provenance import data_condition

            condition = data_condition(datamodule)
            module.save_hyperparameters({"data_condition": condition})
            runtime = self.validate_runtime_config(config)
            (runtime.run.output_dir / "data-condition.json").write_text(
                json.dumps(condition, indent=2) + "\n"
            )
        return module

    def build_trainer(
        self,
        config: Any,
        callbacks: list[Any],
        logger: TensorBoardLogger,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> pl.Trainer:
        if not has_step_schedule(config):
            return super().build_trainer(
                config, callbacks, logger, artifact_store=artifact_store
            )
        from src.tasks.court_detection.scaling.training import build_step_trainer

        accelerator, devices = self.select_devices(config)
        return build_step_trainer(
            config,
            callbacks,
            logger,
            accelerator=accelerator,
            devices=devices,
            artifact_store=artifact_store,
        )

    def callbacks_extra(
        self, config: Any, datamodule: pl.LightningDataModule, logger: TensorBoardLogger
    ) -> list[Any]:
        if not has_step_schedule(config):
            return cast(list[Any], super().callbacks_extra(config, datamodule, logger))
        from src.tasks.court_detection.scaling.training import step_checkpoints

        callbacks = step_checkpoints(config, logger)
        self._record_ckpt_dir_pointer(
            Path(logger.log_dir) / "checkpoints",
            self.validate_runtime_config(config).resolver,
        )
        return list(callbacks)

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        return CourtTrainingConfig.from_config(config).shared

    def prepare_config(self, config: Any) -> None:
        resolve_training_config(config)


__all__ = [
    "CourtDetectionTrainingRunner",
    "resolve_training_config",
    "validate_train_boundary",
]
