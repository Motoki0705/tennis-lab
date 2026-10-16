"""Training runner for fixed-ratio mixed Court source batches."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.data.mixed import (
    CourtMixedDataConfig,
)
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)
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
        return module

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        return CourtTrainingConfig.from_config(config).shared

    def prepare_config(self, config: Any) -> None:
        resolve_training_config(config)


__all__ = [
    "CourtDetectionTrainingRunner",
    "resolve_training_config",
    "validate_train_boundary",
]
