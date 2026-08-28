"""Training runner for fixed-ratio mixed Court source batches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.mixed import (
    CourtMixedDataConfig,
    MixedCourtDetectionDataModule,
)
from src.tasks.court_detection.training.lightning_module_mixed import (
    MixedCourtDetectionLightningModule,
)
from src.tasks.court_detection.training.runner import CourtDetectionTrainingRunner
from src.utils.configuration import PathRole


def resolve_mixed_training_config(
    config: object,
) -> tuple[DictConfig, CourtMixedDataConfig]:
    """Split the task-standard config from the additional mixed-source section."""
    if not isinstance(config, DictConfig):
        raise TypeError("Mixed Court training requires a Hydra DictConfig.")
    unresolved = OmegaConf.to_container(config, resolve=False)
    if not isinstance(unresolved, dict) or "mixed" not in unresolved:
        raise ValueError("Mixed Court training requires a top-level mixed section.")
    standard_mapping = dict(unresolved)
    standard_mapping.pop("mixed")
    standard = OmegaConf.create(standard_mapping)
    runtime = CourtTrainingConfig.from_config(standard)

    mixed_node = config.get("mixed")
    resolved_mixed = OmegaConf.to_container(mixed_node, resolve=True)
    mixed = CourtMixedDataConfig.from_mapping(resolved_mixed, runtime=runtime)
    return standard, mixed


def validate_mixed_train_boundary(config: DictConfig) -> None:
    resolve_mixed_training_config(config)


class MixedCourtDetectionTrainingRunner(CourtDetectionTrainingRunner):
    """Run the standard Court model against a mixed-source DataModule."""

    def __init__(self) -> None:
        super().__init__()
        self._mixed_config: CourtMixedDataConfig | None = None
        self._full_config: DictConfig | None = None

    def run(self, config: Any) -> None:
        standard, mixed = resolve_mixed_training_config(config)
        self._mixed_config = mixed
        self._full_config = cast(
            DictConfig,
            OmegaConf.create(OmegaConf.to_container(config, resolve=False)),
        )
        super().run(standard)

    def _require_mixed_config(self) -> CourtMixedDataConfig:
        if self._mixed_config is None:
            raise RuntimeError("Mixed Court source configuration is unresolved.")
        return self._mixed_config

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return MixedCourtDetectionDataModule(
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
        if not isinstance(datamodule, MixedCourtDetectionDataModule):
            raise TypeError(
                "Mixed Court training requires MixedCourtDetectionDataModule."
            )
        CourtTrainingConfig.from_config(config)
        module = MixedCourtDetectionLightningModule(
            config,
            target_bundle=datamodule.target_bundle_spec,
        )
        module.steps_per_epoch = steps_per_epoch
        return module

    def save_config(self, config: Any, output_dir: Path) -> None:
        super().save_config(config, output_dir)
        if self._full_config is None:
            return
        runtime = CourtTrainingConfig.from_config(config).shared
        config_path = runtime.resolver.resolve_beneath(
            PathRole.OUTPUT,
            output_dir,
            "config.yaml",
        )
        OmegaConf.save(self._full_config, config_path)


__all__ = [
    "MixedCourtDetectionTrainingRunner",
    "resolve_mixed_training_config",
    "validate_mixed_train_boundary",
]
