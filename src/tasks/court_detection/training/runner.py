"""Training runner for composable Court detection."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytorch_lightning as pl
import torch

from src.tasks.base.configuration import TrainingRuntimeConfig
from src.tasks.base.training.runner import BaseTrainingRunner
from src.tasks.court_detection.configuration import (
    CourtQueryLossConfig,
    CourtQueryModelConfig,
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.bundle_state import (
    deserialize_query_checkpoint_state,
)
from src.tasks.court_detection.data.datamodule import CourtDetectionDataModule
from src.tasks.court_detection.training.lightning_module import (
    CourtDetectionLightningModule,
)


class CourtDetectionTrainingRunner(BaseTrainingRunner):
    """Construct the model only after the DataModule resolves its target bundle."""

    def build_datamodule(self, config: Any) -> pl.LightningDataModule:
        return CourtDetectionDataModule(config)

    def build_lightning_module(
        self,
        config: Any,
        datamodule: pl.LightningDataModule,
        *,
        steps_per_epoch: int | None = None,
    ) -> pl.LightningModule:
        if not isinstance(datamodule, CourtDetectionDataModule):
            raise TypeError(
                "Court training requires CourtDetectionDataModule to resolve "
                "the target bundle before model construction."
            )
        runtime = CourtTrainingConfig.from_config(config)
        query_checkpoint_state = self._validated_query_checkpoint_state(
            runtime,
        )
        module = CourtDetectionLightningModule(
            config,
            target_bundle=datamodule.target_bundle_spec,
            query_checkpoint_state=query_checkpoint_state,
        )
        module.steps_per_epoch = steps_per_epoch
        return module

    def validate_runtime_config(self, config: Any) -> TrainingRuntimeConfig:
        return CourtTrainingConfig.from_config(config).shared

    def prepare_config(self, config: Any) -> None:
        """Reject query checkpoint/config mismatches before run side effects."""
        runtime = CourtTrainingConfig.from_config(config)
        self._validated_query_checkpoint_state(runtime)

    def _validated_query_checkpoint_state(
        self,
        runtime: CourtTrainingConfig,
    ) -> Mapping[str, object] | None:
        checkpoint_path = _configured_checkpoint_path(runtime)
        cache_key: object = checkpoint_path
        if (
            isinstance(runtime.model, CourtQueryModelConfig)
            and isinstance(runtime.loss, CourtQueryLossConfig)
        ):
            cache_key = (
                checkpoint_path,
                runtime.model.heads.dense_targets,
                runtime.loss.name,
                runtime.loss.pose.enabled,
                runtime.loss.consistency,
            )
        if getattr(self, "_query_checkpoint_cache_key", None) == cache_key:
            return getattr(self, "_query_checkpoint_cache_state", None)
        state = _query_checkpoint_state(
            runtime,
            require_query_identity=isinstance(runtime.model, CourtQueryModelConfig),
        )
        self._query_checkpoint_cache_key = cache_key
        self._query_checkpoint_cache_state = state
        return state


def _query_checkpoint_state(
    runtime: CourtTrainingConfig,
    *,
    require_query_identity: bool,
) -> Mapping[str, object] | None:
    checkpoint_path = _configured_checkpoint_path(runtime)
    if checkpoint_path is None:
        return None
    checkpoint = _load_checkpoint(checkpoint_path)
    hyperparameters = checkpoint.get("hyper_parameters")
    query_state: object = (
        hyperparameters.get("query_checkpoint_state")
        if isinstance(hyperparameters, Mapping)
        else None
    )
    if not require_query_identity:
        return None
    if not isinstance(runtime.model, CourtQueryModelConfig):  # pragma: no cover
        raise TypeError("Query checkpoint validation requires a query model.")
    if not isinstance(runtime.loss, CourtQueryLossConfig):  # pragma: no cover
        raise TypeError("Query checkpoint validation requires a query loss.")
    if not isinstance(query_state, Mapping):
        raise ValueError(
            "Court query checkpoint lacks its required versioned identity."
        )
    restored = deserialize_query_checkpoint_state(query_state)
    expected_consistency = (
        runtime.loss.consistency if runtime.loss.consistency.enabled else None
    )
    expected_subset = (
        *runtime.model.heads.dense_targets,
        *(("pose",) if runtime.loss.pose.enabled else ()),
    )
    if (
        restored.loss_config_name != runtime.loss.name
        or restored.pose_supervision != runtime.loss.pose.enabled
        or restored.supervision_subset != expected_subset
        or restored.consistency != expected_consistency
    ):
        raise ValueError(
            "Court query supervision identity disagrees with checkpoint."
        )
    return query_state


def _configured_checkpoint_path(runtime: CourtTrainingConfig) -> Path | None:
    """Select the explicitly configured, mutually exclusive checkpoint field."""
    resume = cast(Path | None, runtime.shared.run.resume)
    if resume is not None:
        return resume
    return cast(Path | None, runtime.shared.run.init_weights)


def _load_checkpoint(path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Court checkpoint {path} must contain a mapping payload.")
    return checkpoint


__all__ = ["CourtDetectionTrainingRunner"]
