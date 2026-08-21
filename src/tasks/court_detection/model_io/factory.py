"""Composition factory for one verified Court model/target-bundle pair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.configuration import CourtTrainingConfig
from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtEncoderKind,
    CourtLogits,
    CourtModelSpec,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


def build_court_detection_pair(
    config: object,
    *,
    target_bundle: CourtTargetBundleSpec,
) -> BoundModelIO[Mapping[str, object], CourtLogits, CourtLogits]:
    """Bind one shared-trunk model to the exact resolved target bundle."""
    runtime = CourtTrainingConfig.from_config(config)
    spec = CourtModelSpec(
        target_bundle=target_bundle,
        in_channels=runtime.model.in_channels,
        short_side=runtime.data.augmentation.val_short_side,
        encoder_kind=cast(CourtEncoderKind, runtime.model.encoder.name),
    )
    model = CourtHierarchicalModel.from_config(runtime.model, target_bundle)
    adapter = build_court_model_io(spec, runtime=runtime)
    adapter.validate_model_pair(model)
    return bind_model_io(model, adapter)


def build_court_model_io(
    spec: CourtModelSpec,
    *,
    runtime: CourtTrainingConfig,
) -> CourtModelIOAdapter:
    """Build the one bundle-aware adapter from a validated runtime contract."""
    lora = runtime.model.encoder.lora
    lora_enabled = lora is not None and lora.enabled
    execution_boundary = (
        CourtDINOv3ExecutionBoundary(
            frozen_backbone=(
                runtime.model.encoder.train_mode == "frozen"
                and not lora_enabled
            )
        )
        if spec.encoder_kind == "dinov3"
        else None
    )
    return CourtModelIOAdapter(
        spec,
        loss_config=runtime.loss,
        execution_boundary=execution_boundary,
    )


__all__ = ["build_court_detection_pair", "build_court_model_io"]
