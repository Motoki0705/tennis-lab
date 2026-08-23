"""Composition factory for one verified Court model/target-bundle pair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.configuration import (
    CourtLossConfig,
    CourtModelConfig,
    CourtQueryLossConfig,
    CourtQueryModelConfig,
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtModelIOAdapter,
    CourtQueryDINOv3ExecutionBoundary,
    CourtQueryModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtEncoderKind,
    CourtLogits,
    CourtModelSpec,
    CourtQueryModelSpec,
    CourtQueryRawOutput,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.tasks.court_detection.models.query_encoder.model import CourtQueryEncoderModel

CourtDetectionRawOutput: TypeAlias = CourtLogits | CourtQueryRawOutput
CourtDetectionBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object],
    CourtDetectionRawOutput,
    CourtDetectionRawOutput,
]


def build_court_detection_pair(
    config: object,
    *,
    target_bundle: CourtTargetBundleSpec,
) -> CourtDetectionBoundModelIO:
    """Bind the discriminated legacy or additive model to one exact bundle."""
    runtime = CourtTrainingConfig.from_config(config)
    if isinstance(runtime.model, CourtQueryModelConfig):
        query_spec = CourtQueryModelSpec(
            target_bundle=target_bundle,
            in_channels=runtime.model.in_channels,
            short_side=runtime.data.augmentation.val_short_side,
        )
        query_model = CourtQueryEncoderModel.from_config(runtime.model, target_bundle)
        query_adapter = build_court_query_model_io(query_spec, runtime=runtime)
        query_adapter.validate_model_pair(query_model)
        return cast(
            CourtDetectionBoundModelIO,
            bind_model_io(query_model, query_adapter),
        )
    if not isinstance(runtime.model, CourtModelConfig):
        raise TypeError(f"Unsupported Court model config: {type(runtime.model).__name__}.")
    spec = CourtModelSpec(
        target_bundle=target_bundle,
        in_channels=runtime.model.in_channels,
        short_side=runtime.data.augmentation.val_short_side,
        encoder_kind=cast(CourtEncoderKind, runtime.model.encoder.name),
    )
    model = CourtHierarchicalModel.from_config(runtime.model, target_bundle)
    adapter = build_court_model_io(spec, runtime=runtime)
    adapter.validate_model_pair(model)
    return cast(CourtDetectionBoundModelIO, bind_model_io(model, adapter))


def build_court_model_io(
    spec: CourtModelSpec,
    *,
    runtime: CourtTrainingConfig,
) -> CourtModelIOAdapter:
    """Build the one bundle-aware adapter from a validated runtime contract."""
    if not isinstance(runtime.model, CourtModelConfig):
        raise TypeError("build_court_model_io only accepts the legacy model config.")
    if not isinstance(runtime.loss, CourtLossConfig):
        raise TypeError("build_court_model_io only accepts the legacy loss config.")
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


def build_court_query_model_io(
    spec: CourtQueryModelSpec,
    *,
    runtime: CourtTrainingConfig,
) -> CourtQueryModelIOAdapter:
    """Build the raw query-output seam without claiming training integration."""
    if not isinstance(runtime.model, CourtQueryModelConfig):
        raise TypeError(
            "build_court_query_model_io requires the query-model configuration."
        )
    if not isinstance(runtime.loss, CourtQueryLossConfig):
        raise TypeError(
            "build_court_query_model_io requires the query loss configuration."
        )
    return CourtQueryModelIOAdapter(
        spec,
        loss_config=runtime.loss,
        execution_boundary=CourtQueryDINOv3ExecutionBoundary(
            frozen_backbone=(
                runtime.model.backbone.train_mode == "frozen"
                and not runtime.model.backbone.lora.enabled
            )
        ),
    )


__all__ = [
    "CourtDetectionBoundModelIO",
    "CourtDetectionRawOutput",
    "build_court_detection_pair",
    "build_court_model_io",
    "build_court_query_model_io",
]
