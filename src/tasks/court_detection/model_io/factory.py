"""Composition factory for one verified Court model/target-bundle pair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.configuration import (
    CourtInferenceConfig,
    CourtModelConfig,
    CourtTrainingConfig,
)
from src.tasks.court_detection.data.contracts import CourtTargetBundleSpec
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtModelIOAdapter,
    CourtPoseModelIOAdapter,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtDecodedOutput,
    CourtEncoderKind,
    CourtLogits,
    CourtModelOutput,
    CourtModelSpec,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.utils.configuration import PathResolver

CourtDetectionRawOutput: TypeAlias = CourtLogits | CourtModelOutput
CourtDetectionBoundModelIO: TypeAlias = BoundModelIO[
    Mapping[str, object],
    CourtDetectionRawOutput,
    CourtLogits | CourtDecodedOutput,
]


def build_court_detection_pair(
    config: object,
    *,
    target_bundle: CourtTargetBundleSpec,
) -> CourtDetectionBoundModelIO:
    """Bind one hierarchical model and its exact bundle-aware adapter."""
    runtime = CourtTrainingConfig.from_config(config)
    if not isinstance(runtime.model, CourtModelConfig):
        raise TypeError(f"Unsupported Court model config: {type(runtime.model).__name__}.")
    spec = CourtModelSpec(
        target_bundle=target_bundle,
        in_channels=runtime.model.in_channels,
        short_side=runtime.data.augmentation.val_short_side,
        pose_long_side=runtime.loss.pose.enabled,
        patch_size=runtime.data.augmentation.patch_size,
        encoder_kind=cast(CourtEncoderKind, runtime.model.encoder.name),
    )
    model = CourtHierarchicalModel.from_config(runtime.model, target_bundle)
    adapter = build_court_model_io(spec, runtime=runtime)
    adapter.validate_model_pair(model)
    return cast(CourtDetectionBoundModelIO, bind_model_io(model, adapter))


def build_court_model_io(
    spec: CourtModelSpec,
    *,
    runtime: CourtTrainingConfig | CourtInferenceConfig,
) -> CourtModelIOAdapter | CourtPoseModelIOAdapter:
    """Build the one bundle-aware adapter from a validated runtime contract."""
    if not isinstance(runtime.model, CourtModelConfig):
        raise TypeError("build_court_model_io requires CourtModelConfig.")
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
    if runtime.loss.pose.enabled:
        return CourtPoseModelIOAdapter(
            spec,
            loss_config=runtime.loss,
            execution_boundary=execution_boundary,
        )
    return CourtModelIOAdapter(
        spec,
        loss_config=runtime.loss,
        execution_boundary=execution_boundary,
    )


def build_court_inference_pair(
    config: object, *, resolver: PathResolver, target_bundle: CourtTargetBundleSpec
) -> CourtDetectionBoundModelIO:
    """Build the exact checkpoint model without constructing training services."""
    runtime = CourtInferenceConfig.from_config(config, resolver=resolver)
    spec = CourtModelSpec(
        target_bundle=target_bundle,
        in_channels=runtime.model.in_channels,
        short_side=runtime.short_side,
        pose_long_side=runtime.pose_long_side,
        patch_size=runtime.patch_size,
        encoder_kind=cast(CourtEncoderKind, runtime.model.encoder.name),
    )
    model = CourtHierarchicalModel.from_config(runtime.model, target_bundle)
    return cast(CourtDetectionBoundModelIO, bind_model_io(model, build_court_model_io(spec, runtime=runtime)))


__all__ = [
    "CourtDetectionBoundModelIO",
    "CourtDetectionRawOutput",
    "build_court_detection_pair",
    "build_court_model_io",
]
