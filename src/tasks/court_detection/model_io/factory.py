"""Composition factory for one verified Court model/target-bundle pair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, cast

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.configuration import (
    CourtLossConfig,
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
        raise TypeError(
            f"Unsupported Court model config: {type(runtime.model).__name__}."
        )
    return build_court_inference_pair(
        model_config=runtime.model,
        loss_config=runtime.loss,
        short_side=runtime.data.augmentation.val_short_side,
        pose_long_side=runtime.loss.pose.enabled,
        patch_size=runtime.data.augmentation.patch_size,
        target_bundle=target_bundle,
    )


def build_court_inference_pair(
    *,
    model_config: CourtModelConfig,
    loss_config: CourtLossConfig,
    short_side: int,
    pose_long_side: bool,
    patch_size: int,
    target_bundle: CourtTargetBundleSpec,
) -> CourtDetectionBoundModelIO:
    """Build the saved architecture without parsing training-only run/data fields."""
    spec = CourtModelSpec(
        target_bundle=target_bundle,
        in_channels=model_config.in_channels,
        short_side=short_side,
        encoder_kind=cast(CourtEncoderKind, model_config.encoder.name),
        pose_long_side=pose_long_side,
        patch_size=patch_size,
    )
    model = CourtHierarchicalModel.from_config(model_config, target_bundle)
    adapter = _build_adapter(spec, model_config=model_config, loss_config=loss_config)
    adapter.validate_model_pair(model)
    return cast(CourtDetectionBoundModelIO, bind_model_io(model, adapter))


def build_court_model_io(
    spec: CourtModelSpec,
    *,
    runtime: CourtTrainingConfig,
) -> CourtModelIOAdapter | CourtPoseModelIOAdapter:
    """Build the one bundle-aware adapter from a validated runtime contract."""
    if not isinstance(runtime.model, CourtModelConfig):
        raise TypeError("build_court_model_io requires CourtModelConfig.")
    return _build_adapter(spec, model_config=runtime.model, loss_config=runtime.loss)


def _build_adapter(
    spec: CourtModelSpec,
    *,
    model_config: CourtModelConfig,
    loss_config: CourtLossConfig,
) -> CourtModelIOAdapter | CourtPoseModelIOAdapter:
    lora = model_config.encoder.lora
    lora_enabled = lora is not None and lora.enabled
    execution_boundary = (
        CourtDINOv3ExecutionBoundary(
            frozen_backbone=(
                model_config.encoder.train_mode == "frozen" and not lora_enabled
            )
        )
        if spec.encoder_kind == "dinov3"
        else None
    )
    if loss_config.pose.enabled:
        return CourtPoseModelIOAdapter(
            spec,
            loss_config=loss_config,
            execution_boundary=execution_boundary,
        )
    return CourtModelIOAdapter(
        spec,
        loss_config=loss_config,
        execution_boundary=execution_boundary,
    )


__all__ = [
    "CourtDetectionBoundModelIO",
    "CourtDetectionRawOutput",
    "build_court_detection_pair",
    "build_court_inference_pair",
    "build_court_model_io",
]
