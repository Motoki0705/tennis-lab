"""Composition factory for one verified court model/task-adapter pair."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from torch import Tensor

from src.tasks.base.model_io import BoundModelIO, bind_model_io
from src.tasks.court_detection.configuration import CourtLoRAConfig, CourtTrainingConfig
from src.tasks.court_detection.model_io.adapters import (
    CourtDINOv3ExecutionBoundary,
    CourtKeypointModelIO,
    CourtLineModelIO,
    CourtModelIOAdapter,
    CourtSegmentationModelIO,
)
from src.tasks.court_detection.model_io.contracts import (
    CourtEncoderKind,
    CourtModelIOError,
    CourtModelSpec,
    CourtTask,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel


def build_court_detection_pair(
    config: object,
) -> BoundModelIO[Mapping[str, object], Tensor, Tensor]:
    """Select the model and exact kp/seg/line adapter once."""
    runtime = CourtTrainingConfig.from_config(config)
    spec = CourtModelSpec(
        task=cast(CourtTask, runtime.data.task),
        in_channels=runtime.model.in_channels,
        output_channels=runtime.data.output_channels,
        short_side=runtime.data.augmentation.val_short_side,
        encoder_kind=cast(CourtEncoderKind, runtime.model.encoder.name),
    )
    model = CourtHierarchicalModel.from_config(runtime.model)
    adapter = build_court_model_io(spec, runtime=runtime)
    adapter.validate_model_pair(model)
    return bind_model_io(model, adapter)


def build_court_model_io(
    spec: CourtModelSpec,
    *,
    runtime: CourtTrainingConfig,
) -> CourtModelIOAdapter:
    """Build the exact task adapter from a validated runtime contract."""
    execution_boundary = (
        CourtDINOv3ExecutionBoundary(
            frozen_backbone=(
                runtime.model.encoder.train_mode == "frozen"
                and not cast(CourtLoRAConfig, runtime.model.encoder.lora).enabled
            )
        )
        if spec.encoder_kind == "dinov3"
        else None
    )
    if spec.task == "kp":
        return CourtKeypointModelIO(
            spec,
            focal_gamma=runtime.loss.focal_gamma,
            execution_boundary=execution_boundary,
        )
    if spec.task == "seg":
        return CourtSegmentationModelIO(
            spec,
            ce_weight=runtime.loss.ce_weight,
            dice_weight=runtime.loss.dice_weight,
            execution_boundary=execution_boundary,
        )
    if spec.task == "line":
        return CourtLineModelIO(
            spec,
            bce_weight=runtime.loss.bce_weight,
            dice_weight=runtime.loss.dice_weight,
            pos_weight=runtime.loss.pos_weight,
            execution_boundary=execution_boundary,
        )
    raise CourtModelIOError(f"Unsupported court task {spec.task!r}.")


__all__ = ["build_court_detection_pair", "build_court_model_io"]
