"""Pinned player learning inputs; numerical budgets/policies have no defaults."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    ModelRef,
    after_validator,
)
from src.tasks.player_detection.configuration import (
    AugmentationConfig,
    DetectionEvaluationConfig,
    InputSizeConfig,
)
from src.tasks.player_detection.data.snapshots.contracts import SelectionPolicy

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class InputSize(Document):
    short_side: int = Field(ge=32)
    max_long_side: int = Field(ge=32)

    @_after
    def bounds(self) -> Self:
        if self.short_side > self.max_long_side:
            raise ValueError("input short side exceeds long-side cap")
        return self

    def configuration(self) -> InputSizeConfig:
        return InputSizeConfig(self.short_side, self.max_long_side)


class Augmentation(Document):
    hflip_prob: float = Field(ge=0, le=1)
    short_side_choices: tuple[int, ...] = Field(min_length=1)
    brightness: float = Field(ge=0, le=1)
    contrast: float = Field(ge=0, le=1)
    saturation: float = Field(ge=0, le=1)

    def configuration(self, input_size: InputSize) -> AugmentationConfig:
        return AugmentationConfig.from_mapping(
            self.model_dump(mode="json"),
            "augmentation",
            max_long_side=input_size.max_long_side,
        )


class OperatingPoint(Document):
    score_threshold: float = Field(gt=0, lt=1)
    iou_threshold: float = Field(gt=0, le=1)
    max_detections: int = Field(ge=1)

    def configuration(self) -> DetectionEvaluationConfig:
        return DetectionEvaluationConfig(
            self.score_threshold, self.iou_threshold, self.max_detections
        )


class PlayerLearningRecipe(Document):
    schema_version: Literal["player.annotation-learning-recipe.v1"] = (
        "player.annotation-learning-recipe.v1"
    )
    # Complete shared training section as JSON, validated by BaseTrainingConfig.
    training: ArtifactRef
    selection: SelectionPolicy
    input_size: InputSize
    augmentation: Augmentation
    operating_point: OperatingPoint
    batch_size: int = Field(ge=1)
    evaluation_batch_size: int = Field(ge=1)
    train_samples_per_epoch: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    pin_memory: bool
    seed: int = Field(ge=0, le=2**32 - 1)
    use_checkpoint: bool
    backbone_freeze_keywords: tuple[str, ...]
    backbone_lr_scale: float = Field(gt=0, le=1)
    device: Literal["cuda"] = "cuda"

    @_after
    def complete_batches(self) -> Self:
        self.augmentation.configuration(self.input_size)
        if self.selection.frame_stride != 1:
            raise ValueError(
                "player training uses the complete eligible native-frame pool"
            )
        if self.train_samples_per_epoch < self.batch_size:
            raise ValueError("training must contain at least one full batch")
        if any(not item.strip() for item in self.backbone_freeze_keywords):
            raise ValueError("backbone freeze keywords must be nonempty")
        return self


class PlayerSelectionPolicy(Document):
    schema_version: Literal["player.annotation-selection-policy.v1"] = (
        "player.annotation-selection-policy.v1"
    )
    minimum_validation_frames: int = Field(ge=1)
    minimum_target_instances: int = Field(ge=1)
    minimum_map: float = Field(ge=0, le=1)
    minimum_map_improvement: float = Field(ge=0, le=1)
    maximum_precision_regression: float = Field(ge=0, le=1)
    maximum_recall_regression: float = Field(ge=0, le=1)
    # None means this criterion is explicitly unused, not an assumed zero FP rate.
    maximum_negative_fp_per_image: float | None = Field(ge=0)
    minimum_negative_frames: int = Field(ge=0)

    @_after
    def negative_support(self) -> Self:
        if (
            self.maximum_negative_fp_per_image is not None
            and self.minimum_negative_frames < 1
        ):
            raise ValueError(
                "negative-frame FP criterion requires positive minimum support"
            )
        return self


class PlayerTrainingProvenance(Document):
    request: ArtifactRef
    intent: ArtifactRef
    run_binding: ArtifactRef
    initial_configuration: ArtifactRef
    latest_checkpoint: ArtifactRef
    selected_checkpoint: ArtifactRef
    completed: ArtifactRef


class RunBinding(Document):
    intent_digest: Digest
    recipe: ArtifactRef
    initial_weights: ArtifactRef
    parent_model_digest: Digest
    train_partition: ArtifactRef
    validation_partition: ArtifactRef
    initial_configuration: ArtifactRef
    output: str
    total_updates: int = Field(ge=1)
    epochs: int = Field(ge=1)
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")


class TrainingComplete(Document):
    binding: ArtifactRef
    latest_checkpoint: ArtifactRef
    selected_checkpoint: ArtifactRef
    export: ArtifactRef
    best_validation_map: float = Field(ge=0, le=1)


class PlayerEvaluationMetrics(Document):
    request: ArtifactRef
    comparison: ArtifactRef


class ModelEvaluation(Document):
    model: ModelRef
    predictions: ArtifactRef
    metrics: ArtifactRef


class PlayerComparison(Document):
    schema_version: Literal["player.fixed-validation-comparison.v1"] = (
        "player.fixed-validation-comparison.v1"
    )
    request: ArtifactRef
    validation_partition: ArtifactRef
    support_digest: Digest
    provenance: ArtifactRef
    summary: ArtifactRef
    baseline: ModelEvaluation
    candidate: ModelEvaluation
    label_scope: Literal["pinned_selected_labels_only"] = "pinned_selected_labels_only"
