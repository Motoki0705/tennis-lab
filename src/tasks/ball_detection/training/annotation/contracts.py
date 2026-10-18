from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.ball_detection.data.windows.learning.contracts import WindowPolicy
from src.tasks.ball_detection.training.runtime.execution import CoordinateRuntime
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    after_validator,
)

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class LearningRuntime(Document):
    device: Literal["cpu", "cuda"]
    precision: Literal["fp32", "bf16"]
    num_workers: int = Field(ge=0)
    pin_memory: bool
    prefetch_factor: int = Field(ge=1)
    cpu_threads: int = Field(ge=1)
    compile_mode: Literal["off", "default"]
    compile_recompile_limit: Literal[8] = 8
    jpeg_decoder: Literal["opencv", "nvjpeg"]
    input_verification: Literal["upfront"] = "upfront"
    image_prefetch: bool

    def coordinate_runtime(self) -> CoordinateRuntime:
        runtime = CoordinateRuntime(**self.model_dump(exclude={"device"}))
        return runtime

    @_after
    def explicit_device(self) -> Self:
        self.coordinate_runtime()
        if self.device == "cpu" and (
            self.precision != "fp32"
            or self.pin_memory
            or self.compile_mode != "off"
            or self.jpeg_decoder != "opencv"
        ):
            raise ValueError("CPU learning requires explicit fp32/OpenCV/eager runtime")
        if self.device == "cuda" and self.precision != "bf16":
            raise ValueError("CUDA learning requires BF16")
        return self


class ReplaySampling(Document):
    previously_trained_manifest: ArtifactRef
    untrained_fraction: float = Field(gt=0, lt=1)


class BallLearningRecipe(Document):
    schema_version: Literal["ball.annotation-learning-recipe.v1"] = (
        "ball.annotation-learning-recipe.v1"
    )
    windows: WindowPolicy
    augmentation: ArtifactRef
    runtime: LearningRuntime
    epochs: int = Field(ge=2)
    freeze_epochs: int = Field(ge=1)
    windows_per_epoch: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    seed: int
    learning_rate: float = Field(gt=0)
    encoder_lr_ratio: float = Field(gt=0, le=1)
    warmup_updates: int = Field(ge=0)
    log_every: int = Field(ge=1)
    preview_clips: int = Field(ge=0)
    stress_evaluation: bool
    replay: ReplaySampling | None = None

    @_after
    def training_budget(self) -> Self:
        if (
            self.freeze_epochs >= self.epochs
            or self.warmup_updates >= self.total_updates
        ):
            raise ValueError("invalid freeze/warmup budget")
        return self

    @property
    def total_updates(self) -> int:
        return self.epochs * math.ceil(self.windows_per_epoch / self.batch_size)


class SourceRegressionLimit(Document):
    source: Literal["tracknet", "meiji", "chat_annotation"]
    minimum_frames_per_fps: int = Field(ge=1)
    maximum_regression_px: float = Field(ge=0)


class BallSelectionPolicy(Document):
    schema_version: Literal["ball.annotation-selection-policy.v1"] = (
        "ball.annotation-selection-policy.v1"
    )
    minimum_common_frames_per_fps: int = Field(ge=1)
    maximum_common_error_px: float = Field(ge=0)
    minimum_improvement_px: float = Field(ge=0)
    maximum_full_regression_px: float = Field(ge=0)
    source_regression_limits: tuple[SourceRegressionLimit, ...]

    @_after
    def unique_sources(self) -> Self:
        sources = tuple(limit.source for limit in self.source_regression_limits)
        if len(sources) != len(set(sources)):
            raise ValueError("source regression criteria must be unique")
        return self


class BalancedSelectionPolicy(Document):
    schema_version: Literal["ball.annotation-balanced-selection.v1"] = "ball.annotation-balanced-selection.v1"
    existing_weight: float = Field(default=0.5, ge=0.5, le=0.5)
    new_weight: float = Field(default=0.5, ge=0.5, le=0.5)
    strict_improvement: Literal[True] = True
    maximum_existing_regression_px: float = Field(default=0.0, ge=0.0, le=0.0)
    minimum_frames_per_fps: int = Field(ge=1)


class BallTrainingProvenance(Document):
    schema_version: Literal["ball.annotation-training-provenance.v1"] = (
        "ball.annotation-training-provenance.v1"
    )
    request: ArtifactRef
    intent: ArtifactRef
    prepared: ArtifactRef
    validation_manifest: ArtifactRef
    run_config: ArtifactRef
    latest_checkpoint: ArtifactRef
    selected_checkpoint: ArtifactRef
    best_record: ArtifactRef
    completed: ArtifactRef
    stress: ArtifactRef | None
    initial_state_sha256: Digest


class BallEvaluationMetrics(Document):
    schema_version: Literal["ball.annotation-evaluation-metrics.v1"] = (
        "ball.annotation-evaluation-metrics.v1"
    )
    request: ArtifactRef
    validation_manifest: ArtifactRef
    comparison: ArtifactRef
    test_manifest: ArtifactRef | None = None
