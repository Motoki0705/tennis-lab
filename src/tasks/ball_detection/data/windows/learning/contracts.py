from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from src.tasks.ball_detection.data.windows.intervals import PlayIntervalConfig
from src.tasks.ball_detection.data.windows.temporal import TemporalSamplingConfig
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


class WindowPolicy(Document):
    window_length: Literal[32] = 32
    window_stride: int = Field(ge=1, le=32)
    max_gap_seconds: float = Field(ge=0)
    min_presence_fraction: float = Field(gt=0, le=1)
    min_observed_frames: int = Field(ge=1, le=32)
    timestamp_gap_factor: float = Field(gt=1)
    frame_steps: tuple[int, ...] = Field(min_length=1)

    def play(self) -> PlayIntervalConfig:
        return PlayIntervalConfig(**self.model_dump(exclude={"frame_steps"}))

    def sampling(self) -> TemporalSamplingConfig:
        return TemporalSamplingConfig(self.frame_steps)

    @_after
    def valid_sampling(self) -> Self:
        self.play()
        self.sampling()
        return self


class ClipCoverage(Document):
    clip_id: Digest
    source_id: Digest
    store_clip_id: str
    added: bool
    native_frames: int = Field(gt=0)
    coordinate_teacher_frames: int = Field(ge=0)
    windows_by_frame_step: dict[str, int]
    status: Literal["eligible", "no_eligible_window"]

    @_after
    def valid_counts(self) -> Self:
        if (self.coordinate_teacher_frames > self.native_frames
                or any(count < 0 for count in self.windows_by_frame_step.values())
                or (self.status == "eligible") != bool(sum(self.windows_by_frame_step.values()))):
            raise ValueError("invalid clip teacher/window coverage")
        return self


class TrainingWindows(Document):
    schema_version: Literal["ball.annotation-training-windows.v1"] = "ball.annotation-training-windows.v1"
    publication: ArtifactRef
    snapshot: ArtifactRef
    policy: WindowPolicy
    manifest: ArtifactRef | None
    coverage: tuple[ClipCoverage, ...]
    old_train_clips: int = Field(ge=0)
    added_train_clips: int = Field(gt=0)
    added_eligible_windows: int = Field(ge=0)
    status: Literal["ready", "no_additional_coordinate_teachers"]

    @_after
    def coherent(self) -> Self:
        added = tuple(row for row in self.coverage if row.added)
        if len({row.clip_id for row in self.coverage}) != len(self.coverage) or any(
            set(row.windows_by_frame_step) != {str(step) for step in self.policy.frame_steps} for row in self.coverage
        ):
            raise ValueError("training coverage membership or sampling differs")
        if (len(added) != self.added_train_clips
                or len(self.coverage) - len(added) != self.old_train_clips
                or sum(sum(r.windows_by_frame_step.values()) for r in added) != self.added_eligible_windows):
            raise ValueError("training coverage counts disagree")
        ready = self.added_eligible_windows > 0
        if (self.manifest is not None) != ready or (self.status == "ready") != ready:
            raise ValueError("ready training requires additional eligible coordinate windows")
        return self


class HoldoutWindows(Document):
    schema_version: Literal["ball.annotation-holdout-windows.v1"] = "ball.annotation-holdout-windows.v1"
    snapshot: ArtifactRef
    manifest: ArtifactRef
    split: Literal["validation", "test"]
    existing: ArtifactRef | None = None


class BallEvaluationRecipe(Document):
    schema_version: Literal["ball.annotation-evaluation-recipe.v1"] = "ball.annotation-evaluation-recipe.v1"
    windows: WindowPolicy
    cohorts: Literal["existing_and_new"] = "existing_and_new"
    checkpoint_selection: Literal["balanced_validation"] = "balanced_validation"
    final_comparison: Literal["test_once_after_selection"] = "test_once_after_selection"
