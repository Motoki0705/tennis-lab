"""Learning identities; task adapters own data formats and metric semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, Self, TypeVar, cast

from pydantic import Field

from ..contracts import (
    ArtifactRef,
    Digest,
    Document,
    Identifier,
    ModelRef,
    ProductOutcome,
    after_validator,
)
from ..execution.contracts import QueueConfiguration

_M = TypeVar("_M")


class _Validator(Protocol):
    def __call__(self, function: Callable[[_M], _M]) -> Callable[[_M], _M]: ...


_after = cast(_Validator, after_validator)


class TaskProducts(Document):
    task: Identifier
    products: tuple[ProductOutcome, ...] = Field(min_length=1)


class CampaignCompletion(Document):
    plan: ArtifactRef
    tasks: tuple[TaskProducts, ...] = Field(min_length=1)

    @property
    def complete(self) -> bool:
        return all(p.annotation_complete for t in self.tasks for p in t.products)


class DatasetPartition(Document):
    """Membership extracted and checked by the task, without decoding teachers."""

    manifest: ArtifactRef
    split: Literal["train", "validation", "test"]
    sources: tuple[Digest, ...] = Field(min_length=1)
    clips: tuple[Digest, ...] = Field(min_length=1)

    @_after
    def canonical_membership(self) -> Self:
        for members in (self.sources, self.clips):
            if tuple(sorted(set(members))) != members:
                raise ValueError("partition identities must be sorted and unique")
        return self


class FixedEvaluation(Document):
    task: Identifier
    validation: DatasetPartition
    test: DatasetPartition | None

    @_after
    def split_names(self) -> Self:
        if self.validation.split != "validation" or (
            self.test is not None and self.test.split != "test"
        ):
            raise ValueError("fixed evaluation must explicitly separate val and test")
        return self


class ModelState(Document):
    generation: int = Field(ge=1)
    model: ModelRef


class LearningIntent(Document):
    completion: ArtifactRef
    task: Identifier
    publication: ArtifactRef
    train: DatasetPartition
    parent: ModelState
    recipe: ArtifactRef
    fixed_evaluation: ArtifactRef
    selection_policy: ArtifactRef | None

    @_after
    def coherent(self) -> Self:
        if self.parent.model.task != self.task or self.train.split != "train":
            raise ValueError("learning task/parent or training split mismatch")
        return self


class StageInput(Document):
    intent: ArtifactRef
    stage: Literal["train", "evaluate"]
    attempt: int = Field(ge=1)
    training: ArtifactRef | None
    resume_checkpoint: ArtifactRef | None = None

    @_after
    def dependencies(self) -> Self:
        if (self.stage == "evaluate") != (self.training is not None):
            raise ValueError("only evaluation takes a completed training receipt")
        if self.resume_checkpoint is not None and self.stage != "train":
            raise ValueError("only training can resume its own checkpoint")
        return self


class TrainingReceipt(Document):
    request: ArtifactRef
    candidate: ModelRef
    provenance: ArtifactRef


class EvaluationReceipt(Document):
    request: ArtifactRef
    training: ArtifactRef
    candidate: ModelRef
    baseline: ModelState
    validation: DatasetPartition
    metrics: ArtifactRef


class CompletedStage(Document):
    request: ArtifactRef
    result: ArtifactRef
    execution: ArtifactRef


class SelectionVerdict(Document):
    # References a CompletedStage, never an unverified metrics file.
    evaluation: ArtifactRef
    policy: ArtifactRef
    status: Literal["passed", "rejected"]
    reason: str = Field(min_length=1)


class SelectionRecord(Document):
    verdict: ArtifactRef
    status: Literal["adopted", "rejected", "reevaluation_required"]
    expected: ModelState
    candidate: ModelRef
    observed: ModelState
    adopted: ModelState | None
    reason: str = Field(min_length=1)


class StageCost(Document):
    stage: Identifier
    pss_bytes: int = Field(gt=0)
    storage_bytes: int = Field(ge=0)
    gpu_seconds: float = Field(gt=0)
    timeout_seconds: float = Field(gt=0)

    @_after
    def teardown_margin(self) -> Self:
        if self.timeout_seconds + 10 > self.gpu_seconds:
            raise ValueError(
                "GPU reservation must include ten seconds of teardown margin"
            )
        return self


class LearningRuntime(Document):
    scope: ArtifactRef
    queue: QueueConfiguration
    train: StageCost
    evaluate: StageCost


class LearningRegistration(Document):
    task: Identifier
    status: Literal[
        "ready", "annotation_incomplete", "no_additional_teachers", "not_configured"
    ]
    intent: ArtifactRef | None
    reason: str


class LearningOutcome(Document):
    intent: ArtifactRef
    state: Literal[
        "training_pending",
        "training",
        "evaluation_pending",
        "evaluating",
        "failed",
        "not_configured",
        "rejected",
        "adopted",
        "reevaluation_required",
    ]
    stage: Literal["train", "evaluate", "select"]
    launch: ArtifactRef | None = None
    training: ArtifactRef | None = None
    evaluation: ArtifactRef | None = None
    selection: ArtifactRef | None = None
    reason: str | None = None


class Retry(Document):
    previous: ArtifactRef
    reason: str = Field(min_length=1)
    resume_checkpoint: ArtifactRef | None


class StageFailure(Document):
    request: ArtifactRef
    launch: ArtifactRef
    reason: str = Field(min_length=1)
