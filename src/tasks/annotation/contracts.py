"""Composition inputs: fixed resource costs, concrete operations and campaign runtime."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from src.tasks.base.annotation.collection.contracts import (
    HistoricalSource,
    SourceCandidate,
)
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    Digest,
    Document,
    Identifier,
    PositiveInt,
)
from src.tasks.base.annotation.execution.contracts import QueueConfiguration


class StageCost(Document):
    stage: Identifier
    pss_bytes: PositiveInt
    storage_bytes: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    gpu_seconds: float = Field(ge=0)


class Runtime(Document):
    schema_version: Literal["annotation.runtime.v1"] = "annotation.runtime.v1"
    plan: ArtifactRef
    scope: ArtifactRef
    queue: QueueConfiguration
    codex_binary: str = Field(min_length=1)
    codex_home: str = Field(min_length=1)
    costs: tuple[StageCost, ...] = Field(min_length=1)
    maximum_services_per_attempt: PositiveInt = 100
    maximum_clips_per_prediction_batch: PositiveInt = 4

    def cost(self, stage: str) -> StageCost:
        matching = [cost for cost in self.costs if cost.stage == stage]
        if len(matching) != 1:
            raise ValueError(
                f"exactly one frozen resource cost is required for {stage}"
            )
        return matching[0]


class HistoryEntry(Document):
    source: HistoricalSource
    evidence: ArtifactRef


class HistoryManifest(Document):
    entries: tuple[HistoryEntry, ...]


class CandidateManifest(Document):
    candidates: tuple[SourceCandidate, ...]


class HistoryOperation(Document):
    action: Literal["history"] = "history"
    manifest: ArtifactRef


class AcquireOperation(Document):
    action: Literal["acquire"] = "acquire"
    source_id: Digest
    retry: bool = False


class ScreenOperation(Document):
    action: Literal["screen_prepare"] = "screen_prepare"
    source_id: Digest


class PrepareOperation(Document):
    action: Literal["prepare"] = "prepare"
    source_id: Digest


class PreviewOperation(Document):
    action: Literal["screen_preview"] = "screen_preview"
    service_request: ArtifactRef


class PredictOperation(Document):
    action: Literal["predict"] = "predict"
    task: Literal["ball", "player"]
    clips: tuple[ArtifactRef, ...] = Field(min_length=1)


class ReviewOperation(Document):
    action: Literal["review_service"] = "review_service"
    service_request: ArtifactRef


class PublishOperation(Document):
    action: Literal["publish_ball"] = "publish_ball"
    review: ArtifactRef


class BootstrapModelOperation(Document):
    action: Literal["bootstrap_model"] = "bootstrap_model"
    task: Identifier
    evidence: ArtifactRef


class OperationInput(Document):
    plan: ArtifactRef
    operation: Annotated[
        HistoryOperation
        | AcquireOperation
        | ScreenOperation
        | PrepareOperation
        | PreviewOperation
        | ReviewOperation
        | PublishOperation
        | BootstrapModelOperation
        | PredictOperation,
        Field(discriminator="action"),
    ]

    @property
    def logical_id(self) -> str:
        """Retry authorizes an acquisition attempt; it does not identify new work."""
        result: str
        if isinstance(self.operation, AcquireOperation):
            result = self.model_copy(
                update={"operation": self.operation.model_copy(update={"retry": False})}
            ).sha256
        else:
            result = self.sha256
        return result

    @property
    def stage(self) -> str:
        action = self.operation.action
        if isinstance(self.operation, PredictOperation):
            return "infer_" + self.operation.task
        if action in {"screen_prepare", "screen_preview", "review_service"}:
            return "render"
        return action


class OperationReceipt(Document):
    operation: ArtifactRef
    status: Literal["complete", "not_applicable"]
    results: tuple[ArtifactRef, ...]
    reason: str | None


class RetryWorker(Document):
    action: Literal["retry_worker"] = "retry_worker"
    request: ArtifactRef
    after_attempt: str = Field(min_length=1)


class RetryLearning(Document):
    action: Literal["retry_learning"] = "retry_learning"
    intent: ArtifactRef
    stage: Literal["train", "evaluate"]
    resume_checkpoint: ArtifactRef | None = None


class ResumeWorkers(Document):
    action: Literal["resume_workers"] = "resume_workers"


class RecoveryInput(Document):
    plan: ArtifactRef
    reason: str = Field(min_length=1)
    operation: Annotated[
        RetryWorker | RetryLearning | ResumeWorkers, Field(discriminator="action")
    ]


class RecoveryReceipt(Document):
    request: ArtifactRef
    worker_attempt: str | None = None
    learning_request: ArtifactRef | None = None
