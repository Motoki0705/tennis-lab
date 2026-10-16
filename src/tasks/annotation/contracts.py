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
from src.tasks.base.annotation.images.contracts import PageView
from src.tasks.base.annotation.reproducibility import CodeBoundDocument
from src.tasks.base.annotation.storage.comparison.contracts import ReadLimits


class StageCost(Document):
    stage: Identifier
    pss_bytes: PositiveInt
    storage_bytes: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0)
    gpu_seconds: float = Field(ge=0)


class RuntimeSettings(Document):
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


class Runtime(RuntimeSettings, CodeBoundDocument):
    schema_version: Literal["annotation.runtime.v1"] = "annotation.runtime.v1"
    plan: ArtifactRef
    scope: ArtifactRef


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


class RestoreSourceOperation(Document):
    action: Literal["restore_source"] = "restore_source"
    source_id: Digest
    attempt: Identifier


class PrepareOperation(Document):
    action: Literal["prepare"] = "prepare"
    source_id: Digest


class RetireExcludedOperation(Document):
    action: Literal["retire_excluded"] = "retire_excluded"
    source_id: Digest


class PreviewOperation(Document):
    action: Literal["screen_preview"] = "screen_preview"
    service_request: ArtifactRef


class PredictOperation(Document):
    action: Literal["predict"] = "predict"
    task: Literal["ball", "player"]
    clips: tuple[ArtifactRef, ...] = Field(min_length=1)


class PredictionPreviewOperation(Document):
    action: Literal["prediction_preview"] = "prediction_preview"
    task: Literal["ball", "player"]
    prediction: ArtifactRef
    drawing: Literal["plain", "prediction"]
    view: PageView


class ReviewOperation(Document):
    action: Literal["review_service"] = "review_service"
    service_request: ArtifactRef


class ServiceSessionOperation(Document):
    action: Literal["service_session"] = "service_session"
    session: PositiveInt
    maximum_requests: int = Field(ge=1, le=128)
    maximum_services_per_attempt: PositiveInt
    duration_seconds: float = Field(gt=0, le=600)


class PublishOperation(Document):
    action: Literal["publish_ball"] = "publish_ball"
    review: ArtifactRef


class BootstrapModelOperation(Document):
    action: Literal["bootstrap_model"] = "bootstrap_model"
    task: Identifier
    evidence: ArtifactRef


class InspectStorageOperation(Document):
    action: Literal["inspect_storage"] = "inspect_storage"
    specification: ArtifactRef
    limits: ReadLimits


class CompareStorageOperation(Document):
    action: Literal["compare_storage"] = "compare_storage"
    request: ArtifactRef


class OperationInput(Document):
    plan: ArtifactRef
    operation: Annotated[
        HistoryOperation
        | AcquireOperation
        | RestoreSourceOperation
        | ScreenOperation
        | PrepareOperation
        | RetireExcludedOperation
        | PreviewOperation
        | PredictionPreviewOperation
        | ReviewOperation
        | ServiceSessionOperation
        | PublishOperation
        | BootstrapModelOperation
        | InspectStorageOperation
        | CompareStorageOperation
        | PredictOperation,
        Field(discriminator="action"),
    ]

    @property
    def stage(self) -> str:
        action = self.operation.action
        if action == "restore_source":
            return "acquire"
        if action == "retire_excluded":
            return "prepare"
        if isinstance(self.operation, PredictOperation):
            return "infer_" + self.operation.task
        if action in {
            "screen_prepare",
            "screen_preview",
            "review_service",
            "service_session",
            "prediction_preview",
        }:
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


class CancelPendingStage(Document):
    action: Literal["cancel_pending_stage"] = "cancel_pending_stage"
    launch: ArtifactRef


class ReopenScreening(Document):
    action: Literal["reopen_incomplete_screening"] = "reopen_incomplete_screening"
    request: ArtifactRef
    after_attempt: str = Field(min_length=1)
    selection: ArtifactRef


class ObserveHostSpace(Document):
    action: Literal["observe_host_space"] = "observe_host_space"
    policy: ArtifactRef


class RecoveryInput(Document):
    plan: ArtifactRef
    reason: str = Field(min_length=1)
    operation: Annotated[
        RetryWorker
        | RetryLearning
        | ResumeWorkers
        | CancelPendingStage
        | ReopenScreening
        | ObserveHostSpace,
        Field(discriminator="action"),
    ]


class RecoveryReceipt(Document):
    request: ArtifactRef
    worker_attempt: str | None = None
    learning_request: ArtifactRef | None = None
