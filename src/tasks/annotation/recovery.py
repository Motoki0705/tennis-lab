"""Explicit, recorded recovery; ordinary workflow ticks never retry failed work."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.tasks.ball_detection.training.annotation.adapter import BallLearningAdapter
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    LearningIntent,
)
from src.tasks.base.annotation.plan import references
from src.tasks.base.annotation.workers.contracts import WorkRequest

from .contracts import (
    RecoveryInput,
    RecoveryReceipt,
    ResumeWorkers,
    RetryLearning,
    RetryWorker,
)

if TYPE_CHECKING:
    from .controller import Controller


def recover(controller: Controller, data: RecoveryInput) -> ArtifactRef:
    if data.plan != controller.runtime.plan or not data.reason.strip():
        raise ValueError("recovery requires the current plan and an explicit reason")
    layout, operation = controller.layout, data.operation
    for ref in references(data):
        verify_reference(layout, ref)
    directory = layout.path(
        "campaigns", controller.plan.campaign_id, "recovery", data.sha256
    )
    # All recovery requests for this campaign serialize, including different reasons.
    with artifact_lock(
        layout, "workflow-recovery-" + controller.plan.campaign_id, blocking=False
    ):
        request_ref = write_document(layout, directory / "request.json", data)
        if (directory / "result.json").exists():
            result_ref = annotation_reference(layout, directory / "result.json")
            result = read_document(layout, result_ref, RecoveryReceipt)
            if result.request != request_ref:
                raise ValueError("recovery receipt belongs to another request")
            for ref in references(result):
                verify_reference(layout, ref)
            return result_ref
        result = RecoveryReceipt(request=request_ref)
        if isinstance(operation, ResumeWorkers):
            controller.pool.ledger.resume_campaign(controller.plan.campaign_id)
        elif isinstance(operation, RetryWorker):
            request = read_document(layout, operation.request, WorkRequest)
            if request.adapter == "ball_review" and controller.production is not None:
                pool = controller.production.pool
            elif request.adapter == "source_screening":
                pool = controller.pool
            else:
                raise ValueError("worker recovery has no configured adapter")
            # register checks immutable campaign/request ownership before any launch.
            pool.register(request)
            row = pool.reconcile(request)
            if row["current_attempt"] != operation.after_attempt:
                raise ValueError("worker attempt changed; inspect its current state")
            if row["state"] not in {
                "deferred",
                "failed",
                "cancelled",
                "quota",
                "timed_out",
                "validation_failed",
            }:
                raise ValueError("only a stopped or failed worker can be retried")
            attempt = pool.start(request, retry=True)
            if attempt is None:
                raise ValueError("worker completed while recovery was being requested")
            result = RecoveryReceipt(request=request_ref, worker_attempt=attempt)
        elif isinstance(operation, RetryLearning):
            production = controller.production
            if production is None:
                raise ValueError("learning recovery requires a production workflow")
            intent = read_document(layout, operation.intent, LearningIntent)
            completion = read_document(layout, intent.completion, CampaignCompletion)
            if completion.plan != data.plan or intent.task != "ball":
                raise ValueError("learning recovery belongs to another campaign/task")
            stage_request = production.learning.retry(
                operation.intent,
                BallLearningAdapter(),
                operation.stage,
                reason=data.reason,
                resume_checkpoint=operation.resume_checkpoint,
            )
            result = RecoveryReceipt(
                request=request_ref, learning_request=stage_request
            )
        reference: ArtifactRef = write_document(
            layout, directory / "result.json", result
        )
        return reference
