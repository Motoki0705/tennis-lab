"""Recorded manual recovery and the pinned direct-annotation timeout policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tasks.ball_detection.training.annotation.adapter import BallLearningAdapter
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    LearningIntent,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan, references
from src.tasks.base.annotation.workers.contracts import Launch, WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerBusy
from src.tasks.base.annotation.workers.pool import WorkerPool

from .contracts import (
    AnnotationContinuation,
    AnnotationContinuationPolicy,
    CancelPendingStage,
    ContinueReview,
    ObserveHostSpace,
    RecoveryInput,
    RecoveryReceipt,
    ReopenScreening,
    ResumeWorkers,
    RetryLearning,
    RetryWorker,
)

if TYPE_CHECKING:
    from .controller import Controller
    from .local_controller import LocalAnnotationController


def recover_annotation(
    layout: AnnotationLayout,
    policy_ref: ArtifactRef,
    pool: WorkerPool,
    request_ref: ArtifactRef,
) -> dict[str, Any]:
    """Authorize saved timeout progress; launching remains normal admission.

    The caller serializes campaign ticks. Decisions are immutable and precede the
    ready transition, so interruption on either side is recoverable. Common pool
    locks/CAS still arbitrate against stop, successful finalization and retry.
    """
    request = read_document(layout, request_ref, WorkRequest)
    pool.register(request)
    verification_error = None
    try:
        pool.reconcile(request)
    except (ValueError, OSError) as error:
        verification_error = error
    try:
        with (
            artifact_lock(layout, "worker-" + request.work_id, blocking=False),
            artifact_lock(layout, "worker-services-" + request.work_id),
        ):
            result = _annotation_decision(
                layout, policy_ref, pool, request_ref, verification_error
            )
    except BlockingIOError:
        # The supervisor (possibly a concurrent explicit retry) still owns the
        # work. Inspection may report progress, but cannot authorize or seal it.
        return inspect_annotation(layout, pool, request_ref)
    if result["state"] == "ready" and result["worker_state"] == "timed_out":
        try:
            pool.authorize_retry(
                request,
                after_attempt=result["current_attempt"],
                reason=result["reason"],
                accepted_states=frozenset({"timed_out"}),
            )
        except (ValueError, WorkerBusy):
            current = pool.ledger.task(request.work_id)
            if (
                current["current_attempt"] == result["current_attempt"]
                and current["state"] == "timed_out"
            ):
                raise
            # Never apply an old decision to a newer attempt or completed work.
            return inspect_annotation(layout, pool, request_ref)
    return result


def inspect_annotation(
    layout: AnnotationLayout, pool: WorkerPool, request_ref: ArtifactRef
) -> dict[str, Any]:
    """Read-only snapshot, retaining process state even when saved data is invalid."""
    from .review_progress import camera_status

    request = read_document(layout, request_ref, WorkRequest)
    try:
        status = camera_status(layout, pool.plan, request_ref)
        result: dict[str, Any] = status.model_dump(mode="json")
        result["worker_state"] = status.state
        decision_path = layout.path(
            "campaigns",
            pool.plan.campaign_id,
            "workflow",
            "continuations",
            request.work_id,
            str(status.current_attempt) + ".json",
        )
        decision = (
            read_document(
                layout,
                annotation_reference(layout, decision_path),
                AnnotationContinuation,
            )
            if decision_path.exists()
            else None
        )
        if decision is not None and (
            decision.request != request_ref
            or decision.after_attempt != status.current_attempt
        ):
            raise ValueError("saved decision differs from the inspected camera/attempt")
        if status.completion is not None:
            result["state"] = "complete"
        elif (
            decision is not None
            and decision.action == "hold"
            and status.state not in {"ready", "running", "exited"}
        ):
            result.update(state=decision.state, reason=decision.reason)
        elif status.state == "timed_out" and (
            status.new_processed or status.summary["complete"]
        ):
            result["state"] = "continuing"
        return result
    except (ValueError, OSError) as error:
        row = pool.ledger.task(request.work_id)
        return {
            "request": request_ref.model_dump(mode="json"),
            "state": "invalid_saved_progress",
            "worker_state": row["state"],
            "current_attempt": row["current_attempt"],
            "reason": f"invalid saved progress: {type(error).__name__}: {error}; inspect before explicit recovery",
        }


def _annotation_decision(
    layout: AnnotationLayout,
    policy_ref: ArtifactRef,
    pool: WorkerPool,
    request_ref: ArtifactRef,
    verification_error: ValueError | OSError | None,
) -> dict[str, Any]:
    """Verify and record under work/service ownership; never launch or authorize."""
    from src.tasks.ball_detection.annotation.video.agent.contracts import AgentTask
    from src.tasks.ball_detection.annotation.video.agent.store import VideoRevisionStore

    from .review_progress import camera_status

    policy = read_document(layout, policy_ref, AnnotationContinuationPolicy)
    request = read_document(layout, request_ref, WorkRequest)
    plan = read_document(layout, policy.plan, LocalAnnotationPlan)
    if (
        pool.plan != plan
        or pool.timeout != policy.timeout_seconds
        or pool.resources is None
        or pool.resources.reference != policy.scope
    ):
        raise ValueError("automatic continuation differs from its frozen policy")
    row = pool.ledger.task(request.work_id)
    attempt = row["current_attempt"]
    decision_path = layout.path(
        "campaigns",
        plan.campaign_id,
        "workflow",
        "continuations",
        request.work_id,
        str(attempt) + ".json",
    )
    decision: AnnotationContinuation | None = None
    try:
        if verification_error is not None:
            raise verification_error
        status = camera_status(layout, plan, request_ref)
        if attempt is not None:
            history = pool.ledger.history(plan, request)
            actor = history[-1].launch
            launch = read_document(layout, actor, Launch)
            task = read_document(
                layout,
                annotation_reference(layout, Path(launch.directory) / "request.json"),
                AgentTask,
            )
            store = VideoRevisionStore(layout, task.run)
            if (
                store.run.runtime != policy.runtime
                or launch.timeout_seconds != policy.timeout_seconds
            ):
                raise ValueError(
                    "saved attempt differs from the automatic runtime/timeout"
                )
        result: dict[str, Any] = status.model_dump(mode="json")
        result["worker_state"] = status.state
    except (ValueError, OSError) as error:
        reason = f"invalid saved progress: {type(error).__name__}: {error}; inspect and repair artifacts before explicit recovery"
        result = {
            "request": request_ref.model_dump(mode="json"),
            "state": "invalid_saved_progress",
            "worker_state": row["state"],
            "current_attempt": attempt,
            "reason": reason,
        }
        if attempt is not None:
            decision = AnnotationContinuation(
                policy=policy_ref,
                request=request_ref,
                after_attempt=attempt,
                state="invalid_saved_progress",
                action="hold",
                revision=None,
                processed_before=None,
                processed_after=None,
                remaining=None,
                reason=reason,
            )
            # Repeated ticks retain the original evidence, even if a broken file
            # disappears later. A different attempt needs an explicit decision.
            if not decision_path.exists():
                write_document(layout, decision_path, decision)
        return result
    if status.completion is not None:
        result.update(
            state="complete",
            reason="saved annotation is complete, including final unknowns",
        )
        return result
    if attempt is None or row["state"] in {"ready", "running", "exited"}:
        return result
    if decision_path.exists():
        decision_ref = annotation_reference(layout, decision_path)
        decision = read_document(layout, decision_ref, AnnotationContinuation)
        if (decision.policy, decision.request, decision.after_attempt) != (
            policy_ref,
            request_ref,
            attempt,
        ):
            raise ValueError(
                "automatic decision belongs to another policy/request/attempt"
            )
        if (
            decision.action == "hold"
            and decision.state == "invalid_saved_progress"
            and row["state"] == "timed_out"
            and status.summary["complete"]
        ):
            # Verified repairs can seal a fully saved timeout. Keep the original
            # hold immutable and persist a separate completion decision before
            # finalization, so interruption recovers through the same tick path.
            decision_path = decision_path.with_suffix("") / "completion.json"
            decision = None
    if decision is None:
        action = "hold"
        if row["state"] == "timed_out":
            if status.summary["complete"]:
                action, reason = (
                    "complete",
                    "all target frames were durably processed before timeout; finalize saved annotation",
                )
            elif status.new_processed:
                action, reason = (
                    "continue",
                    (
                        f"90-minute timeout with {len(status.new_processed)} newly processed frames; "
                        f"continue {status.summary['unprocessed']} unprocessed frames"
                    ),
                )
            else:
                reason = "timeout made no new processed-frame progress; inspect worker and explicitly retry after resolving the cause"
        else:
            reason = {
                "quota": "quota or rate limit; resolve service capacity, explicitly resume the campaign and retry",
                "failed": "worker or external service failure; inspect exit/events and explicitly retry after repair",
                "cancelled": "cancelled; preserve saved work and wait for an explicit retry",
                "validation_failed": "worker result validation failed; inspect result and saved artifacts before explicit recovery",
                "deferred": "worker deferred; inspect its reason before an explicit retry",
            }.get(
                row["state"],
                "unexpected stopped state; inspect before explicit recovery",
            )
        decision = AnnotationContinuation(
            policy=policy_ref,
            request=request_ref,
            after_attempt=attempt,
            action=action,
            state=row["state"],
            revision=status.revision,
            processed_before=status.processed_before,
            processed_after=status.processed_after,
            remaining=int(status.summary["unprocessed"]),
            reason=reason,
        )
        decision_ref = write_document(layout, decision_path, decision)
    if decision.action in {"continue", "complete"} and row["state"] != "timed_out":
        # A concurrent stop can supersede an already persisted timeout decision.
        # Retain that immutable evidence without applying it to the new state.
        result["continuation"] = decision_ref.model_dump(mode="json")
        return result
    result.update(
        reason=decision.reason, continuation=decision_ref.model_dump(mode="json")
    )
    if decision.action == "complete":
        if decision.revision != status.revision or not status.summary["complete"]:
            raise ValueError("saved completion decision differs from current progress")
        completion = store.complete(decision.revision, actor, decision.reason)
        result.update(state="complete", completion=completion.model_dump(mode="json"))
    elif decision.action == "continue":
        if decision.revision != status.revision or not status.new_processed:
            raise ValueError(
                "saved continuation decision differs from current progress"
            )
        result["state"] = "ready"
    return result


def recover(
    controller: Controller | LocalAnnotationController, data: RecoveryInput
) -> ArtifactRef:
    from .local_controller import LocalAnnotationController

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
            result_ref: ArtifactRef = annotation_reference(
                layout, directory / "result.json"
            )
            result = read_document(layout, result_ref, RecoveryReceipt)
            if result.request != request_ref:
                raise ValueError("recovery receipt belongs to another request")
            for ref in references(result):
                verify_reference(layout, ref)
            return result_ref
        result = RecoveryReceipt(request=request_ref)
        if isinstance(operation, ResumeWorkers):
            controller.pool.ledger.resume_campaign(controller.plan.campaign_id)
        elif isinstance(operation, ObserveHostSpace):
            from src.tasks.base.annotation.resources.contracts import (
                ObservedHostSpacePolicy,
            )
            from src.tasks.base.annotation.resources.host_space import activate

            policy = read_document(layout, operation.policy, ObservedHostSpacePolicy)
            if policy.plan != data.plan:
                raise ValueError("observed host-space policy belongs to another plan")
            activate(controller.resources, operation.policy)
        elif isinstance(operation, CancelPendingStage):
            launch = read_document(layout, operation.launch, StageLaunch)
            if (
                launch.resource.plan != data.plan
                or launch.scope != controller.runtime.scope
            ):
                raise ValueError(
                    "pending stage belongs to another plan or resource scope"
                )
            manager = controller.resources
            with manager.registry.transaction() as db:
                row = db.execute(
                    "SELECT state,reason FROM resource_requests WHERE id=?",
                    (manager.key(launch.resource),),
                ).fetchone()
            if (
                row is None
                or row["state"] != "cancelled"
                or row["reason"] != data.reason
            ):
                manager.cancel_pending(launch.resource, reason=data.reason)
        elif isinstance(operation, ReopenScreening):
            from .screening_recovery import reopen_incomplete_screening

            if isinstance(controller, LocalAnnotationController):
                raise ValueError("local annotation has no screening stage")
            reopen_incomplete_screening(controller, request_ref, data)
        elif isinstance(operation, RetryWorker):
            request = read_document(layout, operation.request, WorkRequest)
            if (
                request.adapter in {"ball_review", "ball_video_annotation"}
                and controller.production is not None
            ):
                pool = controller.production.pool
            elif request.adapter == "source_screening" and not isinstance(
                controller, LocalAnnotationController
            ):
                source = controller.collection.ledger.source(
                    controller.plan, request.subject_id
                )
                if source["state"] == "prepared":
                    raise ValueError(
                        "prepared source selection is immutable; resume preparation rather than retrying screening"
                    )
                pool = controller.pool
            else:
                raise ValueError("worker recovery has no configured adapter")
            # register checks immutable campaign/request ownership before any launch.
            pool.register(request)
            row = pool.reconcile(request)
            if isinstance(controller, LocalAnnotationController):
                from .review_progress import camera_status

                saved = camera_status(layout, controller.plan, operation.request)
                if saved.completion is not None or (
                    row["state"] == "timed_out" and saved.summary["complete"]
                ):
                    raise ValueError(
                        "camera annotation is already complete; run a tick to recover its completion without another worker"
                    )
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
        elif isinstance(operation, ContinueReview):
            from .review_progress import clip_status, continuation

            production = controller.production
            request = read_document(layout, operation.request, WorkRequest)
            if (
                production is None
                or request.adapter != "ball_review"
                or isinstance(controller, LocalAnnotationController)
            ):
                raise ValueError(
                    "review continuation requires a production ball review"
                )
            production.pool.register(request)
            row = production.pool.reconcile(request)
            if row["current_attempt"] != operation.after_attempt:
                raise ValueError("worker attempt changed; inspect its current state")
            # Recompute verified coverage here; the caller's summary is not evidence.
            allowed, detail = continuation(
                clip_status(layout, controller.plan, operation.request),
                operation.max_attempts,
            )
            if not allowed:
                raise ValueError("review cannot be continued: " + detail)
            production.pool.authorize_retry(
                request, after_attempt=operation.after_attempt, reason=data.reason
            )
        elif isinstance(operation, RetryLearning):
            if isinstance(controller, LocalAnnotationController):
                raise ValueError("local annotation has no learning stage")
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
