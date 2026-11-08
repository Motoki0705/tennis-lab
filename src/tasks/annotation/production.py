"""Compose verified review, publication and learning receipts for a frozen campaign."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.annotation.review.worker import BallReviewWorker
from src.tasks.ball_detection.annotation.video.agent.contracts import (
    DEFAULT_POLICY,
    AnnotationPolicy,
)
from src.tasks.ball_detection.training.annotation.adapter import BallLearningAdapter
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    write_document,
)
from src.tasks.base.annotation.collection.local import LocalInputRegistration
from src.tasks.base.annotation.contracts import (
    ArtifactRef,
    ProductOutcome,
    PublicationReceipt,
)
from src.tasks.base.annotation.learning import Learning
from src.tasks.base.annotation.learning.contracts import (
    CampaignCompletion,
    LearningRuntime,
    StageCost,
    TaskProducts,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import CampaignPlan, LocalAnnotationPlan
from src.tasks.base.annotation.registry import Registry
from src.tasks.base.annotation.resources import CapacityUnavailable, ResourceManager
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerBusy, WorkerStopped
from src.tasks.base.annotation.workers.pool import WorkerPool

from .contracts import (
    AnnotationContinuationPolicy,
    CompletionEvidence,
    LocalAnnotationRuntime,
    OperationInput,
    OperationReceipt,
    PublishOperation,
    Runtime,
)
from .reviews import (
    DECISION_STATES,
    CampaignReview,
    close_reviews,
    requests,
    require_supported,
    validate_closed,
)


class LocalAnnotationWorkflow:
    """Direct camera annotation, ending at saved v3 completion (no publication)."""

    def __init__(
        self,
        layout: AnnotationLayout,
        runtime: LocalAnnotationRuntime,
        plan: LocalAnnotationPlan,
        resources: ResourceManager,
        services: Callable[[WorkRequest, dict[str, Any], WorkerPool], None],
        *,
        test_command: tuple[str, ...] | None = None,
        test_clock: bool = False,
        annotation_policy: AnnotationPolicy = DEFAULT_POLICY,
    ) -> None:
        from src.tasks.ball_detection.annotation.video.agent.worker import (
            BallVideoWorker,
        )
        from src.tasks.base.annotation.collection.local import LocalInputLedger

        runtime_ref = annotation_reference(
            layout,
            layout.path("campaigns", plan.campaign_id, "configuration", "runtime.json"),
        )
        if read_document(layout, runtime_ref, LocalAnnotationRuntime) != runtime:
            raise ValueError("direct annotation requires its saved runtime")
        entries = LocalInputLedger(layout).entries(plan)
        if plan.phase == "benchmark":
            from .benchmark.blank import BlankCondition

            condition = read_document(
                layout,
                annotation_reference(
                    layout,
                    layout.path(
                        "campaigns",
                        plan.campaign_id,
                        "benchmark",
                        "blank-condition.json",
                    ),
                ),
                BlankCondition,
            )
            if (
                condition.plan != runtime.plan
                or condition.runtime != runtime_ref
                or condition.policy != annotation_policy
                or condition.requests != tuple(e.request for e in entries)
                or plan.publish_dataset
            ):
                raise ValueError(
                    "Blank condition differs from its frozen policy/runtime/inputs"
                )
        else:
            self._validate_registration(layout, runtime, runtime_ref, entries, plan)
        cost = runtime.cost("review")
        policy = AnnotationContinuationPolicy(
            plan=runtime.plan, runtime=runtime_ref, scope=runtime.scope
        )
        if cost.timeout_seconds != policy.timeout_seconds:
            raise ValueError(
                "direct annotation requires 5400 seconds per camera attempt"
            )
        self.layout, self.runtime, self.plan, self.services = (
            layout,
            runtime,
            plan,
            services,
        )
        self.policy = write_document(
            layout,
            layout.path(
                "campaigns", plan.campaign_id, "workflow", "continuation-policy.json"
            ),
            policy,
        )
        self.requests = tuple(entry.request for entry in entries)
        if not self.requests:
            raise ValueError(
                "direct annotation requires registered local camera inputs"
            )
        write_document(
            layout,
            layout.path(
                "campaigns", plan.campaign_id, "workflow", "annotation-policy.json"
            ),
            annotation_policy,
        )
        self.pool = WorkerPool(
            layout,
            plan,
            (BallVideoWorker(annotation_policy),),
            codex_binary=runtime.codex_binary,
            codex_home=Path(runtime.codex_home),
            timeout_seconds=policy.timeout_seconds,
            resources=resources,
            worker_pss_bytes=cost.pss_bytes,
            worker_storage_bytes=cost.storage_bytes,
            code_revision=runtime.code_revision,
            test_command=test_command,
            test_clock=test_clock,
        )

    @staticmethod
    def _validate_registration(
        layout: AnnotationLayout,
        runtime: LocalAnnotationRuntime,
        runtime_ref: ArtifactRef,
        entries: tuple[LocalInputRegistration, ...],
        plan: LocalAnnotationPlan,
    ) -> None:
        from .local_videos import LocalVideoConfigurationReceipt

        registration = read_document(
            layout,
            annotation_reference(
                layout,
                layout.path(
                    "campaigns", plan.campaign_id, "configuration", "result.json"
                ),
            ),
            LocalVideoConfigurationReceipt,
        )
        if (registration.plan, registration.runtime, registration.scope) != (
            runtime.plan,
            runtime_ref,
            runtime.scope,
        ):
            raise ValueError(
                "direct annotation runtime differs from its registration receipt"
            )
        if set(entries) != set(registration.registrations):
            raise ValueError(
                "registered camera inputs differ from the completed configuration"
            )

    @staticmethod
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        failures = [
            r
            for r in rows
            if r["state"] in DECISION_STATES | {"invalid_saved_progress"}
        ]
        live = any(
            r["state"] in {"running", "exited"}
            or r.get("worker_state") in {"running", "exited"}
            for r in rows
        )
        return {
            "state": "complete"
            if all(r["state"] == "complete" for r in rows)
            else "blocked"
            if failures
            else "reviewing",
            "live": live,
            "reviews": rows,
            "failures": failures,
            "reason": "saved camera annotations are complete"
            if all(r["state"] == "complete" for r in rows)
            else "resolve stopped camera reasons after live work settles"
            if failures
            else "annotating or continuing unprocessed camera frames",
        }

    def status(self) -> dict[str, Any]:
        from .recovery import inspect_annotation

        rows = []
        for ref in self.requests:
            request = read_document(self.layout, ref, WorkRequest)
            try:
                self.pool.reconcile(request)
                rows.append(inspect_annotation(self.layout, self.pool, ref))
            except (ValueError, OSError) as error:
                rows.append(
                    {
                        "request": ref.model_dump(mode="json"),
                        "state": "invalid_saved_progress",
                        "worker_state": self.pool.ledger.task(request.work_id)["state"],
                        "reason": str(error),
                    }
                )
        return self.summarize(rows)

    def cycle(self) -> dict[str, Any]:
        from .recovery import recover_annotation

        rows = []
        for ref in self.requests:
            rows.append(recover_annotation(self.layout, self.policy, self.pool, ref))
        hold = any(
            r["state"] in DECISION_STATES | {"invalid_saved_progress"} for r in rows
        )
        for ref, row in zip(self.requests, rows, strict=True):
            request = read_document(self.layout, ref, WorkRequest)
            if row["state"] == "invalid_saved_progress" and row["worker_state"] in {
                "running",
                "exited",
            }:
                try:
                    stopped = self.pool.stop(request, reason=row["reason"])
                    row["worker_state"] = stopped["state"]
                except WorkerBusy:
                    pass  # Ownership is not published yet; the next tick retries stopping.
                continue
            if row["state"] == "ready":
                # Authorization is its own durable tick. Normal admission, with
                # the same request/plan/scope, launches on a subsequent tick.
                if row["worker_state"] == "timed_out":
                    continue
                if hold and row["current_attempt"] is None:
                    row.update(
                        state="held",
                        reason="new cameras wait for resolution of stopped cameras",
                    )
                    continue
                try:
                    self.pool.start(request)
                    state = self.pool.reconcile(request)
                    row.update(
                        state=state["state"],
                        current_attempt=state["current_attempt"],
                        reason=state["reason"],
                    )
                except (WorkerBusy, CapacityUnavailable, WorkerStopped) as error:
                    row.update(state="waiting", reason=str(error))
            try:
                self.services(request, row, self.pool)
            except (ValueError, OSError) as error:
                stopped = self.pool.stop(
                    request, reason=f"invalid camera service request: {error}"
                )
                row.update(state=stopped["state"], reason=stopped["reason"])
        return self.summarize(rows)


class ProductionWorkflow:
    def __init__(
        self,
        layout: AnnotationLayout,
        runtime: Runtime,
        plan: CampaignPlan,
        resources: ResourceManager,
        operation: Callable[[OperationInput], dict[str, Any]],
        receipt: Callable[[dict[str, Any]], OperationReceipt],
        services: Callable[[WorkRequest, dict[str, Any], WorkerPool], None],
        publication_failures: Callable[[], list[dict[str, Any]]],
    ) -> None:
        require_supported(plan)
        self.layout, self.runtime, self.plan = layout, runtime, plan
        self.operation, self.receipt, self.services = operation, receipt, services
        self.publication_failures = publication_failures
        cost = runtime.cost("review")
        runtime.cost("publish_ball")
        self.learning_runtime = (
            LearningRuntime(
                code_revision=runtime.code_revision,
                scope=runtime.scope,
                queue=runtime.queue,
                train=self._cost("train_ball"),
                evaluate=self._cost("evaluate_ball"),
            )
            if plan.tasks[0].learning is not None
            else None
        )
        self.pool = WorkerPool(
            layout,
            plan,
            (BallReviewWorker(),),
            codex_binary=runtime.codex_binary,
            codex_home=Path(runtime.codex_home),
            timeout_seconds=cost.timeout_seconds,
            resources=resources,
            worker_pss_bytes=cost.pss_bytes,
            worker_storage_bytes=cost.storage_bytes,
            code_revision=runtime.code_revision,
        )
        self.registry = Registry(layout)
        self.registry.initialize(
            "annotation-production-progress-v1",
            """
            CREATE TABLE IF NOT EXISTS annotation_workflow_progress (
                campaign TEXT PRIMARY KEY, plan_digest TEXT NOT NULL, progress TEXT NOT NULL
            );
        """,
        )
        self.learning = Learning(layout)

    def _cost(self, stage: str) -> StageCost:
        cost = self.runtime.cost(stage)
        return StageCost(
            stage=stage,
            pss_bytes=cost.pss_bytes,
            storage_bytes=cost.storage_bytes,
            gpu_seconds=cost.gpu_seconds,
            timeout_seconds=cost.timeout_seconds,
        )

    def status(self) -> dict[str, Any]:
        with self.registry.transaction() as database:
            row = database.execute(
                "SELECT plan_digest,progress FROM annotation_workflow_progress WHERE campaign=?",
                (self.plan.campaign_id,),
            ).fetchone()
        if row is None:
            return {
                "state": "pending",
                "reason": "production workflow has not advanced",
            }
        if row[0] != self.plan.sha256:
            raise ValueError("workflow progress belongs to another frozen plan")
        result: dict[str, Any] = json.loads(row[1])
        return result

    def cycle(self) -> dict[str, Any]:
        progress = self._cycle()
        # Progress is a display cache; immutable receipts are authoritative on resume.
        with self.registry.transaction() as database:
            database.execute(
                "INSERT INTO annotation_workflow_progress VALUES (?,?,?) "
                "ON CONFLICT(campaign) DO UPDATE SET plan_digest=excluded.plan_digest,progress=excluded.progress",
                (
                    self.plan.campaign_id,
                    self.plan.sha256,
                    json.dumps(progress, ensure_ascii=False),
                ),
            )
        return progress

    def _cycle(self) -> dict[str, Any]:
        reconciled = []
        for request_ref in requests(self.layout, self.plan, self.runtime.plan):
            request = read_document(self.layout, request_ref, WorkRequest)
            self.pool.register(request)
            reconciled.append((request, self.pool.reconcile(request)))
        # A stop that needs a decision may be systematic (prompt, policy, model).
        # Live and explicitly retried work continues; new clips wait for the decision.
        hold = any(row["state"] in DECISION_STATES for _, row in reconciled)
        rows = []
        for request, row in reconciled:
            if row["state"] == "ready":
                if hold and row["current_attempt"] is None:
                    rows.append(
                        {
                            "state": "held",
                            "reason": "new reviews wait for decisions on stopped reviews",
                        }
                    )
                    continue
                try:
                    self.pool.start(request)
                except (WorkerBusy, CapacityUnavailable, WorkerStopped) as error:
                    rows.append({"state": "waiting", "reason": str(error)})
                    continue
                row = self.pool.reconcile(request)
            try:
                self.services(request, row, self.pool)
            except (ValueError, OSError) as error:
                row = self.pool.stop(
                    request, reason=f"invalid review service request: {error}"
                )
            rows.append({"state": row["state"], "reason": row["reason"]})
        failures = [row for row in rows if row["state"] in DECISION_STATES]
        if failures:
            return {
                "state": "blocked",
                "reviews": rows,
                "reason": "review requires explicit resolution; no publication",
                "failures": failures,
            }
        input_failures = self.publication_failures()
        if input_failures:
            states = {row["state"] for row in input_failures}
            active = {"running", "draining", "queued", "waiting"}
            waiting = states <= active | {"pending", "unreconciled"} and (
                bool(states & active) or "unreconciled" not in states
            )
            return {
                "state": "reviewing" if waiting else "blocked",
                "reviews": rows,
                "reason": "waiting for input stage completion before publication or learning"
                if waiting
                else "input stages lack verified completion; explicit recovery is required before publication or learning",
                "failures": input_failures,
            }
        review_ref = close_reviews(self.layout, self.plan, self.runtime.plan)
        if review_ref is None:
            return {
                "state": "reviewing",
                "reviews": rows,
                "reason": "waiting for requested sources and every clip review",
            }
        operation = OperationInput(
            plan=self.runtime.plan, operation=PublishOperation(review=review_ref)
        )
        state = self.operation(operation)
        if state["state"] != "succeeded":
            return {
                "state": "blocked"
                if state["state"] in {"failed", "cancelled", "timed_out", "interrupted"}
                else "publishing",
                "review": review_ref.model_dump(mode="json"),
                "publication": state,
            }
        receipt = self.receipt(state)
        if receipt.status != "complete" or len(receipt.results) != 1:
            raise ValueError(
                "publication operation requires exactly one completed receipt"
            )
        publication_ref = receipt.results[0]
        publication = read_document(self.layout, publication_ref, PublicationReceipt)
        adoptions = validate_closed(
            self.layout, self.plan, self.runtime.plan, review_ref
        )
        review = read_document(self.layout, review_ref, CampaignReview)
        clips = tuple(
            sorted(
                read_document(self.layout, row.request, WorkRequest).subject_id
                for row in review.reviews
            )
        )
        product = self.plan.tasks[0].products[0]
        if (
            publication.task != "ball"
            or publication.product != product.product
            or publication.base_snapshot != product.base_snapshot
            or publication.adoptions != adoptions
            or tuple(sorted(publication.added_clips)) != clips
        ):
            raise ValueError(
                "publication does not cover exactly the campaign's adopted new clips"
            )
        completion = CampaignCompletion(
            plan=self.runtime.plan,
            tasks=(
                TaskProducts(
                    task="ball",
                    products=(
                        ProductOutcome(
                            product=product.product,
                            status="published",
                            publication=publication_ref,
                            reason=None,
                        ),
                    ),
                ),
            ),
        )
        completion_ref = write_document(
            self.layout,
            self.layout.path(
                "campaigns", self.plan.campaign_id, "workflow", "products.json"
            ),
            completion,
        )
        evidence = write_document(
            self.layout,
            self.layout.path(
                "campaigns", self.plan.campaign_id, "workflow", "completion.json"
            ),
            CompletionEvidence(
                review=review_ref,
                publication_operation=receipt.operation,
                completion=completion_ref,
            ),
        )
        adapter = BallLearningAdapter()
        registration = self.learning.register(completion_ref, adapter)
        if registration.intent is None:
            return {
                "state": "complete"
                if registration.status == "no_additional_teachers"
                else "blocked",
                "completion": evidence.model_dump(mode="json"),
                "learning": registration.model_dump(mode="json"),
                "reason": registration.reason,
            }
        if self.learning_runtime is None:
            raise ValueError("registered learning intent requires its frozen runtime")
        outcome = self.learning.tick(
            registration.intent, adapter, self.learning_runtime
        )
        return {
            "state": "complete"
            if outcome.state in {"adopted", "rejected"}
            else "blocked"
            if outcome.state in {"failed", "not_configured", "reevaluation_required"}
            else "learning",
            "completion": evidence.model_dump(mode="json"),
            "learning": outcome.model_dump(mode="json"),
            "reason": outcome.reason,
        }
