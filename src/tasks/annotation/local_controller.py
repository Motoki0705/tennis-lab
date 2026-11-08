"""Controller for registered local cameras, without collection or prediction stages."""

from __future__ import annotations

import os
from typing import Any

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
)
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import LocalAnnotationPlan
from src.tasks.base.annotation.resources import BudgetStopped, ResourceManager
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.pool import WorkerPool

from .configuration import validate_runtime
from .contracts import LocalAnnotationRuntime, OperationInput, OperationReceipt
from .jobs import Operations
from .production import LocalAnnotationWorkflow
from .services import ServiceBroker


class LocalAnnotationController:
    def __init__(
        self,
        layout: AnnotationLayout,
        runtime: LocalAnnotationRuntime,
        *,
        test_command: tuple[str, ...] | None = None,
        test_clock: bool = False,
    ) -> None:
        self.layout, self.runtime = layout, runtime
        self.plan = read_document(layout, runtime.plan, LocalAnnotationPlan)
        if self.plan.phase != "production" or layout.review_namespace is not None:
            raise ValueError(
                "local controller requires production direct annotation inputs"
            )
        self.resources = ResourceManager(layout, runtime.scope)
        self.resources.bind(self.plan)
        self.resources.observe(os.getpid())
        validate_runtime(layout, runtime, self.plan, self.resources.scope.roots)
        self.jobs = Operations(layout, runtime, self.plan, self.resources)
        self.broker = ServiceBroker(
            layout,
            runtime.plan,
            runtime.maximum_services_per_attempt,
            self.operation,
            self.receipt,
        )
        self.production = LocalAnnotationWorkflow(
            layout,
            runtime,
            self.plan,
            self.resources,
            self._services,
            test_command=test_command,
            test_clock=test_clock,
        )
        self.pool = self.production.pool

    def operation(
        self, data: OperationInput, *, attempt: int | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = self.jobs.operation(data, attempt=attempt)
        return result

    def receipt(self, state: dict[str, Any]) -> OperationReceipt:
        receipt: OperationReceipt = self.jobs.receipt(state)
        return receipt

    def _services(
        self, request: WorkRequest, row: dict[str, Any], pool: WorkerPool
    ) -> None:
        self.broker.process(request, row, pool)

    def cycle(self) -> dict[str, Any]:
        # Different controller instances and duplicate ticks serialize decisions
        # and admission. Worker/service locks still protect process ownership.
        with artifact_lock(
            self.layout, "local-annotation-controller-" + self.plan.campaign_id
        ):
            sample = self.resources.poll()
            if sample.stopped_reason is not None:
                for ref in self.production.requests:
                    self.production.pool.cancel_pending(
                        read_document(self.layout, ref, WorkRequest),
                        reason=sample.stopped_reason,
                    )
                raise BudgetStopped(sample.stopped_reason)
            progress = self.production.cycle()
            self.broker.flush()
            # Quota is campaign-wide, including pending resource reservations.
            with self.production.pool.ledger.registry.transaction() as db:
                stop = db.execute(
                    "SELECT reason FROM worker_campaign_stops WHERE campaign=?",
                    (self.plan.campaign_id,),
                ).fetchone()
            if stop is not None:
                for ref in self.production.requests:
                    self.production.pool.cancel_pending(
                        read_document(self.layout, ref, WorkRequest),
                        reason=stop["reason"],
                    )
            return self._stage_status(progress)

    def status(self) -> dict[str, Any]:
        return self._stage_status(self.production.status())

    def _stage_status(self, progress: dict[str, Any]) -> dict[str, Any]:
        stages = [
            self.jobs.stages.status(annotation_reference(self.layout, path))
            for path in self.layout.path(
                "campaigns", self.plan.campaign_id, "jobs"
            ).glob("*/launch.json")
        ]
        live = progress["live"] or any(
            s["state"] in {"running", "draining", "queued"} for s in stages
        )
        return {
            **progress,
            "stages": stages,
            "live": live,
            "state": "reviewing"
            if progress["state"] == "complete" and live
            else progress["state"],
        }
