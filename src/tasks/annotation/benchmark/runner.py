"""Sequential conditions with isolated ball review, shared budget and no publication."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.annotation.inference import BallPredictionAdapter
from src.tasks.ball_detection.annotation.review.worker import BallReviewWorker
from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    write_document,
)
from src.tasks.base.annotation.benchmarks.capture import (
    ObservationLimit,
    directory,
    observe,
    worker_snapshot,
)
from src.tasks.base.annotation.benchmarks.contracts import Observation, WorkerBinding
from src.tasks.base.annotation.benchmarks.inputs import (
    register_inputs,
    require_registered_scope,
)
from src.tasks.base.annotation.benchmarks.validation import comparison, condition
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.reproducibility import require_frozen_code
from src.tasks.base.annotation.resources import (
    BudgetStopped,
    CapacityUnavailable,
    ResourceManager,
)
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerBusy, WorkerStopped
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.utils.paths import PROJECT_ROOT

from ..configuration import validate_runtime
from ..contracts import Runtime
from ..jobs import Operations
from ..reviews import ClipReview, adoption, prediction_requests
from ..services import ServiceBroker
from .contracts import BenchmarkRuntime, ConditionCompletion

_FAILED = {
    "failed",
    "deferred",
    "not_applicable",
    "cancelled",
    "quota",
    "timed_out",
    "validation_failed",
}


class BenchmarkRunner:
    def __init__(
        self,
        layout: AnnotationLayout,
        reference: ArtifactRef,
        condition_id: str,
        *,
        test_command: tuple[str, ...] | None = None,
    ) -> None:
        self.shared = layout.shared()
        self.settings = read_document(layout, reference, BenchmarkRuntime)
        self.comparison, self.conditions = comparison(layout, self.settings.comparison)
        matches = [
            i
            for i, spec in enumerate(self.conditions)
            if spec.condition_id == condition_id
        ]
        if len(matches) != 1:
            raise ValueError("benchmark condition is absent from the frozen comparison")
        self.index = matches[0]
        self.reference = self.comparison.conditions[self.index]
        self.spec, self.plan, self.expected = condition(layout, self.reference)
        required = WorkerBinding(
            stage="review",
            adapter=BallReviewWorker.name,
            adapter_version=BallReviewWorker.version,
            task="ball",
        )
        for spec_ref, spec in zip(
            self.comparison.conditions, self.conditions, strict=True
        ):
            _, plan, _ = condition(layout, spec_ref)
            if (
                plan.phase != "benchmark"
                or [t.task for t in plan.tasks] != ["ball"]
                or spec.bindings != (required,)
            ):
                raise ValueError(
                    "benchmark runner currently requires isolated ball review conditions"
                )
            stages = {s.stage: s.kind for s in plan.resources.stages}
            if stages.get("render") != "cpu":
                raise ValueError(
                    "benchmark review requires an explicit CPU render stage"
                )
        if test_command is not None and layout.mode != "test":
            raise ValueError("synthetic benchmark commands require a test root")
        self.layout = self.shared.for_benchmark(self.plan.campaign_id)
        require_registered_scope(self.shared, self.spec.scope)
        self.resources = ResourceManager(self.shared, self.spec.scope)
        validate_runtime(
            self.layout, self.settings.settings, self.plan, self.resources.scope.roots
        )
        for stage in ("review", "render"):
            self.settings.settings.cost(stage)
        if layout.mode == "production":
            require_frozen_code(PROJECT_ROOT, self.plan)
        self.runtime = Runtime(
            plan=self.spec.plan,
            scope=self.spec.scope,
            **self.settings.settings.model_dump(),
        )
        self.test_command = test_command
        self.pool: WorkerPool | None = None
        self.requests: tuple[ArtifactRef, ...] = ()
        self.runtime_ref: ArtifactRef | None = None
        self.last_observation = 0.0
        self.jobs = Operations(
            self.layout,
            self.runtime,
            self.plan,
            self.resources,
            command="benchmark-stage",
        )
        self.broker = ServiceBroker(
            self.layout,
            self.spec.plan,
            self.runtime.maximum_services_per_attempt,
            self.jobs.operation,
            self.jobs.receipt,
        )

    def _runtime_path(self) -> Path:
        path: Path = self.shared.path(
            "campaigns", self.comparison.benchmark_id, "benchmark", "runtime.json"
        )
        return path

    def _completion_path(self, index: int) -> Path:
        _, plan, _ = condition(self.shared, self.comparison.conditions[index])
        path: Path = self.shared.path(
            "campaigns", plan.campaign_id, "benchmark", "complete.json"
        )
        return path

    def completion(self, index: int) -> ArtifactRef | None:
        path = self._completion_path(index)
        if not path.exists():
            return None
        ref = annotation_reference(self.shared, path)
        result = read_document(self.shared, ref, ConditionCompletion)
        spec_ref = self.comparison.conditions[index]
        spec, plan, expected = condition(self.shared, spec_ref)
        local = self.shared.for_benchmark(plan.campaign_id)
        runtime_ref = annotation_reference(self.shared, self._runtime_path())
        if (
            result.condition != spec_ref
            or result.runtime != runtime_ref
            or read_document(self.shared, runtime_ref, BenchmarkRuntime)
            != self.settings
        ):
            raise ValueError(
                "completed benchmark differs from its pinned runtime or condition"
            )
        work = set()
        for review in result.reviews:
            request = read_document(local, review.request, WorkRequest)
            if request.work_id in work:
                raise ValueError("benchmark completion repeats a worker")
            work.add(request.work_id)
            adoption(local, review.request, review.result, spec.plan)
        if work != set(expected):
            raise ValueError(
                "benchmark completion does not cover its exact frozen cohort"
            )
        observed = read_document(local, result.final_observation, Observation)
        if observed.specification != spec_ref or observed.task_states != {
            "complete": len(expected)
        }:
            raise ValueError("benchmark completion has no final complete observation")
        return ref

    def _initialize(self) -> None:
        for index in range(self.index):
            if self.completion(index) is None:
                raise ValueError(
                    "benchmark conditions must complete in order: 10, 25, 50"
                )
        self.runtime_ref = write_document(
            self.shared, self._runtime_path(), self.settings
        )
        if self.pool is not None:
            return
        register_inputs(self.shared, self.reference, (BallPredictionAdapter(),))
        self.resources.bind(self.plan)
        self.resources.observe(os.getpid())
        self.requests = prediction_requests(
            self.layout, self.plan, self.spec.plan, self.spec.predictions
        )
        if {
            read_document(self.layout, ref, WorkRequest).work_id
            for ref in self.requests
        } != set(self.expected):
            raise ValueError("benchmark request cohort differs from its condition")
        cost = self.runtime.cost("review")
        self.pool = WorkerPool(
            self.layout,
            self.plan,
            (BallReviewWorker(),),
            codex_binary=self.runtime.codex_binary,
            codex_home=Path(self.runtime.codex_home),
            timeout_seconds=cost.timeout_seconds,
            resources=self.resources,
            worker_pss_bytes=cost.pss_bytes,
            worker_storage_bytes=cost.storage_bytes,
            test_command=self.test_command,
        )
        for ref in self.requests:
            self.pool.register(read_document(self.layout, ref, WorkRequest))

    def _observe(self, *, force: bool = False) -> ArtifactRef | None:
        now = time.monotonic()
        if (
            not force
            and now - self.last_observation < self.settings.observation_interval_seconds
        ):
            return None
        ref = observe(self.shared, self.reference)
        self.last_observation = now
        return ref

    def status(self) -> dict[str, Any]:
        states, _, foreign = worker_snapshot(
            self.layout, self.spec, self.plan.campaign_id, self.expected
        )
        complete = self.completion(self.index)
        stop_reason = None
        if self.layout.registry.exists():
            with sqlite3.connect(
                self.layout.registry.as_uri() + "?mode=ro", uri=True
            ) as db:
                if (
                    db.execute(
                        "SELECT name FROM sqlite_master WHERE name='worker_campaign_stops'"
                    ).fetchone()
                    is not None
                ):
                    stopped = db.execute(
                        "SELECT reason FROM worker_campaign_stops WHERE campaign=?",
                        (self.plan.campaign_id,),
                    ).fetchone()
                    stop_reason = None if stopped is None else stopped[0]
        return {
            "state": "workers_complete"
            if complete is not None
            else "blocked"
            if stop_reason is not None or any(s in _FAILED for s in states)
            else "not_started"
            if states == {"unregistered": len(self.expected)}
            else "reviewing",
            "condition": self.spec.condition_id,
            "target_workers": self.spec.target_workers,
            "task_states": states,
            "completion": None
            if complete is None
            else complete.model_dump(mode="json"),
            "other_active_plan_digests": foreign,
            "acceptance": "unverified",
            "reason": stop_reason,
        }

    def stop(self, reason: str) -> tuple[str, ...]:
        if self.pool is None:
            return ()
        pending = []
        self.pool.ledger.stop_campaign(self.plan.campaign_id, reason)
        for ref in self.requests:
            request = read_document(self.layout, ref, WorkRequest)
            row = self.pool.reconcile(request)
            if row["state"] in {"running", "exited"}:
                try:
                    self.pool.stop(request, reason=reason)
                except WorkerBusy:
                    pending.append(request.work_id)
        return tuple(pending)

    def _stopped(self, reason: str, state: str) -> dict[str, Any]:
        pending = self.stop(reason)
        return {
            **self.status(),
            "state": "stopping" if pending else state,
            "reason": reason,
            "pending_workers": pending,
        }

    def tick(self) -> dict[str, Any]:
        with artifact_lock(
            self.shared, "benchmark-run-" + self.comparison.benchmark_id, blocking=False
        ):
            if self.completion(self.index) is not None:
                return self.status()
            self._initialize()
            assert self.pool is not None
            sample = self.resources.poll()
            if sample.stopped_reason:
                return self._stopped(sample.stopped_reason, "budget_stopped")
            count = len(
                list(
                    (directory(self.shared, self.spec) / "observations").glob("*.json")
                )
            )
            if count >= self.spec.maximum_observations:
                return self._stopped("benchmark observation limit reached", "blocked")
            stopped = self.status()["reason"]
            if stopped is not None:
                return self._stopped(stopped, "blocked")
            rows = []
            try:
                for ref in self.requests:
                    request = read_document(self.layout, ref, WorkRequest)
                    row = self.pool.reconcile(request)
                    if row["state"] == "ready":
                        try:
                            self.pool.start(request)
                        except (WorkerBusy, WorkerStopped, CapacityUnavailable):
                            rows.append(row)
                            continue
                        row = self.pool.reconcile(request)
                    try:
                        self.broker.process(request, row, self.pool)
                    except (ValueError, OSError) as error:
                        row = self.pool.stop(
                            request, reason=f"invalid benchmark review service: {error}"
                        )
                    rows.append(row)
                self.broker.flush()
                all_complete = all(row["state"] == "complete" for row in rows)
                observation = self._observe(force=all_complete)
            except BudgetStopped as error:
                return self._stopped(str(error), "budget_stopped")
            except ObservationLimit as error:
                return self._stopped(str(error), "blocked")
            if all_complete:
                assert observation is not None and self.runtime_ref is not None
                reviews = tuple(
                    ClipReview(
                        request=ref,
                        result=ArtifactRef.model_validate_json(row["result"]),
                    )
                    for ref, row in zip(self.requests, rows, strict=True)
                )
                for review in reviews:
                    adoption(self.layout, review.request, review.result, self.spec.plan)
                write_document(
                    self.layout,
                    self._completion_path(self.index),
                    ConditionCompletion(
                        condition=self.reference,
                        runtime=self.runtime_ref,
                        reviews=reviews,
                        final_observation=observation,
                    ),
                )
            return self.status()
