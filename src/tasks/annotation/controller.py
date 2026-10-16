"""Resume annotation campaigns from immutable receipts and shared ledgers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
)
from src.tasks.base.annotation.collection.service import Collection
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan
from src.tasks.base.annotation.predictions import PredictionReceipt, Predictions
from src.tasks.base.annotation.resources import (
    BudgetStopped,
    CapacityUnavailable,
    ResourceManager,
)
from src.tasks.base.annotation.workers.contracts import Launch, WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerBusy, WorkerStopped
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.services import ServiceRequest

from .configuration import validate_runtime
from .contracts import (
    AcquireOperation,
    CandidateManifest,
    OperationInput,
    OperationReceipt,
    PredictOperation,
    PrepareOperation,
    PreviewOperation,
    RetireExcludedOperation,
    ReviewOperation,
    Runtime,
    ScreenOperation,
)
from .jobs import Operations
from .screening import ScreeningAdapter
from .service_sessions import ServiceSessions
from .services import ServiceBroker

if TYPE_CHECKING:
    from .production import ProductionWorkflow


class Controller:
    def __init__(self, layout: AnnotationLayout, runtime: Runtime) -> None:
        self.layout, self.runtime = layout, runtime
        # Only the live run uses these verified, in-memory snapshots. Explicit
        # detailed status rechecks the immutable history, including log hashes.
        self._operation_inputs: dict[Path, OperationInput] = {}
        self._stage_inputs: dict[
            Path, tuple[ArtifactRef, StageLaunch, tuple[str, str], int, Launch | None]
        ] = {}
        self._terminal_stages: dict[str, dict[str, Any]] = {}
        self._stage_results: dict[str, tuple[ArtifactRef, ...]] = {}
        self.plan = load_plan(verify_reference(layout, runtime.plan))
        if self.plan.phase == "benchmark" or layout.review_namespace is not None:
            raise ValueError("benchmark conditions require the dedicated review runner")
        if self.plan.phase == "production":
            from .reviews import require_supported

            require_supported(self.plan)
        self.resources = ResourceManager(layout, runtime.scope)
        self.resources.bind(self.plan)
        self.resources.observe(os.getpid())
        self.collection = Collection(layout, self.plan)
        # Register the prediction receipt table without loading either model.
        Predictions(layout, self.plan)
        validate_runtime(layout, runtime, self.plan, self.resources.scope.roots)
        codex_home = Path(runtime.codex_home)
        screen = runtime.cost("screen")
        self.pool = WorkerPool(
            layout,
            self.plan,
            (ScreeningAdapter(),),
            codex_binary=runtime.codex_binary,
            codex_home=codex_home,
            timeout_seconds=screen.timeout_seconds,
            resources=self.resources,
            worker_pss_bytes=screen.pss_bytes,
            worker_storage_bytes=screen.storage_bytes,
            code_revision=runtime.code_revision,
        )
        self.jobs = Operations(layout, runtime, self.plan, self.resources)
        self.stages = self.jobs.stages
        self.broker = ServiceBroker(
            layout,
            runtime.plan,
            runtime.maximum_services_per_attempt,
            self.operation,
            self.receipt,
        )
        self.service_sessions = ServiceSessions(
            layout, runtime, self.plan.campaign_id, self.operation, self.receipt
        )
        self.production: ProductionWorkflow | None = None
        if self.plan.phase == "production":
            from .production import ProductionWorkflow

            self.production = ProductionWorkflow(
                layout,
                runtime,
                self.plan,
                self.resources,
                self.operation,
                self.receipt,
                self._services,
                self.publication_failures,
            )

    def register(self, candidates: CandidateManifest) -> list[dict[str, Any]]:
        if self.layout.path(
            "campaigns", self.plan.campaign_id, "workflow", "review.json"
        ).exists():
            raise ValueError(
                "campaign review is closed; register sources in a new campaign"
            )
        return [
            item.model_dump(mode="json")
            for item in self.collection.register(candidates.candidates)
        ]

    def operation(
        self, data: OperationInput, *, attempt: int | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = self.jobs.operation(data, attempt=attempt)
        return result

    def receipt(self, state: dict[str, Any]) -> OperationReceipt:
        receipt: OperationReceipt = self.jobs.receipt(state)
        return receipt

    def _services(
        self, request: WorkRequest, row: dict[str, Any], pool: WorkerPool | None = None
    ) -> None:
        self.broker.process(request, row, self.pool if pool is None else pool)

    def retry_screen(
        self, source_id: str, *, resume_quota: bool = False
    ) -> dict[str, Any]:
        """Explicitly retry a screening worker without rerunning its preparation."""
        return self._screen(source_id, retry=True, resume_quota=resume_quota)

    def _screen(
        self, source_id: str, *, retry: bool = False, resume_quota: bool = False
    ) -> dict[str, Any]:
        data = OperationInput(
            plan=self.runtime.plan, operation=ScreenOperation(source_id=source_id)
        )
        state = self.operation(data)
        if state["state"] != "succeeded":
            return state
        receipt = self.receipt(state)
        if receipt.status == "not_applicable":
            return {"state": "not_applicable", "reason": receipt.reason}
        if len(receipt.results) != 1:
            raise ValueError("screening preparation requires exactly one source input")
        request = WorkRequest(
            adapter=ScreeningAdapter.name,
            adapter_version=ScreeningAdapter.version,
            stage="screen",
            subject_id=source_id,
            payload=receipt.results[0],
            inputs=(),
        )
        self.pool.register(request)
        row = self.pool.reconcile(request)
        if resume_quota:
            self.pool.ledger.resume_campaign(self.plan.campaign_id)
        if row["state"] == "ready" or retry:
            try:
                self.pool.start(request, retry=retry)
            except (WorkerBusy, CapacityUnavailable, WorkerStopped) as error:
                return {"state": "waiting", "reason": str(error)}
            row = self.pool.reconcile(request)
        try:
            self._services(request, row)
        except (ValueError, OSError) as error:
            row = self.pool.stop(request, reason=f"invalid service request: {error}")
        return {"state": row["state"], "reason": row["reason"]}

    def _reconcile_workers(self) -> list[dict[str, Any]]:
        with self.pool.ledger.registry.transaction() as db:
            requests = [
                WorkRequest.model_validate_json(row["request"])
                for row in db.execute(
                    "SELECT request FROM worker_tasks WHERE campaign=? ORDER BY id",
                    (self.plan.campaign_id,),
                )
            ]
        reconciled = []
        for request in requests:
            pool = self.pool
            if request.adapter != ScreeningAdapter.name and self.production is not None:
                pool = self.production.pool
            reconciled.append(pool.reconcile(request))
        return reconciled

    def cycle(self) -> dict[str, Any]:
        sample = self.resources.poll()
        if sample.stopped_reason is not None:
            raise BudgetStopped(sample.stopped_reason)
        # Adapter finalization may advance a source before the worker receipt is saved.
        self._reconcile_workers()
        with self.collection.ledger.registry.transaction() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM sources WHERE campaign_id=? ORDER BY id",
                    (self.plan.campaign_id,),
                )
            ]
        lead = sum(
            row["state"] in {"acquiring", "acquired", "screened"} for row in rows
        )
        progress: dict[str, Any] = {}
        # Feed existing downstream work before fetching more sources.
        for row in sorted(rows, key=lambda row: row["state"] == "selected"):
            if row["state"] == "selected":
                if lead >= self.collection.config.maximum_unprepared_sources:
                    continue
                state = self.operation(
                    OperationInput(
                        plan=self.runtime.plan,
                        operation=AcquireOperation(source_id=row["id"]),
                    )
                )
                lead += state["state"] in {"running", "succeeded", "draining"}
            elif row["state"] == "acquired":
                state = self._screen(row["id"])
            elif (
                row["state"] == "excluded"
                and row["selection"] is not None
                and self.collection.config.excluded_source_retention
                == "delete_after_verified_exclusion"
            ):
                intent = self.layout.path(
                    "sources",
                    row["id"],
                    self.plan.preparation.digest,
                    "retirement-intent.json",
                )
                if not intent.exists():
                    screening = self._screen(row["id"])
                    if screening["state"] != "not_applicable":
                        progress[row["id"]] = screening
                        continue
                state = self.operation(
                    OperationInput(
                        plan=self.runtime.plan,
                        operation=RetireExcludedOperation(source_id=row["id"]),
                    )
                )
            elif row["state"] == "screened":
                # Selection can commit just before worker finalization. Settle
                # that finalization before handing its source inputs to prepare.
                if (
                    self.collection.config.source_retention
                    == "delete_after_verified_frames"
                ):
                    screening = self._screen(row["id"])
                    if screening["state"] != "complete":
                        progress[row["id"]] = screening
                        continue
                state = self.operation(
                    OperationInput(
                        plan=self.runtime.plan,
                        operation=PrepareOperation(source_id=row["id"]),
                    )
                )
            elif (
                row["state"] == "prepared"
                and self.collection.config.source_retention
                == "delete_after_verified_frames"
            ):
                # Reconcile the existing prepare job even if its process exited
                # between the prepared DB commit and source retirement receipt.
                state = self.operation(
                    OperationInput(
                        plan=self.runtime.plan,
                        operation=PrepareOperation(source_id=row["id"]),
                    )
                )
            else:
                continue
            progress[row["id"]] = state
        claimed: set[tuple[str, str]] = set()
        seen_operations: set[str] = set()
        for path in sorted(
            self.layout.path("campaigns", self.plan.campaign_id, "jobs").glob(
                "*/operation.json"
            )
        ):
            data = self._operation_inputs.get(path)
            if data is None:
                data = read_document(
                    self.layout, annotation_reference(self.layout, path), OperationInput
                )
                self._operation_inputs[path] = data
            if data.logical_id in seen_operations:
                continue
            seen_operations.add(data.logical_id)
            if isinstance(data.operation, PredictOperation):
                claimed.update(
                    (data.operation.task, ref.digest) for ref in data.operation.clips
                )
                progress[data.logical_id] = self.operation(data)
        with self.collection.ledger.registry.transaction() as db:
            pending = [
                dict(row)
                for row in db.execute(
                    "SELECT t.task,c.manifest FROM annotation_tasks t JOIN clips c ON c.id=t.clip_id WHERE t.campaign_id=? AND t.state='pending' ORDER BY t.task,t.clip_id",
                    (self.plan.campaign_id,),
                )
            ]
        for task in ("ball", "player"):
            clips = [
                ArtifactRef.model_validate_json(row["manifest"])
                for row in pending
                if row["task"] == task
            ]
            clips = [ref for ref in clips if (task, ref.digest) not in claimed]
            if clips:
                data = OperationInput(
                    plan=self.runtime.plan,
                    operation=PredictOperation(
                        task=task,
                        clips=tuple(
                            clips[: self.runtime.maximum_clips_per_prediction_batch]
                        ),
                    ),
                )
                progress[data.logical_id] = self.operation(data)
        production = self.production.cycle() if self.production is not None else None
        self.service_sessions.tick(self.broker)
        return {
            "collection": self.collection.ledger.summary(self.plan),
            "progress": progress,
            "resources": sample.model_dump(mode="json"),
            "workflow": production,
        }

    def _stage_states(
        self, *, details: bool = True
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from src.tasks.base.annotation.learning.contracts import StageInput

        jobs: list[dict[str, Any]] = []
        keys: list[tuple[str, str]] = []
        latest: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        with self.pool.ledger.registry.transaction() as db:
            attempts = dict(
                db.execute(
                    "SELECT id,current_attempt FROM worker_tasks WHERE campaign=?",
                    (self.plan.campaign_id,),
                )
            )

        def stage_status(ref: ArtifactRef) -> dict[str, Any]:
            state = None if details else self._terminal_stages.get(ref.digest)
            if state is None:
                state = self.stages.status(ref)
                if state["state"] in {"succeeded", "failed", "cancelled", "timed_out"}:
                    self._terminal_stages[ref.digest] = dict(state)
            return dict(state)

        def record(
            key: tuple[str, str],
            number: int,
            state: dict[str, Any],
            worker: Launch | None = None,
        ) -> None:
            state["attempt"] = number
            state["superseded"] = (
                worker is not None and attempts.get(worker.work_id) != worker.attempt
            )
            jobs.append(state)
            keys.append(key)
            if key not in latest or number > latest[key][0]:
                latest[key] = (number, state)

        root = self.layout.path("campaigns", self.plan.campaign_id, "jobs")
        for path in sorted(root.glob("*/operation.json")):
            launch_path = path.parent / "launch.json"
            cached = None if details else self._stage_inputs.get(launch_path)
            if cached is None:
                operation_ref = annotation_reference(self.layout, path)
                data = read_document(self.layout, operation_ref, OperationInput)
                identity, number_text = path.parent.name.rsplit("-", 1)
                if (
                    data.plan != self.runtime.plan
                    or identity != data.logical_id
                    or not number_text.isdigit()
                    or int(number_text) < 1
                ):
                    raise ValueError(
                        "stage attempt differs from its immutable operation"
                    )
                number, key = int(number_text), (identity, "operation")
                worker = None
                if isinstance(data.operation, (PreviewOperation, ReviewOperation)):
                    service = read_document(
                        self.layout, data.operation.service_request, ServiceRequest
                    )
                    directory = self.layout.artifact_path(service.task_request).parent
                    worker = read_document(
                        self.layout,
                        annotation_reference(self.layout, directory / "launch.json"),
                        Launch,
                    )
                    if (
                        worker.plan != self.runtime.plan
                        or Path(worker.directory) != directory
                    ):
                        raise ValueError(
                            "service stage differs from its worker plan/attempt"
                        )
                if not launch_path.exists():
                    record(
                        key,
                        number,
                        {
                            "state": "pending",
                            "reason": "operation has no launch receipt",
                            "operation": operation_ref.model_dump(),
                        },
                        worker,
                    )
                    continue
                ref = annotation_reference(self.layout, launch_path)
                launch = read_document(self.layout, ref, StageLaunch)
                if (
                    launch.input != operation_ref
                    or launch.resource.request_id != path.parent.name
                    or launch.resource.plan != self.runtime.plan
                ):
                    raise ValueError("stage launch differs from its operation/attempt")
                self._stage_inputs[launch_path] = (ref, launch, key, number, worker)
            else:
                ref, launch, key, number, worker = cached
            if launch.input is None:
                raise ValueError("operation launch has no immutable operation")
            state = stage_status(ref)
            state["operation"] = launch.input.model_dump()
            record(key, number, state, worker)

        for path in sorted(root.glob("*/launch.json")):
            if (path.parent / "operation.json").exists():
                continue
            cached = None if details else self._stage_inputs.get(path)
            if cached is None:
                ref = annotation_reference(self.layout, path)
                launch = read_document(self.layout, ref, StageLaunch)
                if launch.input is None or launch.resource.plan != self.runtime.plan:
                    raise ValueError("campaign job has no matching input/plan")
                if not launch.resource.request_id.startswith("learning-"):
                    raise ValueError("operation launch has no immutable operation")
                request = read_document(self.layout, launch.input, StageInput)
                if launch.resource.request_id != "learning-" + request.sha256:
                    raise ValueError("learning attempt identity differs from its input")
                key, number = (request.intent.digest, request.stage), request.attempt
                self._stage_inputs[path] = (ref, launch, key, number, None)
            else:
                ref, launch, key, number, _worker = cached
            record(key, number, stage_status(ref))
        for key, state in zip(keys, jobs, strict=True):
            state["superseded"] |= state["attempt"] < latest[key][0]
        return jobs, [state for _, state in latest.values() if not state["superseded"]]

    def publication_failures(self) -> list[dict[str, Any]]:
        """Match registered inputs to their latest successful producing attempts."""
        _, latest = self._stage_states()
        completed: dict[tuple[str, str], set[ArtifactRef]] = {}
        failures: list[dict[str, Any]] = []
        for state in latest:
            if "operation" not in state:
                continue
            data = read_document(
                self.layout,
                ArtifactRef.model_validate(state["operation"]),
                OperationInput,
            )
            operation = data.operation
            if not isinstance(
                operation, (AcquireOperation, PrepareOperation, PredictOperation)
            ):
                continue
            if state["state"] != "succeeded":
                failures.append(state)
                continue
            try:
                receipt = self.receipt(state)
                if receipt.status != "complete":
                    continue
                if isinstance(operation, (AcquireOperation, PrepareOperation)):
                    if len(receipt.results) != 1:
                        raise ValueError("source stage must return exactly one receipt")
                    completed.setdefault(
                        (operation.action, operation.source_id), set()
                    ).add(receipt.results[0])
                else:
                    for reference in receipt.results:
                        prediction = read_document(
                            self.layout, reference, PredictionReceipt
                        )
                        completed.setdefault(
                            ("predict:" + operation.task, prediction.context.clip_id),
                            set(),
                        ).add(reference)
            except (ValueError, OSError) as error:
                failures.append(
                    {**state, "state": "invalid_receipt", "reason": str(error)}
                )
        with self.collection.ledger.registry.transaction() as database:
            sources = database.execute(
                "SELECT id,state,acquisition,preparation FROM sources WHERE campaign_id=?",
                (self.plan.campaign_id,),
            ).fetchall()
            predictions = database.execute(
                "SELECT t.task,t.clip_id,p.receipt FROM annotation_tasks t "
                "LEFT JOIN annotation_predictions p ON p.task_id=t.id "
                "WHERE t.campaign_id=? AND t.state IN ('predicted','reviewing','adopted')",
                (self.plan.campaign_id,),
            ).fetchall()
        expected = [
            (("predict:" + row["task"], row["clip_id"]), row["receipt"])
            for row in predictions
        ]
        for source in sources:
            if source["acquisition"] is not None or source["state"] == "prepared":
                expected.append((("acquire", source["id"]), source["acquisition"]))
            if source["state"] == "prepared":
                expected.append((("prepare", source["id"]), source["preparation"]))
        for key, reference in expected:
            if reference is None or ArtifactRef.model_validate_json(
                reference
            ) not in completed.get(key, set()):
                failures.append(
                    {
                        "state": "unreconciled",
                        "stage": key[0],
                        "subject_id": key[1],
                        "reason": "registered artifact lacks a matching successful receipt from its latest stage attempt; recover explicitly",
                    }
                )
        return failures

    def status(self, *, details: bool = True) -> dict[str, Any]:
        reconciled = self._reconcile_workers()
        jobs, latest = self._stage_states(details=details)
        completed: set[ArtifactRef] = set()
        for stage_state in latest:
            if stage_state["state"] != "succeeded" or "operation" not in stage_state:
                continue
            launch_ref = ArtifactRef.model_validate(stage_state["launch"])
            try:
                results = (
                    None if details else self._stage_results.get(launch_ref.digest)
                )
                if results is None:
                    results = self.receipt(stage_state).results
                    self._stage_results[launch_ref.digest] = results
                completed.update(results)
            except (ValueError, OSError) as error:
                self._stage_results.pop(launch_ref.digest, None)
                stage_state.update(
                    state="invalid_receipt", reason=f"{type(error).__name__}: {error}"
                )
        with self.collection.ledger.registry.transaction() as db:
            tasks = dict(
                db.execute(
                    "SELECT state,count(*) FROM annotation_tasks WHERE campaign_id=? GROUP BY state",
                    (self.plan.campaign_id,),
                )
            )
            source_receipts = [
                ref
                for row in db.execute(
                    "SELECT state,acquisition,preparation FROM sources WHERE campaign_id=?",
                    (self.plan.campaign_id,),
                )
                for ref in (
                    row["acquisition"],
                    row["preparation"] if row["state"] == "prepared" else None,
                )
                if ref is not None
            ]
            prediction_receipts = [
                row[0]
                for row in db.execute(
                    "SELECT p.receipt FROM annotation_tasks t LEFT JOIN annotation_predictions p ON p.task_id=t.id WHERE t.campaign_id=?",
                    (self.plan.campaign_id,),
                )
            ]
        workers = [
            {key: row[key] for key in ("id", "state", "current_attempt", "reason")}
            for row in reconciled
        ]
        summary = self.collection.ledger.summary(self.plan)
        source_states = summary["sources_by_state"]
        live = any(
            job["state"] in {"running", "draining", "queued"} for job in jobs
        ) or any(row["state"] in {"running", "exited"} for row in workers)
        failures = [
            row
            for row in workers
            if row["state"]
            in {"failed", "cancelled", "quota", "timed_out", "validation_failed"}
        ]
        failures += [
            job
            for job in latest
            if job["state"]
            in {"failed", "cancelled", "timed_out", "interrupted", "invalid_receipt"}
        ]
        all_sources = bool(source_states) and all(
            key in {"prepared", "excluded"} for key in source_states
        )
        ledger_ready = all_sources and bool(tasks) and set(tasks) == {"predicted"}
        if ledger_ready and any(
            ref is None or ArtifactRef.model_validate_json(ref) not in completed
            for ref in (*source_receipts, *prediction_receipts)
        ):
            failures.append(
                {
                    "state": "unreconciled",
                    "reason": "ledger artifacts lack verified completion receipts from the latest stage attempts; recover with an explicit new attempt",
                }
            )
        ready = (
            ledger_ready
            and not live
            and not failures
            and bool(latest)
            and all(job["state"] == "succeeded" for job in latest)
            and all(row["state"] in {"complete", "not_applicable"} for row in workers)
        )
        state = (
            "preview_ready"
            if ready
            else "running"
            if live
            else "blocked"
            if failures
            else "pending"
        )
        production = self.production.status() if self.production is not None else None
        if production is not None:
            if production["state"] == "complete":
                failures.extend(self.publication_failures())
            state = (
                "blocked"
                if failures or production["state"] == "blocked"
                else "running"
                if live or production["state"] in {"publishing", "learning"}
                else "complete"
                if production["state"] == "complete"
                else "pending"
            )
        return {
            "state": state,
            "collection": summary,
            "task_states": tasks,
            "workers": workers,
            "stages": jobs,
            "failures": failures,
            "preview_adopts_teachers": False,
            "workflow": production,
        }
