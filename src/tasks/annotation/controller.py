"""Resume the new-source preview flow from immutable receipts and shared ledgers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.tasks.base.annotation.artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_document,
)
from src.tasks.base.annotation.collection.service import Collection
from src.tasks.base.annotation.contracts import ArtifactRef
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.execution.queue import validate_queue
from src.tasks.base.annotation.execution.runner import StageRunner
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan
from src.tasks.base.annotation.predictions import Predictions
from src.tasks.base.annotation.resources import (
    BudgetStopped,
    CapacityUnavailable,
    ResourceManager,
    ResourceRequest,
)
from src.tasks.base.annotation.workers.contracts import Launch, WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerBusy, WorkerStopped
from src.tasks.base.annotation.workers.pool import WorkerPool
from src.tasks.base.annotation.workers.services import ServiceReply, ServiceRequest
from src.utils.paths import PROJECT_ROOT

from .contracts import (
    AcquireOperation,
    CandidateManifest,
    OperationInput,
    OperationReceipt,
    PredictOperation,
    PrepareOperation,
    PreviewOperation,
    Runtime,
    ScreenOperation,
)
from .operations import preview_input
from .screening import ScreeningAdapter


class Controller:
    def __init__(self, layout: AnnotationLayout, runtime: Runtime) -> None:
        self.layout, self.runtime = layout, runtime
        self.plan = load_plan(verify_reference(layout, runtime.plan))
        self.resources = ResourceManager(layout, runtime.scope)
        self.resources.bind(self.plan)
        self.resources.observe(os.getpid())
        self.collection = Collection(layout, self.plan)
        # Register the prediction receipt table without loading either model.
        Predictions(layout, self.plan)
        if self.plan.phase != "preview":
            raise ValueError(
                "this controller currently implements preview inputs only; production adoption is not enabled"
            )
        if len({cost.stage for cost in runtime.costs}) != len(runtime.costs):
            raise ValueError("stage costs must be unique")
        declared = {stage.stage: stage for stage in self.plan.resources.stages}
        for cost in runtime.costs:
            if cost.stage not in declared or (cost.gpu_seconds > 0) != (
                declared[cost.stage].kind == "gpu"
            ):
                raise ValueError(
                    "runtime resource costs differ from declared stage kinds"
                )
            if cost.gpu_seconds and cost.timeout_seconds + 10 > cost.gpu_seconds:
                raise ValueError("GPU timeout must leave the reserved teardown margin")
        codex_home = Path(runtime.codex_home)
        if not codex_home.is_absolute() or codex_home.resolve() != codex_home:
            raise ValueError("Codex state must use an explicit normalized directory")
        if not any(
            codex_home.is_relative_to(Path(root)) for root in self.resources.scope.roots
        ):
            raise ValueError("Codex state must be inside the resource inventory")
        validate_queue(layout, runtime.queue)
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
        )
        self.stages = StageRunner(layout)
        self.pool.ledger.registry.initialize(
            "worker-services.v1",
            """
            CREATE TABLE IF NOT EXISTS worker_service_requests (
                attempt TEXT NOT NULL REFERENCES worker_attempts(id),
                digest TEXT NOT NULL, request TEXT NOT NULL,
                PRIMARY KEY (attempt, digest)
            );
            """,
        )

    def register(self, candidates: CandidateManifest) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.collection.register(candidates.candidates)
        ]

    def operation(
        self, data: OperationInput, *, attempt: int | None = None
    ) -> dict[str, Any]:
        if data.plan != self.runtime.plan or (attempt is not None and attempt < 1):
            raise ValueError(
                "operation belongs to another runtime, or has an invalid attempt"
            )
        jobs = self.layout.path("campaigns", self.plan.campaign_id, "jobs")
        if attempt is None:
            attempts = [
                int(path.name.rsplit("-", 1)[1])
                for path in jobs.glob(data.sha256 + "-*")
            ]
            attempt = max(attempts, default=1)
        cost = self.runtime.cost(data.stage)
        request_id = f"{data.sha256}-{attempt}"
        directory = self.layout.path(
            "campaigns", self.plan.campaign_id, "jobs", request_id
        )
        reference = write_document(self.layout, directory / "operation.json", data)
        command = [
            str(PROJECT_ROOT / ".venv/bin/python"),
            "-m",
            "src.tasks.annotation",
            "stage",
            "--operation",
            str(self.layout.artifact_path(reference)),
            "--sha256",
            reference.digest,
        ]
        if self.layout.mode == "test":
            command += ["--test-root", str(self.layout.root)]
        launch = StageLaunch(
            scope=self.runtime.scope,
            resource=ResourceRequest(
                request_id=request_id,
                plan=self.runtime.plan,
                stage=data.stage,
                pss_bytes=cost.pss_bytes,
                storage_bytes=cost.storage_bytes,
                gpu_seconds=cost.gpu_seconds,
            ),
            command=tuple(command),
            input=reference,
            code_root=str(PROJECT_ROOT),
            timeout_seconds=cost.timeout_seconds,
            queue=self.runtime.queue if cost.gpu_seconds else None,
            test_root=str(self.layout.root) if self.layout.mode == "test" else None,
        )
        path = directory / "launch.json"
        if path.exists():
            launch_ref = annotation_reference(self.layout, path)
            existing = read_document(self.layout, launch_ref, StageLaunch)
            if existing != launch:
                raise ValueError("operation resource/command configuration changed")
            state: dict[str, Any] = self.stages.status(launch_ref)
            # A capacity-limited launch has no owner yet and remains explicitly pending.
            with self.resources.registry.transaction() as db:
                row = db.execute(
                    "SELECT state FROM resource_requests WHERE id=?",
                    (self.resources.key(launch.resource),),
                ).fetchone()
            pending = row is not None and row[0] == "pending"
            if not pending or state["state"] not in {"pending", "interrupted"}:
                return state
        try:
            submitted: dict[str, Any] = self.stages.submit(launch)
            return submitted
        except (CapacityUnavailable, BlockingIOError) as error:
            return {
                "state": "waiting",
                "operation": reference.model_dump(),
                "reason": str(error),
            }

    def receipt(self, state: dict[str, Any]) -> OperationReceipt:
        launch = read_document(
            self.layout, ArtifactRef.model_validate(state["launch"]), StageLaunch
        )
        if state["state"] != "succeeded" or launch.input is None:
            raise ValueError("only a successful stage can provide operation results")
        path = self.layout.artifact_path(launch.input).parent / "result.json"
        receipt: OperationReceipt = read_document(
            self.layout, annotation_reference(self.layout, path), OperationReceipt
        )
        if receipt.operation != launch.input:
            raise ValueError("stage result belongs to a different operation")
        for ref in receipt.results:
            verify_reference(self.layout, ref)
        return receipt

    def _admit_service(self, attempt: str, service: ServiceRequest) -> None:
        encoded = service.canonical_bytes().decode()
        with self.pool.ledger.registry.transaction() as db:
            prior = db.execute(
                "SELECT request FROM worker_service_requests WHERE attempt=? AND digest=?",
                (attempt, service.sha256),
            ).fetchone()
            if prior is not None:
                if prior["request"] != encoded:
                    raise ValueError("registered worker service request changed")
                return
            count = db.execute(
                "SELECT count(*) FROM worker_service_requests WHERE attempt=?",
                (attempt,),
            ).fetchone()[0]
            if count >= self.runtime.maximum_services_per_attempt:
                raise ValueError(
                    "worker exceeded its frozen image-service request count"
                )
            db.execute(
                "INSERT INTO worker_service_requests VALUES (?,?,?)",
                (attempt, service.sha256, encoded),
            )

    def _services(self, request: WorkRequest, row: dict[str, Any]) -> None:
        if row["state"] != "running":
            return
        with self.pool.ledger.registry.transaction() as db:
            raw = db.execute(
                "SELECT launch FROM worker_attempts WHERE id=?",
                (row["current_attempt"],),
            ).fetchone()
        launch = read_document(
            self.layout, ArtifactRef.model_validate_json(raw[0]), Launch
        )
        directory = Path(launch.directory)
        official = annotation_reference(self.layout, directory / "request.json")
        paths = sorted((directory / "work/service-requests").glob("*.json"))
        for path in paths:
            if path.stat().st_size > 32768:
                raise ValueError("worker service request exceeds 32 KiB")
            service_ref = annotation_reference(self.layout, path)
            service = read_document(self.layout, service_ref, ServiceRequest)
            if service.task_request != official or path.stem != service.sha256:
                raise ValueError(
                    "worker service request changed its task/attempt identity"
                )
            reply_path = directory / "services" / service.sha256 / "reply.json"
            if reply_path.exists():
                continue
            result, error = None, None
            try:
                _service, task, _attempt = preview_input(self.layout, service_ref)
                if task.prepared_input != request.payload:
                    raise ValueError("service refers to another worker input")
                self._admit_service(launch.attempt, service)
                op = OperationInput(
                    plan=self.runtime.plan,
                    operation=PreviewOperation(service_request=service_ref),
                )
                state = self.operation(op)
                if state["state"] in {
                    "running",
                    "waiting",
                    "queued",
                    "draining",
                    "pending",
                }:
                    continue
                if state["state"] != "succeeded":
                    raise RuntimeError(
                        f"image stage {state['state']}: {state.get('reason')}"
                    )
                receipt = self.receipt(state)
                if receipt.status != "complete" or len(receipt.results) != 1:
                    raise ValueError("preview stage returned no exact image receipt")
                result = receipt.results[0]
            except (ValueError, RuntimeError) as failure:
                error = f"{type(failure).__name__}: {failure}"
            write_document(
                self.layout,
                reply_path,
                ServiceReply(
                    request_digest=service.sha256,
                    status="succeeded" if result is not None else "failed",
                    result=result,
                    reason=error,
                ),
            )

    def _screen(self, source_id: str) -> dict[str, Any]:
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
        if row["state"] == "ready":
            try:
                self.pool.start(request)
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
        return [self.pool.reconcile(request) for request in requests]

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
            elif row["state"] == "screened":
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
            data = read_document(
                self.layout, annotation_reference(self.layout, path), OperationInput
            )
            if data.sha256 in seen_operations:
                continue
            seen_operations.add(data.sha256)
            if isinstance(data.operation, PredictOperation):
                claimed.update(
                    (data.operation.task, ref.digest) for ref in data.operation.clips
                )
                progress[data.sha256] = self.operation(data)
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
                progress[data.sha256] = self.operation(data)
        return {
            "collection": self.collection.ledger.summary(self.plan),
            "progress": progress,
            "resources": sample.model_dump(mode="json"),
        }

    def status(self) -> dict[str, Any]:
        reconciled = self._reconcile_workers()
        jobs = []
        for path in sorted(
            self.layout.path("campaigns", self.plan.campaign_id, "jobs").glob(
                "*/launch.json"
            )
        ):
            jobs.append(self.stages.status(annotation_reference(self.layout, path)))
        with self.collection.ledger.registry.transaction() as db:
            tasks = dict(
                db.execute(
                    "SELECT state,count(*) FROM annotation_tasks WHERE campaign_id=? GROUP BY state",
                    (self.plan.campaign_id,),
                )
            )
        workers = [
            {key: row[key] for key in ("id", "state", "reason")} for row in reconciled
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
            for job in jobs
            if job["state"] in {"failed", "cancelled", "timed_out", "interrupted"}
        ]
        all_sources = bool(source_states) and all(
            key in {"prepared", "excluded"} for key in source_states
        )
        ready = all_sources and bool(tasks) and set(tasks) == {"predicted"} and not live
        state = (
            "preview_ready"
            if ready
            else "running"
            if live
            else "blocked"
            if failures
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
        }
