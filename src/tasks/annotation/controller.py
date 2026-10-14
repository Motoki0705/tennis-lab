"""Resume annotation campaigns from immutable receipts and shared ledgers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    ReviewOperation,
    Runtime,
    ScreenOperation,
)
from .operations import preview_input
from .screening import ScreeningAdapter

if TYPE_CHECKING:
    from .production import ProductionWorkflow


class Controller:
    def __init__(self, layout: AnnotationLayout, runtime: Runtime) -> None:
        self.layout, self.runtime = layout, runtime
        self.plan = load_plan(verify_reference(layout, runtime.plan))
        if self.plan.phase == "production":
            from .reviews import require_supported

            require_supported(self.plan)
        self.resources = ResourceManager(layout, runtime.scope)
        self.resources.bind(self.plan)
        self.resources.observe(os.getpid())
        self.collection = Collection(layout, self.plan)
        # Register the prediction receipt table without loading either model.
        Predictions(layout, self.plan)
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
        if data.plan != self.runtime.plan or (attempt is not None and attempt < 1):
            raise ValueError(
                "operation belongs to another runtime, or has an invalid attempt"
            )
        jobs = self.layout.path("campaigns", self.plan.campaign_id, "jobs")
        attempts = [
            int(path.name.rsplit("-", 1)[1]) for path in jobs.glob(data.sha256 + "-*")
        ]
        latest = max(attempts, default=0)
        if attempt is None:
            attempt = max(latest, 1)
        elif attempt not in attempts:
            if attempt != latest + 1:
                raise ValueError(
                    "a new attempt must follow the latest recorded attempt"
                )
            if latest:
                previous = annotation_reference(
                    layout=self.layout,
                    path=jobs / f"{data.sha256}-{latest}" / "launch.json",
                )
                previous_state = self.stages.status(previous)
                if previous_state["state"] not in {
                    "failed",
                    "cancelled",
                    "timed_out",
                    "interrupted",
                }:
                    raise ValueError(
                        "only a failed or interrupted operation can be retried"
                    )
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

    def _services(
        self, request: WorkRequest, row: dict[str, Any], pool: WorkerPool | None = None
    ) -> None:
        if row["state"] != "running":
            return
        pool = self.pool if pool is None else pool
        with pool.ledger.registry.transaction() as db:
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
        for number, path in enumerate(paths):
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
                if number >= self.runtime.maximum_services_per_attempt:
                    raise ValueError("worker exceeded its frozen service request count")
                if request.adapter == ScreeningAdapter.name:
                    _service, task, _attempt = preview_input(self.layout, service_ref)
                    if task.prepared_input != request.payload:
                        raise ValueError("service refers to another worker input")
                    op = OperationInput(
                        plan=self.runtime.plan,
                        operation=PreviewOperation(service_request=service_ref),
                    )
                elif request.adapter == "ball_review":
                    from src.tasks.base.annotation.workers.review.service import (
                        service_input,
                    )

                    _service, review = service_input(self.layout, service_ref)
                    if review.input.work != request:
                        raise ValueError("review service refers to another worker")
                    op = OperationInput(
                        plan=self.runtime.plan,
                        operation=ReviewOperation(service_request=service_ref),
                    )
                else:
                    raise ValueError("worker has no registered parent service")
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
                        f"service stage {state['state']}: {state.get('reason')}"
                    )
                receipt = self.receipt(state)
                if receipt.status != "complete" or len(receipt.results) != 1:
                    raise ValueError("service stage returned no exact result receipt")
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

    def cycle(self) -> dict[str, Any]:
        sample = self.resources.poll()
        if sample.stopped_reason is not None:
            raise BudgetStopped(sample.stopped_reason)
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
        production = self.production.cycle() if self.production is not None else None
        return {
            "collection": self.collection.ledger.summary(self.plan),
            "progress": progress,
            "resources": sample.model_dump(mode="json"),
            "workflow": production,
        }

    def _stage_statuses(self) -> list[dict[str, Any]]:
        from src.tasks.base.annotation.learning.contracts import StageInput

        jobs = []
        keys: list[tuple[str, str]] = []
        numbers: list[int] = []
        latest: dict[tuple[str, str], int] = {}
        for path in sorted(
            self.layout.path("campaigns", self.plan.campaign_id, "jobs").glob(
                "*/launch.json"
            )
        ):
            ref = annotation_reference(self.layout, path)
            launch = read_document(self.layout, ref, StageLaunch)
            if launch.input is None or launch.resource.plan != self.runtime.plan:
                raise ValueError("campaign job has no matching input/plan")
            if launch.resource.request_id.startswith("learning-"):
                request = read_document(self.layout, launch.input, StageInput)
                if launch.resource.request_id != "learning-" + request.sha256:
                    raise ValueError("learning attempt identity differs from its input")
                key, number = (request.intent.digest, request.stage), request.attempt
            else:
                data = read_document(self.layout, launch.input, OperationInput)
                number = int(launch.resource.request_id.rsplit("-", 1)[1])
                if (
                    data.plan != self.runtime.plan
                    or launch.resource.request_id != f"{data.sha256}-{number}"
                ):
                    raise ValueError(
                        "operation attempt identity differs from its input"
                    )
                key = (data.sha256, "operation")
            keys.append(key)
            numbers.append(number)
            latest[key] = max(latest.get(key, 0), number)
            jobs.append(self.stages.status(ref))
        for key, number, job in zip(keys, numbers, jobs, strict=True):
            job["superseded"] = number < latest[key]
        return jobs

    def status(self) -> dict[str, Any]:
        jobs = self._stage_statuses()
        with self.collection.ledger.registry.transaction() as db:
            tasks = dict(
                db.execute(
                    "SELECT state,count(*) FROM annotation_tasks WHERE campaign_id=? GROUP BY state",
                    (self.plan.campaign_id,),
                )
            )
            workers = [
                dict(row)
                for row in db.execute(
                    "SELECT id,state,current_attempt,reason FROM worker_tasks WHERE campaign=?",
                    (self.plan.campaign_id,),
                )
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
            if not job["superseded"]
            and job["state"] in {"failed", "cancelled", "timed_out", "interrupted"}
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
        production = self.production.status() if self.production is not None else None
        if production is not None:
            state = (
                "running"
                if live or production["state"] in {"publishing", "learning"}
                else "complete"
                if production["state"] == "complete"
                else "blocked"
                if failures or production["state"] == "blocked"
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
