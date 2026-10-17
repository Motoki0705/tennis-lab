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
from src.tasks.base.annotation.collection.contracts import (
    AcquisitionReceipt,
    SourceCandidate,
)
from src.tasks.base.annotation.collection.service import Collection
from src.tasks.base.annotation.contracts import ArtifactRef, ClipManifest
from src.tasks.base.annotation.execution.contracts import StageLaunch
from src.tasks.base.annotation.execution.queue import validate_queue
from src.tasks.base.annotation.execution.runner import StageRunner
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import load_plan, references
from src.tasks.base.annotation.predictions import (
    PredictionContext,
    PredictionReceipt,
    Predictions,
)
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
        if data.plan != self.runtime.plan or (attempt is not None and attempt < 1):
            raise ValueError(
                "operation belongs to another runtime, or has an invalid attempt"
            )
        jobs = self.layout.path("campaigns", self.plan.campaign_id, "jobs")
        attempts = [
            int(path.name.rsplit("-", 1)[1])
            for path in jobs.glob(data.logical_id + "-*")
        ]
        latest = max(attempts, default=0)
        if attempt is None:
            attempt = max(latest, 1)
            if latest:
                prior = read_document(
                    self.layout,
                    annotation_reference(
                        self.layout,
                        jobs / f"{data.logical_id}-{attempt}" / "operation.json",
                    ),
                    OperationInput,
                )
                if prior.logical_id != data.logical_id:
                    raise ValueError("stage attempt differs from its logical operation")
                if (
                    isinstance(data.operation, AcquireOperation)
                    and isinstance(prior.operation, AcquireOperation)
                    and data.operation.retry
                    and not prior.operation.retry
                ):
                    attempt += 1
                else:
                    # Ordinary resume recovers the pinned retry directive as well.
                    data = prior
        if attempt not in attempts:
            if attempt != latest + 1:
                raise ValueError(
                    "a new attempt must follow the latest recorded attempt"
                )
            if latest:
                previous = annotation_reference(
                    layout=self.layout,
                    path=jobs / f"{data.logical_id}-{latest}" / "launch.json",
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
        request_id = f"{data.logical_id}-{attempt}"
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
            "--attempt",
            str(attempt),
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
        data = read_document(self.layout, launch.input, OperationInput)
        if data.plan != self.runtime.plan or data.stage != launch.resource.stage:
            raise ValueError("stage result differs from the runtime plan/stage")
        operation = data.operation
        if isinstance(operation, AcquireOperation) or (
            isinstance(operation, (ScreenOperation, PrepareOperation))
            and receipt.status == "not_applicable"
        ):
            source = self.collection.ledger.source(self.plan, operation.source_id)
            if source["acquisition"] is None or receipt.results != (
                ArtifactRef.model_validate_json(source["acquisition"]),
            ):
                raise ValueError(
                    "source receipt differs from its registered acquisition"
                )
            acquisition = read_document(
                self.layout, receipt.results[0], AcquisitionReceipt
            )
            candidate = SourceCandidate.model_validate_json(source["candidate"])
            if (
                acquisition.source.canonical_id != operation.source_id
                or acquisition.candidate_digest != candidate.sha256
                or acquisition.configuration != self.plan.preparation
            ):
                raise ValueError("acquisition receipt differs from the frozen source")
            for dependency in references(acquisition):
                verify_reference(self.layout, dependency)
            if receipt.status == "not_applicable" and (
                source["state"] != "excluded"
                or not receipt.reason
                or receipt.reason != source["reason"]
            ):
                raise ValueError("source exclusion differs from its registered reason")
        elif receipt.status == "not_applicable":
            raise ValueError("this operation cannot return a source exclusion")
        if isinstance(data.operation, PredictOperation):
            operation = data.operation
            if receipt.status != "complete" or len(receipt.results) != len(
                operation.clips
            ):
                raise ValueError("prediction stage must return one receipt per clip")
            task = next(
                (task for task in self.plan.tasks if task.task == operation.task), None
            )
            if task is None:
                raise ValueError("prediction task is absent from the frozen plan")
            for clip, ref in zip(operation.clips, receipt.results, strict=True):
                manifest = read_document(self.layout, clip, ClipManifest)
                expected = PredictionContext(
                    task=task.task,
                    manifest=clip,
                    clip_id=manifest.clip_id,
                    model=task.model,
                    configuration=task.inference,
                )
                prediction = read_document(self.layout, ref, PredictionReceipt)
                if (
                    prediction.context != expected
                    or prediction.prediction.request_id != expected.request_id
                ):
                    raise ValueError(
                        "prediction receipt differs from the frozen operation"
                    )
                for dependency in (*references(prediction), *references(manifest)):
                    verify_reference(self.layout, dependency)
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
                self._admit_service(launch.attempt, service)
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
        return {
            "collection": self.collection.ledger.summary(self.plan),
            "progress": progress,
            "resources": sample.model_dump(mode="json"),
            "workflow": production,
        }

    def _stage_states(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from src.tasks.base.annotation.learning.contracts import StageInput

        jobs: list[dict[str, Any]] = []
        keys: list[tuple[str, str]] = []
        latest: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
        for path in sorted(
            self.layout.path("campaigns", self.plan.campaign_id, "jobs").glob(
                "*/operation.json"
            )
        ):
            operation_ref = annotation_reference(self.layout, path)
            data = read_document(self.layout, operation_ref, OperationInput)
            identity, number = path.parent.name.rsplit("-", 1)
            if (
                data.plan != self.runtime.plan
                or identity != data.logical_id
                or not number.isdigit()
                or int(number) < 1
            ):
                raise ValueError("stage attempt differs from its immutable operation")
            launch_path = path.parent / "launch.json"
            if launch_path.exists():
                launch_ref = annotation_reference(self.layout, launch_path)
                launch = read_document(self.layout, launch_ref, StageLaunch)
                if (
                    launch.input != operation_ref
                    or launch.resource.request_id != path.parent.name
                    or launch.resource.plan != self.runtime.plan
                ):
                    raise ValueError("stage launch differs from its operation/attempt")
                state: dict[str, Any] = self.stages.status(launch_ref)
            else:
                state = {
                    "state": "pending",
                    "reason": "operation has no launch receipt",
                }
            state = {
                **state,
                "operation": operation_ref.model_dump(),
                "attempt": int(number),
            }
            key = (identity, "operation")
            jobs.append(state)
            keys.append(key)
            if key not in latest or int(number) > latest[key][0]:
                latest[key] = (int(number), state)
        for path in sorted(
            self.layout.path("campaigns", self.plan.campaign_id, "jobs").glob(
                "*/launch.json"
            )
        ):
            if (path.parent / "operation.json").exists():
                continue
            launch_ref = annotation_reference(self.layout, path)
            launch = read_document(self.layout, launch_ref, StageLaunch)
            if launch.input is None or launch.resource.plan != self.runtime.plan:
                raise ValueError("campaign job has no matching input/plan")
            if not launch.resource.request_id.startswith("learning-"):
                raise ValueError("operation launch has no immutable operation")
            request = read_document(self.layout, launch.input, StageInput)
            if launch.resource.request_id != "learning-" + request.sha256:
                raise ValueError("learning attempt identity differs from its input")
            key = (request.intent.digest, request.stage)
            state = {**self.stages.status(launch_ref), "attempt": request.attempt}
            jobs.append(state)
            keys.append(key)
            if key not in latest or request.attempt > latest[key][0]:
                latest[key] = (request.attempt, state)
        for key, state in zip(keys, jobs, strict=True):
            state["superseded"] = state["attempt"] < latest[key][0]
        return jobs, [state for _, state in latest.values()]

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
                # A verified acquisition exclusion still reconciles its saved
                # acquisition; preparation exclusions do not complete preparation.
                if receipt.status != "complete" and not isinstance(
                    operation, AcquireOperation
                ):
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

    def status(self) -> dict[str, Any]:
        reconciled = self._reconcile_workers()
        jobs, latest = self._stage_states()
        completed: set[ArtifactRef] = set()
        for stage_state in latest:
            if stage_state["state"] != "succeeded" or "operation" not in stage_state:
                continue
            try:
                receipt = self.receipt(stage_state)
                completed.update(receipt.results)
            except (ValueError, OSError) as error:
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
