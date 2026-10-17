"""Restartable local worker lifecycle, independent of any label/model implementation."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

from src.utils.paths import PROJECT_ROOT

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ..contracts import ArtifactRef
from ..paths import AnnotationLayout
from ..plan import (
    CampaignPlan,
    load_plan,
    references,
    require_execution_layout,
    save_plan,
)
from ..reproducibility import require_frozen_code
from ..resources import ResourceManager, ResourceRequest
from ..telemetry.capture import capture, presented_images, verify_report
from .contracts import (
    AttemptResult,
    ExitReceipt,
    Launch,
    TaskAdapter,
    WorkOutcome,
    WorkRequest,
)
from .ledger import WorkerBusy, WorkerLedger, WorkerStopped
from .processes import (
    ProcessIdentity,
    owned_members,
    require_process_backend,
    signal_owned,
)


class WorkerPool:
    def __init__(
        self,
        layout: AnnotationLayout,
        plan: CampaignPlan,
        adapters: tuple[TaskAdapter, ...],
        *,
        codex_binary: str = "codex",
        codex_home: Path | None = None,
        timeout_seconds: float = 3600,
        test_command: tuple[str, ...] | None = None,
        resources: ResourceManager | None = None,
        worker_pss_bytes: int | None = None,
        worker_storage_bytes: int | None = None,
        code_revision: ArtifactRef | None = None,
    ) -> None:
        require_execution_layout(layout, plan)
        if test_command is not None and layout.mode != "test":
            raise ValueError("test commands are forbidden in production")
        if layout.mode == "production" and resources is None:
            raise ValueError("production workers require a persistent resource budget")
        if resources is not None:
            resources.bind(plan)
            resources.observe(os.getpid())
            if worker_pss_bytes is None or worker_storage_bytes is None:
                raise ValueError("worker PSS/storage reservations must be explicit")
        self.resources = resources
        self.code_revision = code_revision
        self.worker_pss_bytes = worker_pss_bytes
        self.worker_storage_bytes = worker_storage_bytes
        self.layout, self.plan = layout, plan
        self.codex_home = (
            codex_home or Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
        ).resolve()
        self.adapters = {adapter.name: adapter for adapter in adapters}
        if len(self.adapters) != len(adapters):
            raise ValueError("adapter names must be unique")
        self.ledger = WorkerLedger(layout)
        self.binary, self.timeout, self.test_command = (
            codex_binary,
            timeout_seconds,
            test_command,
        )
        self.children: dict[str, subprocess.Popen[bytes]] = {}

    def _task(self, work_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self.ledger.task(work_id)
        return result

    def _telemetry(
        self,
        request: WorkRequest,
        launch: Launch,
        launch_ref: ArtifactRef,
        exit_ref: ArtifactRef,
    ) -> ArtifactRef | None:
        if launch.test_command is not None:
            return None
        adapter = self._adapter(request)
        directory = Path(launch.directory)
        catalog = adapter.telemetry_images(self.layout, self.plan, request, directory)
        ref = capture(
            self.layout,
            launch_ref,
            exit_ref,
            catalog,
            classifier=adapter.classify_tool,
            classifier_id=f"{adapter.name}.{adapter.version}",
        )
        write_document(self.layout, directory / "telemetry.ref.json", ref)
        return ref

    def _verify_final(self, final: AttemptResult) -> None:
        for reference in references(final):
            verify_reference(self.layout, reference)
        if final.telemetry is not None:
            report = verify_report(self.layout, final.telemetry)
            if (report.launch, report.execution) != (final.launch, final.exit):
                raise ValueError("telemetry belongs to another attempt")
        elif read_document(self.layout, final.launch, Launch).test_command is None:
            raise ValueError("a real Codex result requires rollout telemetry")

    def _adapter(self, request: WorkRequest) -> TaskAdapter:
        adapter = self.adapters.get(request.adapter)
        if adapter is None or adapter.version != request.adapter_version:
            raise ValueError("exact requested adapter/version is not registered")
        return adapter

    def register(self, request: WorkRequest) -> None:
        adapter = self._adapter(request)
        self.ledger.register(
            self.plan,
            request,
            dependencies=adapter.input_references(self.layout, self.plan, request),
        )

    def _launch(self, row: dict[str, Any]) -> tuple[ArtifactRef, Launch]:
        with self.ledger.registry.transaction() as database:
            attempt = database.execute(
                "SELECT launch FROM worker_attempts WHERE id=?",
                (row["current_attempt"],),
            ).fetchone()
        if attempt is None:
            raise ValueError("current worker attempt has no immutable launch record")
        ref = ArtifactRef.model_validate_json(attempt["launch"])
        return ref, read_document(self.layout, ref, Launch)

    def _mark(
        self,
        work_id: str,
        state: str,
        reason: str | None,
        *,
        attempt: str,
        from_states: tuple[str, ...] = ("running", "exited"),
        result: ArtifactRef | None = None,
    ) -> bool:
        with self.ledger.registry.transaction() as database:
            row = database.execute(
                "SELECT * FROM worker_tasks WHERE id=?", (work_id,)
            ).fetchone()
            if row is None:
                raise KeyError(work_id)
            if row["current_attempt"] != attempt or row["state"] not in from_states:
                return False
            database.execute(
                "UPDATE worker_tasks SET state=?,reason=?,result=? WHERE id=? AND current_attempt=? AND state=?",
                (
                    state,
                    reason,
                    None if result is None else result.canonical_bytes().decode(),
                    work_id,
                    attempt,
                    row["state"],
                ),
            )
            updated = database.execute(
                "UPDATE worker_attempts SET state=?,reason=? WHERE id=? AND work_id=? AND state=?",
                (state, reason, attempt, work_id, row["state"]),
            )
            if updated.rowcount != 1:
                raise ValueError("task and attempt states disagree")
            if state == "quota":
                database.execute(
                    "INSERT OR REPLACE INTO worker_campaign_stops VALUES (?,?)",
                    (row["campaign"], reason or "Codex quota exhausted"),
                )
            if state not in {"running", "exited"}:
                self.ledger.storage.settle(attempt, database=database)
        return True

    def _exit_receipt(
        self, launch_ref: ArtifactRef, directory: Path
    ) -> tuple[ArtifactRef, ExitReceipt]:
        ref = annotation_reference(self.layout, directory / "exit.json")
        execution = read_document(self.layout, ref, ExitReceipt)
        if execution.launch_digest != launch_ref.digest:
            raise ValueError("exit receipt belongs to another launch")
        for reference in references(execution):
            verify_reference(self.layout, reference)
        return ref, execution

    def _settle_dead_slots(self) -> None:
        # Reclaim resource accounting after dispatcher death without adopting labels.
        with self.ledger.registry.transaction() as database:
            rows = [
                dict(row)
                for row in database.execute(
                    "SELECT * FROM worker_tasks WHERE state='running'"
                )
            ]
        for row in rows:
            try:
                with artifact_lock(self.layout, "worker-" + row["id"], blocking=False):
                    row = self._task(row["id"])
                    if row["state"] != "running":
                        continue
                    ref, launch = self._launch(row)
                    directory = Path(launch.directory)
                    self._require_no_members(directory)
                    if (directory / "exit.json").exists():
                        _exit_ref, execution = self._exit_receipt(ref, directory)
                        self._mark(
                            row["id"],
                            "quota" if execution.status == "quota" else "exited",
                            execution.reason if execution.status == "quota" else None,
                            attempt=launch.attempt,
                            from_states=("running",),
                        )
                    else:
                        self._mark(
                            row["id"],
                            "failed",
                            "supervisor disappeared without an exit receipt",
                            attempt=launch.attempt,
                            from_states=("running",),
                        )
            except BlockingIOError:
                continue

    @staticmethod
    def _require_no_members(directory: Path) -> None:
        owner_path = directory / "owner.json"
        if owner_path.exists():
            if owner_path.is_symlink():
                raise ValueError("owner receipt cannot be a symlink")
            owner = ProcessIdentity.from_document(json.loads(owner_path.read_text()))
            if owned_members(owner):
                raise WorkerBusy(
                    "attempt still has owned processes, including orphaned descendants"
                )

    def _recover_unregistered_attempt(
        self, request: WorkRequest, *, cancel_reason: str | None = None
    ) -> None:
        """Account for a resource admission interrupted before worker registration.

        The caller holds both admission and work locks. Pending requests have never
        acquired resources and retain their ID for normal backpressure retries,
        unless a terminal stop explicitly cancels them.
        """
        if self.resources is None:
            return
        with self.ledger.registry.transaction() as database:
            number = (
                database.execute(
                    "SELECT count(*) FROM worker_attempts WHERE work_id=?",
                    (request.work_id,),
                ).fetchone()[0]
                + 1
            )
        attempt = f"{request.work_id}-{number:04d}"
        path = self.layout.path(
            "campaigns",
            self.plan.campaign_id,
            "workers",
            request.work_id,
            f"attempt-{number:04d}",
            "launch.json",
        )
        if not path.exists():
            return
        ref = annotation_reference(self.layout, path)
        launch = read_document(self.layout, ref, Launch)
        if (
            (launch.work_id, launch.attempt, launch.request, launch.resource_scope)
            != (request.work_id, attempt, request, self.resources.reference)
            or load_plan(verify_reference(self.layout, launch.plan)) != self.plan
            or launch.resource_request is None
        ):
            raise ValueError("unregistered launch differs from the worker request")
        with self.resources.registry.transaction() as database:
            resource = database.execute(
                "SELECT request,state,reason FROM resource_requests WHERE id=?",
                (self.resources.key(launch.resource_request),),
            ).fetchone()
        if resource is None:
            return
        if resource["request"] != launch.resource_request.canonical_bytes().decode():
            raise ValueError("unregistered launch has a different resource request")
        if resource["state"] == "pending":
            if cancel_reason is None:
                return
            self.resources.cancel_pending(launch.resource_request, reason=cancel_reason)
            state, reason = "cancelled", cancel_reason
        elif resource["state"] == "cancelled":
            # Recover a stop interrupted between the shared and worker ledgers.
            state, reason = "cancelled", resource["reason"]
        else:
            state = "failed"
            reason = "resource admission interrupted before worker registration"
        with self.ledger.registry.transaction() as database:
            database.execute(
                "INSERT INTO worker_attempts VALUES (?,?,?,?,?,?)",
                (
                    attempt,
                    request.work_id,
                    number,
                    state,
                    ref.canonical_bytes().decode(),
                    reason,
                ),
            )
            database.execute(
                "UPDATE worker_tasks SET state=?,current_attempt=?,reason=?,result=NULL WHERE id=?",
                (state, attempt, reason, request.work_id),
            )

    def start(self, request: WorkRequest, *, retry: bool = False) -> str | None:
        self.register(request)
        # Registration is idempotent; successful results and pending finalization never rerun a model.
        state = self.reconcile(request)
        if state["state"] in {"complete", "not_applicable"}:
            return None
        if self.layout.mode == "production":
            require_frozen_code(
                PROJECT_ROOT,
                self.plan,
                layout=self.layout,
                revision=self.code_revision,
                scope=self.resources.reference if self.resources is not None else None,
            )
        require_process_backend()
        binary = shutil.which(self.binary)
        if self.test_command is None and binary is None:
            raise FileNotFoundError(f"Codex executable not found: {self.binary}")
        try:
            with (
                ExitStack() as resource_stack,
                artifact_lock(self.layout, "worker-admission"),
                artifact_lock(
                    self.layout, "worker-" + request.work_id, blocking=False
                ) as descriptor,
                artifact_lock(self.layout, "worker-services-" + request.work_id),
            ):
                self._recover_unregistered_attempt(request)
                self._settle_dead_slots()
                row = self._task(request.work_id)
                if row["state"] in {"complete", "not_applicable"}:
                    return None
                if row["state"] != "ready" and not retry:
                    raise WorkerStopped(
                        f"explicit retry required: {row['state']}: {row['reason']}"
                    )
                if row["state"] in {"running", "exited"}:
                    raise WorkerBusy("existing attempt requires reconciliation")
                with self.ledger.registry.transaction() as database:
                    stopped = database.execute(
                        "SELECT reason FROM worker_campaign_stops WHERE campaign=?",
                        (self.plan.campaign_id,),
                    ).fetchone()
                    active = [
                        dict(item)
                        for item in database.execute(
                            "SELECT * FROM worker_tasks WHERE state='running'"
                        )
                    ]
                    number = (
                        database.execute(
                            "SELECT count(*) FROM worker_attempts WHERE work_id=?",
                            (request.work_id,),
                        ).fetchone()[0]
                        + 1
                    )
                if stopped is not None:
                    raise WorkerStopped(stopped["reason"])
                limits = [self.plan.resources.codex_workers]
                stage_limit = next(
                    stage.parallel
                    for stage in self.plan.resources.stages
                    if stage.stage == request.stage
                )
                stage_count = 0
                for item in active:
                    _ref, launch = self._launch(item)
                    active_plan = load_plan(verify_reference(self.layout, launch.plan))
                    limits.append(active_plan.resources.codex_workers)
                    if launch.request.stage == request.stage:
                        stage_count += 1
                        stage_limit = min(
                            stage_limit,
                            next(
                                stage.parallel
                                for stage in active_plan.resources.stages
                                if stage.stage == request.stage
                            ),
                        )
                if len(active) >= min(limits) or stage_count >= stage_limit:
                    raise WorkerBusy("shared Codex/stage capacity is full")
                attempt = f"{request.work_id}-{number:04d}"
                directory = self.layout.path(
                    "campaigns",
                    self.plan.campaign_id,
                    "workers",
                    request.work_id,
                    f"attempt-{number:04d}",
                )
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "work").mkdir(exist_ok=True)
                prepared = self._adapter(request).prepare(
                    self.layout, self.plan, request, directory
                )
                if (
                    self.layout.artifact_path(prepared.prompt)
                    != directory / "prompt.md"
                ):
                    raise ValueError(
                        "adapter must publish an immutable prompt in its own attempt"
                    )
                verify_reference(self.layout, prepared.prompt)
                for relative in prepared.writable_directories:
                    path = directory / relative
                    self.layout.artifact_path(
                        ArtifactRef(
                            role="annotation",
                            path=path.relative_to(self.layout.root).as_posix(),
                            digest="0" * 64,
                            size_bytes=0,
                        )
                    )
                    path.mkdir(parents=True, exist_ok=True)
                code_root = PROJECT_ROOT
                plan_ref = annotation_reference(
                    self.layout, save_plan(self.layout, self.plan)
                )
                resource_request = None
                resource_descriptor = None
                if self.resources is not None:
                    assert (
                        self.worker_pss_bytes is not None
                        and self.worker_storage_bytes is not None
                    )
                    resource_request = ResourceRequest(
                        request_id=("benchmark-" + self.plan.campaign_id + "-")
                        + attempt
                        if self.plan.phase == "benchmark"
                        else "worker-" + attempt,
                        plan=plan_ref,
                        stage=request.stage,
                        pss_bytes=self.worker_pss_bytes,
                        storage_bytes=self.worker_storage_bytes,
                        gpu_seconds=0.0,
                    )
                launch = Launch(
                    code_revision=self.code_revision,
                    work_id=request.work_id,
                    attempt=attempt,
                    plan=plan_ref,
                    resource_scope=None
                    if self.resources is None
                    else self.resources.reference,
                    resource_request=resource_request,
                    request=request,
                    prepared=prepared,
                    directory=str(directory),
                    code_root=str(code_root),
                    codex_binary=binary or self.binary,
                    codex_home=str(self.codex_home),
                    timeout_seconds=self.timeout,
                    test_command=self.test_command,
                    test_root=str(self.layout.root)
                    if self.layout.mode == "test"
                    else None,
                )
                ref = write_document(self.layout, directory / "launch.json", launch)
                if self.resources is not None and resource_request is not None:
                    resource_descriptor = resource_stack.enter_context(
                        self.resources.lease(resource_request)
                    )
                with self.ledger.registry.transaction() as database:
                    self.ledger.storage.launch(ref, launch, database=database)
                    database.execute(
                        "INSERT INTO worker_attempts VALUES (?,?,?,'running',?,NULL)",
                        (
                            attempt,
                            request.work_id,
                            number,
                            ref.canonical_bytes().decode(),
                        ),
                    )
                    database.execute(
                        "UPDATE worker_tasks SET state='running',current_attempt=?,reason=NULL,result=NULL WHERE id=?",
                        (attempt, request.work_id),
                    )
                command = [
                    str(code_root / ".venv/bin/python"),
                    "-m",
                    "src.tasks.base.annotation.workers",
                    "--launch",
                    str(directory / "launch.json"),
                    "--sha256",
                    ref.digest,
                    "--lock-fd",
                    str(descriptor),
                ]
                if resource_descriptor is not None:
                    command.extend(("--resource-fd", str(resource_descriptor)))
                if self.layout.mode == "test":
                    command.extend(("--test-root", str(self.layout.root)))
                environment = os.environ.copy()
                environment["PYTHONPATH"] = str(code_root)
                try:
                    with (directory / "supervisor.log").open("ab") as log:
                        process = subprocess.Popen(
                            command,
                            cwd=code_root,
                            env=environment,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                            pass_fds=(descriptor,)
                            if resource_descriptor is None
                            else (descriptor, resource_descriptor),
                        )
                    self.children[attempt] = process
                except OSError as error:
                    self._mark(
                        request.work_id,
                        "failed",
                        f"supervisor launch failed: {error}",
                        attempt=attempt,
                        from_states=("running",),
                    )
                    raise
                return attempt
        except BlockingIOError as error:
            raise WorkerBusy("work is owned by an existing supervisor") from error

    def _registered_task(self, request: WorkRequest) -> dict[str, Any]:
        self._adapter(request)
        row = self._task(request.work_id)
        if (row["campaign"], row["plan_digest"], row["request"]) != (
            self.plan.campaign_id,
            self.plan.sha256,
            request.canonical_bytes().decode(),
        ):
            raise ValueError("worker request differs from the registered campaign")
        return row

    def reconcile(self, request: WorkRequest) -> dict[str, Any]:
        self._registered_task(request)
        try:
            with (
                artifact_lock(self.layout, "worker-" + request.work_id, blocking=False),
                artifact_lock(self.layout, "worker-services-" + request.work_id),
            ):
                return self._reconcile_locked(request)
        except BlockingIOError:
            return self._task(request.work_id)

    def _reconcile_locked(self, request: WorkRequest) -> dict[str, Any]:
        # The caller holds the work lock through receipt recovery and state updates.
        row = self._task(request.work_id)
        if row["state"] in {
            "ready",
            "failed",
            "deferred",
            "cancelled",
            "quota",
            "timed_out",
            "validation_failed",
        }:
            return row
        if row["state"] in {"complete", "not_applicable"}:
            final = read_document(
                self.layout,
                ArtifactRef.model_validate_json(row["result"]),
                AttemptResult,
            )
            self._verify_final(final)
            return row
        ref, launch = self._launch(row)
        directory = Path(launch.directory)
        self._require_no_members(directory)
        child = self.children.pop(launch.attempt, None)
        if child is not None:
            child.wait(timeout=1)
        elif (directory / "owner.json").exists():
            owner = ProcessIdentity.from_document(
                json.loads((directory / "owner.json").read_text())
            )
            with suppress(ChildProcessError):
                os.waitpid(owner.pid, os.WNOHANG)
        exit_path = directory / "exit.json"
        if not exit_path.exists():
            self._mark(
                request.work_id,
                "failed",
                "supervisor disappeared without an exit receipt",
                attempt=launch.attempt,
            )
            return self._task(request.work_id)
        exit_ref, execution = self._exit_receipt(ref, directory)
        if execution.status != "succeeded":
            if launch.test_command is None:
                try:
                    self._telemetry(request, launch, ref, exit_ref)
                except (ValueError, OSError) as error:
                    write_asset(
                        self.layout,
                        directory / "telemetry-error.json",
                        (
                            json.dumps(
                                {
                                    "launch": ref.model_dump(),
                                    "execution": exit_ref.model_dump(),
                                    "error": f"{type(error).__name__}: {error}",
                                }
                            )
                            + "\n"
                        ).encode(),
                    )
            self._mark(
                request.work_id,
                execution.status,
                execution.reason,
                attempt=launch.attempt,
            )
            return self._task(request.work_id)
        final_path = directory / "result.receipt.json"
        if final_path.exists():
            final_ref = annotation_reference(self.layout, final_path)
            final = read_document(self.layout, final_ref, AttemptResult)
            if (final.launch, final.exit) != (ref, exit_ref):
                raise ValueError("finalized result belongs to another attempt")
            self._verify_final(final)
        else:
            try:
                telemetry = self._telemetry(request, launch, ref, exit_ref)
                if telemetry is not None:
                    # Enforce the pinned model and unambiguous response IDs;
                    # the task decides which bound images must be reviewed.
                    presented_images(self.layout, telemetry)
                raw_result = directory / launch.prepared.result_path
                result = write_asset(
                    self.layout,
                    directory / "result.json",
                    verify_reference(
                        self.layout,
                        annotation_reference(self.layout, raw_result),
                    ).read_bytes(),
                )
                outcome = self._adapter(request).validate(
                    self.layout,
                    self.plan,
                    request,
                    directory,
                    result,
                    execution,
                    telemetry,
                )
                outcome = WorkOutcome.model_validate_json(
                    outcome.canonical_bytes()
                )
                if (
                    outcome.adoption is not None
                    and not self.plan.allows_adoption
                ):
                    raise ValueError("preview campaigns cannot adopt teachers")
                for reference in references(outcome):
                    verify_reference(self.layout, reference)
                final = AttemptResult(
                    launch=ref,
                    exit=exit_ref,
                    result=result,
                    outcome=outcome,
                    telemetry=telemetry,
                )
                final_ref = write_document(self.layout, final_path, final)
            except (ValueError, OSError) as error:
                self._mark(
                    request.work_id,
                    "validation_failed",
                    f"{type(error).__name__}: {error}",
                    attempt=launch.attempt,
                )
                return self._task(request.work_id)
        self._mark(
            request.work_id,
            final.outcome.status,
            final.outcome.reason,
            attempt=launch.attempt,
            result=final_ref,
        )
        return self._task(request.work_id)

    def cancel_pending(self, request: WorkRequest, *, reason: str) -> dict[str, Any]:
        """Cancel an unlaunched resource attempt without stopping active processes."""
        if not reason.strip():
            raise ValueError("cancelling requires a reason")
        row = self._registered_task(request)
        if self.resources is None:
            return row
        with artifact_lock(self.layout, "worker-admission"):
            # A concurrent start must finish admission before its state is checked.
            row = self._registered_task(request)
            if row["state"] in {"running", "exited"}:
                return row
            with artifact_lock(self.layout, "worker-" + request.work_id):
                self._recover_unregistered_attempt(request, cancel_reason=reason)
                return self._task(request.work_id)

    def stop(self, request: WorkRequest, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("stop requires a reason")
        row = self._registered_task(request)
        if row["state"] not in {"running", "exited"}:
            row = self.cancel_pending(request, reason=reason)
            if row["state"] not in {"running", "exited"}:
                return row
        _ref, launch = self._launch(row)
        owner_path = Path(launch.directory) / "owner.json"
        if not owner_path.exists():
            raise WorkerBusy(
                "supervisor is still starting; ownership is not published yet"
            )
        owner = ProcessIdentity.from_document(json.loads(owner_path.read_text()))
        signal_owned(owner, signal.SIGTERM)
        deadline = time.monotonic() + 3
        while owned_members(owner) and time.monotonic() < deadline:
            time.sleep(0.02)
        signal_owned(owner, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while owned_members(owner) and time.monotonic() < deadline:
            time.sleep(0.02)
        if owned_members(owner):
            raise WorkerBusy("owned processes have not exited after cancellation")
        self.reconcile(request)
        current = self._task(request.work_id)
        if current["current_attempt"] != launch.attempt or current["state"] in {
            "complete",
            "not_applicable",
        }:
            return current
        try:
            with (
                artifact_lock(self.layout, "worker-" + request.work_id),
                artifact_lock(self.layout, "worker-services-" + request.work_id),
            ):
                current = self._task(request.work_id)
                if current["current_attempt"] != launch.attempt:
                    return current
                self._require_no_members(Path(launch.directory))
                # Recover any receipt committed while stop waited for these locks.
                current = self._reconcile_locked(request)
                if current["state"] in {"complete", "not_applicable"}:
                    return current
                self._mark(
                    request.work_id,
                    "cancelled",
                    reason,
                    attempt=launch.attempt,
                    from_states=(current["state"],),
                )
        except BlockingIOError as error:
            raise WorkerBusy("attempt is still owned during cancellation") from error
        return self._task(request.work_id)
