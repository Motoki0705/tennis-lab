"""Restartable local worker lifecycle, independent of any label/model implementation."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

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
from ..plan import CampaignPlan, load_plan, references, save_plan
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
        timeout_seconds: float = 3600,
        test_command: tuple[str, ...] | None = None,
    ) -> None:
        if test_command is not None and layout.mode != "test":
            raise ValueError("test commands are forbidden in production")
        self.layout, self.plan = layout, plan
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

    def _adapter(self, request: WorkRequest) -> TaskAdapter:
        adapter = self.adapters.get(request.adapter)
        if adapter is None or adapter.version != request.adapter_version:
            raise ValueError("exact requested adapter/version is not registered")
        return adapter

    def register(self, request: WorkRequest) -> None:
        self._adapter(request)
        self.ledger.register(self.plan, request)

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
        result: ArtifactRef | None = None,
    ) -> None:
        with self.ledger.registry.transaction() as database:
            database.execute(
                "UPDATE worker_tasks SET state=?,reason=?,result=? WHERE id=?",
                (
                    state,
                    reason,
                    None if result is None else result.canonical_bytes().decode(),
                    work_id,
                ),
            )
            database.execute(
                "UPDATE worker_attempts SET state=?,reason=? WHERE id=(SELECT current_attempt FROM worker_tasks WHERE id=?)",
                (state, reason, work_id),
            )

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
                    _ref, launch = self._launch(row)
                    directory = Path(launch.directory)
                    self._require_no_members(directory)
                    if (directory / "exit.json").exists():
                        self._mark(row["id"], "exited", None)
                    else:
                        self._mark(
                            row["id"],
                            "failed",
                            "supervisor disappeared without an exit receipt",
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

    def start(self, request: WorkRequest, *, retry: bool = False) -> str | None:
        self.register(request)
        # Registration is idempotent; successful results and pending finalization never rerun a model.
        state = self.reconcile(request)
        if state["state"] in {"complete", "not_applicable"}:
            return None
        require_process_backend()
        binary = shutil.which(self.binary)
        if self.test_command is None and binary is None:
            raise FileNotFoundError(f"Codex executable not found: {self.binary}")
        try:
            with (
                artifact_lock(self.layout, "worker-admission"),
                artifact_lock(
                    self.layout, "worker-" + request.work_id, blocking=False
                ) as descriptor,
            ):
                self._settle_dead_slots()
                row = self.ledger.task(request.work_id)
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
                code_root = Path(__file__).resolve().parents[5]
                launch = Launch(
                    work_id=request.work_id,
                    attempt=attempt,
                    plan=annotation_reference(
                        self.layout, save_plan(self.layout, self.plan)
                    ),
                    request=request,
                    prepared=prepared,
                    directory=str(directory),
                    code_root=str(code_root),
                    codex_binary=binary or self.binary,
                    codex_home=str(
                        Path(os.environ.get("CODEX_HOME", "~/.codex"))
                        .expanduser()
                        .resolve()
                    ),
                    timeout_seconds=self.timeout,
                    test_command=self.test_command,
                    test_root=str(self.layout.root)
                    if self.layout.mode == "test"
                    else None,
                )
                ref = write_document(self.layout, directory / "launch.json", launch)
                with self.ledger.registry.transaction() as database:
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
                    "src.tasks.base.annotation.workers.supervisor",
                    "--launch",
                    str(directory / "launch.json"),
                    "--sha256",
                    ref.digest,
                    "--lock-fd",
                    str(descriptor),
                ]
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
                            pass_fds=(descriptor,),
                        )
                    self.children[attempt] = process
                except OSError as error:
                    self._mark(
                        request.work_id, "failed", f"supervisor launch failed: {error}"
                    )
                    raise
                return attempt
        except BlockingIOError as error:
            raise WorkerBusy("work is owned by an existing supervisor") from error

    def reconcile(self, request: WorkRequest) -> dict[str, Any]:
        self._adapter(request)
        row = self.ledger.task(request.work_id)
        if (row["campaign"], row["plan_digest"], row["request"]) != (
            self.plan.campaign_id,
            self.plan.sha256,
            request.canonical_bytes().decode(),
        ):
            raise ValueError(
                "reconciliation request differs from the registered campaign"
            )
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
            for reference in references(final):
                verify_reference(self.layout, reference)
            return row
        try:
            with artifact_lock(
                self.layout, "worker-" + request.work_id, blocking=False
            ):
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
                    )
                    return self.ledger.task(request.work_id)
                exit_ref = annotation_reference(self.layout, exit_path)
                execution = read_document(self.layout, exit_ref, ExitReceipt)
                if execution.launch_digest != ref.digest:
                    raise ValueError("exit receipt belongs to another launch")
                for reference in references(execution):
                    verify_reference(self.layout, reference)
                if execution.status != "succeeded":
                    self._mark(request.work_id, execution.status, execution.reason)
                    if execution.status == "quota":
                        self.ledger.stop_campaign(
                            self.plan.campaign_id,
                            execution.reason or "Codex quota exhausted",
                        )
                    return self.ledger.task(request.work_id)
                final_path = directory / "result.receipt.json"
                if final_path.exists():
                    final_ref = annotation_reference(self.layout, final_path)
                    final = read_document(self.layout, final_ref, AttemptResult)
                    if (final.launch, final.exit) != (ref, exit_ref):
                        raise ValueError("finalized result belongs to another attempt")
                    for reference in references(final):
                        verify_reference(self.layout, reference)
                else:
                    try:
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
                            launch=ref, exit=exit_ref, result=result, outcome=outcome
                        )
                        final_ref = write_document(self.layout, final_path, final)
                    except (ValueError, OSError) as error:
                        self._mark(
                            request.work_id,
                            "validation_failed",
                            f"{type(error).__name__}: {error}",
                        )
                        return self.ledger.task(request.work_id)
                self._mark(
                    request.work_id,
                    final.outcome.status,
                    final.outcome.reason,
                    result=final_ref,
                )
                return self.ledger.task(request.work_id)
        except BlockingIOError:
            return row

    def stop(self, request: WorkRequest, *, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise ValueError("stop requires a reason")
        row = self.ledger.task(request.work_id)
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
        current = self.reconcile(request)
        if current["state"] in {"complete", "not_applicable"}:
            return current
        self._mark(request.work_id, "cancelled", reason)
        return self.ledger.task(request.work_id)
