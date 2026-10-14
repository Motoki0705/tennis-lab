"""Inherited flock survives dispatcher death; receipts precede releasing ownership."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from ..artifacts import (
    annotation_reference,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ..paths import AnnotationLayout
from ..plan import load_plan, references
from ..reproducibility import require_frozen_code
from ..resources import ResourceManager
from .command import codex_command, worker_environment
from .contracts import ChildIdentity, ChildReceipt, ExitReceipt, Launch
from .processes import capture_supervisor, owned_members, process_snapshot, signal_owned


def event_summary(path: Path) -> tuple[str | None, bool, str | None, bool]:
    thread: str | None = None
    completed, failure, quota = False, None, False
    with path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                failure = "worker emitted malformed JSON events"
                continue
            if not isinstance(event, dict):
                failure = "worker emitted a non-object event"
                continue
            if event.get("type") == "thread.started":
                value = event.get("thread_id")
                if (
                    not isinstance(value, str)
                    or not value
                    or (thread is not None and thread != value)
                ):
                    failure = "worker thread identity is missing or changed"
                else:
                    thread = value
            elif event.get("type") == "turn.completed":
                completed = True
            elif event.get("type") in {"error", "turn.failed"}:
                detail = event.get("error", event.get("message", "worker error"))
                if isinstance(detail, dict):
                    detail = detail.get("message", "worker error")
                failure = str(detail)[:2000]
                quota = quota or any(
                    word in failure.lower()
                    for word in (
                        "usage limit",
                        "rate limit",
                        "quota",
                        "usage_limit",
                        "rate_limit",
                    )
                )
    return thread, completed, failure, quota


def supervise(
    path: Path,
    expected_digest: str,
    lock_fd: int,
    *,
    test_root: Path | None = None,
    resource_fd: int | None = None,
) -> int:
    layout = (
        AnnotationLayout.testing(test_root)
        if test_root is not None
        else AnnotationLayout.production()
    )
    ref = annotation_reference(layout, path)
    if ref.digest != expected_digest:
        raise ValueError("launch artifact changed")
    launch = read_document(layout, ref, Launch)
    directory = Path(launch.directory)
    if path != directory / "launch.json" or not directory.is_relative_to(
        layout.path("campaigns")
    ):
        raise ValueError("launch is outside its declared attempt")
    lock_path = layout.path("cache", "locks", "worker-" + launch.work_id + ".lock")
    if (os.fstat(lock_fd).st_dev, os.fstat(lock_fd).st_ino) != (
        lock_path.stat().st_dev,
        lock_path.stat().st_ino,
    ):
        raise ValueError("supervisor did not inherit its exact work lock")
    if layout.mode != "test" and (
        launch.test_command is not None or launch.test_root is not None
    ):
        raise ValueError("test command/root is forbidden in production")
    plan = load_plan(verify_reference(layout, launch.plan))
    if layout.mode == "production":
        require_frozen_code(Path(launch.code_root), plan)
    for reference in references(launch.request):
        verify_reference(layout, reference)
    owner = capture_supervisor(os.getpid())
    write_asset(
        layout,
        directory / "owner.json",
        (json.dumps(owner.document(), sort_keys=True) + "\n").encode(),
    )
    resources = None
    if launch.resource_scope is not None:
        if resource_fd is None or launch.resource_request is None:
            raise ValueError("resource lease descriptor/request is missing")
        if (launch.resource_request.plan, launch.resource_request.stage) != (
            launch.plan,
            launch.request.stage,
        ):
            raise ValueError("resource reservation differs from the worker plan/stage")
        resources = ResourceManager(layout, launch.resource_scope)
        resources.attach(launch.resource_request, resource_fd, owner)
    elif layout.mode != "test":
        raise ValueError("production supervisor requires a resource reservation")
    started_at, started = datetime.now(UTC).isoformat(), time.monotonic()
    status = "failed"
    reason: str | None = None
    code = 127
    child: subprocess.Popen[bytes] | None = None

    def cancelled(_number: int, _frame: object) -> None:
        raise InterruptedError("worker stopped by its owning dispatcher")

    signal.signal(signal.SIGTERM, cancelled)
    signal.signal(signal.SIGINT, cancelled)
    try:
        with (
            verify_reference(layout, launch.prepared.prompt).open("rb") as prompt,
            (directory / "events.jsonl").open("xb") as events,
            (directory / "stderr.log").open("xb") as errors,
        ):
            try:
                command = (
                    list(launch.test_command)
                    if launch.test_command is not None
                    else codex_command(launch, plan)
                )
                child = subprocess.Popen(
                    command,
                    stdin=prompt,
                    stdout=events,
                    stderr=errors,
                    cwd=directory / "work",
                    env=worker_environment(launch, plan),
                )
                snapshot = process_snapshot(child.pid)
                write_document(
                    layout,
                    directory / "child.json",
                    ChildReceipt(
                        launch_digest=ref.digest,
                        captured_at=datetime.now(UTC).isoformat(),
                        process=None
                        if snapshot is None
                        else ChildIdentity(**snapshot[0].document()),
                        unavailable_reason="child exited before its identity was captured"
                        if snapshot is None
                        else None,
                    ),
                )
                deadline = time.monotonic() + launch.timeout_seconds
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, launch.timeout_seconds)
                    try:
                        code = child.wait(timeout=min(2.0, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        if resources is not None:
                            # Budget exhaustion latches a stop for NEW submissions.
                            # This worker still has its frozen timeout and reservation.
                            resources.watch()
            except subprocess.TimeoutExpired:
                status, reason, code = (
                    "timed_out",
                    "worker exceeded its frozen timeout",
                    124,
                )
            except InterruptedError as error:
                status, reason, code = "cancelled", str(error), 130
            except OSError as error:
                status, reason, code = "failed", f"{type(error).__name__}: {error}", 127
            except (ValueError, RuntimeError, KeyError) as error:
                status, reason, code = (
                    "failed",
                    f"resource monitoring failed: {type(error).__name__}: {error}",
                    1,
                )
            finally:
                # Codex/tools share this session. Clean descendants before an attempt can finish.
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal_owned(owner, signal.SIGTERM, exclude_leader=True)
                deadline = time.monotonic() + 2
                while (
                    any(member.pid != owner.pid for member in owned_members(owner))
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                signal_owned(owner, signal.SIGKILL, exclude_leader=True)
                if child is not None:
                    child.wait(timeout=5)
                events.flush()
                errors.flush()
                os.fsync(events.fileno())
                os.fsync(errors.fileno())
    finally:
        # Closing the inherited descriptor is deliberately left to process exit.
        pass
    thread, completed, failure, quota = event_summary(directory / "events.jsonl")
    if status not in {"timed_out", "cancelled"}:
        if quota:
            status, reason = "quota", failure
        elif code == 0 and completed and thread is not None and failure is None:
            status, reason = "succeeded", None
        else:
            status, reason = (
                "failed",
                reason
                or failure
                or "worker did not finish a successful, identified turn",
            )
    write_document(
        layout,
        directory / "exit.json",
        ExitReceipt(
            launch_digest=ref.digest,
            exit_code=code,
            status=status,
            started_at=started_at,
            ended_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            events=annotation_reference(layout, directory / "events.jsonl"),
            errors=annotation_reference(layout, directory / "stderr.log"),
            thread_id=thread,
            reason=reason,
        ),
    )
    if resources is not None and launch.resource_request is not None:
        resources.finish(launch.resource_request, reason=reason)
    return code if status == "succeeded" else (code or 1)
