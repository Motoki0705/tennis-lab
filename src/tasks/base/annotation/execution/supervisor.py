"""A timeout remains within the queue process group and reaps operation children."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from ..artifacts import (
    annotation_reference,
    artifact_lock,
    read_document,
    verify_reference,
    write_asset,
    write_document,
)
from ..paths import AnnotationLayout
from ..plan import load_plan
from ..resources import ResourceManager
from ..workers.processes import (
    ProcessIdentity,
    capture_supervisor,
    owned_members,
    signal_owned,
)
from .contracts import StageExit, StageLaunch
from .queue import queue_owner
from .runner import stage_directory, stage_lock, validate_launch


def _controllers(owner: ProcessIdentity) -> frozenset[int]:
    """Protect the queue's shell ancestors; only this stage's operation is stopped."""
    protected = {owner.pid}
    pid = os.getpid()
    while pid != owner.pid:
        if pid in protected:
            raise ValueError("stage is not a descendant of its process owner")
        protected.add(pid)
        fields = (
            (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        )
        pid = int(fields[1])
        if pid <= 1:
            raise ValueError("stage escaped the queue's ownership ancestry")
    return frozenset(protected)


def _check_fd(layout: AnnotationLayout, launch: StageLaunch, descriptor: int) -> None:
    path = layout.path("cache", "locks", stage_lock(launch) + ".lock")
    if (os.fstat(descriptor).st_dev, os.fstat(descriptor).st_ino) != (
        path.stat().st_dev,
        path.stat().st_ino,
    ):
        raise ValueError("stage did not inherit its exact job lock")


def supervise(
    layout: AnnotationLayout,
    path: Path,
    digest: str,
    *,
    queued: bool,
    lock_fd: int | None,
    resource_fd: int | None,
) -> int:
    ref = annotation_reference(layout, path)
    if ref.digest != digest:
        raise ValueError("stage launch changed")
    launch = read_document(layout, ref, StageLaunch)
    validate_launch(layout, launch)
    directory = stage_directory(layout, launch)
    if path != directory / "launch.json" or queued != (launch.queue is not None):
        raise ValueError("stage launch path or queue mode is inconsistent")
    manager = ResourceManager(layout, launch.scope)
    with ExitStack() as stack:
        if queued:
            assert launch.queue is not None
            owner = queue_owner(layout, launch.queue)
            lock_fd = stack.enter_context(
                artifact_lock(layout, stage_lock(launch), blocking=False)
            )
            # Idempotent queue replay cannot rerun a completed command.
            if (directory / "exit.json").exists():
                prior: StageExit = read_document(
                    layout,
                    annotation_reference(layout, directory / "exit.json"),
                    StageExit,
                )
                if prior.launch != ref:
                    raise ValueError("stage exit belongs to another launch")
                return prior.exit_code
            resource_fd = stack.enter_context(manager.lease(launch.resource))
        else:
            owner = capture_supervisor(os.getpid())
        if lock_fd is None or resource_fd is None:
            raise ValueError("stage resource/job ownership is missing")
        _check_fd(layout, launch, lock_fd)
        manager.attach(launch.resource, resource_fd, owner)
        protected = _controllers(owner)
        write_asset(
            layout,
            directory / "owner.json",
            (json.dumps(owner.document(), sort_keys=True) + "\n").encode(),
        )
        plan = load_plan(verify_reference(layout, launch.resource.plan))
        stage = next(
            stage
            for stage in plan.resources.stages
            if stage.stage == launch.resource.stage
        )
        environment = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[key] = str(stage.threads_per_process)
        if launch.queue is None:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["PYTHONPATH"] = launch.code_root
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment.pop("TENNIS_ANNOTATION_STAGE_OPERATION", None)
        if launch.input is not None:
            environment["TENNIS_ANNOTATION_STAGE_OPERATION"] = launch.input.digest
        for variable, leaf in (
            ("TMPDIR", "tmp"),
            ("TORCH_HOME", "torch"),
            ("TORCH_EXTENSIONS_DIR", "torch-extensions"),
            ("XDG_CACHE_HOME", "xdg"),
            ("CUDA_CACHE_PATH", "cuda"),
        ):
            cache = layout.path("cache", "runtime", leaf)
            cache.mkdir(parents=True, exist_ok=True)
            environment[variable] = str(cache)
        started, started_at = time.monotonic(), datetime.now(UTC).isoformat()
        child: subprocess.Popen[bytes] | None = None
        status, reason, code = "failed", None, 127

        def cancelled(_number: int, _frame: object) -> None:
            raise InterruptedError("stage was cancelled by its owner")

        signal.signal(signal.SIGTERM, cancelled)
        signal.signal(signal.SIGINT, cancelled)
        with (
            (directory / "stdout.log").open("xb") as output,
            (directory / "stderr.log").open("xb") as error,
        ):
            try:
                child = subprocess.Popen(
                    list(launch.command),
                    cwd=launch.code_root,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=output,
                    stderr=error,
                )
                while True:
                    remaining = launch.timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(
                            launch.command, launch.timeout_seconds
                        )
                    try:
                        code = child.wait(timeout=min(2.0, remaining))
                        status = "succeeded" if code == 0 else "failed"
                        reason = (
                            None if code == 0 else f"operation exited with code {code}"
                        )
                        break
                    except subprocess.TimeoutExpired:
                        manager.watch()
            except subprocess.TimeoutExpired:
                status, reason, code = (
                    "timed_out",
                    "stage exceeded its frozen timeout",
                    124,
                )
            except InterruptedError as failure:
                status, reason, code = "cancelled", str(failure), 130
            except (OSError, ValueError, RuntimeError) as failure:
                status, reason, code = (
                    "failed",
                    f"{type(failure).__name__}: {failure}",
                    1,
                )
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                signal_owned(owner, signal.SIGTERM, exclude_pids=protected)
                deadline = time.monotonic() + 2
                while (
                    any(p.pid not in protected for p in owned_members(owner))
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.02)
                signal_owned(owner, signal.SIGKILL, exclude_pids=protected)
                if child is not None:
                    child.wait(timeout=5)
                if any(p.pid not in protected for p in owned_members(owner)):
                    raise RuntimeError(
                        "operation children remain; stage cannot publish completion"
                    )
                for handle in (output, error):
                    handle.flush()
                    os.fsync(handle.fileno())
        write_document(
            layout,
            directory / "exit.json",
            StageExit.model_validate(
                {
                    "launch": ref,
                    "status": status,
                    "exit_code": code,
                    "started_at": started_at,
                    "ended_at": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": time.monotonic() - started,
                    "stdout": annotation_reference(layout, directory / "stdout.log"),
                    "stderr": annotation_reference(layout, directory / "stderr.log"),
                    "reason": reason,
                }
            ),
        )
        manager.finish(launch.resource, reason=reason)
        return code
