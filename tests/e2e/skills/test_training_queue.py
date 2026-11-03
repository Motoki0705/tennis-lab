"""End-to-end checks for the file-backed training queue entry point."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import runpy
import shlex
import signal
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO
from unittest.mock import patch

import pytest

ROOT = Path(__file__).parents[3]
QUEUE_SCRIPT = ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"


@pytest.mark.parametrize("arguments", [("add", "true"), ("status",), ("serve",)])
def test_standalone_script_fails_with_missing_helper_diagnostic(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    script = tmp_path / "queue.sh"
    script.write_bytes(QUEUE_SCRIPT.read_bytes())
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    result = subprocess.run(
        ["bash", str(script), *arguments],
        cwd=tmp_path,
        env=_queue_env(
            queue_dir,
            lock_file,
            system_lock_file=tmp_path / "unprovisioned/gpu.lock",
        ),
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode != 0
    assert "required training queue helper is missing" in result.stderr
    assert str(script.with_name("queue_history.py")) in result.stderr
    assert "install_queue_bundle.sh" in result.stderr
    assert not list(queue_dir.rglob("*.job"))
    assert not (queue_dir / "worker.pid").exists()
    assert not any(path.exists() for path in _lock_namespace(lock_file))


def _queue_env(
    queue_dir: Path,
    lock_file: Path | None,
    *,
    system_lock_file: Path | None = None,
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TRAINING_QUEUE_LOCK_FILE", "TRAINING_QUEUE_SYSTEM_LOCK_FILE"}
        and not key.startswith("TRAINING_QUEUE_TEST_")
    }
    environment["TRAINING_QUEUE_DIR"] = str(queue_dir)
    # --basetemp may live inside a linked worktree. Never let a "non-Git"
    # fixture discover the real repository's shared GPU lock namespace. A
    # repository explicitly initialized below this ceiling is still visible.
    environment["GIT_CEILING_DIRECTORIES"] = str(queue_dir.parent.parent.resolve())
    environment["TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS"] = "1"
    if lock_file is not None:
        environment["TRAINING_QUEUE_LOCK_FILE"] = str(lock_file)
    if system_lock_file is not None:
        environment["TRAINING_QUEUE_SYSTEM_LOCK_FILE"] = str(system_lock_file)
    return environment


def _queue(
    queue_dir: Path,
    lock_file: Path | None,
    *arguments: str,
    cwd: Path,
    check: bool = True,
    system_lock_file: Path | None = None,
    now_ns: int | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = _queue_env(queue_dir, lock_file, system_lock_file=system_lock_file)
    if now_ns is not None:
        environment["TRAINING_QUEUE_TEST_NOW_NS"] = str(now_ns)
    return subprocess.run(
        ["bash", str(QUEUE_SCRIPT), *arguments],
        cwd=cwd,
        env=environment,
        check=check,
        capture_output=True,
        text=True,
    )


def _add(
    queue_dir: Path,
    lock_file: Path | None,
    command: str,
    *,
    cwd: Path,
    name: str,
    resource: str | None = None,
    require_external_teardown_ack: bool = False,
    system_lock_file: Path | None = None,
    now_ns: int | None = None,
) -> str:
    arguments = ["add", command, "--name", name]
    if resource is not None:
        arguments.extend(["--resource", resource])
    if require_external_teardown_ack:
        arguments.append("--require-external-teardown-ack")
    result = _queue(
        queue_dir,
        lock_file,
        *arguments,
        cwd=cwd,
        system_lock_file=system_lock_file,
        now_ns=now_ns,
    )
    return result.stdout.strip().removeprefix("queued: ")


def _serve(
    queue_dir: Path,
    lock_file: Path | None,
    *,
    cwd: Path,
    system_lock_file: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    environment = _queue_env(
        queue_dir,
        lock_file,
        system_lock_file=system_lock_file,
    )
    if extra_env is not None:
        environment.update(extra_env)
    return subprocess.Popen(
        ["bash", str(QUEUE_SCRIPT), "serve", "--idle-timeout", "0"],
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for(predicate: Callable[[], bool], *, timeout: float = 5) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition was not satisfied before timeout")


def _wait_for_slot_waiter(queue_dirs: list[Path], lock_file: Path) -> None:
    def has_waiter() -> bool:
        return any(
            "state=waiting\n" in state.read_text() and "wait=slot-capacity\n" in state.read_text()
            for queue_dir in queue_dirs for state in (queue_dir / "state").glob("*.state")
        )

    _wait_for(has_waiter)
    # Both actual reservations must remain held while a third worker waits.
    for slot in _lock_namespace(lock_file)[2:]:
        with slot.open("a+") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _wait_for_external_ack_rejection(queue_dir: Path, job: str) -> None:
    state = queue_dir / "state" / f"{job.removesuffix('.job')}.state"
    # Two publications ensure a complete validation cycle after the ack was
    # installed, even if the first write was already in flight.
    def next_publication() -> None:
        inode = state.stat().st_ino
        _wait_for(lambda: state.stat().st_ino != inode)
        assert "wait=external-teardown\n" in state.read_text()

    for _ in range(2):
        next_publication()
    assert (queue_dir / "running" / job).is_file()


def _lock_namespace(lock_file: Path) -> tuple[Path, Path, Path, Path]:
    return (
        lock_file,
        lock_file.with_name(f"{lock_file.name}.gate"),
        lock_file.with_name(f"{lock_file.name}.slot-0"),
        lock_file.with_name(f"{lock_file.name}.slot-1"),
    )


def _provision_lock_namespace(lock_file: Path) -> None:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    for member in _lock_namespace(lock_file):
        member.touch()


def _assert_lock_namespace_reacquirable(lock_file: Path) -> None:
    handles: list[TextIO] = []
    try:
        for member in _lock_namespace(lock_file):
            handle = member.open("a+", encoding="utf-8")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for released_handle in handles:
            fcntl.flock(released_handle, fcntl.LOCK_UN)
            released_handle.close()


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _process_state(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _teardown_events(event_file: Path, job: str) -> list[str]:
    if not event_file.exists():
        return []
    prefix = f"{job} "
    return [
        line.removeprefix(prefix)
        for line in event_file.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix)
    ]


def test_idempotent_add_concurrent_callers_execute_once(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    output = tmp_path / "executions"
    arguments = (
        "add",
        f"printf 'once\\n' >> {shlex.quote(str(output))}",
        "--idempotency-key",
        "campaign/task/clip/attempt-1",
        "--resource",
        "all",
    )
    with ThreadPoolExecutor(max_workers=10) as executor:
        replies = list(
            executor.map(
                lambda _: _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout,
                range(10),
            )
        )
    assert len(set(replies)) == 1
    assert len(list((queue_dir / "jobs").glob("*.job"))) == 1
    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = worker.communicate(timeout=15)
    assert worker.returncode == 0, (stdout, stderr)
    assert output.read_text() == "once\n"
    assert _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout == replies[0]
    assert not list((queue_dir / "jobs").glob("*.job"))


@pytest.mark.parametrize("phase", ["PRE", "POST"])
def test_idempotent_add_recovers_publish_before_reply_crash(
    tmp_path: Path,
    phase: str,
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    ready, fifo, output = (
        tmp_path / "ready",
        tmp_path / "barrier",
        tmp_path / "executions",
    )
    os.mkfifo(fifo)
    arguments = [
        "add",
        f"printf 'once\\n' >> {shlex.quote(str(output))}",
        "--idempotency-key",
        "publish-crash",
    ]
    environment = _queue_env(queue_dir, lock_file)
    environment.update(
        {
            f"TRAINING_QUEUE_TEST_{phase}_PUBLISH_READY_FILE": str(ready),
            f"TRAINING_QUEUE_TEST_{phase}_PUBLISH_BARRIER_FIFO": str(fifo),
        }
    )
    caller = subprocess.Popen(
        ["bash", str(QUEUE_SCRIPT), *arguments],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for(ready.exists)
        job = ready.read_text().strip()
        caller.kill()
        stdout, stderr = caller.communicate(timeout=5)
        assert not stdout, stderr
        if phase == "POST":
            # The workload may even finish before the client retries.
            worker = _serve(queue_dir, lock_file, cwd=tmp_path)
            worker.communicate(timeout=15)
            assert worker.returncode == 0
        result = _queue(queue_dir, lock_file, *arguments, cwd=tmp_path)
        assert result.stdout.strip() == f"queued: {job}"
        if phase == "PRE":
            worker = _serve(queue_dir, lock_file, cwd=tmp_path)
            worker.communicate(timeout=15)
            assert worker.returncode == 0
        assert output.read_text() == "once\n"
        assert list((queue_dir / "done").glob("*.job")) == [queue_dir / "done" / job]
        assert not list((queue_dir / "jobs").glob("*.job"))
    finally:
        if caller.poll() is None:
            caller.kill()
            caller.communicate(timeout=5)


@pytest.mark.parametrize(
    "changed",
    [
        ["--resource", "half"],
        ["--provider", "codex"],
        ["--session", "new-session"],
        ["--issue", "1088"],
        ["--name", "new-name"],
        ["--prune-ckpt"],
        ["--require-external-teardown-ack"],
    ],
)
def test_idempotent_add_rejects_changed_options(
    tmp_path: Path, changed: list[str]
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    arguments = ("add", "true", "--idempotency-key", "immutable")
    _queue(queue_dir, lock_file, *arguments, cwd=tmp_path)
    rejected = _queue(
        queue_dir, lock_file, *arguments, *changed, cwd=tmp_path, check=False
    )
    assert rejected.returncode == 2
    assert "different request" in rejected.stderr
    assert len(list((queue_dir / "jobs").glob("*.job"))) == 1


def test_idempotent_add_hashes_command_and_working_directory(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    arguments = ("add", "printf '%s' '\n$(false);`false`'", "--idempotency-key", "same")
    first = _queue(queue_dir, lock_file, *arguments, cwd=tmp_path)
    assert _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout == first.stdout
    other = tmp_path / "other"
    other.mkdir()
    assert (
        _queue(queue_dir, lock_file, *arguments, cwd=other, check=False).returncode == 2
    )
    changed = _queue(
        queue_dir,
        lock_file,
        "add",
        "true",
        "--idempotency-key",
        "same",
        cwd=tmp_path,
        check=False,
    )
    assert changed.returncode == 2


@pytest.mark.parametrize("action", ["cancel", "clear", "fail"])
def test_idempotent_add_keeps_terminal_job_identity(
    tmp_path: Path, action: str
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    arguments = ("add", "exit 7", "--idempotency-key", "attempt-1")
    first = _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout
    job = first.strip().removeprefix("queued: ")
    if action == "fail":
        worker = _serve(queue_dir, lock_file, cwd=tmp_path)
        worker.communicate(timeout=15)
        terminal = "failed"
    else:
        control = ("cancel", job) if action == "cancel" else ("clear",)
        _queue(queue_dir, lock_file, *control, cwd=tmp_path)
        terminal = "cancelled"
    assert (queue_dir / terminal / job).is_file()
    assert _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout == first
    assert not list((queue_dir / "jobs").glob("*.job"))
    # An intentional retry is a distinct attempt, never an implicit rerun.
    new = _queue(
        queue_dir,
        lock_file,
        "add",
        "exit 7",
        "--idempotency-key",
        "attempt-2",
        cwd=tmp_path,
    )
    assert new.stdout != first


def test_idempotent_add_fails_closed_when_history_was_deleted(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    arguments = ("add", "true", "--idempotency-key", "protected")
    job = (
        _queue(queue_dir, lock_file, *arguments, cwd=tmp_path)
        .stdout.strip()
        .removeprefix("queued: ")
    )
    (queue_dir / "jobs" / job).unlink()
    retry = _queue(queue_dir, lock_file, *arguments, cwd=tmp_path, check=False)
    assert retry.returncode == 1
    assert "no job added" in retry.stderr
    assert not list((queue_dir / "jobs").glob("*.job"))


def test_idempotent_add_recovers_running_job(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    release = tmp_path / "release"
    arguments = (
        "add",
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.05; done",
        "--idempotency-key",
        "running",
    )
    first = _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout
    job = first.strip().removeprefix("queued: ")
    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    try:
        _wait_for(lambda: (queue_dir / "running" / job).is_file())
        assert _queue(queue_dir, lock_file, *arguments, cwd=tmp_path).stdout == first
        assert not list((queue_dir / "jobs").glob("*.job"))
    finally:
        release.touch()
        worker.communicate(timeout=15)
    assert worker.returncode == 0


@pytest.mark.parametrize("damage", ["corrupt", "symlink"])
def test_idempotent_add_rejects_invalid_receipt(tmp_path: Path, damage: str) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    arguments = ("add", "true", "--idempotency-key", "protected")
    _queue(queue_dir, lock_file, *arguments, cwd=tmp_path)
    receipt = next((queue_dir / "control" / "registrations").glob("*.receipt"))
    if damage == "corrupt":
        receipt.write_text("incomplete")
    else:
        target = tmp_path / "receipt"
        receipt.rename(target)
        receipt.symlink_to(target)
    retry = _queue(queue_dir, lock_file, *arguments, cwd=tmp_path, check=False)
    assert retry.returncode == 1
    assert "invalid queue registration" in retry.stderr
    assert len(list((queue_dir / "jobs").glob("*.job"))) == 1


@pytest.mark.parametrize("key", ["", "bad key", "../key", "x" * 257])
def test_idempotent_add_validates_key(tmp_path: Path, key: str) -> None:
    result = _queue(
        tmp_path / "queue",
        tmp_path / "gpu.lock",
        "add",
        "true",
        "--idempotency-key",
        key,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 2


def test_keyless_add_keeps_independent_jobs_and_clear(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    first = _queue(queue_dir, lock_file, "add", "true", cwd=tmp_path).stdout
    second = _queue(queue_dir, lock_file, "add", "true", cwd=tmp_path).stdout
    assert first != second
    _queue(queue_dir, lock_file, "clear", cwd=tmp_path)
    assert not list((queue_dir / "jobs").glob("*.job"))
    assert not list((queue_dir / "cancelled").glob("*.job"))


def test_serve_reports_running_and_honours_shared_gpu_lock(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    result_file = tmp_path / "finished"
    env = _queue_env(queue_dir, lock_file)
    command = f"printf finished > {shlex.quote(str(result_file))}"

    subprocess.run(
        [
            "bash",
            str(QUEUE_SCRIPT),
            "add",
            command,
            "--name",
            "lock-test",
            "--resource",
            "half",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    with lock_file.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = subprocess.Popen(
            ["bash", str(QUEUE_SCRIPT), "serve", "--idle-timeout", "0"],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            deadline = time.monotonic() + 3
            status = ""
            while time.monotonic() < deadline:
                status = subprocess.run(
                    ["bash", str(QUEUE_SCRIPT), "status"],
                    cwd=tmp_path,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                if "worker: RUNNING" in status and "running=1" in status:
                    break
                time.sleep(0.02)
            assert "worker: RUNNING" in status
            assert "running=1" in status
            assert not result_file.exists()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

        stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert result_file.read_text(encoding="utf-8") == "finished"
    final_status = subprocess.run(
        ["bash", str(QUEUE_SCRIPT), "status"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "worker: stopped" in final_status
    assert "queued=0 running=0 done=1 failed=0" in final_status


def test_start_still_launches_detached_worker(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    env = _queue_env(
        queue_dir,
        None,
        system_lock_file=tmp_path / "system/gpu.lock",
    )

    subprocess.run(
        ["bash", str(QUEUE_SCRIPT), "add", "true", "--name", "start-test"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    started = subprocess.run(
        ["bash", str(QUEUE_SCRIPT), "start", "--idle-timeout", "0"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "worker started" in started.stdout

    deadline = time.monotonic() + 3
    status = ""
    while time.monotonic() < deadline:
        status = subprocess.run(
            ["bash", str(QUEUE_SCRIPT), "status"],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if "worker: stopped" in status and "done=1" in status:
            break
        time.sleep(0.02)

    assert "worker: stopped" in status
    assert "queued=0 running=0 done=1 failed=0" in status


def test_detached_start_keeps_worker_identity_until_wrapper_teardown(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    started = tmp_path / "started"
    identity = tmp_path / "identity"
    successor = tmp_path / "successor"
    active_job = _add(
        queue_dir,
        lock_file,
        (
            "trap '' TERM; pgid=$(ps -o pgid= -p $$ | tr -d ' '); "
            f"printf '%s %s\\n' \"$$\" \"$pgid\" > {shlex.quote(str(identity))}; "
            f"touch {shlex.quote(str(started))}; while :; do sleep 0.05; done"
        ),
        cwd=tmp_path,
        name="detached-lifetime",
        resource="all",
    )
    _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor))}",
        cwd=tmp_path,
        name="detached-successor",
        resource="half",
    )
    started_worker = _queue(
        queue_dir, lock_file, "start", "--idle-timeout", "30", cwd=tmp_path
    )
    assert started_worker.returncode == 0
    pgid = 0
    try:
        _wait_for(started.exists)
        leader_pid, pgid = (
            int(value) for value in identity.read_text(encoding="utf-8").split()
        )
        worker_pid = int((queue_dir / "worker.pid").read_text(encoding="utf-8"))
        os.kill(worker_pid, signal.SIGTERM)
        _wait_for(
            lambda: "state=terminating" in _queue(
                queue_dir, lock_file, "list", cwd=tmp_path
            ).stdout
        )
        assert (queue_dir / "worker.pid").read_text(encoding="utf-8").strip() == str(
            worker_pid
        )
        replacement_too_early = _queue(
            queue_dir,
            lock_file,
            "start",
            "--idle-timeout",
            "0",
            cwd=tmp_path,
            check=False,
        )
        assert replacement_too_early.returncode == 1
        assert "already running" in replacement_too_early.stderr
        assert not successor.exists()

        _wait_for(lambda: not _process_group_exists(pgid))
        _wait_for(lambda: not (queue_dir / "worker.pid").exists())
        assert (queue_dir / "failed" / active_job).is_file()
        replacement = _queue(
            queue_dir, lock_file, "start", "--idle-timeout", "0", cwd=tmp_path
        )
        assert replacement.returncode == 0
        _wait_for(successor.exists)
        _wait_for(lambda: not (queue_dir / "worker.pid").exists())
        assert not Path(f"/proc/{leader_pid}").exists()
    finally:
        if pgid and _process_group_exists(pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        if (queue_dir / "worker.pid").is_file():
            with contextlib.suppress(ProcessLookupError, ValueError):
                os.kill(
                    int((queue_dir / "worker.pid").read_text(encoding="utf-8")),
                    signal.SIGTERM,
                )


def test_add_validates_and_persists_resource_declaration(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"

    half_job = _add(
        queue_dir, lock_file, "true", cwd=tmp_path, name="half", resource="half"
    )
    all_job = _add(
        queue_dir, lock_file, "true", cwd=tmp_path, name="all", resource="all"
    )
    default_job = _add(queue_dir, lock_file, "true", cwd=tmp_path, name="default")
    external_job = _add(
        queue_dir,
        lock_file,
        "true",
        cwd=tmp_path,
        name="external",
        require_external_teardown_ack=True,
    )

    assert "# resource: half" in (queue_dir / "jobs" / half_job).read_text()
    assert "# resource: all" in (queue_dir / "jobs" / all_job).read_text()
    assert "# resource: all" in (queue_dir / "jobs" / default_job).read_text()
    external_text = (queue_dir / "jobs" / external_job).read_text(encoding="utf-8")
    expected_ack = queue_dir.resolve() / "control/external-acks" / f"{external_job[:-4]}.ack"
    assert "# external_teardown_ack: 1" in external_text
    assert f"TRAINING_QUEUE_EXTERNAL_TEARDOWN_ACK={expected_ack}" in external_text
    _queue(queue_dir, lock_file, "cancel", external_job, cwd=tmp_path)
    invalid = _queue(
        queue_dir,
        lock_file,
        "add",
        "true",
        "--resource",
        "quarter",
        cwd=tmp_path,
        check=False,
    )
    missing = _queue(
        queue_dir,
        lock_file,
        "add",
        "true",
        "--resource",
        cwd=tmp_path,
        check=False,
    )
    assert invalid.returncode == 2
    assert "must be half or all" in invalid.stderr
    assert missing.returncode == 2
    assert "requires a value" in missing.stderr

    corrupt = queue_dir / "jobs/000_corrupt.job"
    corrupt.write_text(
        "#!/usr/bin/env bash\n# resource: quarter\ntrue\n", encoding="utf-8"
    )
    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = worker.communicate(timeout=5)
    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert (queue_dir / "failed" / corrupt.name).is_file()
    assert "invalid or duplicate resource" in (
        queue_dir / "logs/000_corrupt.log"
    ).read_text()


@pytest.mark.parametrize("invalid_grace", ["0", "invalid"])
def test_internal_term_grace_seam_requires_a_positive_integer(
    tmp_path: Path, invalid_grace: str
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    job = _add(
        queue_dir,
        lock_file,
        "true",
        cwd=tmp_path,
        name="invalid-grace",
        resource="all",
    )

    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env={"TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS": invalid_grace},
    )
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 2, f"stdout={stdout}\nstderr={stderr}"
    assert "TERM grace must be a positive integer (default: 15)" in stderr
    assert (queue_dir / "jobs" / job).is_file()
    assert not (queue_dir / "worker.pid").exists()


def test_external_ack_control_directory_rejects_symlink(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    protected = tmp_path / "protected"
    protected.mkdir()
    queue_dir.mkdir()
    (queue_dir / "control").symlink_to(protected, target_is_directory=True)

    result = _queue(
        queue_dir,
        lock_file,
        "add",
        "true",
        "--require-external-teardown-ack",
        cwd=tmp_path,
        check=False,
    )

    assert result.returncode == 1
    assert "queue control directory must not be a symlink" in result.stderr
    assert not list(protected.iterdir())


def test_half_capacity_status_and_repro_metadata(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    release = tmp_path / "release"
    jobs: list[str] = []
    job_markers: dict[str, Path] = {}
    for index in range(3):
        marker = tmp_path / f"started-{index}"
        command = (
            f"touch {shlex.quote(str(marker))}; "
            f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.02; done"
        )
        job = _add(
            queue_dir,
            lock_file,
            command,
            cwd=tmp_path,
            name=f"half-{index}",
            resource="half",
        )
        jobs.append(job)
        job_markers[job] = marker
    ordered_jobs = jobs  # Admission order is independent of filename order.
    first_markers = [job_markers[job] for job in ordered_jobs[:2]]
    third_marker = job_markers[ordered_jobs[2]]

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    try:
        _wait_for(
            lambda: all(marker.exists() for marker in first_markers),
            timeout=15,
        )
        listing = ""

        def two_running_slots_and_waiter() -> bool:
            nonlocal listing
            listing = _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
            running_slots = re.findall(
                r"resource=half slot=([01]) pid=\d+ pgid=\d+ state=running wait=none",
                listing,
            )
            return (
                len(running_slots) == 2
                and set(running_slots) == {"0", "1"}
                and "state=waiting wait=slot-capacity" in listing
            )

        _wait_for(two_running_slots_and_waiter, timeout=10)
        status = _queue(queue_dir, lock_file, "status", cwd=tmp_path).stdout
        assert not third_marker.exists()
        assert "executing=2 waiting=1" in status
        assert "mode=logical-only mig=unchanged vram-hard-cap=none" in status
        assert listing.count("resource=half") == 3
        assert listing.count("state=running") == 2
        assert "state=waiting" in listing
        assert "wait=slot-capacity" in listing
        _wait_for_slot_waiter([queue_dir], lock_file)
    finally:
        release.touch()
        stdout, stderr = worker.communicate(timeout=10)
    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert third_marker.is_file()
    for job in jobs:
        run = json.loads(
            (queue_dir / "repro" / job.removesuffix(".job") / "run.json").read_text()
        )
        repro = (
            queue_dir / "repro" / job.removesuffix(".job") / "repro.sh"
        ).read_text()
        assert run["resource"] == "half"
        assert run["logical_gpu_slot"] in {"0", "1"}
        assert "export TENNIS_GPU_RESOURCE=half" in repro
        assert "export TENNIS_GPU_SLOT=" in repro


def test_all_is_exclusive_and_preserves_local_fifo(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    release_half = tmp_path / "release-half"
    release_all = tmp_path / "release-all"
    started_half = tmp_path / "started-half"
    started_all = tmp_path / "started-all"
    started_tail = tmp_path / "started-tail"

    _add(
        queue_dir,
        lock_file,
        f"touch {started_half}; while [ ! -f {release_half} ]; do sleep 0.02; done",
        cwd=tmp_path,
        name="half",
        resource="half",
    )
    _add(
        queue_dir,
        lock_file,
        f"touch {started_all}; while [ ! -f {release_all} ]; do sleep 0.02; done",
        cwd=tmp_path,
        name="all",
        resource="all",
    )
    _add(
        queue_dir,
        lock_file,
        f"touch {started_tail}",
        cwd=tmp_path,
        name="tail-half",
        resource="half",
    )

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    _wait_for(started_half.exists)
    assert not started_all.exists()
    assert not started_tail.exists()
    release_half.touch()
    _wait_for(started_all.exists)
    assert not started_tail.exists()
    release_all.touch()
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert started_tail.is_file()


def test_separate_workers_share_the_two_slot_namespace(tmp_path: Path) -> None:
    lock_file = tmp_path / "gpu.lock"
    release = tmp_path / "release"
    workers: list[subprocess.Popen[str]] = []
    for index in range(3):
        queue_dir = tmp_path / f"queue-{index}"
        _add(
            queue_dir,
            lock_file,
            (
                f"touch {tmp_path / f'started-{index}'}; "
                f"while [ ! -f {release} ]; do sleep 0.02; done"
            ),
            cwd=tmp_path,
            name=f"cross-{index}",
            resource="half",
        )
        workers.append(_serve(queue_dir, lock_file, cwd=tmp_path))

    try:
        _wait_for(
            lambda: len(list(tmp_path.glob("started-*"))) == 2,
        )
        _wait_for_slot_waiter([tmp_path / f"queue-{i}" for i in range(3)], lock_file)
        assert len(list(tmp_path.glob("started-*"))) == 2
    finally:
        release.touch()
        outputs = [worker.communicate(timeout=5) for worker in workers]
    assert all(worker.returncode == 0 for worker in workers), outputs
    assert len(list(tmp_path.glob("started-*"))) == 3


def test_explicit_lock_precedes_an_invalid_system_namespace(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    explicit_lock = tmp_path / "explicit/gpu.lock"
    system_lock = tmp_path / "system/gpu.lock"
    system_lock.parent.mkdir(parents=True)
    system_lock.touch()
    marker = tmp_path / "started"
    _add(
        queue_dir,
        explicit_lock,
        f"touch {shlex.quote(str(marker))}",
        cwd=tmp_path,
        name="explicit-precedence",
        resource="half",
        system_lock_file=system_lock,
    )

    worker = _serve(
        queue_dir,
        explicit_lock,
        cwd=tmp_path,
        system_lock_file=system_lock,
    )
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert marker.is_file()
    assert all(member.is_file() for member in _lock_namespace(explicit_lock))
    assert not system_lock.with_name(f"{system_lock.name}.gate").exists()


def test_complete_system_namespace_is_selected_without_creating_fallback(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    system_lock = tmp_path / "system/gpu.lock"
    _provision_lock_namespace(system_lock)
    marker = tmp_path / "started"
    job = _add(
        queue_dir,
        None,
        f"touch {shlex.quote(str(marker))}",
        cwd=tmp_path,
        name="system-lock",
        resource="half",
        system_lock_file=system_lock,
    )

    with system_lock.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        worker = _serve(
            queue_dir,
            None,
            cwd=tmp_path,
            system_lock_file=system_lock,
        )
        _wait_for(
            lambda: "wait=main-shared-lock"
            in _queue(
                queue_dir,
                None,
                "list",
                cwd=tmp_path,
                system_lock_file=system_lock,
            ).stdout
        )
        assert not marker.exists()
        fcntl.flock(lock, fcntl.LOCK_UN)
        stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert marker.is_file()
    assert (queue_dir / "done" / job).is_file()
    assert not (queue_dir / "gpu.lock").exists()


def test_invalid_system_namespaces_fail_closed_without_repair(tmp_path: Path) -> None:
    protected = tmp_path / "protected"
    protected.write_text("unchanged", encoding="utf-8")
    for kind in ("partial", "symlink", "nonregular", "unwritable"):
        case_root = tmp_path / kind
        queue_dir = case_root / "queue"
        system_lock = case_root / "system/gpu.lock"
        system_lock.parent.mkdir(parents=True)
        if kind == "partial":
            system_lock.touch()
        elif kind == "symlink":
            system_lock.symlink_to(protected)
        elif kind == "nonregular":
            system_lock.mkdir()
        else:
            _provision_lock_namespace(system_lock)
            for member in _lock_namespace(system_lock):
                member.chmod(0o400)
        marker = case_root / "started"
        job = _add(
            queue_dir,
            None,
            f"touch {shlex.quote(str(marker))}",
            cwd=case_root,
            name=kind,
            resource="half",
            system_lock_file=system_lock,
        )

        worker = _serve(
            queue_dir,
            None,
            cwd=case_root,
            system_lock_file=system_lock,
        )
        stdout, stderr = worker.communicate(timeout=5)

        assert worker.returncode == 0, f"kind={kind}\nstdout={stdout}\nstderr={stderr}"
        assert not marker.exists()
        assert (queue_dir / "failed" / job).is_file()
        assert "system GPU lock namespace is partial, unsafe, or unusable" in stderr
        assert not (queue_dir / "gpu.lock").exists()
        if kind != "unwritable":
            assert not system_lock.with_name(f"{system_lock.name}.gate").exists()
    assert protected.read_text(encoding="utf-8") == "unchanged"


def test_git_common_root_default_is_shared_by_linked_worktrees(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    linked = tmp_path / "linked"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "queue-test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Queue Test"], cwd=repository, check=True
    )
    (repository / "tracked").write_text("test\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=repository, check=True)
    subprocess.run(
        ["git", "worktree", "add", "-q", "--detach", str(linked), "HEAD"],
        cwd=repository,
        check=True,
    )

    system_lock = tmp_path / "unprovisioned-system/gpu.lock"
    release = tmp_path / "release"
    workers: list[subprocess.Popen[str]] = []
    for index, cwd in enumerate((repository, linked, linked)):
        queue_dir = tmp_path / f"queue-{index}"
        marker = tmp_path / f"started-{index}"
        _add(
            queue_dir,
            None,
            (
                f"touch {shlex.quote(str(marker))}; "
                f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.02; done"
            ),
            cwd=cwd,
            name=f"git-common-{index}",
            resource="half",
            system_lock_file=system_lock,
        )
        workers.append(
            _serve(
                queue_dir,
                None,
                cwd=cwd,
                system_lock_file=system_lock,
            )
        )

    try:
        _wait_for(lambda: len(list(tmp_path.glob("started-*"))) == 2, timeout=10)
        _wait_for_slot_waiter(
            [tmp_path / f"queue-{i}" for i in range(3)], repository / ".training_queue/gpu.lock"
        )
        assert len(list(tmp_path.glob("started-*"))) == 2
        assert all(
            member.is_file()
            for member in _lock_namespace(repository / ".training_queue/gpu.lock")
        )
        assert not (linked / ".training_queue/gpu.lock").exists()
    finally:
        release.touch()
        outputs = [worker.communicate(timeout=10) for worker in workers]

    assert all(worker.returncode == 0 for worker in workers), outputs
    assert len(list(tmp_path.glob("started-*"))) == 3


def test_non_git_default_uses_queue_local_namespace(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    system_lock = tmp_path / "unprovisioned-system/gpu.lock"
    marker = tmp_path / "started"
    _add(
        queue_dir,
        None,
        f"touch {shlex.quote(str(marker))}",
        cwd=tmp_path,
        name="non-git",
        resource="half",
        system_lock_file=system_lock,
    )

    worker = _serve(
        queue_dir,
        None,
        cwd=tmp_path,
        system_lock_file=system_lock,
    )
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert marker.is_file()
    assert all(member.is_file() for member in _lock_namespace(queue_dir / "gpu.lock"))


def test_legacy_failure_and_worker_term_release_capacity(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    (queue_dir / "jobs").mkdir(parents=True)
    legacy = queue_dir / "jobs/000_legacy.job"
    legacy.write_text(
        f"#!/usr/bin/env bash\ntouch {tmp_path / 'legacy-finished'}\n",
        encoding="utf-8",
    )
    # This tests stop/teardown ordering, so pin registration order rather than
    # relying on a host wall clock that can step backwards (observed on WSL).
    _add(
        queue_dir,
        lock_file,
        "false",
        cwd=tmp_path,
        name="failure",
        resource="half",
        now_ns=END_NS,
    )
    started = tmp_path / "term-started"
    _add(
        queue_dir,
        lock_file,
        f"touch {started}; while :; do sleep 0.1; done",
        cwd=tmp_path,
        name="term",
        resource="all",
        now_ns=END_NS + 1,
    )
    successor = tmp_path / "successor"
    _add(
        queue_dir,
        lock_file,
        f"touch {successor}",
        cwd=tmp_path,
        name="successor",
        resource="all",
        now_ns=END_NS + 2,
    )

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    _wait_for(started.exists)
    assert (tmp_path / "legacy-finished").is_file()
    worker.terminate()
    worker.communicate(timeout=5)
    assert worker.returncode != 0
    assert not successor.exists()

    replacement = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = replacement.communicate(timeout=5)
    assert replacement.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert successor.is_file()
    final = _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
    final_status = _queue(queue_dir, lock_file, "status", cwd=tmp_path).stdout
    assert "000_legacy.job resource=all" in final
    assert re.search(
        r"000_legacy\.job resource=all slot=all pid=\d+ pgid=\d+ state=done wait=none",
        final,
    )
    assert final.count("state=failed") == 2
    assert "000_legacy.job resource=all slot=all" in final_status
    assert "failed (latest 10):" in final_status


def test_all_failure_releases_capacity_for_following_half(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    successor_started = tmp_path / "successor-started"
    failed_job = _add(
        queue_dir,
        lock_file,
        "exit 17",
        cwd=tmp_path,
        name="failing-all",
        resource="all",
    )
    successor_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="half-successor",
        resource="half",
    )

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert (queue_dir / "failed" / failed_job).is_file()
    assert (queue_dir / "done" / successor_job).is_file()
    assert successor_started.is_file()
    assert "exit_code=17" in (
        queue_dir / "logs" / f"{failed_job.removesuffix('.job')}.log"
    ).read_text(encoding="utf-8")
    _assert_lock_namespace_reacquirable(lock_file)


def test_cooperative_stop_finishes_active_halves_and_leaves_waiter_queued(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    release = tmp_path / "release"
    jobs_and_markers: dict[str, Path] = {}
    for index in range(3):
        marker = tmp_path / f"stop-started-{index}"
        job = _add(
            queue_dir,
            lock_file,
            (
                f"touch {marker}; "
                f"while [ ! -f {release} ]; do sleep 0.02; done"
            ),
            cwd=tmp_path,
            name=f"stop-{index}",
            resource="half",
        )
        jobs_and_markers[job] = marker
    ordered_jobs = list(jobs_and_markers)
    first_markers = [jobs_and_markers[job] for job in ordered_jobs[:2]]
    waiter_marker = jobs_and_markers[ordered_jobs[2]]

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    _wait_for(lambda: all(marker.exists() for marker in first_markers), timeout=10)
    _queue(queue_dir, lock_file, "stop", cwd=tmp_path)
    release.touch()
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert not waiter_marker.exists()
    status = _queue(queue_dir, lock_file, "status", cwd=tmp_path).stdout
    assert "queued=1 running=0 done=2 failed=0" in status

    replacement = _serve(queue_dir, lock_file, cwd=tmp_path)
    replacement.communicate(timeout=5)
    assert replacement.returncode == 0
    assert waiter_marker.is_file()


def test_cross_queue_waiting_all_prevents_new_half_admission(tmp_path: Path) -> None:
    lock_file = tmp_path / "gpu.lock"
    half_queue = tmp_path / "half-queue"
    all_queue = tmp_path / "all-queue"
    tail_queue = tmp_path / "tail-queue"
    release_half = tmp_path / "release-half"
    release_all = tmp_path / "release-all"
    half_started = tmp_path / "half-started"
    all_started = tmp_path / "all-started"
    tail_started = tmp_path / "tail-started"

    _add(
        half_queue,
        lock_file,
        f"touch {half_started}; while [ ! -f {release_half} ]; do sleep 0.02; done",
        cwd=tmp_path,
        name="first-half",
        resource="half",
    )
    _add(
        all_queue,
        lock_file,
        f"touch {all_started}; while [ ! -f {release_all} ]; do sleep 0.02; done",
        cwd=tmp_path,
        name="waiting-all",
        resource="all",
    )
    _add(
        tail_queue,
        lock_file,
        f"touch {tail_started}",
        cwd=tmp_path,
        name="late-half",
        resource="half",
    )

    half_worker = _serve(half_queue, lock_file, cwd=tmp_path)
    all_worker: subprocess.Popen[str] | None = None
    tail_worker: subprocess.Popen[str] | None = None
    try:
        _wait_for(half_started.exists)
        all_worker = _serve(all_queue, lock_file, cwd=tmp_path)
        _wait_for(
            lambda: "wait=main-exclusive-lock"
            in _queue(all_queue, lock_file, "list", cwd=tmp_path).stdout
        )
        tail_worker = _serve(tail_queue, lock_file, cwd=tmp_path)
        _wait_for(
            lambda: "wait=admission-gate"
            in _queue(tail_queue, lock_file, "list", cwd=tmp_path).stdout
        )
        assert not tail_started.exists()

        release_half.touch()
        _wait_for(all_started.exists)
        assert not tail_started.exists()

        release_all.touch()
        _wait_for(tail_started.exists)
    finally:
        release_half.touch(exist_ok=True)
        release_all.touch(exist_ok=True)
        outputs = [half_worker.communicate(timeout=5)]
        if all_worker is not None:
            outputs.append(all_worker.communicate(timeout=5))
        if tail_worker is not None:
            outputs.append(tail_worker.communicate(timeout=5))

    assert half_worker.returncode == 0, outputs
    assert all_worker is not None and all_worker.returncode == 0, outputs
    assert tail_worker is not None and tail_worker.returncode == 0, outputs


def test_cancel_pending_job_is_idempotent_and_never_launches_it(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    target_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="pending-target",
        resource="half",
    )
    _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="pending-successor",
        resource="half",
    )

    first = _queue(queue_dir, lock_file, "cancel", target_job, cwd=tmp_path)
    second = _queue(queue_dir, lock_file, "cancel", target_job, cwd=tmp_path)
    assert first.stdout.strip() == second.stdout.strip() == "cancelled"
    assert (queue_dir / "cancelled" / target_job).is_file()
    assert not (queue_dir / "state" / f"{target_job[:-4]}.state").exists()
    assert not (
        queue_dir / "cancel-requests" / f"{target_job[:-4]}.cancel"
    ).exists()

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    output = worker.communicate(timeout=5)
    assert worker.returncode == 0, output
    assert not target_started.exists()
    assert successor_started.is_file()


def test_cancel_half_waiting_on_admission_gate_releases_successor(
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "gpu.lock"
    all_queue = tmp_path / "all-queue"
    target_queue = tmp_path / "target-queue"
    all_started = tmp_path / "all-started"
    all_release = tmp_path / "all-release"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    target_job = _add(
        target_queue,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="gate-target",
        resource="half",
    )
    _add(
        target_queue,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="gate-successor",
        resource="half",
    )
    _add(
        all_queue,
        lock_file,
        (
            f"touch {shlex.quote(str(all_started))}; "
            f"while [ ! -f {shlex.quote(str(all_release))} ]; do sleep 0.02; done"
        ),
        cwd=tmp_path,
        name="gate-holder",
        resource="all",
    )

    with lock_file.open("w", encoding="utf-8") as raw_main:
        fcntl.flock(raw_main, fcntl.LOCK_EX)
        all_worker = _serve(all_queue, lock_file, cwd=tmp_path)
        _wait_for(
            lambda: "wait=main-exclusive-lock"
            in _queue(all_queue, lock_file, "list", cwd=tmp_path).stdout
        )
        target_worker = _serve(target_queue, lock_file, cwd=tmp_path)
        _wait_for(
            lambda: "wait=admission-gate"
            in _queue(target_queue, lock_file, "list", cwd=tmp_path).stdout
        )

        cancelled = _queue(
            target_queue, lock_file, "cancel", target_job, cwd=tmp_path
        )
        assert cancelled.stdout.strip() == "cancelled"
        _wait_for(
            lambda: (target_queue / "cancelled" / target_job).is_file()
            and "wait=admission-gate"
            in _queue(target_queue, lock_file, "list", cwd=tmp_path).stdout
        )
        assert not target_started.exists()
        assert not (target_queue / "state" / f"{target_job[:-4]}.state").exists()
        assert not (
            target_queue / "cancel-requests" / f"{target_job[:-4]}.cancel"
        ).exists()

        fcntl.flock(raw_main, fcntl.LOCK_UN)
        _wait_for(all_started.exists)
        assert not successor_started.exists()
        all_release.touch()
        _wait_for(successor_started.exists)

    all_output = all_worker.communicate(timeout=5)
    target_output = target_worker.communicate(timeout=5)
    assert all_worker.returncode == 0, all_output
    assert target_worker.returncode == 0, target_output


def test_cancel_half_waiting_on_slot_capacity_releases_successor(
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "gpu.lock"
    holders_queue = tmp_path / "holders-queue"
    target_queue = tmp_path / "target-queue"
    release = tmp_path / "release"
    for index in range(2):
        _add(
            holders_queue,
            lock_file,
            (
                f"touch {shlex.quote(str(tmp_path / f'holder-{index}'))}; "
                f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.02; done"
            ),
            cwd=tmp_path,
            name=f"holder-{index}",
            resource="half",
        )
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    target_job = _add(
        target_queue,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="slot-target",
        resource="half",
    )
    _add(
        target_queue,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="slot-successor",
        resource="half",
    )

    holders_worker = _serve(holders_queue, lock_file, cwd=tmp_path)
    _wait_for(
        lambda: all((tmp_path / f"holder-{index}").exists() for index in range(2)),
        timeout=10,
    )
    target_worker = _serve(target_queue, lock_file, cwd=tmp_path)
    _wait_for(
        lambda: "wait=slot-capacity"
        in _queue(target_queue, lock_file, "list", cwd=tmp_path).stdout
    )

    cancelled = _queue(target_queue, lock_file, "cancel", target_job, cwd=tmp_path)
    assert cancelled.stdout.strip() == "cancelled"
    _wait_for(
        lambda: (target_queue / "cancelled" / target_job).is_file()
        and "wait=slot-capacity"
        in _queue(target_queue, lock_file, "list", cwd=tmp_path).stdout
    )
    repeated = _queue(target_queue, lock_file, "cancel", target_job, cwd=tmp_path)
    assert repeated.stdout.strip() == "cancelled"
    assert not target_started.exists()
    assert not successor_started.exists()
    assert not (target_queue / "state" / f"{target_job[:-4]}.state").exists()
    assert not (
        target_queue / "cancel-requests" / f"{target_job[:-4]}.cancel"
    ).exists()

    release.touch()
    _wait_for(successor_started.exists)
    holders_output = holders_worker.communicate(timeout=5)
    target_output = target_worker.communicate(timeout=5)
    assert holders_worker.returncode == 0, holders_output
    assert target_worker.returncode == 0, target_output


def test_cancel_all_waiting_on_main_lock_releases_gate_and_successor(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    holder_started = tmp_path / "holder-started"
    holder_release = tmp_path / "holder-release"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    _add(
        queue_dir,
        lock_file,
        (
            f"touch {shlex.quote(str(holder_started))}; "
            f"while [ ! -f {shlex.quote(str(holder_release))} ]; do sleep 0.02; done"
        ),
        cwd=tmp_path,
        name="main-holder",
        resource="half",
    )
    target_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="main-target",
        resource="all",
    )
    _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="main-successor",
        resource="half",
    )

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    _wait_for(holder_started.exists)
    _wait_for(
        lambda: "wait=main-exclusive-lock"
        in _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
    )

    cancelled = _queue(queue_dir, lock_file, "cancel", target_job, cwd=tmp_path)
    assert cancelled.stdout.strip() == "cancelled"
    _wait_for(successor_started.exists)
    assert not target_started.exists()
    assert (queue_dir / "cancelled" / target_job).is_file()
    assert not (queue_dir / "state" / f"{target_job[:-4]}.state").exists()
    assert not (
        queue_dir / "cancel-requests" / f"{target_job[:-4]}.cancel"
    ).exists()

    holder_release.touch()
    output = worker.communicate(timeout=5)
    assert worker.returncode == 0, output


def test_cancel_half_waiting_on_raw_main_lock_releases_slot_and_successor(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    target_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="raw-main-target",
        resource="half",
    )
    successor_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="raw-main-successor",
        resource="all",
    )

    with lock_file.open("w", encoding="utf-8") as raw_main:
        fcntl.flock(raw_main, fcntl.LOCK_EX)
        worker = _serve(queue_dir, lock_file, cwd=tmp_path)
        _wait_for(
            lambda: "wait=main-shared-lock"
            in _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
        )

        cancelled = _queue(queue_dir, lock_file, "cancel", target_job, cwd=tmp_path)
        assert cancelled.stdout.strip() == "cancelled"
        _wait_for(
            lambda: "wait=main-exclusive-lock"
            in _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
        )
        assert not target_started.exists()
        assert not successor_started.exists()
        assert (queue_dir / "cancelled" / target_job).is_file()
        assert not (queue_dir / "state" / f"{target_job[:-4]}.state").exists()
        assert not (
            queue_dir / "cancel-requests" / f"{target_job[:-4]}.cancel"
        ).exists()

        fcntl.flock(raw_main, fcntl.LOCK_UN)
        _wait_for(successor_started.exists)
        output = worker.communicate(timeout=5)

    assert worker.returncode == 0, output
    assert (queue_dir / "done" / successor_job).is_file()
    _assert_lock_namespace_reacquirable(lock_file)


def test_worker_term_while_waiting_for_raw_main_lock_keeps_job_replayable(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    target_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="term-before-admission",
        resource="half",
    )

    with lock_file.open("w", encoding="utf-8") as raw_main:
        fcntl.flock(raw_main, fcntl.LOCK_EX)
        worker = _serve(queue_dir, lock_file, cwd=tmp_path)
        _wait_for(
            lambda: "wait=main-shared-lock"
            in _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
        )
        worker.terminate()
        stdout, stderr = worker.communicate(timeout=5)

        assert worker.returncode != 0, f"stdout={stdout}\nstderr={stderr}"
        assert not target_started.exists()
        assert (queue_dir / "jobs" / target_job).is_file()
        assert not (queue_dir / "running" / target_job).exists()
        assert not (queue_dir / "state" / f"{target_job[:-4]}.state").exists()
        assert not (
            queue_dir / "cancel-requests" / f"{target_job[:-4]}.cancel"
        ).exists()
        assert not (queue_dir / "worker.pid").exists()
        fcntl.flock(raw_main, fcntl.LOCK_UN)

    _assert_lock_namespace_reacquirable(lock_file)
    replacement = _serve(queue_dir, lock_file, cwd=tmp_path)
    replacement_output = replacement.communicate(timeout=5)
    assert replacement.returncode == 0, replacement_output
    assert target_started.is_file()
    assert (queue_dir / "done" / target_job).is_file()


@pytest.mark.parametrize(
    ("active_resource", "successor_resource"),
    [("half", "all"), ("all", "half")],
)
def test_worker_term_reaps_zombie_leader_only_after_live_member_absence(
    tmp_path: Path,
    active_resource: str,
    successor_resource: str,
) -> None:
    queue_dir = tmp_path / "active-queue"
    successor_queue = tmp_path / "successor-queue"
    lock_file = tmp_path / "gpu.lock"
    started = tmp_path / "started"
    identity_file = tmp_path / "identity"
    event_file = tmp_path / "teardown-events"
    successor_started = tmp_path / "successor-started"
    leader_code = "\n".join(
        [
            "import os, signal, time",
            "read_fd, write_fd = os.pipe()",
            "child = os.fork()",
            "if child == 0:",
            "    os.close(read_fd)",
            "    signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "    os.write(write_fd, b'1')",
            "    os.close(write_fd)",
            "    while True:",
            "        time.sleep(0.05)",
            "os.close(write_fd)",
            "os.read(read_fd, 1)",
            "os.close(read_fd)",
            f"event_fd = os.open({str(event_file)!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)",
            "def exit_on_term(*_):",
            "    os.write(event_fd, b'leader-exited\\n')",
            "    os._exit(0)",
            "signal.signal(signal.SIGTERM, exit_on_term)",
            f"open({str(identity_file)!r}, 'w', encoding='utf-8').write(",
            "    f'{os.getpid()} {child} {os.getpgrp()}\\n'",
            ")",
            f"open({str(started)!r}, 'a', encoding='utf-8').close()",
            "while True:",
            "    time.sleep(0.05)",
        ]
    )
    active_job = _add(
        queue_dir,
        lock_file,
        f"exec {shlex.quote(sys.executable)} -c {shlex.quote(leader_code)}",
        cwd=tmp_path,
        name=f"zombie-leader-{active_resource}",
        resource=active_resource,
    )
    successor_job = _add(
        successor_queue,
        lock_file,
        (
            f"printf 'successor-started\\n' >> {shlex.quote(str(event_file))}; "
            f"touch {shlex.quote(str(successor_started))}"
        ),
        cwd=tmp_path,
        name=f"replacement-{successor_resource}",
        resource=successor_resource,
    )
    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env={
            "TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS": "2",
            "TRAINING_QUEUE_TEST_TEARDOWN_EVENT_FILE": str(event_file),
        },
    )
    successor_worker: subprocess.Popen[str] | None = None
    pgid = 0
    try:
        _wait_for(started.exists)
        leader_pid, child_pid, pgid = (
            int(value) for value in identity_file.read_text(encoding="utf-8").split()
        )
        assert leader_pid == pgid
        successor_worker = _serve(successor_queue, lock_file, cwd=tmp_path)
        expected_wait = (
            "main-exclusive-lock"
            if successor_resource == "all"
            else "main-shared-lock"
        )
        _wait_for(
            lambda: f"wait={expected_wait}"
            in _queue(successor_queue, lock_file, "list", cwd=tmp_path).stdout
        )

        worker.terminate()
        _wait_for(
            lambda: "state=terminating wait=process-group-teardown"
            in _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
        )
        _wait_for(
            lambda: "leader-exited"
            in event_file.read_text(encoding="utf-8").splitlines()
            and bool(_process_state(child_pid))
            and not _process_state(child_pid).startswith(("Z", "X"))
        )

        assert _teardown_events(event_file, active_job) == ["term-sent"]
        assert (queue_dir / "running" / active_job).is_file()
        assert not (queue_dir / "failed" / active_job).exists()
        assert not successor_started.exists()
        assert Path(f"/proc/{child_pid}").exists()
        with (
            lock_file.open("a+", encoding="utf-8") as raw_exclusive,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(raw_exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)

        stdout, stderr = worker.communicate(timeout=6)
        assert worker.returncode != 0, f"stdout={stdout}\nstderr={stderr}"
        _wait_for(lambda: (queue_dir / "failed" / active_job).is_file())
        _wait_for(successor_started.exists)
        assert successor_worker is not None
        successor_output = successor_worker.communicate(timeout=5)
        assert successor_worker.returncode == 0, successor_output
        assert (successor_queue / "done" / successor_job).is_file()
        assert not _process_group_exists(pgid)
        assert not Path(f"/proc/{leader_pid}").exists()
        assert not Path(f"/proc/{child_pid}").exists()
        assert _teardown_events(event_file, active_job) == [
            "term-sent",
            "kill-sent",
            "live-members-absent",
            "leader-reap-called",
            "leader-reaped",
            "pgid-absent",
            "terminal-failed",
        ]

        _assert_lock_namespace_reacquirable(lock_file)
        with event_file.open("a", encoding="utf-8") as events:
            events.write("lock-reacquired\n")
        all_events = event_file.read_text(encoding="utf-8").splitlines()
        assert all_events.index(f"{active_job} pgid-absent") < all_events.index(
            "successor-started"
        )
        assert all_events.index("successor-started") < all_events.index(
            "lock-reacquired"
        )
    finally:
        if pgid and _process_group_exists(pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)
        if successor_worker is not None and successor_worker.poll() is None:
            successor_worker.terminate()
            successor_worker.communicate(timeout=5)


def test_prelaunch_cancel_kill_orders_absence_reap_and_never_executes(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    event_file = tmp_path / "teardown-events"
    wrapper_ready = tmp_path / "pre-pid-ready"
    launcher_ready = tmp_path / "prelaunch-ready"
    barrier = tmp_path / "pre-pid-barrier"
    os.mkfifo(barrier)
    barrier_fd = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    active_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="prelaunch-kill",
        resource="all",
    )
    successor_job = _add(
        queue_dir,
        lock_file,
        (
            f"printf 'successor-started\\n' >> {shlex.quote(str(event_file))}; "
            f"touch {shlex.quote(str(successor_started))}"
        ),
        cwd=tmp_path,
        name="prelaunch-successor",
        resource="half",
    )
    environment = _queue_env(queue_dir, lock_file)
    environment.update(
        {
            "TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS": "2",
            "TRAINING_QUEUE_TEST_PRE_PID_READY_FILE": str(wrapper_ready),
            "TRAINING_QUEUE_TEST_PRE_PID_BARRIER_FIFO": str(barrier),
            "TRAINING_QUEUE_TEST_PRELAUNCH_IGNORE_TERM": "1",
            "TRAINING_QUEUE_TEST_PRELAUNCH_READY_FILE": str(launcher_ready),
            "TRAINING_QUEUE_TEST_TEARDOWN_EVENT_FILE": str(event_file),
        }
    )
    worker = subprocess.Popen(
        ["bash", str(QUEUE_SCRIPT), "serve", "--idle-timeout", "0"],
        cwd=tmp_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    cancel_process: subprocess.Popen[str] | None = None
    child_pid = 0
    try:
        _wait_for(wrapper_ready.exists)
        _wait_for(launcher_ready.exists)
        _wrapper_pid, child_pid = (
            int(value)
            for value in wrapper_ready.read_text(encoding="utf-8").split()
        )
        assert int(launcher_ready.read_text(encoding="utf-8")) == child_pid
        assert Path(f"/proc/{child_pid}").exists()
        assert not target_started.exists()

        cancel_process = subprocess.Popen(
            ["bash", str(QUEUE_SCRIPT), "cancel", active_job],
            cwd=tmp_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cancel_marker = (
            queue_dir / "cancel-requests" / f"{active_job.removesuffix('.job')}.cancel"
        )
        _wait_for(cancel_marker.exists)
        os.write(barrier_fd, b"release\n")
        _wait_for(
            lambda: _teardown_events(event_file, active_job) == ["term-sent"]
        )

        assert bool(_process_state(child_pid))
        assert not _process_state(child_pid).startswith(("Z", "X"))
        assert (queue_dir / "running" / active_job).is_file()
        assert not (queue_dir / "cancelled" / active_job).exists()
        assert not target_started.exists()
        assert not successor_started.exists()
        with (
            lock_file.open("a+", encoding="utf-8") as raw_exclusive,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(raw_exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)

        cancel_stdout, cancel_stderr = cancel_process.communicate(timeout=6)
        assert cancel_process.returncode == 0, cancel_stderr
        assert cancel_stdout.strip() == "terminating"
        stdout, stderr = worker.communicate(timeout=6)
        assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        assert (queue_dir / "cancelled" / active_job).is_file()
        assert (queue_dir / "done" / successor_job).is_file()
        assert not target_started.exists()
        assert successor_started.is_file()
        assert not Path(f"/proc/{child_pid}").exists()
        assert not _process_group_exists(child_pid)
        assert _teardown_events(event_file, active_job) == [
            "term-sent",
            "kill-sent",
            "live-members-absent",
            "leader-reap-called",
            "leader-reaped",
            "pgid-absent",
            "terminal-cancelled",
        ]
        all_events = event_file.read_text(encoding="utf-8").splitlines()
        assert all_events.index(f"{active_job} pgid-absent") < all_events.index(
            f"{active_job} terminal-cancelled"
        )
        assert all_events.index(f"{active_job} terminal-cancelled") < all_events.index(
            "successor-started"
        )
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        os.close(barrier_fd)
        if cancel_process is not None and cancel_process.poll() is None:
            cancel_process.terminate()
            cancel_process.communicate(timeout=5)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)
        if child_pid and Path(f"/proc/{child_pid}").exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


@pytest.mark.parametrize("declared_resource", ["half", "all", None])
def test_worker_term_holds_capacity_until_owned_process_group_is_gone(
    tmp_path: Path, declared_resource: str | None
) -> None:
    queue_dir = tmp_path / "active-queue"
    successor_queue = tmp_path / "successor-queue"
    lock_file = tmp_path / "gpu.lock"
    started = tmp_path / "started"
    identity_file = tmp_path / "identity"
    successor = tmp_path / "successor"
    command = (
        "trap '' TERM; "
        "bash -c 'trap \"\" TERM; while :; do sleep 0.05; done' & child=$!; "
        "pgid=$(ps -o pgid= -p $$ | tr -d ' '); "
        f"printf '%s %s %s\\n' \"$$\" \"$child\" \"$pgid\" > {shlex.quote(str(identity_file))}; "
        f"touch {shlex.quote(str(started))}; "
        "while :; do sleep 0.05; done"
    )
    if declared_resource is None:
        (queue_dir / "jobs").mkdir(parents=True)
        active_job = "000_legacy-term-tree.job"
        (queue_dir / "jobs" / active_job).write_text(
            f"#!/usr/bin/env bash\n{command}\n", encoding="utf-8"
        )
        expected_resource = "all"
    else:
        active_job = _add(
            queue_dir,
            lock_file,
            command,
            cwd=tmp_path,
            name=f"term-{declared_resource}-tree",
            resource=declared_resource,
        )
        expected_resource = declared_resource
    successor_resource = "all" if expected_resource == "half" else "half"
    successor_job = _add(
        successor_queue,
        lock_file,
        f"touch {shlex.quote(str(successor))}",
        cwd=tmp_path,
        name="blocked-successor",
        resource=successor_resource,
    )

    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env={"TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS": "2"},
    )
    successor_worker: subprocess.Popen[str] | None = None
    pgid = 0
    try:
        _wait_for(started.exists)
        leader_pid, child_pid, pgid = (
            int(value) for value in identity_file.read_text(encoding="utf-8").split()
        )
        assert leader_pid == pgid
        assert _process_group_exists(pgid)
        successor_worker = _serve(successor_queue, lock_file, cwd=tmp_path)
        expected_wait = (
            "main-exclusive-lock"
            if successor_resource == "all"
            else "main-shared-lock"
        )
        _wait_for(
            lambda: f"wait={expected_wait}"
            in _queue(successor_queue, lock_file, "list", cwd=tmp_path).stdout
        )

        termination_started = time.monotonic()
        worker.terminate()
        terminating = ""
        status = ""

        def observe_terminating() -> bool:
            nonlocal terminating, status
            terminating = _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
            status = _queue(queue_dir, lock_file, "status", cwd=tmp_path).stdout
            return "state=terminating" in terminating and "executing=1" in status

        _wait_for(observe_terminating)
        assert f"resource={expected_resource}" in terminating
        assert f"pid={leader_pid} pgid={pgid}" in terminating
        assert "state=terminating wait=process-group-teardown" in terminating
        assert "executing=1" in status
        assert _process_group_exists(pgid)
        assert Path(f"/proc/{leader_pid}").exists()
        assert Path(f"/proc/{child_pid}").exists()
        assert not successor.exists()
        with (
            lock_file.open("a+", encoding="utf-8") as raw_exclusive,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(raw_exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Repeated worker TERM during the grace period must not interrupt the
        # wrapper-owned teardown or release its allocation early.
        worker.terminate()
        stdout, stderr = worker.communicate(timeout=5)
        assert worker.returncode != 0, f"stdout={stdout}\nstderr={stderr}"
        assert time.monotonic() - termination_started >= 1.8
        assert not _process_group_exists(pgid)
        assert not Path(f"/proc/{leader_pid}").exists()
        assert (queue_dir / "failed" / active_job).is_file()
        final = _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout
        assert f"pid={leader_pid} pgid={pgid} state=failed" in final

        _wait_for(successor.exists)
        successor_output = successor_worker.communicate(timeout=5)
        assert successor_worker.returncode == 0, successor_output
        assert (successor_queue / "done" / successor_job).is_file()
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        if pgid and _process_group_exists(pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        if worker.poll() is None:
            worker.terminate()
        try:
            worker.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate(timeout=5)
        if successor_worker is not None and successor_worker.poll() is None:
            successor_worker.terminate()
            successor_worker.communicate(timeout=5)


def test_running_cancel_waits_for_required_external_teardown_ack(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    started = tmp_path / "started"
    identity = tmp_path / "identity"
    successor = tmp_path / "successor"
    active_job = _add(
        queue_dir,
        lock_file,
        (
            "trap '' TERM; pgid=$(ps -o pgid= -p $$ | tr -d ' '); "
            f"printf '%s %s\\n' \"$$\" \"$pgid\" > {shlex.quote(str(identity))}; "
            f"touch {shlex.quote(str(started))}; while :; do sleep 0.05; done"
        ),
        cwd=tmp_path,
        name="external-owner",
        resource="all",
        require_external_teardown_ack=True,
    )
    successor_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor))}",
        cwd=tmp_path,
        name="external-successor",
        resource="half",
    )
    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    pgid = 0
    ack = queue_dir / "control" / "external-acks" / f"{active_job[:-4]}.ack"
    try:
        _wait_for(started.exists)
        leader_pid, pgid = (
            int(value) for value in identity.read_text(encoding="utf-8").split()
        )
        cancelled = _queue(queue_dir, lock_file, "cancel", active_job, cwd=tmp_path)
        assert cancelled.stdout.strip() == "terminating"
        _wait_for(
            lambda: "state=terminating wait=external-teardown"
            in _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout,
            timeout=5,
        )
        assert not _process_group_exists(pgid)
        assert not Path(f"/proc/{leader_pid}").exists()
        assert (queue_dir / "running" / active_job).is_file()
        assert not (queue_dir / "cancelled" / active_job).exists()
        assert not successor.exists()
        assert not ack.exists()
        with (
            lock_file.open("a+", encoding="utf-8") as raw_exclusive,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(raw_exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)

        protected = tmp_path / "protected-ack-target"
        protected.write_text(f"{active_job}\n", encoding="utf-8")
        ack.symlink_to(protected)
        _wait_for_external_ack_rejection(queue_dir, active_job)
        ack.unlink()
        ack.write_text("wrong-job.job\n", encoding="utf-8")
        _wait_for_external_ack_rejection(queue_dir, active_job)
        ack.unlink()

        temporary_ack = ack.parent / ".tmp.test-ack"
        temporary_ack.write_text(f"{active_job}\n", encoding="utf-8")
        os.replace(temporary_ack, ack)
        _wait_for(lambda: (queue_dir / "cancelled" / active_job).is_file())
        _wait_for(successor.exists)
        output = worker.communicate(timeout=5)
        assert worker.returncode == 0, output
        assert (queue_dir / "done" / successor_job).is_file()
        assert not ack.exists()
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        if (queue_dir / "running" / active_job).exists():
            ack.unlink(missing_ok=True)
            ack.write_text(f"{active_job}\n", encoding="utf-8")
        if pgid and _process_group_exists(pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)


def test_terminating_half_never_allows_three_live_cross_queue_halves(
    tmp_path: Path,
) -> None:
    lock_file = tmp_path / "gpu.lock"
    active_queue = tmp_path / "active"
    started = tmp_path / "active-started"
    identity = tmp_path / "active-identity"
    active_job = _add(
        active_queue,
        lock_file,
        (
            "trap '' TERM; pgid=$(ps -o pgid= -p $$ | tr -d ' '); "
            f"printf '%s %s\\n' \"$$\" \"$pgid\" > {shlex.quote(str(identity))}; "
            f"touch {shlex.quote(str(started))}; while :; do sleep 0.05; done"
        ),
        cwd=tmp_path,
        name="terminating-half",
        resource="half",
    )
    release = tmp_path / "release-successors"
    successor_workers: list[subprocess.Popen[str]] = []
    successor_markers: list[Path] = []
    for index in range(2):
        queue_dir = tmp_path / f"successor-{index}"
        marker = tmp_path / f"successor-started-{index}"
        successor_markers.append(marker)
        _add(
            queue_dir,
            lock_file,
            (
                f"touch {shlex.quote(str(marker))}; "
                f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.02; done"
            ),
            cwd=tmp_path,
            name=f"successor-half-{index}",
            resource="half",
        )

    active_worker = _serve(
        active_queue,
        lock_file,
        cwd=tmp_path,
        extra_env={"TRAINING_QUEUE_TEST_TERM_GRACE_SECONDS": "2"},
    )
    pgid = 0
    try:
        _wait_for(started.exists)
        _leader_pid, pgid = (
            int(value) for value in identity.read_text(encoding="utf-8").split()
        )
        for index in range(2):
            successor_workers.append(
                _serve(tmp_path / f"successor-{index}", lock_file, cwd=tmp_path)
            )
        _wait_for(lambda: sum(marker.exists() for marker in successor_markers) == 1)
        _wait_for_slot_waiter([tmp_path / f"successor-{i}" for i in range(2)], lock_file)
        assert sum(marker.exists() for marker in successor_markers) == 1

        active_worker.terminate()
        _wait_for(
            lambda: "state=terminating" in _queue(
                active_queue, lock_file, "list", cwd=tmp_path
            ).stdout
        )
        assert _process_group_exists(pgid)
        assert sum(marker.exists() for marker in successor_markers) == 1
        _wait_for(lambda: not _process_group_exists(pgid))
        _wait_for(lambda: (active_queue / "failed" / active_job).is_file())
        _wait_for(lambda: all(marker.exists() for marker in successor_markers))
        release.touch()
        active_output = active_worker.communicate(timeout=5)
        successor_outputs = [worker.communicate(timeout=5) for worker in successor_workers]
        assert active_worker.returncode != 0, active_output
        assert all(worker.returncode == 0 for worker in successor_workers), successor_outputs
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        release.touch(exist_ok=True)
        if pgid and _process_group_exists(pgid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
        if active_worker.poll() is None:
            active_worker.terminate()
        active_worker.communicate(timeout=5)
        for successor_worker in successor_workers:
            if successor_worker.poll() is None:
                successor_worker.terminate()
            successor_worker.communicate(timeout=5)


def test_worker_term_after_move_fails_job_without_launch_and_releases_capacity(
    tmp_path: Path,
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    ready = tmp_path / "post-move-ready"
    barrier = tmp_path / "post-move-barrier"
    os.mkfifo(barrier)
    barrier_fd = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    active_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="post-move-term",
        resource="half",
    )
    successor_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="exclusive-successor",
        resource="all",
    )
    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env={
            "TRAINING_QUEUE_TEST_POST_MOVE_READY_FILE": str(ready),
            "TRAINING_QUEUE_TEST_POST_MOVE_BARRIER_FIFO": str(barrier),
        },
    )
    replacement: subprocess.Popen[str] | None = None
    try:
        _wait_for(ready.exists)
        wrapper_pid = int(ready.read_text(encoding="utf-8"))
        assert (queue_dir / "running" / active_job).is_file()
        assert not target_started.exists()

        worker.terminate()
        _wait_for((queue_dir / "stop").exists)
        with contextlib.suppress(ProcessLookupError):
            os.kill(wrapper_pid, signal.SIGTERM)
        os.write(barrier_fd, b"release\n")
        stdout, stderr = worker.communicate(timeout=5)

        assert worker.returncode != 0, f"stdout={stdout}\nstderr={stderr}"
        assert not target_started.exists()
        assert (queue_dir / "failed" / active_job).is_file()
        assert not (queue_dir / "running" / active_job).exists()
        assert (queue_dir / "jobs" / successor_job).is_file()
        assert not (queue_dir / "running" / successor_job).exists()
        state_file = queue_dir / "state" / f"{active_job[:-4]}.state"
        state = state_file.read_text(encoding="utf-8")
        assert "state=failed" in state
        assert "wait=signal" in state
        assert "state=running" not in state
        assert not (
            queue_dir / "cancel-requests" / f"{active_job[:-4]}.cancel"
        ).exists()
        assert (queue_dir / "stop").is_file()
        assert not (queue_dir / "worker.pid").exists()
        assert not (queue_dir / "control/worker.protocol").exists()
        assert not _process_state(wrapper_pid)
        assert not list((queue_dir / "state").glob(".launch.*"))
        _assert_lock_namespace_reacquirable(lock_file)

        replacement = _serve(queue_dir, lock_file, cwd=tmp_path)
        _wait_for(successor_started.exists)
        replacement_output = replacement.communicate(timeout=5)
        assert replacement.returncode == 0, replacement_output
        assert (queue_dir / "done" / successor_job).is_file()
        assert not (queue_dir / "stop").exists()
        assert not (queue_dir / "worker.pid").exists()
        assert not (queue_dir / "control/worker.protocol").exists()
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        os.close(barrier_fd)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)
        if replacement is not None and replacement.poll() is None:
            replacement.terminate()
            replacement.communicate(timeout=5)


@pytest.mark.parametrize("stop_during_empty_scan", [False, True])
def test_cooperative_stop_after_move_requeues_without_launch(
    tmp_path: Path, stop_during_empty_scan: bool
) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    ready = tmp_path / "post-move-ready"
    barrier = tmp_path / "post-move-barrier"
    os.mkfifo(barrier)
    barrier_fd = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    empty_ready = tmp_path / "empty-scan-ready"
    empty_barrier = tmp_path / "empty-scan-barrier"
    os.mkfifo(empty_barrier)
    empty_fd = os.open(empty_barrier, os.O_RDWR | os.O_NONBLOCK)
    worker_env = {
        "TRAINING_QUEUE_TEST_POST_MOVE_READY_FILE": str(ready),
        "TRAINING_QUEUE_TEST_POST_MOVE_BARRIER_FIFO": str(barrier),
    }
    if stop_during_empty_scan:
        # Freeze an actual empty history result. The wrapper requeues and exits
        # before the supervisor sees that stale result and counts active jobs.
        binary = tmp_path / "bin"
        binary.mkdir()
        python = binary / "python3"
        python.write_text(
            f"#!{sys.executable}\n"
            "import subprocess, sys\nfrom pathlib import Path\n"
            f"result = subprocess.run([{sys.executable!r}, *sys.argv[1:]], capture_output=True)\n"
            f"ready = Path({str(empty_ready)!r})\n"
            "if result.returncode == 0 and 'order' in sys.argv and sys.argv[-1] == 'first' and not result.stdout.strip() and not ready.exists():\n"
            "    ready.write_text('empty')\n"
            f"    with open({str(empty_barrier)!r}) as barrier: barrier.readline()\n"
            "sys.stdout.buffer.write(result.stdout)\n"
            "sys.stderr.buffer.write(result.stderr)\n"
            "raise SystemExit(result.returncode)\n"
        )
        python.chmod(0o755)
        worker_env["PATH"] = f"{binary}:{os.environ['PATH']}"
    target_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="post-move-stop",
        resource="all",
    )
    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env=worker_env,
    )
    replacement: subprocess.Popen[str] | None = None
    try:
        _wait_for(ready.exists)
        wrapper_pid = int(ready.read_text(encoding="utf-8"))
        assert (queue_dir / "running" / target_job).is_file()
        assert not target_started.exists()
        if stop_during_empty_scan:
            _wait_for(empty_ready.exists)

        stop_result = _queue(queue_dir, lock_file, "stop", cwd=tmp_path)
        assert "stop requested" in stop_result.stdout
        os.write(barrier_fd, b"release\n")
        if stop_during_empty_scan:

            def wrapper_exited() -> bool:
                state = _process_state(wrapper_pid)
                return not state or state.startswith("Z")

            _wait_for(wrapper_exited)
            assert (queue_dir / "jobs" / target_job).is_file()
            os.write(empty_fd, b"release\n")
        stdout, stderr = worker.communicate(timeout=5)

        assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        if stop_during_empty_scan:
            assert "queue empty for 0s; exiting." in stdout
        assert not target_started.exists()
        assert (queue_dir / "jobs" / target_job).is_file()
        assert not (queue_dir / "running" / target_job).exists()
        assert not (queue_dir / "state" / f"{target_job[:-4]}.state").exists()
        assert not (
            queue_dir / "cancel-requests" / f"{target_job[:-4]}.cancel"
        ).exists()
        assert not (queue_dir / "stop").exists()
        assert not (queue_dir / "worker.pid").exists()
        assert not (queue_dir / "control/worker.protocol").exists()
        assert not _process_state(wrapper_pid)
        assert not list((queue_dir / "state").glob(".launch.*"))
        _assert_lock_namespace_reacquirable(lock_file)

        replacement = _serve(queue_dir, lock_file, cwd=tmp_path)
        replacement_output = replacement.communicate(timeout=5)
        assert replacement.returncode == 0, replacement_output
        assert target_started.is_file()
        assert (queue_dir / "done" / target_job).is_file()
        assert not (queue_dir / "jobs" / target_job).exists()
        assert not (queue_dir / "running" / target_job).exists()
        assert not (queue_dir / "stop").exists()
        assert not (queue_dir / "worker.pid").exists()
        assert not (queue_dir / "control/worker.protocol").exists()
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        os.write(empty_fd, b"release\n")
        os.close(empty_fd)
        os.close(barrier_fd)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)
        if replacement is not None and replacement.poll() is None:
            replacement.terminate()
            replacement.communicate(timeout=5)


def test_wrapper_term_during_pid_capture_reaps_gated_child(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    target_started = tmp_path / "target-started"
    successor_started = tmp_path / "successor-started"
    ready = tmp_path / "pre-pid-ready"
    barrier = tmp_path / "pre-pid-barrier"
    os.mkfifo(barrier)
    barrier_fd = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    active_job = _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(target_started))}",
        cwd=tmp_path,
        name="pre-pid-term",
        resource="half",
    )
    _add(
        queue_dir,
        lock_file,
        f"touch {shlex.quote(str(successor_started))}",
        cwd=tmp_path,
        name="pre-pid-successor",
        resource="all",
    )
    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env={
            "TRAINING_QUEUE_TEST_PRE_PID_READY_FILE": str(ready),
            "TRAINING_QUEUE_TEST_PRE_PID_BARRIER_FIFO": str(barrier),
        },
    )
    child_pid = 0
    try:
        _wait_for(ready.exists)
        wrapper_pid, child_pid = (
            int(value) for value in ready.read_text(encoding="utf-8").split()
        )
        assert Path(f"/proc/{child_pid}").exists()
        assert not target_started.exists()

        os.kill(wrapper_pid, signal.SIGTERM)
        os.write(barrier_fd, b"release\n")
        stdout, stderr = worker.communicate(timeout=5)

        assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
        assert not Path(f"/proc/{child_pid}").exists()
        assert not target_started.exists()
        assert successor_started.is_file()
        assert (queue_dir / "failed" / active_job).is_file()
        assert not (queue_dir / "running" / active_job).exists()
        state_file = queue_dir / "state" / f"{active_job[:-4]}.state"
        state = state_file.read_text(encoding="utf-8")
        assert "state=failed" in state
        assert "wait=signal" in state
        assert f"pid={child_pid}" in state
        assert not (
            queue_dir / "cancel-requests" / f"{active_job[:-4]}.cancel"
        ).exists()
        assert not list((queue_dir / "state").glob(".launch.*"))
        _assert_lock_namespace_reacquirable(lock_file)
    finally:
        os.close(barrier_fd)
        if worker.poll() is None:
            worker.terminate()
        worker.communicate(timeout=5)
        if child_pid and Path(f"/proc/{child_pid}").exists():
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGTERM)


def test_rejects_symlink_in_shared_lock_namespace(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    protected = tmp_path / "protected"
    protected.write_text("unchanged", encoding="utf-8")
    lock_file.touch()
    lock_file.with_name(f"{lock_file.name}.slot-0").symlink_to(protected)
    job = _add(
        queue_dir,
        lock_file,
        "true",
        cwd=tmp_path,
        name="unsafe-lock-namespace",
        resource="half",
    )

    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = worker.communicate(timeout=5)

    assert worker.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    assert protected.read_text(encoding="utf-8") == "unchanged"
    assert (queue_dir / "failed" / job).is_file()
    assert "GPU lock namespace is unavailable" in (
        queue_dir / "logs" / job.removesuffix(".job")
    ).with_suffix(".log").read_text(encoding="utf-8")


DAY_NS = 24 * 60 * 60 * 10**9
END_NS = 1_792_368_000 * 10**9


def _completion(queue: Path, job: str) -> Path:
    return queue / "control/completions" / f"{job[:-4]}.json"


def _finished_job(tmp_path: Path, state: str) -> tuple[Path, Path, str]:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    job = (
        _queue(
            queue_dir,
            lock_file,
            "add",
            "true" if state == "done" else "exit 7",
            "--name",
            "history",
            "--idempotency-key",
            "history-attempt",
            cwd=tmp_path,
            now_ns=END_NS - 20 * DAY_NS,
        )
        .stdout.strip()
        .removeprefix("queued: ")
    )
    if state == "cancelled":
        _queue(queue_dir, lock_file, "cancel", job, cwd=tmp_path, now_ns=END_NS)
    else:
        worker = _serve(
            queue_dir,
            lock_file,
            cwd=tmp_path,
            extra_env={"TRAINING_QUEUE_TEST_NOW_NS": str(END_NS)},
        )
        stdout, stderr = worker.communicate(timeout=15)
        assert worker.returncode == 0, (stdout, stderr)
    assert (queue_dir / state / job).is_file()
    return queue_dir, lock_file, job


@pytest.mark.parametrize("state", ["done", "failed", "cancelled"])
@pytest.mark.parametrize("offset_ns", [-1, 0, 1])
def test_retention_uses_terminal_time_and_strict_seven_day_boundary(
    tmp_path: Path,
    state: str,
    offset_ns: int,
) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, state)
    record = json.loads(_completion(queue_dir, job).read_text())
    assert record["finished_at_ns"] == END_NS
    assert record["retention_started_at_ns"] == END_NS
    assert record["time_source"] == "recorded"
    stem = job[:-4]
    log = queue_dir / "logs" / f"{stem}.log"
    log.write_text("terminal log")
    state_file = queue_dir / "state" / f"{stem}.state"
    state_file.write_text(f"state={state}\n")
    owner = queue_dir / "state" / f"{stem}.owner"
    owner.write_text(f"pid=999999999\nstart_time=1\njob={job}\n")
    cancel = queue_dir / "cancel-requests" / f"{stem}.cancel"
    cancel.touch()
    ack = queue_dir / "control/external-acks" / f"{stem}.ack"
    ack.write_text(job)
    repro = queue_dir / "repro" / stem / "predictions" / "proof"
    repro.parent.mkdir(parents=True, exist_ok=True)
    repro.write_bytes(b"unchanged reproducibility bundle\x00")

    def repro_hashes() -> dict[str, str]:
        root = queue_dir / "repro" / stem
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    digest = repro_hashes()

    for _ in range(2):
        _queue(
            queue_dir,
            lock_file,
            "status",
            cwd=tmp_path,
            now_ns=END_NS + 7 * DAY_NS + offset_ns,
        )
        for path in (queue_dir / state / job, log, state_file, owner, cancel, ack):
            assert path.exists() == (offset_ns <= 0), path
        assert repro_hashes() == digest
    assert ("expired_at_ns" in json.loads(_completion(queue_dir, job).read_text())) == (
        offset_ns > 0
    )
    assert len(list((queue_dir / "control/registrations").glob("*.receipt"))) == 1


@pytest.mark.parametrize("state", ["done", "failed", "cancelled"])
def test_expired_keyed_registration_remains_idempotent_concurrently(
    tmp_path: Path,
    state: str,
) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, state)
    arguments = (
        "add",
        "true" if state == "done" else "exit 7",
        "--name",
        "history",
        "--idempotency-key",
        "history-attempt",
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        replies = list(
            executor.map(
                lambda _: (
                    _queue(
                        queue_dir,
                        lock_file,
                        *arguments,
                        cwd=tmp_path,
                        now_ns=END_NS + 8 * DAY_NS,
                    ).stdout
                ),
                range(8),
            )
        )
    assert set(replies) == {f"queued: {job}\n"}
    assert not list((queue_dir / "jobs").glob("*.job"))
    assert not (queue_dir / state / job).exists()
    rejected = _queue(
        queue_dir,
        lock_file,
        *arguments,
        "--resource",
        "half",
        cwd=tmp_path,
        now_ns=END_NS + 8 * DAY_NS,
        check=False,
    )
    assert rejected.returncode == 2
    assert "different request" in rejected.stderr
    listing = _queue(
        queue_dir,
        lock_file,
        "list",
        "--include-expired",
        cwd=tmp_path,
        now_ns=END_NS + 8 * DAY_NS,
    ).stdout
    assert (
        f"{job} resource=- slot=- pid=- pgid=- state={state} wait=history-expired history=expired"
        in listing
    )
    assert (
        _queue(
            queue_dir,
            lock_file,
            "cancel",
            job,
            cwd=tmp_path,
            now_ns=END_NS + 8 * DAY_NS,
        ).stdout.strip()
        == state
    )


def test_new_filenames_sort_newest_first_across_dates_and_are_unique(
    tmp_path: Path,
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    midnight = int(datetime(2026, 10, 19, tzinfo=UTC).timestamp()) * 10**9
    jobs = []
    for timestamp in (midnight - 1, midnight, midnight + 1):
        job = (
            _queue(
                queue_dir,
                lock_file,
                "add",
                "true",
                "--name",
                "readable-name",
                cwd=tmp_path,
                now_ns=timestamp,
            )
            .stdout.strip()
            .removeprefix("queued: ")
        )
        jobs.append(job)
    assert sorted(jobs) == jobs[::-1]
    assert "2026-10-18T23-59-59-999999999Z" in jobs[0]
    assert "2026-10-19T00-00-00-000000000Z" in jobs[1]
    assert all(job.endswith("_readable-name.job") for job in jobs)
    with ThreadPoolExecutor(max_workers=4) as executor:
        replies = list(
            executor.map(
                lambda _: (
                    _queue(
                        queue_dir,
                        lock_file,
                        "add",
                        "true",
                        "--name",
                        "same-time",
                        cwd=tmp_path,
                        now_ns=midnight + 2,
                    ).stdout
                ),
                range(8),
            )
        )
    assert len(set(replies)) == 8


def test_fifo_separates_new_display_order_and_mixed_legacy_jobs(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    output = tmp_path / "order"
    legacy = f"{END_NS - DAY_NS}_1_legacy.job"
    (queue_dir / "jobs" / legacy).write_text(
        f"#!/usr/bin/env bash\necho legacy >> {output}\n"
    )
    jobs = []
    for index in range(3):
        jobs.append(
            _queue(
                queue_dir,
                lock_file,
                "add",
                f"echo {index} >> {output}",
                "--name",
                str(index),
                cwd=tmp_path,
                now_ns=END_NS + index,
            )
            .stdout.strip()
            .removeprefix("queued: ")
        )
    assert sorted(jobs) == jobs[::-1]
    assert all(job < legacy for job in jobs)
    pending = _queue(queue_dir, lock_file, "list", cwd=tmp_path).stdout.splitlines()
    assert [line.split()[0] for line in pending] == [legacy, *jobs]
    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = worker.communicate(timeout=15)
    assert worker.returncode == 0, (stdout, stderr)
    assert output.read_text().splitlines() == ["legacy", "0", "1", "2"]
    assert sorted(p.name for p in (queue_dir / "done").glob("*.job")) == sorted(
        [legacy, *jobs]
    )


@pytest.mark.parametrize("state", ["queued", "waiting", "running", "terminating"])
def test_retention_protects_nonterminal_jobs_and_logs(
    tmp_path: Path, state: str
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    job = (
        _queue(
            queue_dir,
            lock_file,
            "add",
            "true",
            cwd=tmp_path,
            now_ns=END_NS - 30 * DAY_NS,
        )
        .stdout.strip()
        .removeprefix("queued: ")
    )
    path = queue_dir / "jobs" / job
    if state in {"running", "terminating"}:
        path = path.rename(queue_dir / "running" / job)
    state_file = queue_dir / "state" / f"{job[:-4]}.state"
    state_file.write_text(f"state={state}\nfinished_at_ns={END_NS - 30 * DAY_NS}\n")
    log = queue_dir / "logs" / f"{job[:-4]}.log"
    log.write_text("protected")
    _queue(queue_dir, lock_file, "list", cwd=tmp_path, now_ns=END_NS)
    assert path.is_file() and state_file.is_file()
    assert log.read_text() == "protected"
    assert not _completion(queue_dir, job).exists()


def test_legacy_history_gets_first_observation_grace_without_renaming(
    tmp_path: Path,
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    jobs = []
    for index, state in enumerate(("done", "failed", "cancelled")):
        job = f"{END_NS - 100 * DAY_NS}_{index}_legacy.job"
        jobs.append(job)
        path = queue_dir / state / job
        path.write_text("#!/usr/bin/env bash\ntrue\n")
        os.utime(path, ns=(END_NS - 100 * DAY_NS, END_NS - 100 * DAY_NS))
        (queue_dir / "logs" / f"{job[:-4]}.log").write_text("old log")
    orphan = queue_dir / "logs/orphan.log"
    orphan.write_text("no terminal evidence")
    observed = _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    assert "3 legacy terminal jobs" in observed.stderr
    assert "logs without terminal evidence" in observed.stderr
    for job in jobs:
        record = json.loads(_completion(queue_dir, job).read_text())
        assert record["finished_at_ns"] is None
        assert record["retention_started_at_ns"] == END_NS
        assert record["time_source"] == "legacy-observed"
    # Repeated uses do not reset the migration grace period.
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS + 7 * DAY_NS)
    assert (
        sum(
            len(list((queue_dir / state).glob("*.job")))
            for state in ("done", "failed", "cancelled")
        )
        == 3
    )
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS + 7 * DAY_NS + 1)
    assert all(
        not list((queue_dir / state).glob("*.job"))
        for state in ("done", "failed", "cancelled")
    )
    assert list((queue_dir / "logs").glob("*.log")) == [orphan]
    listing = _queue(
        queue_dir,
        lock_file,
        "list",
        "--include-expired",
        cwd=tmp_path,
        now_ns=END_NS + 8 * DAY_NS,
    )
    assert (
        "orphan.job resource=- slot=- pid=- pgid=- state=unknown wait=history-orphaned history=orphaned"
        in listing.stdout
    )


def test_retention_during_terminal_publication_keeps_live_wrapper_log(
    tmp_path: Path,
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    job = (
        _queue(
            queue_dir,
            lock_file,
            "add",
            "echo proof",
            cwd=tmp_path,
            now_ns=END_NS - 20 * DAY_NS,
        )
        .stdout.strip()
        .removeprefix("queued: ")
    )
    ready, barrier = tmp_path / "ready", tmp_path / "barrier"
    os.mkfifo(barrier)
    descriptor = os.open(barrier, os.O_RDWR | os.O_NONBLOCK)
    worker = _serve(
        queue_dir,
        lock_file,
        cwd=tmp_path,
        extra_env={
            "TRAINING_QUEUE_TEST_NOW_NS": str(END_NS),
            "TRAINING_QUEUE_TEST_PRE_COMPLETE_READY_FILE": str(ready),
            "TRAINING_QUEUE_TEST_PRE_COMPLETE_BARRIER_FIFO": str(barrier),
        },
    )
    try:
        _wait_for(ready.exists)
        assert (queue_dir / "done" / job).is_file()
        # Simulate a consumer accessing the queue while terminal publication is
        # paused, with the wall clock well beyond the registration timestamp.
        _queue(
            queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS + 30 * DAY_NS
        )
        assert (queue_dir / "logs" / f"{job[:-4]}.log").is_file()
        assert not _completion(queue_dir, job).exists()
    finally:
        os.write(descriptor, b"release\n")
        os.close(descriptor)
        stdout, stderr = worker.communicate(timeout=10)
    assert worker.returncode == 0, (stdout, stderr)
    assert (
        json.loads(_completion(queue_dir, job).read_text())["finished_at_ns"] == END_NS
    )


@pytest.mark.parametrize(
    "command", ["add", "status", "list", "cancel", "clear", "stop", "start", "serve"]
)
def test_public_queue_commands_trigger_retention(tmp_path: Path, command: str) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, "cancelled")
    arguments = [command]
    if command == "add":
        arguments.append("true")
    elif command == "cancel":
        arguments.append(job)
    elif command in {"start", "serve"}:
        arguments.extend(["--idle-timeout", "0"])
    _queue(queue_dir, lock_file, *arguments, cwd=tmp_path, now_ns=END_NS + 8 * DAY_NS)
    if command == "start":
        _wait_for(lambda: not (queue_dir / "worker.pid").exists())
    assert not (queue_dir / "cancelled" / job).exists()
    assert "expired_at_ns" in json.loads(_completion(queue_dir, job).read_text())


@pytest.mark.parametrize(
    "damage",
    [
        "log-symlink",
        "log-directory",
        "state-symlink",
        "owner-symlink",
        "completion-symlink",
        "job-directory",
        "ack-fifo",
        "control-symlink",
        "corrupt-with-symlink",
    ],
)
def test_retention_fails_closed_on_unsafe_history(tmp_path: Path, damage: str) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, "done")
    protected_file = tmp_path / "protected"
    protected_file.write_text("must remain")
    log = queue_dir / "logs" / f"{job[:-4]}.log"
    if damage in {"log-symlink", "corrupt-with-symlink"}:
        log.unlink()
        log.symlink_to(protected_file)
        if damage == "corrupt-with-symlink":
            _completion(queue_dir, job).write_text("{")
    elif damage == "log-directory":
        log.unlink()
        log.mkdir()
    elif damage == "job-directory":
        (queue_dir / "done" / job).unlink()
        (queue_dir / "done" / job).mkdir()
    elif damage == "ack-fifo":
        os.mkfifo(queue_dir / "control/external-acks" / f"{job[:-4]}.ack")
    elif damage == "control-symlink":
        control = queue_dir / "control"
        control.rename(queue_dir / "saved-control")
        control.symlink_to(queue_dir / "saved-control", target_is_directory=True)
    else:
        path = (
            _completion(queue_dir, job)
            if damage == "completion-symlink"
            else queue_dir / "state" / f"{job[:-4]}.{damage.split('-')[0]}"
        )
        path.unlink(missing_ok=True)
        path.symlink_to(protected_file)
    result = _queue(
        queue_dir,
        lock_file,
        "status",
        cwd=tmp_path,
        now_ns=END_NS + 8 * DAY_NS,
        check=False,
    )
    assert result.returncode == 1
    assert "error: queue" in result.stderr
    assert (queue_dir / "done" / job).exists()
    assert protected_file.read_text() == "must remain"


def test_legacy_cancelled_invalid_resource_job_does_not_block_queue(
    tmp_path: Path,
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    job = "1790611056370267633_1076243_i935-context-shard-00124-meiji-r13-20260929.job"
    terminal = queue_dir / "cancelled" / job
    terminal.write_text("#!/usr/bin/env bash\ntrue\n")
    state = queue_dir / "state" / f"{job[:-4]}.state"
    state.write_text(
        "state=failed\nresource=invalid\nslot=-\npid=-\npgid=-\nwait=invalid-resource\n"
    )
    log = queue_dir / "logs" / f"{job[:-4]}.log"
    log.write_text("invalid resource in legacy job\n")
    expected = {path: path.read_bytes() for path in (terminal, state, log)}
    healthy = "1790611056370267634_1_healthy.job"
    (queue_dir / "done" / healthy).write_text("true\n")
    healthy_log = queue_dir / "logs" / f"{healthy[:-4]}.log"
    healthy_log.write_text("healthy legacy log")

    for timestamp in (END_NS, END_NS + 8 * DAY_NS, END_NS + 30 * DAY_NS):
        for arguments in (("status",), ("add", "true", "--name", "still-available")):
            result = _queue(
                queue_dir, lock_file, *arguments, cwd=tmp_path, now_ns=timestamp
            )
            reports = [
                line
                for line in result.stderr.splitlines()
                if "inconsistent jobs" in line
            ]
            assert len(reports) == 1
            assert "retaining 1 inconsistent jobs" in reports[0]
            assert "list --include-expired" in reports[0]
        assert {path: path.read_bytes() for path in expected} == expected
        assert not _completion(queue_dir, job).exists()
        assert healthy_log.exists() == (timestamp == END_NS)
        assert (queue_dir / "done" / healthy).exists() == (timestamp == END_NS)
    listing = _queue(
        queue_dir,
        lock_file,
        "list",
        "--include-expired",
        cwd=tmp_path,
        now_ns=END_NS + 30 * DAY_NS,
    )
    detail = next(
        line
        for line in listing.stdout.splitlines()
        if line.startswith(job) and "history=inconsistent" in line
    )
    assert "directories=cancelled" in detail
    assert "state failed disagrees with history cancelled" in detail


@pytest.mark.parametrize(
    "damage",
    [
        "state-conflict",
        "completion-conflict",
        "corrupt-completion",
        "json-completion",
        "unicode-completion",
        "unhashable-completion",
        "state-syntax",
        "state-empty",
        "state-value",
        "state-pid",
        "state-resource",
        "state-unicode",
        "owner-syntax",
        "owner-identity",
        "owner-pid-zero",
        "owner-pid-huge",
        "owner-pid-unicode",
        "missing-terminal",
        "duplicate-terminal",
    ],
)
def test_inconsistent_jobs_are_reported_and_retained_without_blocking_collection(
    tmp_path: Path, damage: str
) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, "done")
    state = queue_dir / "state" / f"{job[:-4]}.state"
    owner = queue_dir / "state" / f"{job[:-4]}.owner"
    terminal = queue_dir / "done" / job
    record = json.loads(_completion(queue_dir, job).read_text())
    healthy = f"{END_NS}_1_healthy.job"
    (queue_dir / "done" / healthy).write_text("true\n")
    healthy_log = queue_dir / "logs" / f"{healthy[:-4]}.log"
    healthy_log.write_text("healthy")
    _completion(queue_dir, healthy).write_text(json.dumps({**record, "job": healthy}))
    if damage == "corrupt-completion":
        _completion(queue_dir, job).write_text('{"version": 1}')
    elif damage == "json-completion":
        _completion(queue_dir, job).write_text("{")
    elif damage == "unicode-completion":
        _completion(queue_dir, job).write_bytes(b"\xff")
    elif damage == "unhashable-completion":
        _completion(queue_dir, job).write_text(
            json.dumps({**record, "time_source": []})
        )
    elif damage == "completion-conflict":
        _completion(queue_dir, job).write_text(
            json.dumps({**record, "state": "failed"})
        )
    elif damage == "state-conflict":
        state.write_text("state=failed\n")
    elif damage == "state-syntax":
        state.write_text("state=done\nstate=failed\n")
    elif damage == "state-empty":
        state.write_text("")
    elif damage == "state-value":
        state.write_text("state=nonsense\n")
    elif damage == "state-pid":
        state.write_text("state=done\npid=nonsense\n")
    elif damage == "state-resource":
        state.write_text("state=done\nresource=nonsense\n")
    elif damage == "state-unicode":
        state.write_bytes(b"\xff")
    elif damage == "owner-syntax":
        owner.write_text("broken\n")
    elif damage in {"owner-identity", "owner-pid-zero"}:
        owner.write_text(
            f"pid={0 if damage == 'owner-pid-zero' else 999999999}\nstart_time=1\njob={job if damage == 'owner-pid-zero' else 'other.job'}\n"
        )
    elif damage in {"owner-pid-huge", "owner-pid-unicode"}:
        pid = "9" * 5000 if damage == "owner-pid-huge" else "\u0661"
        owner.write_text(f"pid={pid}\nstart_time=1\njob={job}\n")
    elif damage == "missing-terminal":
        terminal.unlink()
    elif damage == "duplicate-terminal":
        (queue_dir / "cancelled" / job).write_bytes(terminal.read_bytes())
    files = [path for path in queue_dir.rglob(f"{job[:-4]}.*") if path.is_file()]
    expected = {path: path.read_bytes() for path in files}
    for timestamp in (END_NS + DAY_NS, END_NS + 8 * DAY_NS, END_NS + 30 * DAY_NS):
        result = _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=timestamp)
        assert "retaining 1 inconsistent jobs" in result.stderr
        assert {path: path.read_bytes() for path in expected} == expected
        assert terminal.exists() == (damage != "missing-terminal")
        assert healthy_log.exists() == (timestamp == END_NS + DAY_NS)
    listing = _queue(
        queue_dir,
        lock_file,
        "list",
        "--include-expired",
        cwd=tmp_path,
        now_ns=END_NS + 30 * DAY_NS,
    )
    assert any(
        line.startswith(job) and "history=inconsistent" in line and "reason=" in line
        for line in listing.stdout.splitlines()
    )


def test_interrupted_expiry_resumes_from_tombstone(tmp_path: Path) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, "done")
    path = _completion(queue_dir, job)
    record = json.loads(path.read_text())
    record["expired_at_ns"] = END_NS + 8 * DAY_NS
    path.write_text(json.dumps(record))
    (queue_dir / "done" / job).unlink()
    log = queue_dir / "logs" / f"{job[:-4]}.log"
    assert log.is_file()
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS + 8 * DAY_NS)
    assert not log.exists()
    assert not (queue_dir / "state" / f"{job[:-4]}.state").exists()
    assert json.loads(path.read_text()) == record


def test_collected_tombstones_do_not_add_path_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    # Load the real maintenance entry point so filesystem calls can be counted
    # without timing thresholds or tracing unrelated shell subprocesses.
    monkeypatch.setattr(sys, "path", sys.path.copy())
    maintenance = runpy.run_path(str(QUEUE_SCRIPT.with_name("queue_history.py")))
    monkeypatch.setenv("TRAINING_QUEUE_TEST_NOW_NS", str(END_NS + 8 * DAY_NS))
    measurements = []
    for first, stop in ((0, 1), (1, 1024)):
        for index in range(first, stop):
            job = f"{END_NS}_{index}_collected.job"
            _completion(queue_dir, job).write_text(
                json.dumps(
                    dict(
                        version=1,
                        job=job,
                        state="done",
                        finished_at_ns=END_NS,
                        retention_started_at_ns=END_NS,
                        time_source="recorded",
                        expired_at_ns=END_NS + 8 * DAY_NS,
                    )
                )
            )
        with (
            patch.object(os, "stat", wraps=os.stat) as stats,
            patch.object(os, "lstat", wraps=os.lstat) as lstats,
            patch.object(os, "scandir", wraps=os.scandir) as scans,
            patch.object(Path, "unlink", autospec=True) as unlinks,
        ):
            maintenance["maintain"](queue_dir)
        scan_counts = Counter(Path(call.args[0]) for call in scans.call_args_list)
        assert scan_counts and max(scan_counts.values()) == 1
        assert unlinks.call_count == 0
        measurements.append((stats.call_count + lstats.call_count, scan_counts))
    assert measurements[0] == measurements[1]


@pytest.mark.parametrize(
    ("directory", "suffix", "retained"),
    [
        ("logs", ".log", False),
        ("done", ".job", False),
        ("state", ".state", False),
        ("state", ".owner", True),
        ("cancel-requests", ".cancel", False),
        ("control/external-acks", ".ack", False),
        ("jobs", ".job", True),
        ("running", ".job", True),
        ("failed", ".job", True),
        ("control/completions", ".json", True),
    ],
)
def test_collected_history_rechecks_reappearing_files(
    tmp_path: Path, directory: str, suffix: str, retained: bool
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    job = f"{END_NS}_1_collected.job"
    record = dict(
        version=1,
        job=job,
        state="done",
        finished_at_ns=END_NS,
        retention_started_at_ns=END_NS,
        time_source="recorded",
        expired_at_ns=END_NS + 8 * DAY_NS,
    )
    completion = _completion(queue_dir, job)
    completion.write_text(json.dumps(record))
    path = queue_dir / directory / f"{job[:-4]}{suffix}"
    contents = "reappeared\n"
    if suffix == ".state":
        contents = "state=done\n"
    elif suffix == ".owner":
        process = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
        start_time = process[19]
        contents = f"pid={os.getpid()}\nstart_time={start_time}\njob={job}\n"
    elif suffix == ".json":
        contents = "{"
    path.write_text(contents)
    result = _queue(
        queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS + 9 * DAY_NS
    )
    assert path.exists() == retained
    if retained:
        assert path.read_text() == contents
    if directory in {"failed", "control/completions"}:
        assert "retaining 1 inconsistent jobs" in result.stderr
    if suffix != ".json":
        assert json.loads(completion.read_text()) == record


def test_legacy_registration_survives_migration_and_history_expiry(
    tmp_path: Path,
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    arguments = ("add", "true", "--name", "old", "--idempotency-key", "legacy")
    new_job = (
        _queue(queue_dir, lock_file, *arguments, cwd=tmp_path, now_ns=END_NS)
        .stdout.strip()
        .removeprefix("queued: ")
    )
    legacy = f"{END_NS - 20 * DAY_NS}_123_old.job"
    (queue_dir / "jobs" / new_job).rename(queue_dir / "done" / legacy)
    receipt = next((queue_dir / "control/registrations").glob("*.receipt"))
    digest, _job = receipt.read_text().split()
    receipt.write_text(f"{digest} {legacy}\n")
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    result = _queue(
        queue_dir, lock_file, *arguments, cwd=tmp_path, now_ns=END_NS + 8 * DAY_NS
    )
    assert result.stdout == f"queued: {legacy}\n"
    assert not (queue_dir / "done" / legacy).exists()
    assert not list((queue_dir / "jobs").glob("*.job"))


def test_unconfirmed_legacy_terminal_state_stays_protected(tmp_path: Path) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS)
    job = f"{END_NS - 50 * DAY_NS}_123_unconfirmed.job"
    terminal = queue_dir / "done" / job
    terminal.write_text("#!/usr/bin/env bash\ntrue\n")
    (queue_dir / "state" / f"{job[:-4]}.state").write_text("state=terminating\n")
    log = queue_dir / "logs" / f"{job[:-4]}.log"
    log.write_text("keep")
    for timestamp in (END_NS, END_NS + 30 * DAY_NS):
        _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=timestamp)
        assert terminal.is_file() and log.read_text() == "keep"
        assert not _completion(queue_dir, job).exists()


def test_expired_benchmark_snapshot_reports_expiry_before_collection(
    tmp_path: Path,
) -> None:
    queue_dir, lock_file, job = _finished_job(tmp_path, "failed")
    _queue(queue_dir, lock_file, "status", cwd=tmp_path, now_ns=END_NS + 8 * DAY_NS)
    output = tmp_path / "collection"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/benchmarks/pipeline_stop_collection.py"),
            "--queue",
            str(queue_dir),
            "--report",
            str(tmp_path / "missing-report"),
            "--job",
            job[:-4],
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        env={**_queue_env(queue_dir, lock_file), "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Queue history expired" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("marker", [None, "wrong-pid reverse-name-v1", "legacy-v0"])
def test_new_names_cannot_be_added_to_legacy_or_unknown_workers(
    tmp_path: Path, marker: str | None
) -> None:
    queue_dir, lock_file = tmp_path / "queue", tmp_path / "gpu.lock"
    _queue(queue_dir, lock_file, "status", cwd=tmp_path)
    # A live PID models the old worker's liveness protocol without starting one.
    (queue_dir / "worker.pid").write_text(str(os.getpid()))
    if marker is not None:
        (queue_dir / "control/worker.protocol").write_text(marker)
    rejected = _queue(queue_dir, lock_file, "add", "true", cwd=tmp_path, check=False)
    assert rejected.returncode == 1
    assert "incompatible legacy queue worker" in rejected.stderr
    assert not list((queue_dir / "jobs").glob("*.job"))
    (queue_dir / "worker.pid").unlink()
    _queue(queue_dir, lock_file, "add", "true", cwd=tmp_path)
    worker = _serve(queue_dir, lock_file, cwd=tmp_path)
    stdout, stderr = worker.communicate(timeout=10)
    assert worker.returncode == 0, (stdout, stderr)
    assert len(list((queue_dir / "done").glob("*.job"))) == 1
    assert not (queue_dir / "control/worker.protocol").exists()
