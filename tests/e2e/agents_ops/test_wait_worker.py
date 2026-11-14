"""Manual launch -> read-only wait -> durable termination over public CLI.

Reuse the manual reservation seed and runner's admission assertions. The fake
launch uses real process identities and real agent_exec.sh status; socket gates
replace timing guesses. Three representative turns, plus lost-runner, missing
run and timeout cases. No real agents, GitHub writes, GPU, or fixed test sleeps.
Measure cases/setup with --durations=0; keep each case under 5s where possible.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import (
    ENTRY,
    latest,
    manual,
    run,
    state_dir,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

ROOT = Path(__file__).resolve().parents[3]
# The waiting command itself contains the exact text that used to match pgrep.
WAIT_ENTRY = ENTRY + "\n# run-worker --child 10 --resume\n"


def command(runner: Path, *args: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        WAIT_ENTRY,
        str(runner),
        "--repo",
        "o/r",
        "--parent",
        "1",
        *args,
    ]


@contextmanager
def listening(address: Path) -> Iterator[socket.socket]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(address))
        server.listen(4)
        server.settimeout(15)
        yield server


@contextmanager
def connected(server: socket.socket) -> Iterator[socket.socket]:
    connection, _ = server.accept()
    with connection:
        connection.settimeout(15)
        yield connection


def message(connection: socket.socket) -> bytes:
    with connection.makefile("rb") as reader:
        return reader.readline()


@dataclass
class Running:
    p: ParentEnv
    process: subprocess.Popen[str]
    connection: socket.socket
    runner: Path
    env: dict[str, str]

    def wait(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command(self.runner, "wait-worker", "--child", "10", *args),
            cwd=self.p.repo,
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )


@contextmanager
def running(
    p: ParentEnv, mode: str = "initial", code: int = 0, reason: str = "completed"
) -> Iterator[Running]:
    if mode == "resume":
        run(p)
    runner = p.root / "fake-wait-runner"
    runner.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from fake_wait_worker_runner import main\nraise SystemExit(main())\n"
    )
    runner.chmod(0o700)
    with tempfile.TemporaryDirectory(prefix="wait-worker-") as short:
        env = {
            **p.env,
            "MANUAL_STATE_DIR": str(state_dir(p)),
            "MANUAL_RUNNER_CALLS": str(p.root / "runner-calls.jsonl"),
            "MANUAL_RUNNER_EXIT": str(code),
            "WAIT_TERMINAL_REASON": reason,
            "WAIT_REAL_RUNNER": str(ROOT / ".agents/ops/lib/agent_exec.sh"),
            "WAIT_RUNNER_SOCKET": str(Path(short) / "runner.sock"),
        }
        with listening(Path(env["WAIT_RUNNER_SOCKET"])) as server:
            process = subprocess.Popen(
                command(
                    runner,
                    "run-worker",
                    "--child",
                    "10",
                    "--prompt-file",
                    p.note(),
                    *(["--resume"] if mode == "resume" else []),
                    *(["--planning"] if mode == "planning" else []),
                ),
                cwd=p.repo,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                with connected(server) as connection:
                    assert message(connection) == b"ready\n"
                    yield Running(p, process, connection, runner, env)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=10)


@pytest.mark.parametrize(
    ("mode", "code", "reason"),
    [
        ("initial", 0, "completed"),
        ("resume", 7, "runtime_error"),
        ("planning", 143, "stopped"),
    ],
)
def test_wait_uses_latest_run_and_releases_locks_even_with_matching_command_line(
    reserved: ParentEnv, mode: str, code: int, reason: str
) -> None:
    p = reserved
    with running(p, mode, code, reason) as worker:
        current = latest(p)
        with tempfile.TemporaryDirectory(prefix="wait-status-") as short:
            address = Path(short) / "status.sock"
            with listening(address) as server:
                waiter = subprocess.Popen(
                    command(
                        worker.runner, "wait-worker", "--child", "10", "--timeout", "10"
                    ),
                    cwd=p.repo,
                    env={**worker.env, "WAIT_STATUS_SOCKET": str(address)},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    with connected(server) as connection:
                        observed = json.loads(message(connection))
                        assert observed["result"] == "running"
                        assert observed["runner_alive"] is True
                        assert waiter.poll() is None
                        assert (
                            b"run-worker --child 10 --resume"
                            in Path(f"/proc/{waiter.pid}/cmdline").read_bytes()
                        )
                        # A real mutation needs both the parent operation and
                        # registry locks, and must finish while wait is paused.
                        if mode == "initial":
                            note = subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "parent",
                                    "--repo",
                                    "o/r",
                                    "--parent",
                                    "1",
                                    "note",
                                    "--body-file",
                                    p.note("待機中も親の独立作業を記録する。"),
                                ],
                                cwd=p.repo,
                                env=p.env,
                                text=True,
                                capture_output=True,
                                timeout=8,
                                check=False,
                            )
                            assert note.returncode == 0, note.stderr
                        assert "worker_ended" not in task(p)
                        worker.connection.sendall(b"F")
                        launch_out, launch_err = worker.process.communicate(timeout=10)
                        assert worker.process.returncode == code, launch_err
                        connection.sendall(b"C")
                    server.close()  # Subsequent status calls pass the one-shot gate.
                    stdout, stderr = waiter.communicate(timeout=10)
                    assert waiter.returncode == 0, stderr
                finally:
                    if waiter.poll() is None:
                        waiter.kill()
                    waiter.communicate(timeout=10)
        value = json.loads(stdout)
        record = json.loads((current / "result.json").read_text())
        assert value["wait_state"] == "ended"
        assert value["run_id"] == record["run_id"]
        assert value["runner_summary"]["result"] == ("failed" if code else "completed")
        assert value["runner_summary"]["turn_end_reason"] == reason
        assert value["runner_summary"]["exit_code"] == code
        assert all(
            value[k] is True
            for k in (
                "termination_recorded",
                "cleanup_complete",
                "worker_ended_recorded",
            )
        )
        assert (
            value["final_file"]
            == json.loads(launch_out)["final_file"]
            == str(current / "final.md")
        )
        assert value["delegation_id"] == task(p)["delegation_id"]
        assert len(task(p)["worker_runs"]) == (2 if mode == "resume" else 1)


def test_already_ended_returns_without_wait_or_state_writes(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p)
    registry = state_dir(p) / "registry.json"
    before = registry.read_bytes()
    gh_calls = p.calls()
    result = manual(
        p,
        "wait-worker",
        "--child",
        "10",
        "--timeout",
        "2",
        CLAUDE_CODE_SESSION_ID="",
        CODEX_THREAD_ID="",
        CODEX_SESSION_ID="",
        PARENT_FAIL_GET_PATH="user",
    )
    assert json.loads(result.stdout)["wait_state"] == "ended"
    assert registry.read_bytes() == before
    assert p.calls() == gh_calls


def test_no_current_run_fails_explicitly(reserved: ParentEnv) -> None:
    p = reserved
    result = p.cli("wait-worker", "--child", "10", ok=False)
    assert result.returncode == 2
    assert "no recorded run" in result.stderr


@pytest.mark.parametrize("when", ["before_wait", "during_wait"])
def test_killed_runner_is_liveness_unknown_without_rewriting_evidence(
    reserved: ParentEnv, when: str
) -> None:
    p = reserved
    with running(p) as worker:
        current = latest(p)
        record = json.loads((current / "result.json").read_text())
        if when == "before_wait":
            os.kill(record["runner"]["pid"], signal.SIGKILL)
            worker.process.communicate(timeout=10)
            result = worker.wait("--timeout", "5")
        else:
            with tempfile.TemporaryDirectory(prefix="wait-lost-") as short:
                address = Path(short) / "status.sock"
                with listening(address) as server:
                    waiter = subprocess.Popen(
                        command(
                            worker.runner,
                            "wait-worker",
                            "--child",
                            "10",
                            "--timeout",
                            "5",
                        ),
                        cwd=p.repo,
                        env={**worker.env, "WAIT_STATUS_SOCKET": str(address)},
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    try:
                        with connected(server) as connection:
                            assert (
                                json.loads(message(connection))["runner_alive"] is True
                            )
                            os.kill(record["runner"]["pid"], signal.SIGKILL)
                            worker.process.communicate(timeout=10)
                            connection.sendall(b"C")
                        server.close()
                        stdout, stderr = waiter.communicate(timeout=10)
                        result = subprocess.CompletedProcess(
                            waiter.args, waiter.returncode, stdout, stderr
                        )
                    finally:
                        if waiter.poll() is None:
                            waiter.kill()
                        waiter.communicate(timeout=10)
        assert result.returncode == 2, result.stderr
        value = json.loads(result.stdout)
        assert value["wait_state"] == "liveness_unknown"
        assert value["runner_summary"]["result"] == "running"
        assert value["termination_recorded"] is False
        assert value["worker_ended_recorded"] is False
        assert value["final_file"] is None
        assert value["recovery"].startswith(".agents/ops/lib/README.md#")
        assert json.loads((current / "result.json").read_text()) == record


def test_timeout_does_not_stop_worker_or_release_reservation(
    reserved: ParentEnv,
) -> None:
    p = reserved
    with running(p) as worker:
        current = latest(p)
        before = (current / "result.json").read_bytes()
        with tempfile.TemporaryDirectory(prefix="wait-timeout-") as short:
            address = Path(short) / "status.sock"
            with listening(address) as server:
                waiter = subprocess.Popen(
                    command(
                        worker.runner, "wait-worker", "--child", "10", "--timeout", "2"
                    ),
                    cwd=p.repo,
                    env={**worker.env, "WAIT_STATUS_SOCKET": str(address)},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                try:
                    with connected(server) as connection:
                        assert json.loads(message(connection))["runner_alive"] is True
                        # Do not acknowledge status: the wait deadline must
                        # also bound a status subprocess that never returns.
                        stdout, stderr = waiter.communicate(timeout=6)
                        assert waiter.returncode == 124, stderr
                        assert connection.recv(1) == b""
                finally:
                    if waiter.poll() is None:
                        waiter.kill()
                    waiter.communicate(timeout=10)
        value = json.loads(stdout)
        assert value["wait_state"] == "timeout" and value["error"] == "wait_timeout"
        assert (current / "result.json").read_bytes() == before
        assert worker.process.poll() is None and "worker_ended" not in task(p)
        worker.connection.sendall(b"F")
        worker.process.communicate(timeout=10)
        assert worker.wait("--timeout", "2").returncode == 0
