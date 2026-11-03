"""Join actual parent CLI run provenance to the collector; no real agent calls.

Contract: producer-created reservations/runs must progress through not_started,
running and terminal states, and a redelegation with retained history is unstarted.
Use the existing disposable fake-gh/delegate seed and a fake common runner with
socket acknowledgements. Never construct worker_runs in this integration test.
Two terminal outcomes, no sleep; measure setup/call/teardown with --durations=0.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_human_agreement import delegation_args
from tests.e2e.agents_ops.test_manual_worker_runs import (
    ENTRY,
    ended,
    latest,
    state_dir,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


@pytest.mark.parametrize(
    ("exit_code", "expected"), [(0, "turn_completed"), (7, "failed")]
)
def test_manual_cli_records_are_observed_across_launch_registration_and_redelegation(
    reserved: ParentEnv, exit_code: int, expected: str
) -> None:
    p = reserved
    collect = importlib.import_module("worker_status").collect_status

    def child() -> dict[str, Any]:
        value: dict[str, Any] = collect(state_dir=state_dir(p), parent=1)
        return dict(value["parents"][0]["children"][0])

    def register(directory: Path) -> None:
        p.cli("record-worker-run", "--child", "10", "--run-dir", str(directory))

    assert task(p)["worker_runs"] == []
    assert child()["status"] == "not_started"
    original_delegation = task(p)["delegation_id"]
    runner = p.root / "fake-status-runner"
    runner.write_text(
        f"#!{sys.executable}\nimport sys\nsys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from fake_worker_status_runner import main\nraise SystemExit(main())\n"
    )
    runner.chmod(0o700)
    with tempfile.TemporaryDirectory(prefix="worker-status-") as short:
        address = str(Path(short) / "runner.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(address)
            server.listen(1)
            server.settimeout(30)
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    ENTRY,
                    str(runner),
                    "--repo",
                    "o/r",
                    "--parent",
                    "1",
                    "run-worker",
                    "--child",
                    "10",
                    "--prompt-file",
                    p.note(),
                ],
                cwd=p.repo,
                env={
                    **p.env,
                    "MANUAL_STATE_DIR": str(state_dir(p)),
                    "MANUAL_RUNNER_CALLS": str(p.root / "runner-calls.jsonl"),
                    "MANUAL_RUNNER_EXIT": str(exit_code),
                    "WORKER_STATUS_TEST_SOCKET": address,
                },
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(30)
                    with connection.makefile("rb") as ready:
                        assert ready.readline(7) == b"ready\n"
                    current_run = latest(p)
                    assert child()["status"] == "running"
                    assert child()["run"]["identity_verified"] is True
                    register(current_run)
                    assert len(task(p)["worker_runs"]) == 1
                    assert child()["status"] == "running"
                    connection.sendall(b"F")
                stdout, stderr = proc.communicate(timeout=30)
                assert proc.returncode == exit_code, stderr
                assert json.loads(stdout)["runner_summary"]["result"] == (
                    "failed" if exit_code else "completed"
                )
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                proc.communicate(timeout=10)
    observed = child()
    assert observed["status"] == expected
    assert observed["run"]["turn_end_reason"] == (
        "cli_error" if exit_code else "completed"
    )
    # Also exercise creation of a registered entry, rather than only the
    # idempotent registration of run-worker's initial entry.
    external = p.root / "external-run"
    shutil.copytree(current_run, external)
    result = json.loads((external / "result.json").read_text())
    result.update(run_dir=str(external), session_root=str(external))
    (external / "result.json").write_text(json.dumps(result))
    session = json.loads((external / "session.json").read_text())
    session["latest_run"] = str(external)
    (external / "session.json").write_text(json.dumps(session))
    register(external)
    assert task(p)["worker_runs"][-1]["kind"] == "registered"
    assert child()["status"] == expected
    assert child()["run"]["kind"] == "registered"
    ended(p)
    p.cli(
        *delegation_args(Path(task(p)["worktree"])),
        "--restart-evidence-file",
        p.note("前のrunnerの終了を確認した。"),
    )
    current = task(p)
    assert current["delegation_id"] != original_delegation
    assert len(current["worker_runs"]) == 2
    assert all(
        run["delegation_id"] == original_delegation for run in current["worker_runs"]
    )
    assert child()["status"] == "not_started"
    serialized = json.dumps(child())
    assert original_delegation not in serialized and str(external) not in serialized
