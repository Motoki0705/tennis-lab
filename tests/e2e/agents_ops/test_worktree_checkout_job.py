"""Worktree deletion and main updates share the public checkout coordination."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_git_sync import (
    ROOT,
    SUPERVISOR,
    SyncEnv,
    git,
    install_git_hook,
    wait_file,
    wait_for_flock,
)
from tests.e2e.agents_ops.test_git_sync import control as control
from tests.e2e.agents_ops.test_git_sync import control_factory as control_factory
from tests.e2e.agents_ops.test_git_sync import sync as sync
from tests.e2e.agents_ops.test_worktree_cleanup import FAKE_RUNNER


@dataclass
class ExpiryWorld:
    sync: SyncEnv
    worktree: Path
    state: Path
    registry: Path
    imported: Path
    ready: Path
    hold: Path
    runtime: Path

    def command(self, *args: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "spin",
            "worktrees",
            *args,
            "--state-dir",
            str(self.state),
        ]

    def parent(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.registry.read_text())["parents"][
            "o/r#1"
        ]
        return value

    def deleted(self, process: subprocess.Popen[str]) -> None:
        stdout, stderr = process.communicate("\n", timeout=10)
        assert process.returncode == 0, stdout + stderr
        result = json.loads(stdout[stdout.index("{") :])
        assert result["results"][0]["deleted"] is True
        assert not self.worktree.exists()
        assert str(self.worktree) not in git(self.sync.repo, "worktree", "list")
        row = self.parent()
        assert row["status"] == "released"
        assert row["tasks"]["2"]["stopped"] is True
        assert row["tasks"]["2"]["worker_ended"]["delegation_id"] == "a" * 32
        assert row["worktree_expiry"]["result"] == "stopped"
        terminal = json.loads((self.runtime / "result.json").read_text())
        assert terminal["cleanup_complete"] is True
        assert terminal["turn_end_reason"] == "stopped"

    @contextmanager
    def cleanup(self) -> Iterator[subprocess.Popen[str]]:
        with subprocess.Popen(
            self.command("cleanup"),
            cwd=self.sync.repo,
            env=self.sync.control.environment,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        ) as process:
            try:
                yield process
            finally:
                if process.poll() is None:
                    try:
                        process.communicate("\n", timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.communicate(timeout=5)
                        raise


@pytest.fixture
def expiry(sync: SyncEnv, monkeypatch: pytest.MonkeyPatch) -> ExpiryWorld:
    ops = sync.repo / ".agents/ops"
    # Keep the existing fake dispatcher status CLI, while running the real
    # public spin -> checkout_job -> worktree module in this disposable main.
    fake_dispatcher = (ops / "dispatcher/__main__.py").read_bytes()
    shutil.copytree(
        ROOT / ".agents/ops",
        ops,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (ops / "dispatcher/__main__.py").write_bytes(fake_dispatcher)
    assert not git(sync.repo, "diff", "--name-only", "HEAD")
    work = sync.repo / ".claude/worktrees/expired"
    git(sync.repo, "worktree", "add", "-qb", "expired", str(work))
    directory = sync.directory.parent
    world = ExpiryWorld(
        sync,
        work,
        directory / "expiry-test",
        directory / "parent-sessions/registry.json",
        sync.repo.parent / "cleanup-imported",
        sync.repo.parent / "cleanup-ready",
        sync.repo.parent / "hold-cleanup",
        directory / "logs/expiry/runtime",
    )
    monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
    monkeypatch.setenv("XDG_STATE_HOME", sync.control.environment["XDG_STATE_HOME"])
    cleanup = importlib.import_module("worktree_cleanup.cleanup")
    assert cleanup.Cleaner(sync.repo, directory=world.state).run()[0] == 0
    # Inject an expired persisted clock without changing the actual admin
    # identity. Subsequent public CLI invocations use that same record.
    records_path = world.state / "worktrees.json"
    records = json.loads(records_path.read_text())
    assert len(records["records"]) == 1
    row = next(iter(records["records"].values()))
    row.update(created_at_ns=0, created_at="1970-01-01T00:00:00.000000000Z")
    records_path.write_text(json.dumps(records))
    world.runtime.mkdir(parents=True)
    (world.runtime / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fake",
                "result": "running",
                "contract": {"cwd": str(work), "task_id": "a" * 32},
                "cleanup_complete": False,
            }
        )
    )
    runner = ops / "lib/agent_exec.sh"
    runner.write_text(f"#!{sys.executable}\n" + FAKE_RUNNER)
    runner.chmod(0o700)
    sync.control.environment["EXPIRY_RUN_CALLS"] = str(directory / "runner-calls")
    world.registry.parent.mkdir(parents=True, exist_ok=True)
    world.registry.write_text(
        json.dumps(
            {
                "version": 1,
                "parents": {
                    "o/r#1": {
                        "session": "codex:expiry-test",
                        "generation": 1,
                        "status": "active",
                        "tasks": {
                            "2": {
                                "delegated": True,
                                "delegation_id": "a" * 32,
                                "worktree": str(work),
                                "worker_runs": [
                                    {
                                        "delegation_id": "a" * 32,
                                        "run_dir": str(world.runtime),
                                        "kind": "initial",
                                        "recorded_at": "2026-09-21T00:00:00Z",
                                    }
                                ],
                            }
                        },
                        "reviews": [],
                        "owner_runtime": {
                            "session": "codex:expiry-test",
                            "generation": 1,
                            "cwd": str(work),
                            "process": None,
                        },
                    }
                },
            }
        )
    )
    # This coordination fixture has a fake runner and no native target process.
    # Isolate host/other-worker process churn, as the dedicated expiry tests do.
    # Native stop/ancestry contracts live in test_worktree_cleanup and
    # test_worktree_generation_races; checkout coordination/deletion stay real.
    source = ops / "worktree_cleanup/cleanup.py"
    text = source.read_text().replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n"
        f"from pathlib import Path as _TestPath\n_TestPath({str(world.imported)!r}).touch()\n",
        1,
    )
    source.write_text(
        text
        + f"""
_test_remove = Cleaner.remove
from worktree_cleanup import processes as _test_processes
_test_processes.snapshot = lambda: {{}}
def _test_blocked_remove(self, *args, **kwargs):
    if _TestPath({str(world.hold)!r}).exists():
        import sys
        _TestPath({str(world.ready)!r}).touch()
        sys.stdin.readline()
    return _test_remove(self, *args, **kwargs)
Cleaner.remove = _test_blocked_remove
"""
    )
    return world


@pytest.mark.parametrize("manual_off", [False, True])
def test_deletion_defers_update_and_preserves_resume_and_operator_off(
    expiry: ExpiryWorld, manual_off: bool
) -> None:
    sync = expiry.sync
    target = sync.publish({"app.py": "updated\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="inactive", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    sync.control.write(state)
    sync.directory.mkdir(parents=True, exist_ok=True)
    ticket = sync.directory / "resume.json"
    ticket.write_text(
        json.dumps(
            [
                {
                    "unit": SUPERVISOR,
                    "restart": True,
                    "phase": "stopped",
                    "main_pid": int(state["supervisor_pids"][0]),
                }
            ]
        )
    )
    saved_ticket = ticket.read_bytes()
    expiry.hold.touch()
    with expiry.cleanup() as job:
        wait_file(expiry.ready)
        assert expiry.worktree.is_dir()
        assert expiry.parent()["status"] == "active"
        assert sync.run("checkout_jobs_running", 75)["resume_pending"] is True
        assert ticket.read_bytes() == saved_ticket
        assert git(sync.repo, "rev-parse", "HEAD") != target
        assert ["systemctl", "--user", "start", SUPERVISOR] not in sync.control.calls()
        if manual_off:
            result = sync.control.run("off", "dispatcher")
            assert result.returncode == 0, result.stdout + result.stderr
        unit_file_state = sync.control.read()["units"][SUPERVISOR]["UnitFileState"]
        expiry.deleted(job)
    assert ticket.read_bytes() == saved_ticket
    sync.run("fast_forwarded")  # Also validates both finite-unit launchers.
    assert not ticket.exists()
    state = sync.control.read()
    assert state["dispatcher"]["enabled"] is not manual_off
    assert state["units"][SUPERVISOR]["ActiveState"] == (
        "inactive" if manual_off else "active"
    )
    assert state["units"][SUPERVISOR]["UnitFileState"] == unit_file_state
    assert expiry.parent()["status"] == "released"


def test_update_holds_cleanup_before_import_stop_and_deletion(
    expiry: ExpiryWorld,
) -> None:
    sync = expiry.sync
    sync.publish({"app.py": "updated\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    sync.control.write(state)
    ready, release = [
        sync.repo.parent / name for name in ("merge-ready", "merge-release")
    ]
    install_git_hook(
        sync,
        f"""
if 'merge' in args:
    Path({str(ready)!r}).touch()
    with open({str(release)!r}) as stream: stream.read()
sys.exit(subprocess.call([real, *args]))
""",
    )
    os.mkfifo(release)
    with sync.start() as updater:
        try:
            wait_file(ready)
            ticket = sync.directory / "resume.json"
            saved_ticket = ticket.read_bytes()
            with expiry.cleanup() as job:
                try:
                    wait_for_flock(job)
                    assert not expiry.imported.exists()
                    assert expiry.worktree.is_dir()
                    assert expiry.parent()["status"] == "active"
                    assert ticket.read_bytes() == saved_ticket
                finally:
                    # FIFO EOF releases the updater without a polling sleep.
                    with release.open("w"):
                        pass
                stdout, stderr = updater.communicate(timeout=15)
                assert updater.returncode == 0, stdout + stderr
                expiry.deleted(job)
            assert expiry.imported.exists()
            assert not ticket.exists()
            state = sync.control.read()
            assert state["units"][SUPERVISOR]["ActiveState"] == "active"
            assert state["units"][SUPERVISOR]["UnitFileState"] == "enabled"
            assert state["dispatcher"]["enabled"] is True
        finally:
            if updater.poll() is None:
                updater.terminate()
                updater.communicate(timeout=5)
