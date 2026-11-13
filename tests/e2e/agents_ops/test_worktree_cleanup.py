"""Expiry contracts using real throwaway Git worktrees and injected time/stops."""

from __future__ import annotations

import copy
import fcntl
import importlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
BORN = 1_790_000_000_123_456_789
LIFETIME = 336 * 3600 * 10**9


class Lab:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.repo = tmp / "repo"
        self.repo.mkdir()
        self.state_root = tmp / "xdg/tennis-lab-agents"
        self.state = tmp / "expiry"
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp / "xdg"))
        monkeypatch.delenv("TRAINING_QUEUE_DIR", raising=False)
        monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
        monkeypatch.setenv("GIT_AUTHOR_NAME", "expiry-test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "expiry@example.com")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "expiry-test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "expiry@example.com")
        monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
        self.module = importlib.import_module("worktree_cleanup.cleanup")
        self.inventory = importlib.import_module("worktree_cleanup.inventory")
        self.lifecycle = importlib.import_module("worktree_cleanup.lifecycle")
        self.storage = importlib.import_module("worktree_cleanup.storage")
        self.now = BORN
        monkeypatch.setattr(self.inventory, "birth_time", lambda fd: BORN)
        self.git("init", "-q", "-b", "main")
        (self.repo / ".gitignore").write_text(".claude/worktrees/\n.venv\nignored/\n")
        (self.repo / "tracked").write_text("original\n")
        self.git("add", ".")
        self.git("commit", "-qm", "initial")

    def git(self, *args: str, cwd: Path | None = None) -> str:
        proc = subprocess.run(
            ["git", "-C", str(cwd or self.repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    def add(self, name: str = "task", *, path: Path | None = None) -> Path:
        path = path or self.repo / ".claude/worktrees" / name
        self.git("worktree", "add", "-qb", name, str(path))
        return path

    def cleaner(self) -> Any:
        return self.module.Cleaner(
            self.repo, directory=self.state, clock=lambda: self.now
        )

    def run(self, *, cleanup: bool = False, dry_run: bool = False) -> dict[str, Any]:
        code, report = self.cleaner().run(cleanup=cleanup, dry_run=dry_run)
        assert code == 0, report
        return dict(report)

    def records(self) -> dict[str, Any]:
        return dict(json.loads((self.state / "worktrees.json").read_text())["records"])


@pytest.fixture
def lab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Lab:
    return Lab(tmp_path, monkeypatch)


@pytest.mark.parametrize("offset,removed", [(-1, False), (0, True), (1, True)])
def test_exact_336_hour_boundary_for_dirty_locked_unmerged_worktree(
    lab: Lab, offset: int, removed: bool
) -> None:
    path = lab.add()
    (path / "local-commit").write_text("unmerged and unpushed")
    lab.git("add", "local-commit", cwd=path)
    lab.git("commit", "-qm", "local work", cwd=path)
    head = lab.git("rev-parse", "HEAD", cwd=path)
    (path / "tracked").write_text("dirty")
    (path / "untracked").write_text("discard me")
    (path / "ignored").mkdir()
    (path / "ignored/checkpoint").write_text("discard me too")
    admin = Path(lab.git("rev-parse", "--absolute-git-dir", cwd=path))
    (admin / "index.lock").write_text("Git operation in progress")
    (admin / "MERGE_HEAD").write_text(head)
    lab.git("worktree", "lock", str(path))
    lab.now = BORN + LIFETIME + offset
    report = lab.run(cleanup=True)
    assert report["entries"][0]["selected"] is removed
    assert report["entries"][0]["discarded_changes"] is True
    assert path.exists() is not removed
    assert (str(path) in lab.git("worktree", "list", "--porcelain")) is not removed
    assert lab.git("rev-parse", "task") == head  # branch recovery remains possible
    if removed:
        assert report["results"] == [
            {
                "path": str(path),
                "head": head,
                "decided_at": lab.storage.utc(lab.now),
                "deleted": True,
            }
        ]
        log = [
            json.loads(line)
            for line in (lab.state / "results.jsonl").read_text().splitlines()
        ]
        assert [r["phase"] for r in log] == ["intent", "done"]
        assert log[-1]["deleted"] is True


def test_migrates_existing_birth_and_never_uses_mtime_or_commits(lab: Lab) -> None:
    path = lab.add()
    lab.now = BORN + LIFETIME - 1
    for candidate in (path, lab.repo / ".git/worktrees/task"):
        os.utime(candidate, (1, 1))
    first = lab.run()["entries"][0]
    assert first["created_at"] == "2026-09-21T14:13:20.123456789Z"
    assert first["source"] == "birth_time"
    assert first["expired"] is False
    assert first["deadline"] == "2026-10-05T14:13:20.123456789Z"
    lab.now += 1
    assert lab.run(cleanup=True, dry_run=True)["entries"][0]["selected"] is True
    assert path.is_dir()
    lab.run(cleanup=True)
    assert not path.exists()


def test_first_observed_persists_and_is_not_reset_when_birth_becomes_available(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: None)
    lab.now = BORN + 100 * 10**9
    first = lab.run()["entries"][0]
    assert first["source"] == "first_observed"
    assert first["created_at"] == lab.storage.utc(BORN + 100 * 10**9)
    lab.now += LIFETIME - 1
    monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: BORN)
    assert lab.run(cleanup=True)["entries"][0]["expired"] is False
    assert path.is_dir()
    lab.now += 1
    lab.run(cleanup=True)
    assert not path.exists()
    assert len(lab.records()) == 1


def test_recreation_at_same_path_gets_new_creation_record(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    first = lab.run()["entries"][0]
    lab.git("worktree", "remove", str(path))
    lab.now = BORN + LIFETIME
    monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: lab.now)
    lab.add("replacement", path=path)
    report = lab.run(cleanup=True)
    assert len(report["entries"]) == 1
    assert report["entries"][0]["id"] != first["id"]
    assert report["entries"][0]["source"] == "birth_time"
    assert report["entries"][0]["expired"] is False
    assert path.is_dir()


def test_keeps_main_external_worktrees_and_symlink_targets(
    lab: Lab, tmp_path: Path
) -> None:
    path = lab.add()
    external = lab.add("outside", path=tmp_path / "outside")
    shared = lab.repo / ".venv"
    shared.mkdir()
    (shared / "keep").write_text("shared environment")
    (path / ".venv").symlink_to(shared, target_is_directory=True)
    (path / "outside-link").symlink_to(external, target_is_directory=True)
    (path / "ignored").mkdir()
    (path / "ignored/main-link").symlink_to(lab.repo, target_is_directory=True)
    lab.now += LIFETIME
    report = lab.run(cleanup=True)
    assert not path.exists()
    assert external.is_dir()
    assert (shared / "keep").read_text() == "shared environment"
    assert (lab.repo / "tracked").read_text() == "original\n"
    assert (
        next(e for e in report["entries"] if e["path"] == str(external))["target"]
        is False
    )
    assert lab.git("worktree", "list", "--porcelain").count("worktree ") == 2


@pytest.mark.parametrize("component", ["leaf", "ancestor"])
def test_refuses_worktree_paths_redirected_by_symlink(
    lab: Lab, component: str, tmp_path: Path
) -> None:
    path = lab.add()
    outside = tmp_path / "outside"
    if component == "leaf":
        path.rename(outside)
        path.symlink_to(outside, target_is_directory=True)
        keep = outside / "tracked"
    else:
        root = path.parent
        root.rename(outside)
        root.symlink_to(outside, target_is_directory=True)
        keep = outside / "task/tracked"
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True)
    assert code == 1
    assert "symlink" in report["entries"][0]["scope_error"]
    assert keep.read_text() == "original\n"
    assert str(path) in lab.git("worktree", "list", "--porcelain")


@pytest.mark.parametrize("stage", ["rename", "unregister", "rmtree"])
def test_failure_is_not_success_and_retry_finishes_both_registration_and_files(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    path = lab.add()
    lab.now += LIFETIME
    with monkeypatch.context() as patch:

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise OSError("injected deletion failure")

        if stage == "rename":
            patch.setattr(lab.inventory, "quarantine_tree", fail)
        elif stage == "unregister":
            patch.setattr(lab.inventory, "unregister", fail)
        else:
            patch.setattr(lab.inventory, "remove_tree", fail)
        code, report = lab.cleaner().run(cleanup=True)
        assert code == 1
        assert report["outcome"] == "partial_failure"
        assert report["results"][0]["deleted"] is False
        assert "injected" in report["results"][0]["error"]
    report = lab.run(cleanup=True)
    assert report["results"][0]["deleted"] is True
    assert not path.exists()
    assert str(path) not in lab.git("worktree", "list", "--porcelain")
    assert not list(path.parent.glob(".expired-*"))


def test_crash_after_rename_replays_durable_intent(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    lab.now += LIFETIME

    class Crash(BaseException):
        pass

    real = lab.storage.State.save
    with monkeypatch.context() as patch:

        def interrupt(state: Any) -> None:
            if any(r["phase"] == "quarantined" for r in state.records.values()):
                raise Crash()
            real(state)

        patch.setattr(lab.storage.State, "save", interrupt)
        with pytest.raises(Crash):
            lab.cleaner().run(cleanup=True)
    assert not path.exists()
    assert next(iter(lab.records().values()))["phase"] == "stopping"
    lab.run(cleanup=True)
    assert not list(path.parent.glob(".expired-*"))
    assert str(path) not in lab.git("worktree", "list", "--porcelain")


def test_retry_does_not_delete_replacement_after_unregistration(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    lab.now += LIFETIME
    with monkeypatch.context() as patch:
        patch.setattr(
            lab.inventory,
            "remove_tree",
            lambda *a: (_ for _ in ()).throw(OSError("interrupt")),
        )
        assert lab.cleaner().run(cleanup=True)[0] == 1
    monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: lab.now)
    lab.add("replacement", path=path)
    (path / "new-work").write_text("keep")
    lab.run(cleanup=True)
    assert (path / "new-work").read_text() == "keep"
    assert str(path) in lab.git("worktree", "list", "--porcelain")
    assert not list(path.parent.glob(".expired-*"))


def test_concurrent_scan_and_cleanup_fail_without_mutation(lab: Lab) -> None:
    path = lab.add()
    lab.state.mkdir()
    environment = {**os.environ, "PYTHONPATH": str(ROOT / ".agents/ops")}
    with (lab.state / "cleanup.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for command in ("scan", "cleanup"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "worktree_cleanup",
                    command,
                    "--repo",
                    str(lab.repo),
                    "--state-dir",
                    str(lab.state),
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            assert result.returncode == 3
            assert "holds cleanup.lock" in result.stderr
    assert path.is_dir()
    assert not (lab.state / "worktrees.json").exists()
    lab.run()


def test_corrupt_state_is_not_reinitialized(lab: Lab) -> None:
    lab.add()
    lab.state.mkdir()
    (lab.state / "worktrees.json").write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(lab.storage.CleanupError, match="duplicate"):
        lab.run(cleanup=True)
    assert (lab.repo / ".claude/worktrees/task").is_dir()


def test_stop_failure_preserves_registration_and_returns_failure(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    lab.now += LIFETIME
    monkeypatch.setattr(
        lab.lifecycle.Stopper,
        "stop",
        lambda *a: (_ for _ in ()).throw(
            lab.storage.CleanupError("runner still active")
        ),
    )
    code, report = lab.cleaner().run(cleanup=True)
    assert code == 1
    assert path.is_dir()
    assert report["results"][0]["deleted"] is False
    assert "runner still active" in report["results"][0]["error"]
    assert str(path) in lab.git("worktree", "list", "--porcelain")


def test_public_spin_scan_and_dry_run_do_not_delete(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    lab.add()
    shutil.copytree(
        ROOT / ".agents/ops",
        lab.repo / ".agents/ops",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        ROOT / ".spin", lab.repo / ".spin", ignore=shutil.ignore_patterns("__pycache__")
    )
    (lab.repo / "pyproject.toml").write_text(
        '[tool.spin]\npackage="src"\n[tool.spin.commands]\nEnvironment=[".spin/cmds.py:worktrees"]\n'
    )
    for args in (("scan",), ("cleanup", "--dry-run")):
        proc = subprocess.run(
            [sys.executable, "-m", "spin", "worktrees", *args],
            cwd=lab.repo,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        assert proc.returncode == 0, proc.stderr
        # spin may prefix its package diagnostic; the JSON itself is unchanged.
        report = json.loads(proc.stdout[proc.stdout.index("{") :])
        entry = report["entries"][0]
        assert entry["source"] in {"birth_time", "first_observed"}
        assert {
            "created_at",
            "deadline",
            "selected",
            "discarded_changes",
        } <= entry.keys()
    assert (lab.repo / ".claude/worktrees/task").is_dir()


def test_real_statx_returns_creation_not_directory_mtime(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Use libc's real statx through a freshly imported function, not the fixture.
    module = importlib.reload(lab.inventory)
    path = lab.add()
    admin = Path(lab.git("rev-parse", "--absolute-git-dir", cwd=path))
    os.utime(admin, (1, 1))
    value = module.identity(admin, birth=True)["birth_time_ns"]
    if value is not None:
        seconds = int(
            subprocess.check_output(["stat", "--format=%W", str(admin)], text=True)
        )
        assert value // 10**9 == seconds
        assert seconds > 1


def save_registry(lab: Lab, row: dict[str, Any]) -> Path:
    directory = lab.state_root / "parent-sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "registry.json"
    path.write_text(json.dumps({"version": 1, "parents": {"o/r#1": row}}))
    return path


def parent_row() -> dict[str, Any]:
    return {
        "session": "codex:test-parent",
        "status": "active",
        "generation": 1,
        "tasks": {},
        "reviews": [],
    }


FAKE_RUNNER = """import json,os,pathlib,sys
directory=pathlib.Path(sys.argv[sys.argv.index("--run-dir")+1])
path=directory/"result.json"
record=json.loads(path.read_text())
if sys.argv[1]=="stop":
    record.update(result="failed",turn_end_reason="stopped",cleanup_complete=not bool(os.environ.get("EXPIRY_UNCONFIRMED")))
    path.write_text(json.dumps(record))
    with open(os.environ["EXPIRY_RUN_CALLS"],"a") as stream: stream.write(str(directory)+"\\n")
print(json.dumps({"schema_version":1,"run_id":record["run_id"],"runner_alive":False,"agent_alive":False,"agent_group_alive":False}))
"""


def fake_run(
    lab: Lab, path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    runner = lab.repo.parent / "fake-runner"
    runner.write_text(f"#!{sys.executable}\n" + FAKE_RUNNER)
    runner.chmod(0o700)
    calls = lab.repo.parent / "run-calls"
    monkeypatch.setenv("EXPIRY_RUN_CALLS", str(calls))
    directory = lab.state_root / "logs/example"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "fake",
                "result": "running",
                "contract": {"cwd": str(path), "task_id": "a" * 32},
                "cleanup_complete": False,
            }
        )
    )
    # stop_run's default argument deliberately pins the production runner, so
    # inject only the Runtime construction, keeping real inspect/ended checks.
    runtime_class = lab.lifecycle.Runtime
    monkeypatch.setattr(
        lab.lifecycle, "Runtime", lambda original: runtime_class(runner)
    )
    return directory, calls


@pytest.mark.parametrize("unconfirmed", [False, True])
def test_manual_worker_stop_requires_runner_cleanup_proof(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, unconfirmed: bool
) -> None:
    path = lab.add()
    directory, calls = fake_run(lab, path, monkeypatch)
    row = parent_row()
    row["tasks"]["2"] = {
        "delegated": True,
        "delegation_id": "a" * 32,
        "worktree": str(path),
        "worker_runs": [
            {
                "delegation_id": "a" * 32,
                "run_dir": str(directory),
                "kind": "initial",
                "recorded_at": "2026-09-21T00:00:00Z",
            }
        ],
    }
    registry = save_registry(lab, row)
    if unconfirmed:
        monkeypatch.setenv("EXPIRY_UNCONFIRMED", "1")
    cleaner = lab.cleaner()
    cleaner.stopper.timeout = 0
    lab.now += LIFETIME
    code, report = cleaner.run(cleanup=True)
    task = json.loads(registry.read_text())["parents"]["o/r#1"]["tasks"]["2"]
    assert task["stopped"] is True
    assert calls.exists()
    assert code == int(unconfirmed)
    assert path.exists() is unconfirmed
    assert ("worker_ended" in task) is not unconfirmed
    if unconfirmed:
        assert report["results"][0]["deleted"] is False
        assert "unconfirmed" in report["results"][0]["error"]
    else:
        assert task["worker_ended"]["delegation_id"] == "a" * 32


def test_parent_ownership_is_released_and_native_process_is_stopped(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    row = parent_row()
    ident = {"pid": 123456, "boot_id": "fake", "start_ticks": "1"}
    row["owner_runtime"] = {
        "cwd": str(path),
        "process": ident,
        "session": row["session"],
        "generation": row["generation"],
    }
    registry = save_registry(lab, row)
    stopped = []
    monkeypatch.setattr(lab.lifecycle.processes, "agents", lambda root: [])
    monkeypatch.setattr(
        lab.lifecycle.processes,
        "stop_tree",
        lambda identity, **kwargs: stopped.append(identity),
    )
    lab.now += LIFETIME
    lab.run(cleanup=True)
    current = json.loads(registry.read_text())["parents"]["o/r#1"]
    assert stopped and all(item == ident for item in stopped)
    assert current["status"] == "released"
    assert current["worktree_expiry"]["result"] == "stopped"
    assert not path.exists()


@pytest.mark.parametrize("phase", ["reserved", "starting"])
def test_dispatcher_jobs_finish_and_free_child_reservations(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    path = lab.add()
    row = parent_row()
    directory, _ = fake_run(lab, path, monkeypatch)
    if phase == "reserved":
        runtime_dir = lab.repo.parent / "not-launched"
    else:
        runtime_dir = directory.parent / "dispatch"
        runtime_dir.mkdir()
        shutil.move(str(directory), str(runtime_dir / "runtime"))
    row["tasks"]["2"] = {
        "delegated": True,
        "delegation_id": "a" * 32,
        "worktree": str(path),
    }
    job = {
        "id": "job",
        "phase": phase,
        "role": "implementation",
        "child": 2,
        "delegation_id": "a" * 32,
        "directory": str(runtime_dir),
        "task_id": "a" * 32,
        "cwd": str(path),
        "runner": str(lab.repo.parent / "fake-runner"),
        "worker": None,
        "runner_exit": None,
    }
    row["dispatch"] = {"jobs": {"job": job}, "phase": "waiting", "events": []}
    registry = save_registry(lab, row)
    monkeypatch.setattr(
        lab.lifecycle, "managed", lambda current: current.get("dispatch")
    )
    waited = []
    if phase == "starting":
        attempts = []

        @contextmanager
        def delayed_launcher(lock: Path) -> Any:
            attempts.append(lock)
            if len(attempts) == 1:
                # The launcher records its exit after the runner has stopped,
                # while cleanup is outside the shared registry lock.
                data = json.loads(registry.read_text())
                data["parents"]["o/r#1"]["dispatch"]["jobs"]["job"]["runner_exit"] = 143
                registry.write_text(json.dumps(data))
                yield None
            else:
                yield 1

        monkeypatch.setattr(lab.lifecycle, "gate", delayed_launcher)
        monkeypatch.setattr(
            lab.lifecycle,
            "Event",
            lambda: SimpleNamespace(wait=lambda duration: waited.append(duration)),
        )
    lab.now += LIFETIME
    lab.run(cleanup=True)
    current = json.loads(registry.read_text())["parents"]["o/r#1"]
    assert current["tasks"]["2"]["stopped"] is True
    assert current["tasks"]["2"]["worker_ended"]["delegation_id"] == "a" * 32
    assert current["dispatch"]["jobs"]["job"]["phase"] == "ended"
    if phase == "starting":
        assert waited
        assert current["dispatch"]["jobs"]["job"]["runner_exit"] == 143
    assert not path.exists()


@pytest.mark.parametrize("running", [False, True])
def test_queue_references_are_cancelled_before_deletion(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, running: bool
) -> None:
    path = lab.add("task", path=lab.repo / ".claude/worktrees/task with spaces")
    queue = lab.repo / ".training_queue"
    directory = queue / ("running" if running else "jobs")
    directory.mkdir(parents=True)
    name = "1_task.job"
    (directory / name).write_text(
        "cd " + str(path).replace(" ", "\\ ") + "\npython train.py\n"
    )
    if running:
        fake = lab.repo.parent / "fake-queue.sh"
        fake.write_text("""#!/bin/bash
set -eu
[[ "$1" == cancel ]]
job="$2"
mkdir -p "$TRAINING_QUEUE_DIR/cancelled" "$TRAINING_QUEUE_DIR/control/completions"
mv "$TRAINING_QUEUE_DIR/running/$job" "$TRAINING_QUEUE_DIR/cancelled/$job"
printf '{"version":1,"job":"%s","state":"cancelled","time_source":"recorded","finished_at_ns":123,"retention_started_at_ns":123}' "$job" > "$TRAINING_QUEUE_DIR/control/completions/${job%.job}.json"
""")
        monkeypatch.setattr(lab.lifecycle, "QUEUE", fake)
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert (queue / "cancelled" / name).is_file()
    assert not (queue / "jobs" / name).exists()
    assert not (queue / "running" / name).exists()
    assert not path.exists()


def test_queue_stop_failure_never_deletes_or_claims_success(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    queue = lab.repo / ".training_queue/running"
    queue.mkdir(parents=True)
    (queue / "1_task.job").write_text(f"cd {path}\npython train.py\n")
    fake = lab.repo.parent / "fake-queue.sh"
    fake.write_text("exit 17\n")
    monkeypatch.setattr(lab.lifecycle, "QUEUE", fake)
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True)
    assert code == 1
    assert report["results"][0]["deleted"] is False
    assert "queue cancellation failed" in report["results"][0]["error"]
    assert path.exists()


def test_missing_registered_worktree_is_unregistered_at_expiry(lab: Lab) -> None:
    path = lab.add()
    shutil.rmtree(path)
    lab.now += LIFETIME
    report = lab.run(cleanup=True)
    assert report["entries"][0]["discarded_changes"] is None
    assert report["results"][0]["deleted"] is True
    assert str(path) not in lab.git("worktree", "list", "--porcelain")


def test_symlink_inserted_after_stop_does_not_redirect_removal(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = lab.add()
    outside = tmp_path / "shared"
    (outside / "task").mkdir(parents=True)
    (outside / "task/keep").write_text("shared data")

    def redirect(stopper: Any, root: Path, stamp: str) -> None:
        if not path.parent.is_symlink():
            path.parent.rename(tmp_path / "old-worktrees")
            path.parent.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(lab.lifecycle.Stopper, "stop", redirect)
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True)
    assert code == 1
    assert report["results"][0]["deleted"] is False
    assert (outside / "task/keep").read_text() == "shared data"
    assert (tmp_path / "old-worktrees/task/tracked").read_text() == "original\n"


class NativeForest:
    def __init__(
        self, lab: Lab, monkeypatch: pytest.MonkeyPatch, *, newborn: str | None = None
    ) -> None:
        self.proc = lab.lifecycle.processes
        self.nodes: dict[int, Any] = {}
        self.signals: list[tuple[int, int]] = []
        self.frozen: set[int] = set()
        self.now = 0.0
        self.calls = 0
        self.newborn = newborn
        for pid, parent in ((1, 0), (101, 1), (102, 101), (900, 1)):
            self.add(pid, parent)
        monkeypatch.setattr(self.proc, "snapshot", self.snapshot)
        monkeypatch.setattr(
            self.proc,
            "process_identity",
            lambda pid: self.nodes.get(pid, {}).get("identity"),
        )
        monkeypatch.setattr(
            self.proc, "threads_stopped", lambda item: item["pid"] in self.frozen
        )
        monkeypatch.setattr(self.proc, "open_pidfd", lambda pid: pid + 1000)
        monkeypatch.setattr(self.proc, "signal_pidfd", self.signal)
        real_close = os.close
        monkeypatch.setattr(
            os, "close", lambda fd: None if fd in {1101, 1102, 1103} else real_close(fd)
        )
        monkeypatch.setattr(self.proc.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(self.proc, "Event", lambda: SimpleNamespace(wait=self.wait))

    def add(self, pid: int, parent: int) -> None:
        self.nodes[pid] = {
            "identity": {"pid": pid, "boot_id": "boot", "start_ticks": str(pid)},
            "parent": parent,
        }

    def snapshot(self) -> dict[int, Any]:
        result = copy.deepcopy(self.nodes)
        self.calls += 1
        if self.calls == 1 and self.newborn:
            self.add(
                103,
                102
                if self.newborn == "child"
                else 900
                if self.newborn == "other"
                else 1,
            )
        return result

    def wait(self, duration: float) -> None:
        self.now += 1

    def signal(self, fd: int, sig: int) -> None:
        pid = fd - 1000
        self.signals.append((pid, sig))
        if sig == self.proc.signal.SIGSTOP:
            self.frozen.add(pid)
        else:
            if sig == self.proc.signal.SIGKILL:
                owned = {101, 102, 103} if self.newborn == "child" else {101, 102}
                assert all(p in self.frozen for p in self.nodes if p in owned)
            self.nodes.pop(pid, None)
            for child in self.nodes.values():
                if child["parent"] == pid:
                    child["parent"] = 1


def test_native_stop_pins_children_and_parent_without_pid_reuse(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    forest = NativeForest(lab, monkeypatch)
    parent = dict(forest.nodes[101]["identity"])
    forest.proc.stop_tree(parent)
    assert set(forest.nodes) == {1, 900}
    signals = list(forest.signals)
    forest.add(101, 1)
    forest.nodes[101]["identity"]["start_ticks"] = "reused"
    forest.proc.stop_tree(parent)
    assert forest.signals == signals


def test_native_stop_finds_child_born_after_snapshot_and_reparented_during_kill(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    forest = NativeForest(lab, monkeypatch, newborn="child")
    forest.proc.stop_tree(forest.nodes[101]["identity"])
    assert set(forest.nodes) == {1, 900}
    assert (103, forest.proc.signal.SIGKILL) in forest.signals
    assert all(sig != forest.proc.signal.SIGTERM for _, sig in forest.signals)


def test_native_stop_preserves_new_child_of_an_existing_other_process(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    forest = NativeForest(lab, monkeypatch, newborn="other")
    forest.proc.stop_tree(forest.nodes[101]["identity"])
    assert set(forest.nodes) == {1, 900, 103}
    assert not any(pid in {900, 103} for pid, _ in forest.signals)


def test_unattributed_new_orphan_blocks_cleanup_and_replay_until_it_exits(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    forest = NativeForest(lab, monkeypatch, newborn="orphan")
    parent = dict(forest.nodes[101]["identity"])
    monkeypatch.setattr(
        forest.proc,
        "agents",
        lambda root: (
            [{"identity": parent, "sessions": []}] if 101 in forest.nodes else []
        ),
    )
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True)
    assert code == 1
    assert report["results"][0]["deleted"] is False
    assert path.exists()
    assert 103 in forest.nodes
    assert not any(pid == 103 for pid, sig in forest.signals)
    assert lab.cleaner().run(cleanup=True)[0] == 1
    assert path.exists()
    forest.nodes.pop(103)
    lab.run(cleanup=True)
    assert not path.exists()


def test_legacy_parent_session_is_linked_by_existing_herdr_process_stamp(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    path = lab.add()
    row = parent_row()
    registry = save_registry(lab, row)
    directory = lab.state_root / "herdr-labels/panes"
    directory.mkdir(parents=True)
    (directory / "pane.json").write_text(
        json.dumps(
            {
                "version": 1,
                "owner": hashlib.sha256(row["session"].encode()).hexdigest(),
                "processes": [{"pid": 123, "start": "456"}],
                "state": "active",
            }
        )
    )
    remaining = [
        {
            "identity": {"pid": 123, "start_ticks": "456", "boot_id": "fake"},
            "sessions": [],
        }
    ]
    monkeypatch.setattr(
        lab.lifecycle.processes,
        "agents",
        lambda root: list(remaining) if root == path else [],
    )
    monkeypatch.setattr(
        lab.lifecycle.processes,
        "stop_tree",
        lambda identity, **kwargs: remaining.clear(),
    )
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert json.loads(registry.read_text())["parents"]["o/r#1"]["status"] == "released"


def test_split_registry_interruption_is_replayed_before_reading_ownership(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    row = parent_row()
    row["owner_runtime"] = {
        "cwd": str(path),
        "process": {"pid": 123, "boot_id": "fake", "start_ticks": "1"},
        "session": row["session"],
        "generation": row["generation"],
    }
    registry = save_registry(lab, row)
    Store = importlib.import_module("parent.state").Store
    store = Store(registry.parent)
    with store.locked():
        store.migrate()
    monkeypatch.setattr(lab.lifecycle.processes, "agents", lambda root: [])
    monkeypatch.setattr(
        lab.lifecycle.processes, "stop_tree", lambda identity, **kwargs: None
    )
    ExpiryStore = lab.lifecycle.ExpiryStore
    original = ExpiryStore.write_json

    class Crash(BaseException):
        pass

    def interrupt(target: Path, value: Any) -> None:
        if (
            target == registry
            and value.get("parents", {}).get("o/r#1", {}).get("status") == "released"
        ):
            raise Crash()
        original(target, value)

    lab.now += LIFETIME
    with monkeypatch.context() as patch:
        patch.setattr(ExpiryStore, "write_json", staticmethod(interrupt))
        with pytest.raises(Crash):
            lab.cleaner().run(cleanup=True)
    assert json.loads(registry.read_text())["parents"]["o/r#1"]["status"] == "active"
    assert (registry.parent / "worktree-expiry-transaction.json").exists()
    report = lab.run(cleanup=True)
    assert report["results"][0]["deleted"] is True
    assert (
        Store(registry.parent).read_data()["parents"]["o/r#1"]["status"] == "released"
    )
    assert not (registry.parent / "worktree-expiry-transaction.json").exists()


def test_runner_claimed_during_rename_is_stopped_using_original_contract(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    directory, calls = fake_run(lab, path, monkeypatch)
    external = lab.repo.parent / "explicit-run"
    shutil.move(str(directory), external)
    monkeypatch.setattr(
        lab.lifecycle.processes,
        "run_directories",
        lambda physical: {external} if physical.name.startswith(".expired-") else set(),
    )
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert str(external) in calls.read_text()
    assert (
        json.loads((external / "result.json").read_text())["cleanup_complete"] is True
    )
    assert not path.exists()


def test_completed_fallback_record_does_not_expire_recycled_admin_inode(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = lab.inventory.opened_identity
    monkeypatch.setattr(
        lab.inventory,
        "opened_identity",
        lambda path, *, birth=False: (
            {"device": 42, "inode": 123, "birth_time_ns": None} if birth else real(path)
        ),
    )
    path = lab.add()
    first = lab.run()["entries"][0]
    lab.now += LIFETIME
    lab.run(cleanup=True)
    lab.git("worktree", "prune", "--expire=now")
    lab.add("replacement", path=path)
    report = lab.run(cleanup=True)
    current = report["entries"][0]
    assert current["id"] != first["id"]
    assert current["source"] == "first_observed"
    assert current["created_at"] == lab.storage.utc(lab.now)
    assert current["selected"] is False
    assert path.exists()


def test_worktree_move_preserves_original_creation_and_deadline(lab: Lab) -> None:
    path = lab.add()
    before = lab.run()["entries"][0]
    moved = path.with_name("moved")
    lab.git("worktree", "move", str(path), str(moved))
    lab.now += LIFETIME
    report = lab.run(cleanup=True)
    after = report["entries"][0]
    assert after["id"] == before["id"]
    assert after["deadline"] == before["deadline"]
    assert not moved.exists()


def test_explicit_runner_directory_is_discovered_for_deleted_cwd(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    external = lab.repo.parent / "explicit-run"
    real_iter, real_stat, real_bytes = Path.iterdir, Path.stat, Path.read_bytes
    entry = Path("/proc/777")
    monkeypatch.setattr(
        Path,
        "iterdir",
        lambda directory: (
            iter([entry]) if directory == Path("/proc") else real_iter(directory)
        ),
    )
    monkeypatch.setattr(
        Path,
        "stat",
        lambda candidate, *a, **kw: (
            SimpleNamespace(st_uid=os.getuid())
            if candidate == entry
            else real_stat(candidate, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda candidate: (
            os.fsencode(f"AGENT_RUNTIME_RUN_DIR={external}\0")
            if candidate == entry / "environ"
            else real_bytes(candidate)
        ),
    )
    monkeypatch.setattr(os, "readlink", lambda candidate: str(path) + " (deleted)")
    assert lab.lifecycle.processes.run_directories(path) == {external}


@pytest.mark.parametrize("point", ["before_pin", "after_pin"])
def test_unregistration_keeps_replacement_created_in_the_last_gap(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    path = lab.add()
    original_admin = Path(lab.git("rev-parse", "--absolute-git-dir", cwd=path))
    fired: list[bool] = []

    def replace() -> None:
        if fired:
            return
        fired.append(True)
        lab.git("worktree", "add", "--force", "-qb", "replacement", str(path))
        (path / "new-work").write_text("keep this replacement")
        monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: lab.now)

    if point == "before_pin":
        real_save = lab.storage.State.save
        saves = []

        def save(state: Any) -> None:
            real_save(state)
            if any(r["phase"] == "quarantined" for r in state.records.values()):
                saves.append(True)
                if len(saves) == 2:
                    replace()

        monkeypatch.setattr(lab.storage.State, "save", save)
    else:
        old = original_admin.stat()
        real_scandir = os.scandir

        def scandir(fd: Any) -> Any:
            if isinstance(fd, int):
                st = os.fstat(fd)
                if (st.st_dev, st.st_ino) == (old.st_dev, old.st_ino):
                    replace()
            return real_scandir(fd)

        monkeypatch.setattr(os, "scandir", scandir)
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert fired
    assert path.is_dir(), "replacement worktree was removed"
    assert (path / "new-work").read_text() == "keep this replacement"
    assert lab.git("branch", "--show-current", cwd=path) == "replacement"
    assert str(path) in lab.git("worktree", "list", "--porcelain")


def test_external_removal_observation_retires_fallback_identity(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = lab.inventory.opened_identity
    monkeypatch.setattr(
        lab.inventory,
        "opened_identity",
        lambda path, *, birth=False: (
            {"device": 42, "inode": 123, "birth_time_ns": None} if birth else real(path)
        ),
    )
    path = lab.add()
    first = lab.run()["entries"][0]
    lab.git("worktree", "remove", str(path))
    assert lab.run()["entries"] == []
    retired_phase = lab.records()[first["id"]]["phase"]
    lab.now += LIFETIME
    lab.add("replacement", path=path)
    report = lab.run(cleanup=True)
    current = report["entries"][0]
    assert path.is_dir(), "reused administration inode inherited the old deadline"
    assert retired_phase == "absent"
    assert current["id"] != first["id"]
    assert current["created_at"] == lab.storage.utc(lab.now)
    assert current["source"] == "first_observed"
    assert current["selected"] is False
    assert path.exists()


@pytest.mark.parametrize(
    "binding", ["old_session", "old_generation", "unbound", "old_dispatch"]
)
def test_stale_parent_runtime_does_not_release_new_owner_or_children(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    old = lab.add()
    row = parent_row()
    row["generation"] = 2
    saved = {
        "cwd": str(old),
        "process": {"pid": 123, "boot_id": "fake", "start_ticks": "1"},
        "session": row["session"],
        "generation": row["generation"],
    }
    if binding == "old_session":
        saved["session"] = "codex:old-owner"
    elif binding == "old_generation":
        saved["generation"] = 1
    elif binding == "unbound":
        saved.pop("session")
        saved.pop("generation")
    else:
        saved["cwd"] = str(lab.repo / ".claude/worktrees/new-owner")
        row["dispatch"] = {
            "session": row["session"],
            "jobs": {
                "old": {
                    "role": "parent",
                    "cwd": str(old),
                    "authorization": {"generation": 1},
                    "phase": "ended",
                }
            },
        }
        monkeypatch.setattr(
            lab.lifecycle, "managed", lambda current: current.get("dispatch")
        )
    row["owner_runtime"] = saved
    row["tasks"]["2"] = {
        "worktree": str(lab.repo / ".claude/worktrees/other"),
        "stopped": False,
    }
    registry = save_registry(lab, row)
    monkeypatch.setattr(lab.lifecycle.processes, "agents", lambda root: [])
    stopped = []
    monkeypatch.setattr(
        lab.lifecycle.processes,
        "stop_tree",
        lambda identity, **kwargs: stopped.append(identity),
    )
    lab.now += LIFETIME
    lab.run(cleanup=True)
    current = json.loads(registry.read_text())["parents"]["o/r#1"]
    assert current["status"] == "active"
    assert current["tasks"]["2"]["stopped"] is False
    assert not stopped


def test_observed_registration_loss_does_not_rebind_pending_deletion_to_reused_inode(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = lab.inventory.opened_identity
    monkeypatch.setattr(
        lab.inventory,
        "opened_identity",
        lambda fd, *, birth=False: (
            {"device": 42, "inode": 123, "birth_time_ns": None} if birth else real(fd)
        ),
    )
    path = lab.add()
    lab.run()
    lab.now += LIFETIME
    with monkeypatch.context() as patch:
        patch.setattr(
            lab.inventory,
            "remove_tree",
            lambda *a: (_ for _ in ()).throw(OSError("interrupt")),
        )
        assert lab.cleaner().run(cleanup=True)[0] == 1
    lab.run()
    lab.git("worktree", "prune", "--expire=now")
    lab.add("replacement", path=path)
    (path / "new-work").write_text("keep")
    lab.run(cleanup=True)
    assert (path / "new-work").read_text() == "keep"
    assert lab.git("branch", "--show-current", cwd=path) == "replacement"
