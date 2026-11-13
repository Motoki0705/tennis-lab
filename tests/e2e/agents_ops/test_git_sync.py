"""Pinned updates and operational handoffs in disposable Git/systemd worlds."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_ops_control import (
    ROOT,
    SUPERVISOR,
    ControlEnv,
)
from tests.e2e.agents_ops.test_ops_control import (
    control as control,
)
from tests.e2e.agents_ops.test_ops_control import (
    control_factory as control_factory,
)

SYNC = ROOT / ".agents/ops/git_sync/sync.py"
LAUNCHER = ROOT / ".agents/ops/checkout_job.py"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def wait_file(path: Path) -> None:
    deadline = time.monotonic() + 8
    while not path.exists():
        assert time.monotonic() < deadline, f"missing process handshake: {path}"
        time.sleep(0.02)


def wait_for_flock(process: subprocess.Popen[str] | subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 8
    while True:
        assert process.poll() is None, (
            "process ran before the checkout lock was released"
        )
        pids = [process.pid]
        for pid in pids:
            try:
                if "lock" in Path(f"/proc/{pid}/wchan").read_text():
                    return
                pids.extend(
                    int(child)
                    for child in Path(f"/proc/{pid}/task/{pid}/children")
                    .read_text()
                    .split()
                )
            except ProcessLookupError:
                continue
            except FileNotFoundError:
                continue
        assert time.monotonic() < deadline, "process did not reach the lock"
        time.sleep(0.02)


@dataclass
class SyncEnv:
    control: ControlEnv
    origin: Path
    producer: Path

    @property
    def repo(self) -> Path:
        return Path(self.control.repo)

    @property
    def directory(self) -> Path:
        return (
            Path(self.control.environment["XDG_STATE_HOME"])
            / "tennis-lab-agents/git-sync"
        )

    def publish(self, changes: dict[str, str]) -> str:
        for name, content in changes.items():
            path = self.producer / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        git(self.producer, "add", "-A")
        if changes:
            git(self.producer, "add", "-f", "--", *changes)
        git(self.producer, "commit", "-qm", "upstream update")
        git(self.producer, "push", "-q", "origin", "main")
        return git(self.producer, "rev-parse", "HEAD")

    def start(self) -> subprocess.Popen[str]:
        return subprocess.Popen(
            [sys.executable, str(SYNC), "--repo", str(self.repo)],
            env=self.control.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run(self, reason: str, code: int = 0) -> dict[str, Any]:
        with self.start() as process:
            stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == code, stdout + stderr
        result: dict[str, Any] = json.loads(stdout)
        assert result["reason"] == reason, result
        assert result == json.loads((self.directory / "status.json").read_text())
        log = [
            json.loads(row)
            for row in (self.directory / "runs.jsonl").read_text().splitlines()
        ]
        assert log[-1]["status"] == result
        assert reason in stderr
        return result


@pytest.fixture
def sync(control: ControlEnv) -> Iterator[SyncEnv]:
    repo = control.repo
    git(repo, "checkout", "-qb", "main")
    for root in (repo,):
        git(root, "config", "user.name", "Git Sync Test")
        git(root, "config", "user.email", "sync@example.invalid")
    (repo / "app.py").write_text("old\n")
    (repo / ".gitignore").write_text("data/\n.venv\n.claude/worktrees/\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "baseline")
    origin, producer = repo.parent / "origin.git", repo.parent / "producer"
    subprocess.run(["git", "clone", "--bare", "-q", str(repo), str(origin)], check=True)
    git(repo, "remote", "add", "origin", str(origin))
    subprocess.run(["git", "clone", "-q", str(origin), str(producer)], check=True)
    git(producer, "config", "user.name", "Git Sync Test")
    git(producer, "config", "user.email", "sync@example.invalid")
    # A real harmless process supplies /proc signal-handler evidence to the
    # fake user manager. It never reads or writes the actual checkout.
    with ExitStack() as stack:
        processes = [
            stack.enter_context(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import signal,sys; signal.signal(signal.SIGHUP, lambda *args: None); print('ready', flush=True); sys.stdin.read()",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
            )
            for _ in range(3)
        ]
        try:
            for process in processes:
                assert process.stdout is not None
                assert process.stdout.readline().strip() == "ready"
            state = control.read()
            state["supervisor_pids"] = [str(process.pid) for process in processes]
            state["units"][SUPERVISOR]["MainPID"] = state["supervisor_pids"][0]
            control.write(state)
            yield SyncEnv(control, origin, producer)
        finally:
            for process in processes:
                process.communicate(timeout=5)


def test_fast_forward_keeps_local_data_venv_and_worktree_and_reports_dependencies(
    sync: SyncEnv,
) -> None:
    shared = sync.repo.parent / "shared-venv"
    shared.mkdir()
    (shared / "sentinel").write_text("shared")
    (sync.repo / ".venv").symlink_to(shared, target_is_directory=True)
    (sync.repo / "data").mkdir()
    (sync.repo / "data/keep").write_text("local data")
    work = sync.repo / ".claude/worktrees/feature"
    git(sync.repo, "worktree", "add", "-qb", "feature", str(work))
    old = git(work, "rev-parse", "HEAD")
    (work / "app.py").write_text("worker edits\n")
    target = sync.publish({"app.py": "new\n", "uv.lock": "new lock\n"})
    result = sync.run("fast_forwarded")
    assert result["outcome"] == "updated"
    assert result["fetched_commit"] == target == git(sync.repo, "rev-parse", "HEAD")
    assert result["dependency_changes"] == ["uv.lock"]
    assert result["submodule_changes"] == []
    assert git(work, "rev-parse", "HEAD") == old
    assert (work / "app.py").read_text() == "worker edits\n"
    assert (shared / "sentinel").read_text() == "shared"
    assert (sync.repo / ".venv").is_symlink()
    assert (sync.repo / "data/keep").read_text() == "local data"
    status = json.loads(sync.control.run("status", "git-sync").stdout)
    assert status["targets"][0]["last_run"] == result
    sync.run("already_current")
    assert not git(sync.repo, "for-each-ref", "refs/tennis-lab/git-sync")


@pytest.mark.parametrize("staged", [False, True])
def test_tracked_changes_are_never_stashed_or_overwritten(
    sync: SyncEnv, staged: bool
) -> None:
    head = git(sync.repo, "rev-parse", "HEAD")
    sync.publish({"app.py": "upstream\n"})
    (sync.repo / "app.py").write_text("my work\n")
    if staged:
        git(sync.repo, "add", "app.py")
    sync.run("tracked_changes", 75)
    assert git(sync.repo, "rev-parse", "HEAD") == head
    assert (sync.repo / "app.py").read_text() == "my work\n"
    assert git(sync.repo, "stash", "list") == ""


@pytest.mark.parametrize(
    "marker",
    [
        "MERGE_HEAD",
        "CHERRY_PICK_HEAD",
        "REVERT_HEAD",
        "rebase-merge",
        "rebase-apply",
        "sequencer",
        "BISECT_LOG",
        "index.lock",
    ],
)
def test_in_progress_git_operations_are_distinguished(
    sync: SyncEnv, marker: str
) -> None:
    sync.publish({"app.py": "new\n"})
    head = git(sync.repo, "rev-parse", "HEAD")
    path = sync.repo / ".git" / marker
    if marker in {"rebase-merge", "rebase-apply", "sequencer"}:
        path.mkdir()
    else:
        path.write_text(head + "\n")
    result = sync.run("git_operation_in_progress", 75)
    assert marker in result["operations"]
    assert git(sync.repo, "rev-parse", "HEAD") == head
    assert path.exists()


@pytest.mark.parametrize("branch", ["feature", "detached"])
def test_only_main_in_primary_checkout_is_updated(sync: SyncEnv, branch: str) -> None:
    sync.publish({"app.py": "new\n"})
    git(
        sync.repo,
        "checkout",
        "--detach" if branch == "detached" else "-b",
        *([] if branch == "detached" else [branch]),
    )
    result = sync.run("non_main_branch", 75)
    assert result["branch"] == ("HEAD" if branch == "detached" else branch)
    assert result["fetched_commit"] is None


@pytest.mark.parametrize("diverged", [False, True])
def test_local_commits_are_not_rebased_or_reset(sync: SyncEnv, diverged: bool) -> None:
    if diverged:
        sync.publish({"upstream": "remote"})
    (sync.repo / "local").write_text("unpublished")
    git(sync.repo, "add", "local")
    git(sync.repo, "commit", "-qm", "local commit")
    before = git(sync.repo, "rev-parse", "HEAD")
    sync.run("diverged" if diverged else "local_ahead", 75)
    assert git(sync.repo, "rev-parse", "HEAD") == before


@pytest.mark.parametrize(
    "kind", ["untracked", "ignored", "symlink_parent", "directory"]
)
def test_untracked_and_ignored_collisions_preserve_local_contents(
    sync: SyncEnv, kind: str
) -> None:
    if kind == "ignored":
        path = sync.repo / "data/file"
        path.parent.mkdir()
        changes = {"data/file": "remote"}
    elif kind == "symlink_parent":
        outside = sync.repo.parent / "outside"
        outside.mkdir()
        (outside / "sentinel").write_text("external")
        path = sync.repo / "new-dir"
        path.symlink_to(outside, target_is_directory=True)
        changes = {"new-dir/file": "remote"}
    elif kind == "directory":
        path = sync.repo / "new-file"
        path.mkdir()
        (path / "inside").write_text("local")
        changes = {"new-file": "remote"}
    else:
        path = sync.repo / "new-file"
        changes = {"new-file": "remote"}
    if kind not in {"symlink_parent", "directory"}:
        path.write_text("local")
    target = sync.publish(changes)
    result = sync.run("untracked_collision", 75)
    assert result["fetched_commit"] == target
    assert path.exists()
    if kind == "symlink_parent":
        assert path.is_symlink()
        assert (path / "sentinel").read_text() == "external"
        assert not (path / "file").exists()
    elif kind == "directory":
        assert (path / "inside").read_text() == "local"
    else:
        assert path.read_text() == "local"


def test_new_tracked_data_can_coexist_with_unrelated_ignored_data(
    sync: SyncEnv,
) -> None:
    (sync.repo / "data").mkdir()
    (sync.repo / "data/local").write_text("local")
    # The upstream intentionally tracks this ignored directory's other file.
    (sync.producer / "data").mkdir()
    (sync.producer / "data/remote").write_text("remote")
    git(sync.producer, "add", "-f", "data/remote")
    sync.publish({})
    sync.run("fast_forwarded")
    assert (sync.repo / "data/local").read_text() == "local"
    assert (sync.repo / "data/remote").read_text() == "remote"


def test_submodule_pointer_changes_leave_its_checkout_untouched(sync: SyncEnv) -> None:
    source = sync.repo.parent / "submodule-source"
    source.mkdir()
    git(source, "init", "-q", "-b", "main")
    git(source, "config", "user.name", "Submodule Test")
    git(source, "config", "user.email", "submodule@example.invalid")
    (source / "model").write_text("v1")
    git(source, "add", "model")
    git(source, "commit", "-qm", "model v1")
    first = git(source, "rev-parse", "HEAD")
    git(
        sync.repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(source),
        "third_party/model",
    )
    git(sync.repo, "commit", "-qm", "pin model")
    git(sync.repo, "push", "-q", "origin", "main")
    git(sync.producer, "pull", "-q", "--ff-only")
    (source / "model").write_text("v2")
    git(source, "commit", "-qam", "model v2")
    second = git(source, "rev-parse", "HEAD")
    git(
        sync.producer,
        "update-index",
        "--cacheinfo",
        f"160000,{second},third_party/model",
    )
    git(sync.producer, "commit", "-qm", "advance model pin")
    git(sync.producer, "push", "-q", "origin", "main")
    module = sync.repo / "third_party/model"
    (module / "local-note").write_text("keep")
    # Even a configured recursive merge must not update the module.
    git(sync.repo, "config", "submodule.recurse", "true")
    result = sync.run("fast_forwarded")
    assert result["submodule_changes"] == [
        {"path": "third_party/model", "before": first, "after": second}
    ]
    assert git(module, "rev-parse", "HEAD") == first
    assert (module / "model").read_text() == "v1"
    assert (module / "local-note").read_text() == "keep"
    # An intentionally lagging module does not block the next main-only update.
    sync.publish({"app.py": "next\n"})
    sync.run("fast_forwarded")
    assert git(module, "rev-parse", "HEAD") == first


def test_linked_checkout_is_rejected_before_fetch(sync: SyncEnv) -> None:
    work = sync.repo.parent / "worker"
    git(sync.repo, "worktree", "add", "-qb", "worker", str(work))
    old = git(sync.repo, "rev-parse", "HEAD")
    sync.publish({"app.py": "new\n"})
    process = subprocess.run(
        [sys.executable, str(SYNC), "--repo", str(work)],
        env=sync.control.environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert process.returncode == 75
    assert json.loads(process.stdout)["reason"] == "not_main_checkout"
    assert git(sync.repo, "rev-parse", "HEAD") == old == git(work, "rev-parse", "HEAD")


def test_shared_venv_path_change_is_refused(sync: SyncEnv) -> None:
    target = sync.publish({".venv/version": "unsafe"})
    result = sync.run("protected_path_change", 75)
    assert result["paths"] == [".venv/version"]
    assert git(sync.repo, "rev-parse", "HEAD") != target


def test_bus_and_managed_evidence_errors_prevent_update(sync: SyncEnv) -> None:
    target = sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["manager"] = "bus_unreachable"
    sync.control.write(state)
    sync.run("operation_failed", 2)
    assert git(sync.repo, "rev-parse", "HEAD") != target
    state.pop("manager")
    state["dispatcher"]["parents"] = [{"jobs": []}]
    sync.control.write(state)
    result = sync.run("operation_failed", 2)
    assert "incomplete managed-job evidence" in result["error"]
    assert git(sync.repo, "rev-parse", "HEAD") != target


def install_git_hook(sync: SyncEnv, source: str) -> None:
    path = Path(sync.control.environment["PATH"].split(os.pathsep)[0]) / "git"
    real = shutil.which("git")
    path.write_text(
        f"#!{sys.executable}\nimport os, subprocess, sys, time\nfrom pathlib import Path\nreal = {real!r}\nargs = sys.argv[1:]\n"
        + source
    )
    path.chmod(0o700)


def test_local_git_hooks_cannot_run_dependency_or_submodule_updates(
    sync: SyncEnv,
) -> None:
    sync.publish({"app.py": "new\n"})
    marker = sync.repo.parent / "hook-ran"
    for name in ("post-merge", "reference-transaction"):
        hook = sync.repo / ".git/hooks" / name
        hook.write_text(
            f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n"
        )
        hook.chmod(0o700)
    sync.run("fast_forwarded")
    assert not marker.exists()


def test_fetched_sha_survives_fetch_head_and_tracking_ref_replacement(
    sync: SyncEnv,
) -> None:
    old = git(sync.repo, "rev-parse", "HEAD")
    target = sync.publish({"app.py": "pinned\n"})
    install_git_hook(
        sync,
        f"""
code = subprocess.call([real, *args])
if 'fetch' in args and code == 0:
    subprocess.check_call([real, 'update-ref', 'refs/remotes/origin/main', {old!r}])
    Path('.git/FETCH_HEAD').write_text({old!r} + '\\n')
sys.exit(code)
""",
    )
    result = sync.run("fast_forwarded")
    assert result["fetched_commit"] == target == git(sync.repo, "rev-parse", "HEAD")
    assert git(sync.repo, "rev-parse", "origin/main") == old


@pytest.mark.parametrize(
    "unit",
    [
        "tennis-lab-cleanup.service",
        "tennis-lab-ops-report.service",
        "tennis-lab-ops-triage.service",
        "tennis-lab-dispatch-judgement.service",
    ],
)
def test_legacy_running_finite_units_defer_without_stopping_them(
    sync: SyncEnv, unit: str
) -> None:
    target = sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][unit]["ActiveState"] = "active"
    sync.control.write(state)
    result = sync.run("finite_jobs_running", 75)
    assert result["units"] == [unit]
    assert git(sync.repo, "rev-parse", "HEAD") != target
    assert not any("kill" in call or "stop" in call for call in sync.control.calls())


@pytest.mark.parametrize("kind", ["judgement", "child", "reserved", "slot"])
def test_managed_children_reservations_and_judgement_defer(
    sync: SyncEnv, kind: str
) -> None:
    sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    if kind == "judgement":
        state["dispatcher"]["active_judgement"] = "active"
    else:
        state["dispatcher"]["parents"] = [
            {
                "children_pending": [12] if kind == "child" else [],
                "occupies_slot": kind == "slot",
                "jobs": [{"phase": "reserved" if kind == "reserved" else "ended"}],
            }
        ]
    sync.control.write(state)
    sync.run("managed_jobs_running", 75)
    assert not any("kill" in call or "stop" in call for call in sync.control.calls())


@pytest.mark.parametrize(
    "active,enabled,internal,restarts",
    [
        (True, True, True, True),
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
    ],
)
def test_supervisor_resume_preserves_each_off_state(
    sync: SyncEnv, active: bool, enabled: bool, internal: bool, restarts: bool
) -> None:
    sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(
        ActiveState="active" if active else "inactive",
        UnitFileState="enabled" if enabled else "disabled",
    )
    state["dispatcher"]["enabled"] = internal
    sync.control.write(state)
    sync.run("fast_forwarded")
    calls = sync.control.calls()
    assert (["systemctl", "--user", "start", SUPERVISOR] in calls) == restarts
    assert any("kill" in call for call in calls) == active
    assert sync.control.read()["dispatcher"]["enabled"] == internal
    assert not any("enable" in call or "shutdown" in call for call in calls)


@pytest.mark.parametrize("interrupts", [[1], [1, 2]])
def test_interrupted_delayed_hup_waits_for_old_pid_before_resuming(
    sync: SyncEnv, interrupts: list[int]
) -> None:
    target = sync.publish({"app.py": "new\n"})
    before = git(sync.repo, "rev-parse", "HEAD")
    state = sync.control.read()
    old_pid = state["units"][SUPERVISOR]["MainPID"]
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    state.update(hup_exit_after_polls=8, interrupt_hup_calls=interrupts)
    sync.control.write(state)
    result = sync.run("operation_failed", 2)
    assert git(sync.repo, "rev-parse", "HEAD") == before
    ticket = sync.directory / "resume.json"
    if len(interrupts) == 2:
        assert result["resume_pending"] is True
        saved = json.loads(ticket.read_text())
        assert saved[0]["phase"] in {"stop_requested", "stopping"}
        assert saved[0]["main_pid"] == int(old_pid)
        sync.run("fast_forwarded")
        assert git(sync.repo, "rev-parse", "HEAD") == target
    else:
        assert result["resume_pending"] is False
    assert not ticket.exists()
    current = sync.control.read()["units"][SUPERVISOR]
    assert current["ActiveState"] == "active"
    assert current["MainPID"] != old_pid
    calls = sync.control.calls()
    exited = next(
        i for i, call in enumerate(calls) if call[:2] == ["resident_exited", SUPERVISOR]
    )
    started = calls.index(["systemctl", "--user", "start", SUPERVISOR])
    assert exited < started
    assert sync.control.read()["dispatcher"]["enabled"] is True


def test_off_during_pause_is_rechecked_and_not_undone(sync: SyncEnv) -> None:
    sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    state["disable_after_pause"] = True
    sync.control.write(state)
    sync.run("fast_forwarded")
    assert ["systemctl", "--user", "start", SUPERVISOR] not in sync.control.calls()
    assert not sync.control.read()["dispatcher"]["enabled"]


def test_starting_or_legacy_supervisor_is_not_sent_an_unsupported_pause_signal(
    sync: SyncEnv,
) -> None:
    target = sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(
        ActiveState="active", UnitFileState="enabled", MainPID="0"
    )
    state["dispatcher"]["enabled"] = True
    sync.control.write(state)
    sync.run("supervisor_not_maintenance_ready", 75)
    assert git(sync.repo, "rev-parse", "HEAD") != target
    assert not any("kill" in call for call in sync.control.calls())


def test_restart_failure_is_visible_and_next_invocation_recovers_its_ticket(
    sync: SyncEnv,
) -> None:
    target = sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    state["fail_commands"] = [f"start:{SUPERVISOR}"]
    sync.control.write(state)
    sync.run("operation_failed", 2)
    assert git(sync.repo, "rev-parse", "HEAD") == target
    assert (sync.directory / "resume.json").exists()
    state = sync.control.read()
    state.pop("fail_commands")
    sync.control.write(state)
    sync.run("already_current")
    assert not (sync.directory / "resume.json").exists()
    assert sync.control.read()["units"][SUPERVISOR]["ActiveState"] == "active"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_index_flags_cannot_hide_tracked_changes(sync: SyncEnv, flag: str) -> None:
    sync.publish({"app.py": "remote\n"})
    git(sync.repo, "update-index", flag, "app.py")
    (sync.repo / "app.py").write_text("hidden local edit\n")
    sync.run("index_flags", 75)
    assert (sync.repo / "app.py").read_text() == "hidden local edit\n"


@pytest.mark.parametrize(
    "unit", ["tennis-lab-ops-report.service", "tennis-lab-board-web.service"]
)
def test_cached_unguarded_units_block_update_even_when_their_source_was_replaced(
    sync: SyncEnv, unit: str
) -> None:
    target = sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][unit]["ExecStart"] = "/usr/bin/python old-unprotected-job.py"
    sync.control.write(state)
    result = sync.run("unguarded_job_units", 75)
    assert result["units"] == [unit]
    assert git(sync.repo, "rev-parse", "HEAD") != target


def test_board_web_is_quiesced_and_restored_with_its_enablement_intact(
    sync: SyncEnv,
) -> None:
    sync.publish({"app.py": "remote\n"})
    unit = "tennis-lab-board-web.service"
    state = sync.control.read()
    state["units"][unit].update(ActiveState="active", UnitFileState="enabled")
    sync.control.write(state)
    sync.run("fast_forwarded")
    calls = sync.control.calls()
    assert calls.index(["systemctl", "--user", "stop", unit]) < calls.index(
        ["systemctl", "--user", "start", unit]
    )
    assert sync.control.read()["units"][unit]["UnitFileState"] == "enabled"


def test_saved_resume_ticket_does_not_start_supervisor_from_another_branch(
    sync: SyncEnv,
) -> None:
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="inactive", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    sync.control.write(state)
    sync.directory.mkdir(parents=True)
    ticket = sync.directory / "resume.json"
    ticket.write_text(json.dumps([{"unit": SUPERVISOR, "restart": True}]))
    git(sync.repo, "checkout", "-qb", "feature")
    result = sync.run("non_main_branch", 75)
    assert result["resume_pending"] is True
    assert ticket.exists()
    assert ["systemctl", "--user", "start", SUPERVISOR] not in sync.control.calls()


@pytest.mark.parametrize("partial", [False, True])
def test_merge_failure_resumes_only_a_consistent_checkout(
    sync: SyncEnv, partial: bool
) -> None:
    sync.publish({"app.py": "new\n"})
    before = git(sync.repo, "rev-parse", "HEAD")
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    sync.control.write(state)
    install_git_hook(
        sync,
        f"""
if 'merge' in args:
    if {partial!r}: Path('app.py').write_text('partial checkout\\n')
    sys.exit(11)
sys.exit(subprocess.call([real, *args]))
""",
    )
    result = sync.run("operation_failed", 2)
    assert git(sync.repo, "rev-parse", "HEAD") == before
    assert result["resume_pending"] == partial
    assert result["checkout_blocked"] == partial
    assert (sync.directory / "resume.json").exists() == partial
    assert (
        sync.control.read()["units"][SUPERVISOR]["ActiveState"] == "active"
    ) != partial
    if partial:
        assert result["resume_blocked_reason"] == "tracked_changes"
        assert (sync.repo / "app.py").read_text() == "partial checkout\n"
        marker = sync.repo.parent / "unsafe-job-ran"
        blocked = subprocess.run(
            [
                sys.executable,
                str(LAUNCHER),
                "--repo",
                str(sync.repo),
                "--",
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert blocked.returncode == 75
        assert not marker.exists()
        # Simulate a human repair without having the updater reset any files.
        (sync.repo / "app.py").write_text("old\n")
        (Path(sync.control.environment["PATH"].split(os.pathsep)[0]) / "git").unlink()
        sync.run("fast_forwarded")
        assert not (sync.directory / "resume.json").exists()
        assert sync.control.read()["units"][SUPERVISOR]["ActiveState"] == "active"


def test_abandoned_update_blocks_imports_until_clean_main_is_revalidated(
    sync: SyncEnv,
) -> None:
    before = git(sync.repo, "rev-parse", "HEAD")
    target = sync.publish({"app.py": "new\n"})
    barrier = sync.repo / ".git/ops-code.blocked"
    barrier.write_text(
        json.dumps({"schema_version": 1, "before": before, "target": target})
    )
    git(sync.repo, "fetch", "-q", "origin")
    blocked = subprocess.run(
        [sys.executable, str(LAUNCHER), "--repo", str(sync.repo), "--supervisor"],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert blocked.returncode == 75
    assert "checkout recovery pending" in blocked.stderr
    result = sync.run("fast_forwarded")
    assert result["checkout_blocked"] is False
    assert not barrier.exists()


def test_partial_checkout_with_only_added_source_keeps_new_jobs_blocked(
    sync: SyncEnv,
) -> None:
    sync.publish({"new_code.py": "complete module\n"})
    install_git_hook(
        sync,
        """
if 'merge' in args:
    Path('new_code.py').write_text('partial module\\n')
    sys.exit(11)
sys.exit(subprocess.call([real, *args]))
""",
    )
    result = sync.run("operation_failed", 2)
    assert result["checkout_blocked"] is True
    assert result["resume_blocked_reason"] == "untracked_collision"
    assert (sync.repo / "new_code.py").read_text() == "partial module\n"


def test_corrupt_checkout_marker_is_not_silently_cleared(sync: SyncEnv) -> None:
    barrier = sync.repo / ".git/ops-code.blocked"
    barrier.write_text('{"schema_version": 99}')
    result = sync.run("operation_failed", 2)
    assert result["checkout_blocked"] is True
    assert "invalid checkout update marker" in result["error"]
    assert barrier.read_text() == '{"schema_version": 99}'


def test_fetch_error_is_not_reported_as_a_skip_or_success(sync: SyncEnv) -> None:
    git(
        sync.repo,
        "remote",
        "set-url",
        "origin",
        str(sync.repo.parent / "missing-origin"),
    )
    result = sync.run("operation_failed", 2)
    assert result["outcome"] == "failed"
    assert "does not appear to be a git repository" in result["stderr"]


@pytest.fixture
def running_job(sync: SyncEnv) -> Iterator[subprocess.Popen[str]]:
    ready = sync.repo.parent / "job-ready"
    command = [
        sys.executable,
        str(LAUNCHER),
        "--repo",
        str(sync.repo),
        "--",
        sys.executable,
        "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).touch(); sys.stdin.read()",
        str(ready),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE, text=True) as process:
        try:
            wait_file(ready)
            yield process
        finally:
            process.communicate(timeout=5)


def test_running_process_holds_shared_code_lock_through_exec(
    sync: SyncEnv, running_job: subprocess.Popen[str]
) -> None:
    sync.publish({"app.py": "new\n"})
    assert running_job.poll() is None
    sync.run("checkout_jobs_running", 75)


def test_update_blocks_new_job_before_it_reads_checkout(sync: SyncEnv) -> None:
    sync.publish({"app.py": "new\n"})
    ready, release, read = [
        sync.repo.parent / name for name in ("merge-ready", "merge-release", "job-read")
    ]
    install_git_hook(
        sync,
        f"""
if 'merge' in args:
    Path({str(ready)!r}).touch()
    deadline = time.monotonic() + 8
    while not Path({str(release)!r}).exists():
        if time.monotonic() > deadline: sys.exit(9)
        time.sleep(.02)
sys.exit(subprocess.call([real, *args]))
""",
    )
    with sync.start() as updater:
        try:
            wait_file(ready)
            job = subprocess.Popen(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "--repo",
                    str(sync.repo),
                    "--",
                    sys.executable,
                    "-c",
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(pathlib.Path(sys.argv[2]).read_text())",
                    str(read),
                    str(sync.repo / "app.py"),
                ]
            )
            try:
                wait_for_flock(job)
                assert not read.exists()
                release.touch()
                stdout, stderr = updater.communicate(timeout=15)
                assert updater.returncode == 0, stdout + stderr
                job.wait(timeout=5)
                assert job.returncode == 0
                assert read.read_text() == "new\n"
            finally:
                if job.poll() is None:
                    job.terminate()
                    job.wait(timeout=5)
        finally:
            release.touch()
            if updater.poll() is None:
                updater.terminate()
                updater.wait(timeout=5)


def test_operator_off_waits_for_update_then_leaves_supervisor_off(
    sync: SyncEnv,
) -> None:
    sync.publish({"app.py": "new\n"})
    state = sync.control.read()
    state["units"][SUPERVISOR].update(ActiveState="active", UnitFileState="enabled")
    state["dispatcher"]["enabled"] = True
    sync.control.write(state)
    ready, release = [
        sync.repo.parent / name for name in ("pause-ready", "pause-release")
    ]
    install_git_hook(
        sync,
        f"""
if 'merge' in args:
    Path({str(ready)!r}).touch()
    deadline = time.monotonic() + 8
    while not Path({str(release)!r}).exists():
        if time.monotonic() > deadline: sys.exit(9)
        time.sleep(.02)
sys.exit(subprocess.call([real, *args]))
""",
    )
    with sync.start() as updater:
        try:
            wait_file(ready)
            with subprocess.Popen(
                [sys.executable, "-m", "spin", "ops", "off", "dispatcher"],
                cwd=sync.repo,
                env=sync.control.environment,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as off:
                try:
                    wait_for_flock(off)
                    release.touch()
                    stdout, stderr = updater.communicate(timeout=15)
                    assert updater.returncode == 0, stdout + stderr
                    stdout, stderr = off.communicate(timeout=10)
                    assert off.returncode == 0, stdout + stderr
                finally:
                    if off.poll() is None:
                        os.killpg(off.pid, signal.SIGTERM)
                        off.wait(timeout=5)
        finally:
            release.touch()
            if updater.poll() is None:
                updater.terminate()
                updater.wait(timeout=5)
    assert sync.control.read()["units"][SUPERVISOR]["ActiveState"] == "inactive"
    assert sync.control.read()["units"][SUPERVISOR]["UnitFileState"] == "linked"
    assert not sync.control.read()["dispatcher"]["enabled"]
