"""Checkout protection belongs to native CLI bootstraps, independent of callers."""

from __future__ import annotations

import fcntl
import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.agents_ops.checkout_cli import (
    ENTRIES,
    PARENT,
    ROOT,
    Entry,
    environment,
    install_cli,
    line,
    running_cli,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def code_is_locked(repo: Path) -> bool:
    with (repo / ".git/ops-code.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry.name)
def test_native_entry_holds_lock_before_first_project_import(
    repo: Path,
    tmp_path: Path,
    entry: Entry,
) -> None:
    install_cli(repo, entry)
    # Launch from elsewhere: the code location, not cwd, defines protection.
    with running_cli(
        repo,
        entry,
        cwd=tmp_path,
        arguments=("--private-probe", "private-value", "-m", "private-module"),
    ) as process:
        assert line(process) == "project-import"
        assert code_is_locked(repo)
        deferred = subprocess.run(
            [
                sys.executable,
                str(ROOT / ".agents/ops/git_sync/sync.py"),
                "--repo",
                str(repo),
                "--state-dir",
                str(tmp_path / "sync-state"),
            ],
            env=environment(repo),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        assert deferred.returncode == 75, deferred.stdout + deferred.stderr
        status = json.loads(deferred.stdout)
        assert status["reason"] == "checkout_jobs_running"
        assert status["lock_holders"] == [
            {"pid": process.pid, "entry": entry.description, "mode": "read"}
        ]
        assert "private-value" not in deferred.stdout + deferred.stderr
        assert "private-module" not in deferred.stdout + deferred.stderr
    assert process.returncode == 0
    assert not code_is_locked(repo)


def test_cli_waits_for_updater_before_project_import(repo: Path) -> None:
    install_cli(repo)
    with (repo / ".git/ops-code.lock").open("a") as updater:
        fcntl.flock(updater, fcntl.LOCK_EX)
        with running_cli(repo) as process:
            assert "waiting for git-sync" in line(process, stderr=True)
            assert process.stdout is not None
            assert not select.select([process.stdout], [], [], 0)[0]
            fcntl.flock(updater, fcntl.LOCK_UN)
            assert line(process) == "project-import"
            assert code_is_locked(repo)
    assert process.returncode == 0


@pytest.mark.parametrize("failure", ["recovery", "lock-unavailable"])
def test_lock_failure_refuses_cli_before_import(repo: Path, failure: str) -> None:
    install_cli(repo)
    if failure == "recovery":
        (repo / ".git/ops-code.blocked").write_text("interrupted update")
    else:
        (repo / ".git/ops-code.lock").mkdir()
    result = subprocess.run(
        PARENT.command(repo),
        env=environment(repo),
        capture_output=True,
        text=True,
        check=False,
        timeout=8,
    )
    assert result.returncode == 75
    assert not result.stdout
    assert "checkout CLI refused" in result.stderr
    if failure == "recovery":
        assert "checkout recovery pending" in result.stderr


def test_worktree_source_does_not_lock_main_but_main_source_always_does(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_cli(repo)
    for args in (
        ("config", "user.name", "Checkout Test"),
        ("config", "user.email", "checkout@example.invalid"),
        ("add", "-f", ".agents"),
        ("commit", "-qm", "bootstrap"),
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "-qb", "worker", str(worktree)],
        check=True,
    )
    unrelated = tmp_path / "unrelated"
    subprocess.run(["git", "init", "-q", str(unrelated)], check=True)
    monkeypatch.setenv("GIT_DIR", str(unrelated / ".git"))
    monkeypatch.setenv("GIT_COMMON_DIR", str(unrelated / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(unrelated))
    with (repo / ".git/ops-code.lock").open("a") as updater:
        fcntl.flock(updater, fcntl.LOCK_EX)
        with running_cli(worktree, cwd=repo) as worker_cli:
            assert line(worker_cli) == "project-import"
        assert worker_cli.returncode == 0
    with running_cli(
        repo, cwd=worktree, arguments=("--repo", str(worktree))
    ) as main_cli:
        assert line(main_cli) == "project-import"
        assert code_is_locked(repo)
    assert main_cli.returncode == 0
    assert not (unrelated / ".git/ops-code.lock").exists()


@pytest.mark.parametrize("launch", ["exec", "fork"])
def test_external_child_cannot_keep_cli_lock_after_parent_exit(
    repo: Path,
    launch: str,
) -> None:
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    child = (
        "import fcntl,json,os,pathlib\n"
        "os.write(int(os.environ['READY_FD']), b'ready\\n')\n"
        "os.read(int(os.environ['RELEASE_FD']), 1)\n"
        f"stream = pathlib.Path({str(repo / '.git/ops-code.lock')!r}).open('a')\n"
        "try:\n"
        "    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "    locked = False\n"
        "except BlockingIOError:\n"
        "    locked = True\n"
        "stream.close()\n"
        "os.write(int(os.environ['READY_FD']), json.dumps({'locked': locked}).encode()+b'\\n')\n"
    )
    if launch == "exec":
        spawn = (
            f"subprocess.Popen([sys.executable, '-c', {child!r}], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            "stderr=subprocess.DEVNULL, close_fds=False)\n"
        )
    else:
        spawn = (
            "if os.fork() == 0:\n"
            "    null = os.open('/dev/null', os.O_RDWR)\n"
            "    for fd in range(3): os.dup2(null, fd)\n"
            f"    exec({child!r})\n"
            "    os._exit(0)\n"
        )
    install_cli(
        repo, project="import os,subprocess,sys\n" + spawn + "raise SystemExit(0)\n"
    )
    env = {
        **environment(repo),
        "READY_FD": str(ready_write),
        "RELEASE_FD": str(release_read),
    }
    try:
        with (
            os.fdopen(ready_read) as ready,
            subprocess.Popen(
                PARENT.command(repo),
                env=env,
                pass_fds=(ready_write, release_read),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as parent,
        ):
            assert select.select([ready], [], [], 8)[0]
            assert ready.readline().strip() == "ready"
            stdout, stderr = parent.communicate(timeout=8)
            assert parent.returncode == 0, stdout + stderr
            os.write(release_write, b"x")
            assert select.select([ready], [], [], 8)[0]
            assert json.loads(ready.readline()) == {"locked": False}
            assert not code_is_locked(repo)
    finally:
        for fd in (ready_write, release_read, release_write):
            os.close(fd)
