"""Documented manual CLI entry points cannot bypass checkout coordination."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"


@pytest.mark.parametrize(
    "entry",
    [
        "report",
        "cleanup",
        "dispatch-file",
        "dispatch-module",
        "judgement",
        "supervisor",
        "board-web",
        "worktree-scan",
        "worktree-dry-run",
        "worktree-cleanup",
    ],
)
def test_direct_operational_entry_requires_its_launcher(
    tmp_path: Path, entry: str
) -> None:
    commands = {
        "report": [str(OPS / "ops-report/ops_report.py"), "--help"],
        "cleanup": [str(OPS / "cleanup/cleanup.py"), "--help"],
        "dispatch-file": [str(OPS / "dispatcher/dispatch.py"), "run", "--explicit"],
        "dispatch-module": ["-m", "dispatcher", "run", "--explicit"],
        "judgement": [
            "-m",
            "dispatcher.judgement",
            "--repo",
            "o/r",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        "supervisor": ["-m", "dispatcher", "supervise", "--once"],
        "board-web": ["-m", "board", "web", "--port", "8799"],
        "worktree-scan": ["-m", "worktree_cleanup", "scan"],
        "worktree-dry-run": ["-m", "worktree_cleanup", "cleanup", "--dry-run"],
        "worktree-cleanup": ["-m", "worktree_cleanup", "cleanup"],
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TENNIS_LAB_CHECKOUT_")
    }
    env.update(
        PYTHONPATH=str(OPS),
        XDG_STATE_HOME=str(tmp_path / "state"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("gh", "claude", "codex"):
        fake = bin_dir / tool
        fake.write_text("#!/bin/sh\nexit 77\n")
        fake.chmod(0o700)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    result = subprocess.run(
        [sys.executable, *commands[entry]],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 75, result.stdout + result.stderr
    assert "checkout_job.py" in result.stderr
    assert not (tmp_path / "run").exists()


def test_environment_flag_without_an_inherited_lock_descriptor_is_refused(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".git/ops-code.lock").touch()
    result = subprocess.run(
        [sys.executable, str(OPS / "ops-report/ops_report.py"), "--help"],
        env={
            **os.environ,
            "TENNIS_LAB_CHECKOUT_ROOT": str(tmp_path),
            "TENNIS_LAB_CHECKOUT_FD": "1",
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 75
    assert "not the checkout lock descriptor" in result.stderr


def test_worktree_target_must_share_the_launcher_repository(tmp_path: Path) -> None:
    launcher_root, target = tmp_path / "launcher", tmp_path / "target"
    for repo in (launcher_root, target):
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    state = tmp_path / "expiry-state"
    result = subprocess.run(
        [
            sys.executable,
            str(OPS / "checkout_job.py"),
            "--repo",
            str(launcher_root),
            "--",
            sys.executable,
            "-m",
            "worktree_cleanup",
            "cleanup",
            "--repo",
            str(target),
            "--state-dir",
            str(state),
        ],
        env={**os.environ, "PYTHONPATH": str(OPS)},
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "must share the launcher repository" in result.stderr
    assert not state.exists()
