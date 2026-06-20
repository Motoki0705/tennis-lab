"""training-queue reproducibility-capture tests (issue #533).

Runs the bash queue script end-to-end in a throwaway git repo and asserts that a
finished job leaves a reproducibility bundle (run.json / repro.sh /
uncommitted.patch) with the declared provider/session/issue and captured git
state, and that the job actually saw TENNIS_RUN_ID / TENNIS_REPRO_DIR.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _run(cwd: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=cwd, env=env, capture_output=True, text=True
    )


def test_repro_bundle_is_captured(tmp_path: Path) -> None:
    repo = tmp_path / "wt"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("hello\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "remote", "add", "origin", "https://example.com/foo.git")
    # Uncommitted change that must show up in the patch.
    (repo / "a.txt").write_text("hello\ndirty\n")

    env = {**os.environ, "TRAINING_QUEUE_DIR": str(tmp_path / ".tq")}
    add = _run(
        repo,
        env,
        "add",
        'echo RAN $TENNIS_RUN_ID; test -n "$TENNIS_REPRO_DIR"',
        "--name",
        "smoke",
        "--provider",
        "claude",
        "--session",
        "sess-1",
        "--issue",
        "533",
    )
    assert add.returncode == 0, add.stderr
    start = _run(repo, env, "start", "--idle-timeout", "2")
    assert start.returncode == 0, start.stderr

    for _ in range(60):
        status = _run(repo, env, "status").stdout
        if "worker: stopped" in status and "done=1" in status:
            break
        time.sleep(0.5)
    else:
        raise AssertionError(f"worker did not finish a job: {status}")

    repro_dirs = list((tmp_path / ".tq" / "repro").glob("*"))
    assert len(repro_dirs) == 1, repro_dirs
    rd = repro_dirs[0]

    run = json.loads((rd / "run.json").read_text())
    assert run["provider"] == "claude"
    assert run["session"] == "sess-1"
    assert run["issue"] == "533"
    assert run["name"] == "smoke"
    assert run["commit"] and run["branch"]
    assert run["remote"] == "https://example.com/foo.git"

    assert (rd / "repro.sh").exists()
    patch = (rd / "uncommitted.patch").read_text()
    assert "+dirty" in patch

    # The job itself ran with the injected env vars.
    logs = list((tmp_path / ".tq" / "logs").glob("*.log"))
    assert logs and any("RAN " in lf.read_text() for lf in logs)
