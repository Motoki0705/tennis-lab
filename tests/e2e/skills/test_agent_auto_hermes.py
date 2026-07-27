"""Command-level tests for the agent-auto Hermes wrapper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[3]
WRAPPER = REPOSITORY_ROOT / ".agents/skills/agent-auto/scripts/hermes-auto.sh"


def _write_hermes_stub(bin_dir: Path, log_path: Path) -> None:
    stub = bin_dir / "hermes"
    stub.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$HERMES_STUB_LOG"
if [[ "$*" == *"--resume stub-session"* ]]; then
  printf 'resume response\n'
else
  printf 'initial response\n'
fi
printf '\nsession_id: stub-session\n' >&2
""",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    log_path.touch()


def _run_wrapper(
    tmp_path: Path, *, prompt: str, resume: str | None = None
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log_path = tmp_path / "hermes-calls.log"
    _write_hermes_stub(bin_dir, log_path)
    log_dir = tmp_path / "logs"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["HERMES_STUB_LOG"] = str(log_path)
    command = [
        str(WRAPPER),
        "--dir",
        str(tmp_path),
        "--log-dir",
        str(log_dir),
        "--name",
        "test",
        "--no-yolo",
        prompt,
    ]
    if resume is not None:
        command[1:1] = ["--resume", resume]
    return subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _single_run_dir(tmp_path: Path) -> Path:
    run_dirs = list((tmp_path / "logs").glob("*-test-*"))
    assert len(run_dirs) == 1
    return run_dirs[0]


def test_initial_and_resume_runs_are_logged_and_detected(tmp_path: Path) -> None:
    initial = _run_wrapper(tmp_path, prompt="initial task")

    assert initial.returncode == 0
    initial_dir = _single_run_dir(tmp_path)
    initial_summary = (initial_dir / "summary.txt").read_text()
    assert "status=success" in initial_summary
    assert "session_id=stub-session" in initial_summary
    assert "initial response" in (initial_dir / "result.txt").read_text()

    resumed = _run_wrapper(tmp_path, prompt="follow-up task", resume="stub-session")

    assert resumed.returncode == 0
    run_dirs = list((tmp_path / "logs").glob("*-test-*"))
    assert len(run_dirs) == 2
    resumed_dir = sorted(run_dirs)[-1]
    summary = (resumed_dir / "summary.txt").read_text()
    assert "status=success" in summary
    assert "resumed_session=true" in summary
    assert "resume response" in (resumed_dir / "result.txt").read_text()
    assert "--resume stub-session" in (tmp_path / "hermes-calls.log").read_text()


def test_missing_session_id_is_a_failed_run(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "hermes"
    stub.write_text(
        "#!/usr/bin/env bash\nprintf 'response without session'\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    log_dir = tmp_path / "logs"
    result = subprocess.run(
        [str(WRAPPER), "--dir", str(tmp_path), "--log-dir", str(log_dir), "task"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    run_dir = next(log_dir.iterdir())
    summary = (run_dir / "summary.txt").read_text()
    assert "status=failed" in summary
    assert "failure_reason=missing_session_id" in summary
