"""End-to-end checks for the file-backed training queue entry point."""

from __future__ import annotations

import fcntl
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).parents[3]
QUEUE_SCRIPT = ROOT / ".agents/skills/training-queue/scripts/training_queue.sh"


def _queue_env(queue_dir: Path, lock_file: Path) -> dict[str, str]:
    return {
        **os.environ,
        "TRAINING_QUEUE_DIR": str(queue_dir),
        "TRAINING_QUEUE_LOCK_FILE": str(lock_file),
    }


def test_serve_reports_running_and_honours_shared_gpu_lock(tmp_path: Path) -> None:
    queue_dir = tmp_path / "queue"
    lock_file = tmp_path / "gpu.lock"
    result_file = tmp_path / "finished"
    env = _queue_env(queue_dir, lock_file)
    command = f"printf finished > {shlex.quote(str(result_file))}"

    subprocess.run(
        ["bash", str(QUEUE_SCRIPT), "add", command, "--name", "lock-test"],
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
    env = {**os.environ, "TRAINING_QUEUE_DIR": str(queue_dir)}

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
