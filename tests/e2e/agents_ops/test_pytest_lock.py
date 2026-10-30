"""The historical pytest entrypoint allows concurrent runs across worktrees."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[3] / ".agents/ops/pytest_locked.sh"
FAKE_PYTHON = r"""
import json, os, sys, time
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
label = os.environ["FAKE_LABEL"]
with (root / "trace.jsonl").open("a") as out:
    out.write(json.dumps({"label": label, "args": sys.argv[1:], "cwd": os.getcwd()}) + "\n")
(root / (label + ".entered")).touch()
if label == "first":
    deadline = time.monotonic() + 10
    while not (root / "release").exists():
        if time.monotonic() > deadline:
            sys.exit(98)
        time.sleep(0.01)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
"""


def fake_worktree(root: Path, name: str) -> Path:
    repo = root / name
    destination = repo / ".agents/ops/pytest_locked.sh"
    destination.parent.mkdir(parents=True)
    shutil.copyfile(WRAPPER, destination)
    python = repo / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f"#!{sys.executable}\n" + FAKE_PYTHON)
    python.chmod(0o755)
    return destination


@pytest.mark.parametrize("explicit_lock", [False, True])
def test_pytest_runs_concurrently_even_when_old_lock_is_held(
    tmp_path: Path, explicit_lock: bool
) -> None:
    first_wrapper = fake_worktree(tmp_path, "worktree-a")
    second_wrapper = fake_worktree(tmp_path, "worktree-b")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_LOCK"}
    env.update(XDG_STATE_HOME=str(tmp_path / "state"), FAKE_ROOT=str(tmp_path))
    lock = tmp_path / "state/tennis-lab-agents/pytest.lock"
    if explicit_lock:
        lock = tmp_path / "campaign/pytest.lock"
        env["PYTEST_LOCK"] = str(lock)
    lock.parent.mkdir(parents=True)
    processes: list[subprocess.Popen[str]] = []
    with lock.open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for label, wrapper in (("first", first_wrapper), ("second", second_wrapper)):
                process = subprocess.Popen(
                    ["bash", str(wrapper), f"tests/{label}.py", "-q"],
                    env={**env, "FAKE_LABEL": label},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append(process)
                deadline = time.monotonic() + 10
                while not (tmp_path / f"{label}.entered").exists():
                    assert process.poll() is None
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            # Both entered before releasing the first test or the old host lock.
            assert processes[0].poll() is None
            (tmp_path / "release").touch()
            for process in processes:
                output, errors = process.communicate(timeout=15)
                assert process.returncode == 0, output + errors
        finally:
            (tmp_path / "release").touch()
            for pending in processes:
                if pending.poll() is None:
                    pending.kill()
                    pending.communicate(timeout=5)
    events = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert [e["label"] for e in events] == ["first", "second"]
    assert events[0]["args"] == ["-m", "pytest", "-n", "4", "tests/first.py", "-q"]
    assert events[1]["cwd"] == str(second_wrapper.parents[2])


@pytest.mark.parametrize(
    "arguments",
    [["-n", "0"], ["-n8"], ["--numprocesses=8"], ["--numprocesses", "2"]],
)
def test_worker_count_override_and_exit_status_are_preserved(
    tmp_path: Path, arguments: list[str]
) -> None:
    wrapper = fake_worktree(tmp_path, "worker")
    result = subprocess.run(
        ["bash", str(wrapper), *arguments],
        env={
            **os.environ,
            "PYTEST_LOCK": "obsolete-relative.lock",
            "FAKE_ROOT": str(tmp_path),
            "FAKE_LABEL": "ended",
            "FAKE_EXIT": "7",
        },
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 7, result.stderr
    event = json.loads((tmp_path / "trace.jsonl").read_text())
    assert event["args"] == ["-m", "pytest", *arguments]
    assert not (wrapper.parents[2] / "obsolete-relative.lock").exists()


def test_missing_worktree_python_is_reported(tmp_path: Path) -> None:
    wrapper = fake_worktree(tmp_path, "worker")
    (wrapper.parents[2] / ".venv/bin/python").unlink()
    result = subprocess.run(
        ["bash", str(wrapper)], capture_output=True, check=False, timeout=10
    )
    assert result.returncode == 2
    assert b"repository .venv/bin/python is missing" in result.stderr
