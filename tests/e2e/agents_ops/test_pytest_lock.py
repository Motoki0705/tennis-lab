"""Two worktrees request fake pytest runs through the host-wide lock."""

from __future__ import annotations

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
import fcntl, json, os, sys, time
from pathlib import Path
root = Path(os.environ["FAKE_ROOT"])
label = os.environ["FAKE_LABEL"]
with (root / "active.lock").open("a") as active:
    try:
        fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(99)
    with (root / "trace.jsonl").open("a") as out:
        out.write(json.dumps({"label": label, "event": "start", "args": sys.argv[1:], "cwd": os.getcwd()}) + "\n")
    (root / (label + ".entered")).touch()
    if label == "first":
        deadline = time.monotonic() + 10
        while not (root / "release").exists():
            if time.monotonic() > deadline:
                sys.exit(98)
            time.sleep(0.01)
    with (root / "trace.jsonl").open("a") as out:
        out.write(json.dumps({"label": label, "event": "end"}) + "\n")
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
def test_pytest_is_exclusive_across_worktrees(
    tmp_path: Path, explicit_lock: bool
) -> None:
    first_wrapper = fake_worktree(tmp_path, "worktree-a")
    second_wrapper = fake_worktree(tmp_path, "worktree-b")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_LOCK"}
    env.update(XDG_STATE_HOME=str(tmp_path / "state"), FAKE_ROOT=str(tmp_path))
    if explicit_lock:
        env["PYTEST_LOCK"] = str(tmp_path / "campaign/pytest.lock")
    first = subprocess.Popen(
        ["bash", str(first_wrapper), "tests/a.py", "-q"],
        env={**env, "FAKE_LABEL": "first"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    second = None
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "first.entered").exists():
            assert first.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        second = subprocess.Popen(
            ["bash", str(second_wrapper), "tests/b.py", "-q"],
            env={**env, "FAKE_LABEL": "second"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.15)
        assert second.poll() is None
        assert not (tmp_path / "second.entered").exists()
        (tmp_path / "release").touch()
        for process in (first, second):
            output, errors = process.communicate(timeout=15)
            assert process.returncode == 0, output + errors
    finally:
        (tmp_path / "release").touch()
        for pending in (first, second):
            if pending is not None and pending.poll() is None:
                pending.kill()
                pending.communicate(timeout=5)
    events = [
        json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()
    ]
    assert [(e["label"], e["event"]) for e in events] == [
        ("first", "start"),
        ("first", "end"),
        ("second", "start"),
        ("second", "end"),
    ]
    assert events[0]["args"] == ["-m", "pytest", "-n", "4", "tests/a.py", "-q"]
    assert events[2]["cwd"] == str(second_wrapper.parents[2])
    exited = subprocess.run(
        ["bash", str(second_wrapper)],
        env={**env, "FAKE_LABEL": "ended", "FAKE_EXIT": "7"},
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert exited.returncode == 7


@pytest.mark.parametrize(
    "arguments",
    [["-n", "auto"], ["-n8"], ["--numprocesses=8"], ["--numprocesses", "2"]],
)
def test_worker_count_override_is_rejected(
    tmp_path: Path, arguments: list[str]
) -> None:
    wrapper = fake_worktree(tmp_path, "worker")
    result = subprocess.run(
        ["bash", str(wrapper), *arguments],
        env={**os.environ, "PYTEST_LOCK": str(tmp_path / "pytest.lock")},
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert b"fixed at -n 4" in result.stderr
    assert not (tmp_path / "pytest.lock").exists()


def test_relative_lock_is_rejected_before_pytest(tmp_path: Path) -> None:
    wrapper = fake_worktree(tmp_path, "worker")
    result = subprocess.run(
        ["bash", str(wrapper)],
        env={**os.environ, "PYTEST_LOCK": "relative.lock"},
        capture_output=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == 2
    assert b"absolute path" in result.stderr
