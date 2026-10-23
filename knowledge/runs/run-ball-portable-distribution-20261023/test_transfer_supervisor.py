"""Detached transfers must keep a terminal receipt, including failed commands."""

# mypy: disallow_untyped_decorators=False

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("transfer_supervisor", Path(__file__).with_name("transfer_supervisor.py"))
assert spec is not None and spec.loader is not None
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


@pytest.mark.parametrize("returncode", [0, 7])
def test_terminal_receipt_preserves_the_command_result(tmp_path: Path, returncode: int) -> None:
    state = tmp_path / "result.json"
    argv = [sys.executable, "-c", f"raise SystemExit({returncode})"]
    assert supervisor.supervise(state, argv) == returncode
    result = json.loads(state.read_text())
    assert result["returncode"] == returncode
    assert result["status"] == ("completed" if returncode == 0 else "failed")
    assert result["argv"] == argv
    assert result["command_pid"] != result["pid"]
    assert result["elapsed_seconds"] > 0
    assert result["finished_at"] >= result["started_at"]
    before = state.read_bytes()
    with pytest.raises(ValueError, match="new receipt"):
        supervisor.supervise(state, argv)
    assert state.read_bytes() == before


def test_failure_to_start_still_leaves_a_failure_receipt(tmp_path: Path) -> None:
    state = tmp_path / "result.json"
    assert supervisor.supervise(state, [str(tmp_path / "missing-executable")]) == 1
    result = json.loads(state.read_text())
    assert result["status"] == "failed"
    assert result["returncode"] == 1
    assert "error" in result


def test_detached_launch_uses_its_own_session_and_file_streams(tmp_path: Path) -> None:
    state, stdout, stderr = (tmp_path / name for name in ("result.json", "stdout.log", "stderr.log"))
    started = supervisor.launch(state, stdout, stderr, [sys.executable, "-c", "print('finished')"])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = json.loads(state.read_text()) if state.exists() else {}
        if result.get("status") == "completed":
            break
        time.sleep(0.01)
    else:
        pytest.fail("Detached command did not publish its successful terminal receipt")
    assert result["pid"] == started["supervisor_pid"]
    assert result["session_id"] == result["pid"]
    assert result["process_group_id"] == result["pid"]
    assert result["returncode"] == 0
    assert stdout.read_text().strip() == "finished"
    assert stderr.read_text() == ""
