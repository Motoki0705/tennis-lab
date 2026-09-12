from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.command_supervisor import supervise


@pytest.mark.parametrize(("command", "timeout", "code", "outcome"), [
    ("exit 0", 5, 0, "succeeded"),
    ("exit 124", 5, 124, "failed"),
    ("exit 143", 5, 143, "failed"),
    ("sleep 30", .05, 124, "timed_out"),
])
def test_observed_outcomes(command: str, timeout: float, code: int, outcome: str, tmp_path: Path) -> None:
    destination = tmp_path / "outcome.json"
    assert supervise(command, timeout, destination) == code
    assert json.loads(destination.read_text()) == {"exit_code": code, "outcome": outcome}
