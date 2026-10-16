"""Retired ball entrypoints cannot dispatch legacy execution or adoption."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("command", ["run", "qa", "intake"])
def test_retired_ball_commands_are_rejected(command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.tennis_scene.chat_annotation.local_agent", command],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
