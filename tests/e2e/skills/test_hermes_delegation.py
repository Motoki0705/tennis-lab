"""Tests for the Hermes delegation skill's CLI adapter."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3]
    / ".agents"
    / "skills"
    / "hermes-delegation"
    / "scripts"
    / "hermes_delegate.py"
)
SPEC = importlib.util.spec_from_file_location("hermes_delegate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
hermes_delegate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hermes_delegate)


def test_extract_session_id_uses_latest_value() -> None:
    stderr = "warning\nsession_id: old\nresumed\nsession_id: current\n"

    assert hermes_delegate.extract_session_id(stderr) == "current"


def test_strip_display_reasoning_preserves_answer() -> None:
    stdout = "┌─ Reasoning ─┐\ninternal thought\n└────┘\nFinal answer\n"

    assert hermes_delegate.strip_display_reasoning(stdout) == "Final answer\n"


def test_build_command_resumes_without_shell_interpolation() -> None:
    command = hermes_delegate.build_command(
        hermes="hermes",
        prompt='explain "alignment"; do not edit',
        session_id="session-1",
        model="provider/model",
        provider="provider",
        toolsets="terminal",
        max_turns=12,
        source="tool",
        ignore_user_config=True,
    )

    assert command[:8] == [
        "hermes",
        "chat",
        "--query",
        'explain "alignment"; do not edit',
        "--quiet",
        "--pass-session-id",
        "--source",
        "tool",
    ]
    assert command[8:] == [
        "--ignore-user-config",
        "--resume",
        "session-1",
        "--model",
        "provider/model",
        "--provider",
        "provider",
        "--toolsets",
        "terminal",
        "--max-turns",
        "12",
    ]


def test_main_persists_session_and_returns_only_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    state_file = tmp_path / "state.json"
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout="Hermes answer\n", stderr="session_id: sid-1\n"
        )

    monkeypatch.setattr(hermes_delegate.shutil, "which", lambda _: "/usr/bin/hermes")
    monkeypatch.setattr(hermes_delegate.subprocess, "run", fake_run)

    assert hermes_delegate.main(["--session-file", str(state_file), "first task"]) == 0
    assert capsys.readouterr() == ("Hermes answer\n", "")
    assert json.loads(state_file.read_text(encoding="utf-8"))["session_id"] == "sid-1"
    assert "--resume" not in calls[0]

    assert hermes_delegate.main(["--session-file", str(state_file), "follow-up"]) == 0
    assert calls[1][calls[1].index("--resume") + 1] == "sid-1"


def test_resume_required_does_not_create_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state_file = tmp_path / "missing.json"

    assert (
        hermes_delegate.main(
            ["--session-file", str(state_file), "--resume-required", "follow-up"]
        )
        == 1
    )
    assert not state_file.exists()
    assert "refusing to start a new session" in capsys.readouterr().err
