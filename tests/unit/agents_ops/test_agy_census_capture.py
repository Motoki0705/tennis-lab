"""Synthetic streams test observations, not a guessed list of model behavior."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/github_actions"))
census: Any = importlib.import_module("agy_census_capture")


def stream(*steps: dict[str, Any], result: dict[str, Any] | None = None) -> str:
    events = [{"event": "init", "conversation_id": "test", "init": {}}]
    events.extend({"event": "step_update", "step_update": step} for step in steps)
    if result is not None:
        events.append({"event": "result", "result": result})
    return "\n".join(json.dumps(event) for event in events)


def step(
    index: int, command: str, state: str = "ACTIVE", **info: Any
) -> dict[str, Any]:
    return {
        "conversation_id": "test",
        "step_index": index,
        "state": state,
        "tool_info": {
            "name": "run_command",
            "parameters": {"CommandLine": command},
            **info,
        },
    }


def test_soft_denial_survives_done_without_error_or_output() -> None:
    raw = stream(
        step(2, "python3 -c pass"),
        step(2, "python3 -c pass", "DONE"),
        result={"status": "SUCCESS", "response": ""},
    )
    capture = census.capture_stream(
        raw, 'Print mode: soft-denying tool confirmation "RunCommand" at step 2'
    )
    assert [(x["command"], x["outcome"]) for x in capture["commands"]] == [
        ("python3 -c pass", "soft_denied")
    ]
    assert "structured_output" not in capture["result"]


def test_repeated_command_is_two_attempts_but_active_done_are_one() -> None:
    capture = census.capture_stream(
        stream(
            step(2, "cat a"),
            step(2, "cat a", "DONE", output="one"),
            step(4, "cat a"),
            step(4, "cat a", "DONE", output="two"),
        )
    )
    assert len(capture["commands"]) == 2
    assert capture["tool_counts"] == {"run_command": 2}
    groups = census.summarize([{**capture, "run_id": "repeat"}])
    assert groups[0]["count"] == groups[0]["examples"][0]["count"] == 2


def test_explicit_denial_can_be_followed_by_successful_output() -> None:
    raw = stream(
        step(2, "find . -type f", "ERROR", error={"message": "Permission denied"}),
        step(4, "cat a", "DONE", output="ok"),
        result={"status": "SUCCESS", "structured_output": {"attempted": True}},
    )
    capture = census.capture_stream(raw)
    assert [x["outcome"] for x in capture["commands"]] == ["denied", "completed"]


def test_missing_step_and_unfinished_attempt_remain_visible() -> None:
    capture = census.capture_stream(
        stream(step(2, "cat a")),
        'Print mode: soft-denying tool confirmation "RunCommand" at step 7',
    )
    assert capture["commands"][0]["outcome"] == "unknown"
    assert capture["uncaptured_soft_denial_steps"] == [7]
    assert capture["result"] is None
    without_log = census.capture_stream(stream(step(2, "cat a", "DONE")))
    assert without_log["commands"][0]["outcome"] == "unknown"


@pytest.mark.parametrize(
    "raw",
    [
        '{"event":"new-unknown-event"}',
        stream(
            {"step_index": 2, "state": "DONE", "tool_info": {"name": "run_command"}}
        ),
    ],
)
def test_incomplete_capture_fails_loudly(raw: str) -> None:
    with pytest.raises(ValueError):
        census.capture_stream(raw)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "gh issue view 12 --json number,title",
            "gh issue view <N> --json number,title",
        ),
        ("rg -n 'hello world' src/code.py", "rg -n <PATTERN> <PATH>"),
        ("head -n20 a | wc -l", "head -n<N> <ARG> | wc -l"),
        ("ls -la", "ls -la"),
    ],
)
def test_descriptive_forms_retain_flags_and_operators(
    command: str, expected: str
) -> None:
    assert census.command_form(command) == expected


def test_grammar_uses_raw_text_and_does_not_expand_combined_flags() -> None:
    assert census.grammar_allows("ls -l -a")
    assert not census.grammar_allows("ls -la")
    assert not census.grammar_allows("cat a | head -n 1")
    assert not census.grammar_allows("cat a\n")


@pytest.mark.parametrize("fault", ["none", "missing", "extra", "changed"])
def test_trajectory_reconciliation_checks_both_directions(fault: str) -> None:
    capture = census.capture_stream(
        stream(
            step(2, "cat a", "DONE", output="ok"),
            step(4, "python3 -c pass", "ERROR", error={"message": "Permission denied"}),
        )
    )
    trajectory = [
        {"conversation_id": "test", "step": 4, "command": "python3 -c pass"},
        {"conversation_id": "test", "step": 2, "command": "cat a"},
    ]
    if fault == "missing":
        trajectory.append({"conversation_id": "test", "step": 6, "command": "ls"})
    elif fault == "extra":
        trajectory.pop()
    elif fault == "changed":
        trajectory[0]["command"] = "python3 -c 'pass'"
    result = census.reconcile_commands(capture, trajectory)
    assert result["matches"] == (fault == "none")
    assert len(result["missing"]) == (fault in {"missing", "changed"})
    assert len(result["extra"]) == (fault in {"extra", "changed"})
