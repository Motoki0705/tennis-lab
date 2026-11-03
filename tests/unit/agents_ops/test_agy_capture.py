"""Synthetic streams test observations, not a guessed list of model behavior."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/lib"))
census: Any = importlib.import_module("agy_capture")


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


def read_step(
    index: int, path: Any, state: str = "ACTIVE", **info: Any
) -> dict[str, Any]:
    return {
        "conversation_id": "test",
        "step_index": index,
        "state": state,
        "tool_info": {"name": "read_file", "parameters": {"Path": path}, **info},
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


def test_invalid_or_truncated_stream_retains_prior_attempts_but_is_not_valid() -> None:
    capture = census.capture_stream(
        stream(step(2, "cat private.txt")) + '\n{"event":', strict=False
    )
    assert capture["result"] is None
    assert capture["commands"][0]["command"] == "cat private.txt"
    assert capture["errors"]


@pytest.mark.parametrize(
    "bad",
    [
        {"event": "result", "result": None},
        {
            "event": "step_update",
            "step_update": {"step_index": 3, "state": "ACTIVE", "tool_info": {}},
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": [],
                "state": "ACTIVE",
                "tool_info": {"name": "run_command"},
            },
        },
        {
            "event": "step_update",
            "step_update": {
                "step_index": 3,
                "state": "ACTIVE",
                "tool_info": {"name": [], "parameters": {}},
            },
        },
    ],
)
def test_bad_event_does_not_discard_captured_commands_or_become_valid(
    bad: dict[str, Any],
) -> None:
    raw = stream(step(2, "cat a")) + "\n" + json.dumps(bad)
    capture = census.capture_stream(raw, strict=False)
    assert capture["errors"]
    assert capture["commands"][0]["command"] == "cat a"
    with pytest.raises(ValueError):
        census.capture_stream(raw)


@pytest.mark.parametrize(
    "command",
    [
        "/home/private/person/program --private-secret",
        "gh issue private-subcommand 1",
        "cat --private-secret=file.txt a",
        "gh issue view 1 --json private_field",
        "gh issue view 1 --json=private_field",
        "python3 -c 'private source'",
        "cat /home/private/path",
        "rg 'private pattern' private-file.txt",
    ],
)
def test_public_forms_cannot_copy_private_executables_flags_or_operands(
    command: str,
) -> None:
    result = census.public_command_summary(
        {"commands": [{"step": 2, "command": command, "outcome": "unknown"}]}
    )
    assert result["attempted_count"] == 1
    assert "private" not in json.dumps(result)


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


def test_hook_restores_truncated_command_once_and_retains_provenance() -> None:
    full = 'python3 -c "' + "x" * 600 + '" && cat a'
    shown = full[:512] + "…"
    capture = census.capture_stream(
        stream(step(2, shown, "ERROR", error={"message": "denied by pre-tool hook"}))
    )
    record = json.dumps({"step": 2, "command": full, "decision": "deny"})
    census.restore_hook_commands(capture, record + "\n" + record)
    census.restore_hook_commands(capture, record)
    assert len(capture["commands"]) == 1
    command = capture["commands"][0]
    assert command["command"] == command["parameters"]["CommandLine"] == full
    assert command["stream_command"] == shown
    assert command["form"] == "python3 -c <CODE> && cat <ARG>"
    assert command["grammar_allowed"] is False
    assert command["outcome"] == "hook_denied"
    assert command["command_capture_source"] == "pre_tool_hook"
    assert capture["restored_stream_commands"] == 1


@pytest.mark.parametrize("fault", ["changed", "missing", "conflicting", "executed"])
def test_hook_disagreement_cannot_be_hidden_as_truncation(fault: str) -> None:
    capture = census.capture_stream(stream(step(2, "cat a", "DONE", output="ok")))
    record = {"step": 2, "command": "cat a", "decision": "allow"}
    if fault == "changed":
        record["command"] = "cat b"
    elif fault == "missing":
        record["step"] = 4
    elif fault == "executed":
        record["decision"] = "deny"
    raw = json.dumps(record)
    if fault == "conflicting":
        raw += "\n" + json.dumps({**record, "command": "cat b"})
    with pytest.raises(ValueError):
        census.restore_hook_commands(capture, raw)


def test_production_requires_a_hook_for_every_stream_command() -> None:
    capture = census.capture_stream(stream(step(2, "ls -la", "DONE", output="listing")))
    with pytest.raises(ValueError, match="no hook decision"):
        census.restore_hook_commands(capture, [], require_hook=True)


def test_read_hook_records_never_reconstruct_a_shell_command() -> None:
    capture = census.capture_stream(
        stream(
            read_step(
                1,
                "../private-file",
                "ERROR",
                error={
                    "type": "synthetic_denial",
                    "message": "Denied by pre-tool hook",
                },
            ),
            step(2, "cat a", "DONE", output="ok"),
        )
    )
    records = [
        {
            "tool": "read_file",
            "step": 1,
            "path": "../private-file",
            "command": None,
            "decision": "deny",
        },
        {"tool": "run_command", "step": 2, "command": "cat a", "decision": "allow"},
    ]
    census.restore_hook_commands(capture, records, require_hook=True)
    assert capture["commands"][0]["hook_decision"] == "allow"
    assert capture["reads"][0]["hook_decision"] == "deny"


def test_a_denial_error_cannot_hide_execution_output() -> None:
    capture = census.capture_stream(
        stream(
            step(
                2,
                "ls -la",
                "ERROR",
                output="Command exited with code 1",
                error={"message": "denied by pre-tool hook"},
            )
        )
    )
    with pytest.raises(ValueError, match="observed executing"):
        census.restore_hook_commands(
            capture,
            [{"step": 2, "command": "ls -la", "decision": "deny"}],
            require_hook=True,
        )


def test_read_observations_preserve_shapes_and_deduplicate_updates() -> None:
    capture = census.capture_stream(
        stream(
            read_step(2, "../not-available.txt"),
            read_step(
                2,
                "../not-available.txt",
                "ERROR",
                error={
                    "type": "synthetic_denial",
                    "message": "Denied by pre-tool hook",
                },
            ),
            read_step(
                4,
                "inside.txt",
                parameters={"Path": "inside.txt", "StartLine": 2, "EndLine": 5},
            ),
            read_step(4, "inside.txt", "DONE", output="synthetic content"),
        )
    )
    assert not capture["commands"]
    assert [(entry["state"], entry["outcome"]) for entry in capture["reads"]] == [
        ("ERROR", "hook_denied"),
        ("DONE", "completed"),
    ]
    assert capture["reads"][1]["parameters"] == {
        "Path": "inside.txt",
        "StartLine": 2,
        "EndLine": 5,
    }
    summary = census.public_command_summary(capture)
    assert summary["read_file"]["attempted_count"] == 2
    assert "inside.txt" not in json.dumps(
        summary
    ) and "not-available.txt" not in json.dumps(summary)


@pytest.mark.parametrize(
    "fault", ["orphan", "conflict", "wrong_tool", "ambiguous", "hidden_done"]
)
def test_read_reconciliation_rejects_ambiguous_or_contradictory_evidence(
    fault: str,
) -> None:
    steps = [read_step(2, "inside.txt", "DONE", output="synthetic content")]
    records = [
        {"tool": "read_file", "step": 2, "path": "inside.txt", "decision": "allow"}
    ]
    if fault == "orphan":
        records[0]["step"] = 4
    elif fault == "conflict":
        records.append({**records[0], "decision": "deny"})
    elif fault == "wrong_tool":
        records = [
            {
                "tool": "run_command",
                "step": 2,
                "command": "cat inside.txt",
                "decision": "allow",
            }
        ]
    elif fault == "ambiguous":
        steps.append({**steps[0], "conversation_id": "another-conversation"})
    else:
        records[0]["decision"] = "deny"
        steps.append(
            read_step(
                2,
                "inside.txt",
                "ERROR",
                output="",
                error={
                    "type": "synthetic_denial",
                    "message": "Denied by pre-tool hook",
                },
            )
        )
    capture = census.capture_stream(stream(*steps))
    with pytest.raises(ValueError):
        census.restore_hook_commands(capture, records, require_hook=True)


@pytest.mark.parametrize("path", [None, 17])
def test_invalid_read_target_never_becomes_a_valid_capture(path: Any) -> None:
    raw = stream(read_step(2, path))
    capture = census.capture_stream(raw, strict=False)
    assert capture["errors"]
    with pytest.raises(ValueError):
        census.capture_stream(raw)
