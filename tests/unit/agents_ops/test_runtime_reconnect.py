"""A retry notification is recoverable only before a complete, verified final."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def events_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("lib.agent_events")


NOTICE: dict[str, Any] = {
    "type": "error",
    "message": "Reconnecting... 2/5 (workspace routing discovery timed out)",
}
START: list[dict[str, Any]] = [
    {"type": "thread.started", "thread_id": "probe-session"},
    {"type": "turn.started"},
]
FINAL: dict[str, Any] = {
    "type": "item.completed",
    "item": {"type": "agent_message", "text": "Completed report"},
}
DONE: dict[str, Any] = {
    "type": "turn.completed",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.mark.parametrize(
    "case", ["missing_runtime", "missing_errors", "null_errors", "explicit_empty"]
)
def test_missing_runtime_errors_is_not_an_explicit_empty_array(
    events_module: Any, case: str
) -> None:
    record: dict[str, Any] = {"runtime": {"runtime_errors": []}}
    if case == "missing_runtime":
        record.pop("runtime")
    elif case == "missing_errors":
        record["runtime"].pop("runtime_errors")
    elif case == "null_errors":
        record["runtime"]["runtime_errors"] = None
    assert events_module.runtime_errors_failed(record) is (case != "explicit_empty")


@pytest.mark.parametrize(
    "case",
    [
        "recovered",
        "unfinished",
        "fatal",
        "turn_failed",
        "late_error",
        "after_message",
        "before_start",
        "unknown_reason",
        "bad_counter",
    ],
)
def test_reconnection_requires_ordered_completion(
    events_module: Any, case: str
) -> None:
    notice = dict(NOTICE)
    if case == "unknown_reason":
        notice["message"] = "Reconnecting... 2/5 (unknown failure)"
    if case == "bad_counter":
        notice["message"] = (
            "Reconnecting... 6/5 (workspace routing discovery timed out)"
        )
    errors: list[dict[str, Any]] = (
        [{"type": "turn.failed", "error": {"message": "fatal"}}]
        if case == "turn_failed"
        else (
            [{"type": "error", "message": "authentication failed"}]
            if case == "fatal"
            else [notice]
        )
    )
    sequence: list[dict[str, Any]] = START + errors + [FINAL, DONE]
    if case == "unfinished":
        sequence = START + errors
    elif case == "late_error":
        sequence = START + [FINAL, DONE] + errors
    elif case == "after_message":
        sequence = START + [FINAL] + errors + [DONE]
    elif case == "before_start":
        sequence = errors + START + [FINAL, DONE]
    events = events_module.Events("codex", {"model": "fixture", "effort": "max"}, None)
    for event in sequence:
        events.feed(json.dumps(event).encode())
    assert events.failed is (case != "recovered")
    assert events.errors == errors  # Original diagnostics never disappear.


@pytest.mark.parametrize(
    "change",
    ["none", "exit", "reaped", "cleanup", "wrong_final", "wrong_errors", "failed"],
)
def test_saved_reconnection_diagnostics_need_original_evidence(
    events_module: Any,
    tmp_path: Path,
    change: str,
) -> None:
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in START + [NOTICE, FINAL, DONE])
    )
    (tmp_path / "final.md").write_text("Completed report")
    record: dict[str, Any] = {
        "run_dir": str(tmp_path),
        "result": "completed",
        "turn_end_reason": "completed",
        "exit_code": 0,
        "cli_exit_code": 0,
        "cleanup_complete": True,
        "terminated_pids": [],
        "contract": {
            "settings": {
                "agent": "codex",
                "selected": {"model": "fixture", "effort": "max"},
            }
        },
        "runtime": {
            "session_id": "probe-session",
            "turn_id": "probe-turn",
            "runtime_errors": [NOTICE],
        },
    }
    if change == "exit":
        record["cli_exit_code"] = 1
    if change == "reaped":
        record["terminated_pids"] = [123]
    if change == "cleanup":
        record["cleanup_complete"] = False
    if change == "wrong_final":
        (tmp_path / "final.md").write_text("Unrelated final")
    if change == "wrong_errors":
        record["runtime"]["runtime_errors"] = [{"type": "error", "message": "fatal"}]
    if change == "failed":
        record.update(result="failed", turn_end_reason="runtime_error", exit_code=1)
    assert events_module.runtime_errors_failed(record) is (change != "none")
