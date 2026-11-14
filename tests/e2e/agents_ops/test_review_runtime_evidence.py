"""Review recovery checks real runner artifacts produced by an offline Codex CLI."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_agent_runtime import Harness
from tests.e2e.agents_ops.test_review_model_selection import choice_files

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture
def runner_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> tuple[Path, Any]:
    for path in (OPS, OPS / "review", OPS / "lib"):
        monkeypatch.syspath_prepend(str(path))
    recovery = importlib.import_module("review_recovery")
    h = Harness(tmp_path)
    _, selection_path, _ = choice_files(h, monkeypatch, "gpt-6-astra")
    schema = tmp_path / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
                "additionalProperties": False,
            }
        )
    )
    scenario = getattr(request, "param", "success")
    proc = h.run(
        "agent-runtime",
        scenario=scenario,
        extra=(
            "--profile",
            "review",
            "--review-selection-file",
            str(selection_path),
            "--mode",
            "read-only",
            "--json-schema",
            str(schema),
            "--out",
            str(tmp_path / "agent_output.md"),
        ),
    )
    assert proc.returncode == (0 if scenario in {"success", "web_search"} else 1), (
        proc.stderr
    )
    (tmp_path / "prompt.md").write_bytes(
        (tmp_path / "agent-runtime-prompt.md").read_bytes()
    )
    return tmp_path, recovery


def runtime(path: Path) -> dict[str, Any]:
    value = json.loads((path / "agent-runtime/result.json").read_text())
    assert isinstance(value, dict)
    return value


def save(path: Path, record: dict[str, Any]) -> None:
    (path / "agent-runtime/result.json").write_text(json.dumps(record))


def events(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (path / "agent-runtime/events.jsonl").read_text().splitlines()
    ]


def save_events(path: Path, rows: list[dict[str, Any]]) -> None:
    (path / "agent-runtime/events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


@pytest.mark.parametrize(
    "runner_review",
    [
        "different_final",
        "wrong_model",
        "wrong_effort",
        "missing_transcript",
        "missing_completion",
    ],
    indirect=True,
)
def test_runner_validation_failures_are_not_recoverable(
    runner_review: tuple[Path, Any],
) -> None:
    path, recovery = runner_review
    record = runtime(path)
    assert record["turn_end_reason"] in {"invalid_output", "metadata_error"}
    assert record["cleanup_complete"] is True
    assert not (path / "agent_output.md").exists()
    with pytest.raises(ValueError, match="not recoverable"):
        recovery.runtime_evidence(path)


@pytest.mark.parametrize("reason", ["completed", "runtime_error", "cli_error"])
def test_matching_completed_events_allow_only_verified_runtime_failures(
    runner_review: tuple[Path, Any], reason: str
) -> None:
    path, recovery = runner_review
    record = runtime(path)
    if reason != "completed":
        record.update(
            result="failed", turn_end_reason=reason, exit_code=7, cli_exit_code=7
        )
        (path / "agent_output.md").unlink()
        if reason == "runtime_error":
            error = {"type": "error", "message": "transport error after completion"}
            record["runtime"]["runtime_errors"] = [error]
            save_events(path, [*events(path), error])
        save(path, record)
    assert recovery.runtime_evidence(path)["turn_end_reason"] == reason


@pytest.mark.parametrize(
    "reason",
    [
        "invalid_output",
        "metadata_error",
        "usage_limit",
        "timeout",
        "stopped",
        "descendants_reaped",
        "cleanup_failed",
        "startup_failed",
        None,
        "unknown_reason",
    ],
)
def test_runtime_reason_is_not_waived_by_a_valid_final(
    runner_review: tuple[Path, Any], reason: str | None
) -> None:
    path, recovery = runner_review
    record = runtime(path)
    record.update(result="failed", turn_end_reason=reason, exit_code=1)
    save(path, record)
    with pytest.raises(ValueError, match="not recoverable"):
        recovery.runtime_evidence(path)


@pytest.mark.parametrize(
    "change",
    [
        "final",
        "session",
        "turn",
        "effective_model",
        "missing_completion",
        "evidence_error",
        "recovery_error",
        "post_completion_message",
        "runtime_errors",
        "missing_events",
    ],
)
def test_inconsistent_or_missing_runtime_evidence_is_rejected(
    runner_review: tuple[Path, Any], change: str
) -> None:
    path, recovery = runner_review
    record, rows = runtime(path), events(path)
    if change == "final":
        # Both JSON objects satisfy the pinned schema; content correspondence is
        # essential even when the runner never copied final into --out.
        (path / "agent_output.md").unlink()
        (path / "agent-runtime/final.md").write_text('{"message":"approve"}')
    elif change == "session":
        next(e for e in rows if e["type"] == "thread.started")["thread_id"] = (
            "another-session"
        )
    elif change == "turn":
        next(e for e in rows if e["type"] == "turn.started")["turn_id"] = "another-turn"
    elif change == "effective_model":
        record["runtime"]["effective"]["model"]["value"] = "another-model"
    elif change == "missing_completion":
        rows = [e for e in rows if e["type"] != "turn.completed"]
    elif change in {"evidence_error", "recovery_error"}:
        record[change] = "execution destination/effective settings disagree"
    elif change == "post_completion_message":
        rows.append(next(e for e in rows if e["type"] == "item.completed"))
    elif change == "runtime_errors":
        record["runtime"]["runtime_errors"] = [
            {"type": "error", "message": "unrecorded error"}
        ]
    save(path, record)
    save_events(path, rows)
    if change == "missing_events":
        (path / "agent-runtime/events.jsonl").unlink()
    with pytest.raises((ValueError, OSError)):
        recovery.runtime_evidence(path)


@pytest.mark.parametrize("runner_review", ["web_search"], indirect=True)
def test_recovery_replays_the_shared_codex_event_boundary(
    runner_review: tuple[Path, Any],
) -> None:
    path, recovery = runner_review
    raw = (path / "agent-runtime/events.jsonl").read_text()
    assert '"web_search"' in raw and '"id":"item_1"' in raw
    assert recovery.runtime_evidence(path)["result"] == "completed"
