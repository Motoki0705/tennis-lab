"""AutoDream trigger boundaries and the input contracts at its seams."""

from __future__ import annotations

import importlib
import json
import sys
from datetime import timedelta
from pathlib import Path

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path.insert(0, str(OPS))
autodream = importlib.import_module("shared_memory.autodream")
plans = importlib.import_module("shared_memory.autodream_plan")
states = importlib.import_module("shared_memory.autodream_state")
checks = importlib.import_module("shared_memory.validate")


@pytest.mark.parametrize(
    ("hours", "main_commits", "memory_commits", "expected"),
    [
        (167, 15, 0, ()),
        (167, 10, 3, ()),
        (168, 9, 2, ()),
        (168, 10, 0, ("main_commits",)),
        (168, 3, 3, ("memory_commits",)),
        (240, 10, 3, ("main_commits", "memory_commits")),
        (None, 9, 2, ()),
        (None, 10, 0, ("main_commits",)),
        (None, 3, 3, ("memory_commits",)),
    ],
)
def test_periodic_trigger(
    hours: int | None, main_commits: int, memory_commits: int, expected: tuple[str, ...]
) -> None:
    trigger = autodream.decide_trigger(
        elapsed=None if hours is None else timedelta(hours=hours),
        main_commits=main_commits,
        memory_commits=memory_commits,
        report=checks.AuditReport("2026-01-01", [], [], [], []),
        generated_drift=0,
    )
    assert trigger.reasons == expected
    assert trigger.elapsed_seconds == (None if hours is None else hours * 3600)


@pytest.mark.parametrize(
    "reason",
    [
        "broken_citations",
        "stale_entries",
        "limits_exceeded",
        "generated_drift",
        "forced",
    ],
)
def test_immediate_trigger_bypasses_age_and_activity(reason: str) -> None:
    citation = checks.Problem("entry.md", "citation", "missing anchor")
    stale = checks.StaleEntry("entry.md", "2025-10-02", 91)
    limit = checks.LimitUsage("entry.md", "lines", 41, 40, 1.025, True)
    report = checks.AuditReport(
        "2026-01-01",
        [stale] if reason == "stale_entries" else [],
        [citation] if reason == "broken_citations" else [],
        [limit] if reason == "limits_exceeded" else [],
        [citation] if reason == "broken_citations" else [],
    )
    trigger = autodream.decide_trigger(
        elapsed=timedelta(hours=1),
        main_commits=0,
        memory_commits=0,
        report=report,
        generated_drift=int(reason == "generated_drift"),
        force=reason == "forced",
    )
    assert trigger.reasons == (reason,)


def test_reaching_capacity_is_only_a_signal() -> None:
    report = checks.AuditReport(
        "2026-01-01",
        [],
        [],
        [checks.LimitUsage("index.md", "entries", 30, 30, 1, False)],
        [],
    )
    trigger = autodream.decide_trigger(
        elapsed=timedelta(hours=1),
        main_commits=0,
        memory_commits=0,
        report=report,
        generated_drift=0,
    )
    assert trigger.reasons == ()
    assert trigger.limits_exceeded == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"repair_index": 1}, "JSON Schema"),
        ({"other": None}, "JSON Schema"),
        ({"operations": {}}, "JSON Schema"),
        (
            {"operations": [{"op": "delete", "layer": "gotcha", "slug": "old"}]},
            "JSON Schema",
        ),
        (
            {
                "operations": [
                    {
                        "op": "delete",
                        "layer": "gotcha",
                        "slug": "old",
                        "reason": "Retired.",
                        "body": "Ignored.",
                    }
                ]
            },
            "JSON Schema",
        ),
        (
            {
                "operations": [
                    {
                        "op": "update",
                        "layer": "gotcha",
                        "slug": "old",
                        "reason": "Rechecked.",
                        "expected_sha256": None,
                    }
                ]
            },
            "JSON Schema",
        ),
        (
            {
                "operations": [
                    {
                        "op": "update",
                        "layer": "gotcha",
                        "slug": "old",
                        "reason": "Rechecked.",
                        "citations": [{"path": "AGENTS.md", "extra": "ignored"}],
                    }
                ]
            },
            "JSON Schema",
        ),
        ({"summary": "Line 1\nLine 2"}, "one line"),
        ({"summary": "password=fixture-only"}, "credential assignment"),
    ],
)
def test_plan_rejects_wrong_schema_without_echoing_values(
    change: dict[str, object], message: str
) -> None:
    raw = json.dumps(
        {"summary": "Checked.", "operations": [], "repair_index": False, **change}
    )
    with pytest.raises(ValueError, match=message) as failure:
        plans.parse_plan(raw)
    assert "fixture-only" not in str(failure.value)


@pytest.mark.parametrize(
    "raw",
    [
        'Notes\n{"summary":"Checked","operations":[],"repair_index":false}',
        '{"summary":"Checked","summary":"Duplicate","operations":[],"repair_index":false}',
        '{"summary":NaN,"operations":[],"repair_index":false}',
    ],
)
def test_plan_never_extracts_or_repairs_malformed_json(raw: str) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        plans.parse_plan(raw)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("{", "invalid AutoDream state JSON"),
        ('{"schema_version":99}', "unsupported AutoDream state schema"),
        ('{"schema_version":true}', "unsupported AutoDream state schema"),
        ('{"schema_version":1}', "invalid AutoDream state fields"),
    ],
)
def test_bad_state_is_preserved(tmp_path: Path, raw: str, message: str) -> None:
    state = tmp_path / states.STATE_FILE
    state.write_text(raw)
    with pytest.raises(ValueError, match=message):
        states.read_state(state)
    assert state.read_text() == raw


def test_version_one_state_preserves_completed_baseline(tmp_path: Path) -> None:
    path = tmp_path / states.STATE_FILE
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "last_run_at": "2026-01-03T00:00:00+00:00",
                "last_completed_at": "2026-01-02T00:00:00+00:00",
                "last_consolidated_sha": "a" * 40,
                "last_outcome": "skipped_trigger",
                "last_trigger": {},
                "last_pr": None,
                "last_merge_sha": None,
                "last_error": None,
            }
        )
    )
    state = states.read_state(path)
    assert state.schema_version == 2 and state.pending_publication is None
    assert state.last_completed_at == "2026-01-02T00:00:00+00:00"
    assert state.last_consolidated_sha == "a" * 40


def test_state_replace_failure_preserves_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / states.STATE_FILE
    record = states.State(
        last_run_at="2026-01-01T00:00:00+00:00", last_outcome="skipped_trigger"
    )
    states.write_state(state, record)
    before = state.read_bytes()

    def refuse(self: Path, target: Path) -> Path:
        raise OSError("replace refused")

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="replace refused"):
        states.write_state(state, record)
    assert state.read_bytes() == before
    assert list(tmp_path.iterdir()) == [state]
