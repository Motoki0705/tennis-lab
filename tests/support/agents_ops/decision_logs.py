"""Small causal histories with explicit missing evidence, no Git or agent runs."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
schema: Any = importlib.import_module("decision_logs.schema")
graph: Any = importlib.import_module("decision_logs.graph")
projection: Any = importlib.import_module("decision_logs.projection")
journal: Any = importlib.import_module("decision_logs.journal")
aggregation: Any = importlib.import_module("decision_logs.aggregation")
safety: Any = importlib.import_module("worker_logs.safety")
PublicationError: Any = importlib.import_module("worker_logs.journal").PublicationError

ACTIVITY = "ac71c7009ad14dd29786e10c958a0be1"
ACTOR = "2172d88be10a4c3b931ad63228d5784d"
CONDITION = "f0bff9febfa5411f880dc5cde6271292"
AT = "2026-11-10T08:00:00+00:00"


def missing(reason: str = "not_captured") -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def observed(value: Any) -> dict[str, Any]:
    return {"status": "observed", "value": value}


def evidence() -> dict[str, Any]:
    return {
        "kind": "public",
        "url": "https://github.com/o/r/issues/10#issuecomment-123",
        "revision": None,
        "content_sha256": "b" * 64,
    }


def source() -> dict[str, Any]:
    return {"kind": "parent_choice", "actor_key": ACTOR, "evidence": [evidence()]}


def base(kind: str, data: dict[str, Any], *, at: str = AT) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": schema.new_id(),
        "kind": kind,
        "scope": {
            "repo": "o/r",
            "parent": 1,
            "issue": 10,
            "activity_key": ACTIVITY,
            "delegation": {"key": "d-" + "a" * 32, "turn": 1},
            "review_key": None,
        },
        "time": {
            "occurred_at": missing("source_clock_unavailable"),
            "received_at": observed(at),
            "recorded_at": at,
        },
        "data": data,
    }


def result(
    *,
    previous: dict[str, Any] | None = None,
    action: dict[str, Any] | None = None,
    at: str = AT,
) -> dict[str, Any]:
    return base(
        "result",
        {
            "action_id": action["id"] if action else None,
            "previous_result_id": previous["id"] if previous else None,
            "verifies": [],
            "status": "failure",
            "summary": "CI log retrieval failed; the code was already checked.",
            "changes": dict.fromkeys(
                schema.CHANGE_FIELDS, "unchanged" if previous else "unconfirmed"
            ),
            "evidence": [evidence()],
            "observer": "machine",
            "final_report": missing("missing_final"),
            "auxiliary": {
                k: missing("source_did_not_report")
                for k in (
                    "selected_model",
                    "effective_model",
                    "elapsed_seconds",
                    "wait_seconds",
                    "usage",
                )
            },
            "repeat": None,
        },
        at=at,
    )


def decision(
    inputs: list[dict[str, Any]] | None = None, *, operation: str = "retry_fetch"
) -> dict[str, Any]:
    return base(
        "decision",
        {
            "inputs": [r["id"] for r in inputs or []],
            "basis": [] if inputs else [evidence()],
            "evaluation": "retry",
            "reason": "Reuse the code check and fetch only the missing CI evidence.",
            "next_actions": [
                {
                    "id": schema.new_id(),
                    "operation": operation,
                    "summary": "Fetch the CI job log",
                    "actor": "parent",
                    "target": "PR 42, Python job",
                    "instruction": "Fetch the existing job log",
                    "conditions": "Same code and acceptance criteria",
                    "expected_result": "Pinned evidence or an explicit retrieval failure",
                }
            ],
            "reuse_scope": ["Existing code verification"],
            "recheck_scope": ["CI job evidence"],
            "unresolved": ["CI log not received"],
            "resume_conditions": [],
            "source": source(),
        },
    )


def action(
    decision: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
    state: str = "requested",
    results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plan = decision["data"]["next_actions"][0]
    return base(
        "action",
        {
            "decision_id": decision["id"],
            "plan_id": plan["id"],
            "decision_missing_reason": None,
            "operation": plan["operation"],
            "previous_action_id": previous["id"] if previous else None,
            "state": state,
            "summary": plan["summary"],
            "result_ids": [r["id"] for r in results or []],
        },
    )


def correction(
    target: dict[str, Any], replacement: dict[str, Any] | None = None
) -> dict[str, Any]:
    return base(
        "correction",
        {
            "target_id": target["id"],
            "replacement_id": replacement["id"] if replacement else None,
            "reason": "The earlier observation was incomplete; retain its original record.",
            "source": source(),
        },
    )


def history() -> list[dict[str, Any]]:
    initial = decision(operation="start_ai")
    request = action(initial)
    report = result(action=request)
    report["data"].update(
        observer="worker_report", status="partial", final_report=observed(evidence())
    )
    ended = action(initial, previous=request, state="ended", results=[report])
    followup = decision([report])
    fetch = action(followup, state="decided_not_executed")
    return [initial, request, report, ended, followup, fetch]


def repetitions(kind: str = "state_check") -> list[dict[str, Any]]:
    seed = result()
    first = result(previous=seed, at="2026-11-10T08:01:00+00:00")
    second = result(previous=first, at="2026-11-10T08:03:00+00:00")
    for record in (first, second):
        record["data"]["repeat"] = {
            "kind": kind,
            "condition_key": CONDITION,
            "target": evidence(),
        }
    return [seed, first, second]
