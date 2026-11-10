"""Explicit causal references, lifecycle snapshots and missing-state views."""

from __future__ import annotations

from collections import deque
from typing import Any

from worker_logs.journal import require

from .schema import validate_record

TRANSITIONS = {
    "decided_not_executed": {
        "decided_not_executed",
        "requested",
        "rejected_before_start",
        "unknown",
    },
    "requested": {"requested", "running", "rejected_before_start", "ended", "unknown"},
    "running": {"running", "ended", "unknown"},
    "unknown": {"unknown", "requested", "running", "ended", "rejected_before_start"},
    "ended": {"ended"},
    "rejected_before_start": {"rejected_before_start"},
}


def references(record: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Record dependencies, not timestamp-based guesses or plan identifiers."""
    data = record["data"]
    result: dict[str, tuple[str, ...]] = {}

    def add(key: str | None, kinds: tuple[str, ...]) -> None:
        if key is not None:
            require(key not in result or result[key] == kinds, "invalid_log_reference")
            result[key] = kinds

    if record["kind"] == "result":
        add(data["action_id"], ("action",))
        add(data["previous_result_id"], ("result",))
        for key in data["verifies"]:
            add(key, ("result",))
    elif record["kind"] == "decision":
        for key in data["inputs"]:
            add(key, ("result",))
    elif record["kind"] == "action":
        add(data["decision_id"], ("decision",))
        add(data["previous_action_id"], ("action",))
        for key in data["result_ids"]:
            add(key, ("result",))
    else:
        for field in ("target_id", "replacement_id"):
            add(data[field], ("result", "decision", "action", "correction"))
    return result


def validate_graph(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate a complete DAG, including action histories and reference types.

    An action snapshot precedes its results. A later terminal snapshot can link
    those results; results point to the earlier snapshot, avoiding a false cycle.
    """
    require(isinstance(records, list), "record_schema")
    rows = [validate_record(record) for record in records]
    by_id = {r["id"]: r for r in rows}
    require(len(by_id) == len(rows), "record_already_exists")
    dependencies = {key: references(row) for key, row in by_id.items()}
    outgoing: dict[str, list[str]] = {key: [] for key in by_id}
    degree = {key: len(refs) for key, refs in dependencies.items()}
    for key, refs in dependencies.items():
        row = by_id[key]
        for ref, kinds in refs.items():
            require(
                ref in by_id and by_id[ref]["kind"] in kinds, "invalid_log_reference"
            )
            require(
                all(
                    row["scope"][f] == by_id[ref]["scope"][f]
                    for f in ("repo", "parent")
                ),
                "invalid_log_reference",
            )
            outgoing[ref].append(key)
    ready = deque(key for key, n in degree.items() if n == 0)
    ordered = []
    while ready:
        key = ready.popleft()
        ordered.append(key)
        for successor in outgoing[key]:
            degree[successor] -= 1
            if degree[successor] == 0:
                ready.append(successor)
    require(len(ordered) == len(rows), "invalid_log_reference")

    plans: dict[str, str] = {}
    roots: dict[tuple[str, str], str] = {}
    successors: set[str] = set()
    ancestors: dict[str, set[str]] = {}
    for key in ordered:
        row = by_id[key]
        data = row["data"]
        if row["kind"] == "decision":
            for plan in data["next_actions"]:
                require(
                    plan["id"] not in plans and plan["id"] not in by_id,
                    "invalid_log_reference",
                )
                plans[plan["id"]] = key
        elif row["kind"] == "action":
            if data["decision_id"] is not None:
                decision = by_id[data["decision_id"]]
                plan = next(
                    (
                        p
                        for p in decision["data"]["next_actions"]
                        if p["id"] == data["plan_id"]
                    ),
                    None,
                )
                require(
                    plan is not None and plan["operation"] == data["operation"],
                    "invalid_log_reference",
                )
            prior = data["previous_action_id"]
            ancestors[key] = set()
            if prior is None and data["decision_id"] is not None:
                identity = (data["decision_id"], data["plan_id"])
                require(identity not in roots, "invalid_log_reference")
                roots[identity] = key
            elif prior is not None:
                before = by_id[prior]
                require(
                    before["scope"] == row["scope"]
                    and all(
                        before["data"][f] == data[f]
                        for f in ("decision_id", "plan_id", "operation")
                    )
                    and data["state"] in TRANSITIONS[before["data"]["state"]]
                    and set(before["data"]["result_ids"]) <= set(data["result_ids"])
                    and prior not in successors,
                    "invalid_log_reference",
                )
                successors.add(prior)
                ancestors[key] = ancestors[prior] | {prior}
            for result_id in data["result_ids"]:
                require(
                    by_id[result_id]["data"]["action_id"] in ancestors[key],
                    "invalid_log_reference",
                )
        elif row["kind"] == "result":
            action_id = data["action_id"]
            if action_id is not None:
                require(
                    by_id[action_id]["scope"] == row["scope"], "invalid_log_reference"
                )
                require(
                    by_id[action_id]["data"]["state"] != "decided_not_executed",
                    "invalid_log_reference",
                )
            if data["repeat"] is not None and action_id is not None:
                operation = by_id[action_id]["data"]["operation"]
                allowed = (
                    ("retry_fetch",)
                    if data["repeat"]["kind"] == "fetch_retry"
                    else ("fetch", "retry_fetch", "verify")
                )
                require(operation in allowed, "invalid_log_reference")
        elif row["kind"] == "correction":
            target = by_id[data["target_id"]]
            require(target["scope"] == row["scope"], "invalid_log_reference")
            if data["replacement_id"] is not None:
                replacement = by_id[data["replacement_id"]]
                require(
                    replacement["kind"] == target["kind"]
                    and replacement["scope"] == target["scope"],
                    "invalid_log_reference",
                )
    return {key: by_id[key] for key in ordered}


def states(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Derived observations for readers; no inferred receipt or acceptance."""
    rows = validate_graph(records)
    decided = {
        key
        for r in rows.values()
        if r["kind"] == "decision"
        for key in r["data"]["inputs"]
    }
    verified = {
        key
        for r in rows.values()
        if r["kind"] == "result" and r["data"]["observer"] == "parent_verification"
        for key in r["data"]["verifies"]
    }
    return {
        key: {
            "decision": "recorded" if key in decided else "undecided",
            "parent_confirmation": "recorded" if key in verified else "unrecorded",
            "final_report": r["data"]["final_report"]["status"],
        }
        for key, r in rows.items()
        if r["kind"] == "result"
    }
