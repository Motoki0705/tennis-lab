"""Immutable v1 records and v2 result observations with strict object keys."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Callable
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from worker_logs.journal import PublicationError, decode, require
from worker_logs.schema import digest, fields, integer, json_bytes

VERSION = 1
OPERATIONS = (
    "continue_ai",
    "start_ai",
    "retry_fetch",
    "retry_publication",
    "review",
    "merge",
    "fetch",
    "verify",
    "publish",
    "stop",
    "resume",
    "handoff",
    "record",
)
ACTION_STATES = (
    "decided_not_executed",
    "requested",
    "rejected_before_start",
    "running",
    "ended",
    "unknown",
)
CHANGE_FIELDS = {"code", "requirements", "instruction", "conditions", "verified_scope"}
COMMON = {"schema_version", "id", "kind", "scope", "time", "data"}
DATA_FIELDS = {
    "result": {
        "action_id",
        "previous_result_id",
        "verifies",
        "status",
        "summary",
        "changes",
        "evidence",
        "observer",
        "final_report",
        "auxiliary",
        "repeat",
    },
    "decision": {
        "inputs",
        "basis",
        "evaluation",
        "reason",
        "next_actions",
        "reuse_scope",
        "recheck_scope",
        "unresolved",
        "resume_conditions",
        "source",
    },
    "action": {
        "decision_id",
        "plan_id",
        "decision_missing_reason",
        "previous_action_id",
        "operation",
        "state",
        "summary",
        "result_ids",
    },
    "correction": {"target_id", "replacement_id", "reason", "source"},
}


def new_id() -> str:
    """Random public identity, independent of private IDs and keys."""
    return uuid4().hex


def text(value: Any) -> None:
    require(isinstance(value, str) and bool(value.strip()), "record_schema")


def choice(value: Any, options: tuple[str, ...]) -> None:
    require(isinstance(value, str) and value in options, "record_schema")


def public_id(value: Any) -> None:
    require(digest(value, 32), "invalid_public_key")
    parsed = UUID(hex=value)
    require(parsed.version == 4 and parsed.hex == value, "invalid_public_key")


def nullable_id(value: Any) -> None:
    if value is not None:
        public_id(value)


def ids(value: Any) -> None:
    require(isinstance(value, list), "record_schema")
    for item in value:
        public_id(item)
    require(len(value) == len(set(value)), "invalid_log_reference")


def texts(value: Any) -> None:
    require(isinstance(value, list), "record_schema")
    for item in value:
        text(item)


def timestamp(value: Any) -> datetime:
    text(value)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise PublicationError("record_time") from None
    require(parsed.tzinfo is not None, "record_time")
    return parsed


def available(value: Any, check: Callable[[Any], Any]) -> None:
    require(isinstance(value, dict), "record_schema")
    if value.get("status") == "observed":
        fields(value, {"status", "value"})
        check(value["value"])
    else:
        fields(value, {"status", "reason"})
        require(value["status"] == "unavailable", "record_schema")
        text(value["reason"])


def evidence(value: Any) -> None:
    require(isinstance(value, dict), "record_source")
    if value.get("kind") == "public":
        fields(value, {"kind", "url", "revision", "content_sha256"})
        require(
            isinstance(value["url"], str) and value["url"].startswith("https://"),
            "record_source",
        )
        require(
            value["revision"] is None or digest(value["revision"], 40), "record_source"
        )
        require(
            value["content_sha256"] is None or digest(value["content_sha256"]),
            "record_source",
        )
        require(
            value["revision"] is not None or value["content_sha256"] is not None,
            "record_source",
        )
    else:
        fields(value, {"kind", "reason"})
        require(value["kind"] == "unavailable", "record_source")
        text(value["reason"])


def evidence_list(value: Any) -> None:
    require(isinstance(value, list), "record_source")
    for item in value:
        evidence(item)


def scope(value: Any) -> None:
    fields(
        value, {"repo", "parent", "issue", "activity_key", "delegation", "review_key"}
    )
    require(
        isinstance(value["repo"], str)
        and bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value["repo"])),
        "record_repository",
    )
    require(integer(value["parent"]), "record_issue")
    require(value["issue"] is None or integer(value["issue"]), "record_issue")
    public_id(value["activity_key"])
    nullable_id(value["review_key"])
    delegation = value["delegation"]
    if delegation is not None:
        fields(delegation, {"key", "turn"})
        require(
            isinstance(delegation["key"], str)
            and bool(re.fullmatch(r"d-[a-f0-9]{32}", delegation["key"]))
            and integer(delegation["turn"])
            and value["issue"] is not None,
            "record_turn",
        )


def source(value: Any) -> None:
    fields(value, {"kind", "actor_key", "evidence"})
    choice(value["kind"], ("settings", "human_instruction", "parent_choice"))
    public_id(value["actor_key"])
    evidence_list(value["evidence"])
    require(bool(value["evidence"]), "record_source")


def nonnegative(value: Any) -> None:
    require(
        type(value) in (int, float)
        and value >= 0
        and (type(value) is int or math.isfinite(value)),
        "record_schema",
    )


def token_count(value: Any) -> None:
    require(type(value) is int and value >= 0, "record_schema")


def measurement(value: Any) -> None:
    fields(value, {"value", "source"})
    nonnegative(value["value"])
    evidence(value["source"])


def selected_model(value: Any) -> None:
    fields(value, {"model", "effort", "specified_by", "source"})
    text(value["model"])
    text(value["effort"])
    choice(value["specified_by"], ("settings", "human_instruction", "parent_choice"))
    evidence(value["source"])


def effective_model(value: Any) -> None:
    fields(value, {"model", "effort", "source"})
    available(value["model"], text)
    available(value["effort"], text)
    evidence(value["source"])


def usage(value: Any) -> None:
    fields(
        value,
        {
            "basis",
            "counter_key",
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "total_tokens",
            "source",
        },
    )
    choice(value["basis"], ("turn", "cumulative"))
    public_id(value["counter_key"])
    for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        available(value[key], token_count)
    evidence(value["source"])


def auxiliary(value: Any) -> None:
    checks = {
        "selected_model": selected_model,
        "effective_model": effective_model,
        "elapsed_seconds": measurement,
        "wait_seconds": measurement,
        "usage": usage,
    }
    fields(value, set(checks))
    for key, check in checks.items():
        available(value[key], check)


def result(value: dict[str, Any]) -> None:
    nullable_id(value["action_id"])
    nullable_id(value["previous_result_id"])
    ids(value["verifies"])
    choice(
        value["status"],
        ("success", "partial", "failure", "rejected", "interrupted", "unknown"),
    )
    text(value["summary"])
    fields(value["changes"], CHANGE_FIELDS)
    for changed in value["changes"].values():
        choice(changed, ("changed", "unchanged", "unconfirmed"))
    require(
        value["previous_result_id"] is not None
        or all(v == "unconfirmed" for v in value["changes"].values()),
        "invalid_log_reference",
    )
    evidence_list(value["evidence"])
    require(bool(value["evidence"]), "record_source")
    choice(value["observer"], ("machine", "worker_report", "parent_verification"))
    require(
        not value["verifies"] or value["observer"] == "parent_verification",
        "record_schema",
    )
    available(value["final_report"], evidence)
    auxiliary(value["auxiliary"])
    if value["repeat"] is not None:
        fields(value["repeat"], {"kind", "condition_key", "target"})
        choice(value["repeat"]["kind"], ("state_check", "fetch_retry"))
        public_id(value["repeat"]["condition_key"])
        evidence(value["repeat"]["target"])
        require(value["observer"] == "machine", "record_schema")


def decision(value: dict[str, Any]) -> None:
    ids(value["inputs"])
    evidence_list(value["basis"])
    require(
        bool(value["inputs"]) or any(e["kind"] == "public" for e in value["basis"]),
        "invalid_log_reference",
    )
    choice(
        value["evaluation"],
        (
            "adopt",
            "request_changes",
            "investigate",
            "reuse",
            "retry",
            "continue",
            "wait",
            "stop",
            "complete",
            "reject",
            "handoff",
        ),
    )
    text(value["reason"])
    require(isinstance(value["next_actions"], list), "record_schema")
    plans = []
    for plan in value["next_actions"]:
        fields(
            plan,
            {
                "id",
                "operation",
                "summary",
                "actor",
                "target",
                "instruction",
                "conditions",
                "expected_result",
            },
        )
        public_id(plan["id"])
        plans.append(plan["id"])
        choice(plan["operation"], OPERATIONS)
        choice(plan["actor"], ("parent", "worker", "external"))
        for key in (
            "summary",
            "target",
            "instruction",
            "conditions",
            "expected_result",
        ):
            text(plan[key])
    require(len(plans) == len(set(plans)), "invalid_log_reference")
    for key in ("reuse_scope", "recheck_scope", "unresolved", "resume_conditions"):
        texts(value[key])
    require(
        value["evaluation"] not in ("wait", "stop") or bool(value["resume_conditions"]),
        "record_decision",
    )
    source(value["source"])


def action(value: dict[str, Any]) -> None:
    nullable_id(value["decision_id"])
    nullable_id(value["plan_id"])
    if value["decision_id"] is None:
        require(value["plan_id"] is None, "invalid_log_reference")
        text(value["decision_missing_reason"])
    else:
        public_id(value["plan_id"])
        require(value["decision_missing_reason"] is None, "record_schema")
    nullable_id(value["previous_action_id"])
    choice(value["operation"], OPERATIONS)
    choice(value["state"], ACTION_STATES)
    text(value["summary"])
    ids(value["result_ids"])
    require(
        not value["result_ids"]
        or value["state"] in ("ended", "rejected_before_start", "unknown"),
        "record_schema",
    )


def correction(value: dict[str, Any]) -> None:
    public_id(value["target_id"])
    nullable_id(value["replacement_id"])
    require(value["target_id"] != value["replacement_id"], "invalid_log_reference")
    text(value["reason"])
    source(value["source"])


def validate_record(value: Any) -> dict[str, Any]:
    """Validate structure, without treating private captured text as publishable."""
    if isinstance(value, (bytes, str)):
        value = decode(value)
    fields(value, COMMON)
    require(
        type(value["schema_version"]) is int and value["schema_version"] in {1, 2},
        "record_version",
    )
    public_id(value["id"])
    choice(value["kind"], tuple(DATA_FIELDS))
    scope(value["scope"])
    fields(value["time"], {"occurred_at", "received_at", "recorded_at"})
    for key in ("occurred_at", "received_at"):
        available(value["time"][key], timestamp)
    timestamp(value["time"]["recorded_at"])
    extended = value["schema_version"] == 2
    require(not extended or value["kind"] == "result", "record_version")
    fields(
        value["data"],
        DATA_FIELDS[value["kind"]] | ({"observation"} if extended else set()),
    )
    {
        "result": result,
        "decision": decision,
        "action": action,
        "correction": correction,
    }[value["kind"]](value["data"])
    if extended:
        from .trace_schema import validate

        validate(
            value["data"]["observation"], value["scope"], value["data"]["final_report"]
        )
    return copy.deepcopy(value)


__all__ = ["new_id", "validate_record", "json_bytes"]
