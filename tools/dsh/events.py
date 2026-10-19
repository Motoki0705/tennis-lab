"""Read pinned headless NDJSON, retaining valid prefixes on interrupted runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.dsh.accounting import TOKEN_FIELDS


class EventError(ValueError):
    """A diagnostic that contains no provider-controlled text."""


def _required_fields(event: dict[str, Any]) -> tuple[str, ...]:
    fields = {
        "session": ("sessionId", "cwd"),
        "status": ("phase", "turn"),
        "text": ("text",),
        "thinking": ("text",),
        "tool_call": ("callId", "tool", "input"),
        "tool_result": ("callId", "status", "result"),
        "final": ("text",),
        "error": ("message",),
    }[event["type"]]
    if event["type"] == "status":
        if event.get("phase") in {"step_start", "step_end"}:
            fields += ("step",)
        elif event.get("phase") == "turn_end":
            fields += ("reason",)
    return fields


def _validate_bounded_fields(event: dict[str, Any]) -> bool:
    """Check pinned boundJsonLine shapes; return whether only type survived."""
    kind = event["type"]
    truncated = event.get("truncated") is True
    if "truncated" in event and (not truncated or kind == "final"):
        # finish() writes final directly, bypassing every bound and marker.
        raise EventError("Invalid NDJSON truncation marker.")
    if truncated and set(event) == {"type", "truncated"}:
        return True
    missing = set(_required_fields(event)) - event.keys()
    # The scalar-only form drops input and reason, never individual scalars.
    droppable = {"tool_call": {"input"}, "status": {"reason"}}.get(kind, set())
    if missing - (droppable if truncated else set()):
        raise EventError("Missing NDJSON event field.")
    string_fields = {
        "session": ("sessionId", "cwd"),
        "text": ("text",),
        "thinking": ("text",),
        "final": ("text",),
        "error": ("message",),
    }.get(kind, ())
    if any(not isinstance(event[field], str) for field in string_fields) or (
        kind == "status"
        and event.get("phase") == "turn_end"
        and "reason" in event
        and not isinstance(event["reason"], dict)
    ):
        raise EventError("Invalid NDJSON event field.")
    return False


def validate_event(event: Any, *, fresh: bool, allow_truncated: bool) -> None:
    allowed = {
        "session",
        "status",
        "text",
        "thinking",
        "tool_call",
        "tool_result",
        "final",
        "error",
    }
    if (
        not isinstance(event, dict)
        or event.get("type") not in allowed
        or (event.get("truncated") and not allow_truncated)
    ):
        raise EventError("Unexpected NDJSON event.")
    if allow_truncated and _validate_bounded_fields(event):
        return
    kind = event["type"]
    if kind == "status" and (
        event.get("phase") not in {"turn_start", "step_start", "step_end", "turn_end"}
        or type(event.get("turn")) is not int
        or event["turn"] < 1
        or (fresh and event["turn"] != 1)
    ):
        raise EventError("Invalid NDJSON status.")
    if (
        kind == "status"
        and event["phase"] in {"step_start", "step_end"}
        and (type(event.get("step")) is not int or event["step"] < 1)
    ):
        raise EventError("Invalid NDJSON step.")
    if kind in {"tool_call", "tool_result"} and not isinstance(
        event.get("callId"), str
    ):
        raise EventError("Invalid NDJSON callId.")
    if kind == "tool_call" and (
        not isinstance(event.get("tool"), str)
        # parseArguments can emit any JSON value or raw argument text. Only the
        # smoke requires a dict; the runner checks presence above, except when
        # boundJsonLine explicitly dropped the structured input.
        or not allow_truncated
        and not isinstance(event.get("input"), dict)
    ):
        raise EventError("Invalid NDJSON tool input.")
    if kind == "tool_result" and (
        not isinstance(event.get("result"), str)
        or event.get("status") not in {"completed", "error"}
    ):
        raise EventError("Invalid NDJSON tool result.")
    if kind == "status" and event["phase"] == "step_end" and "usage" in event:
        usage = event["usage"]
        if (
            not isinstance(usage, dict)
            or any(field not in usage for field in ("inputTokens", "outputTokens"))
            or any(
                type(usage[field]) is not int or usage[field] < 0
                for field in TOKEN_FIELDS
                if field in usage
            )
        ):
            raise EventError("Invalid NDJSON usage.")


def truncation_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe observed loss without guessing which retained values were cut."""
    truncated = [event for event in events if event.get("truncated") is True]
    return {
        "truncated_events": len(truncated),
        "truncated_event_types": dict(Counter(event["type"] for event in truncated)),
        "truncated_missing_fields": dict(
            Counter(
                f"{event['type']}.{field}"
                for event in truncated
                for field in _required_fields(event)
                if field not in event
            )
        ),
    }


def event_prefix(
    data: bytes, *, fresh: bool = False, allow_truncated: bool = True
) -> tuple[list[dict[str, Any]], bool]:
    events: list[dict[str, Any]] = []
    lines = data.splitlines(keepends=True)
    for line in lines:
        if not line.endswith(b"\n"):
            return events, False
        try:
            event = json.loads(line.decode("utf-8"))
            validate_event(event, fresh=fresh, allow_truncated=allow_truncated)
        except (UnicodeError, ValueError, TypeError):
            return events, False
        events.append(event)
    return events, bool(events)


def parse_events(path: Path) -> list[dict[str, Any]]:
    """The smoke's original strict fresh-session parsing contract."""
    events, complete = event_prefix(
        path.read_bytes(), fresh=True, allow_truncated=False
    )
    if not complete:
        raise EventError(
            "Invalid/incomplete NDJSON; inspect the private events/stderr."
        )
    return events
