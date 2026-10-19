"""Read pinned headless NDJSON, retaining valid prefixes on interrupted runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.dsh.accounting import TOKEN_FIELDS


class EventError(ValueError):
    """A diagnostic that contains no provider-controlled text."""


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
        or not isinstance(event.get("input"), dict)
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
