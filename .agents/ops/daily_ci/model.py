"""Versioned identities and the fixed maintenance window (clock injection is test-only)."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
LIVE = {"freezing", "active", "blocked"}


def utcnow() -> datetime:
    return datetime.now(UTC)


def instant(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("a timezone is required")
    return result


def window(now: datetime) -> tuple[str, datetime, datetime]:
    if now.tzinfo is None:
        raise ValueError("a timezone is required")
    day = now.astimezone(JST).date()
    return (
        f"daily-ci-{day.isoformat()}",
        datetime.combine(day, time(6), JST),
        datetime.combine(day, time(9), JST),
    )


def sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("a full lowercase commit SHA is required")
    return value


def positive(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("a positive integer is required")
    return value


def text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a nonempty string is required")
    return value


def request_id(repo: str, cycle_id: str, generation: int, workflow: str) -> str:
    return hashlib.sha256(
        f"{repo.casefold()}:{cycle_id}:{generation}:{workflow}".encode()
    ).hexdigest()[:32]


def generation(
    repo: str, cycle: dict[str, Any], target: str, kind: str
) -> dict[str, Any]:
    number = len(cycle["generations"]) + 1
    return {
        "number": number,
        "target_sha": sha(target),
        "kind": kind,
        "requests": {
            workflow: {
                "request_id": request_id(repo, cycle["cycle_id"], number, workflow),
                "state": "planned",
                "requested_at": None,
                "run_id": None,
                "attempt": None,
                "remote_status": None,
                "remote_conclusion": None,
                "observation": None,
                "error": None,
                "stop_state": None,
            }
            for workflow in cycle["workflows"]
        },
    }
