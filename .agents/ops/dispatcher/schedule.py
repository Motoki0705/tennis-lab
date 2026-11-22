"""Twice-daily wall-clock slots. Never trigger judgment from issue changes."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lib.state_files import read_state_text
from parent.policy import object_value
from parent.state import Store

from .diagnostics import RESTORE_STATE, DispatcherError
from .outcomes import RESULTS
from .records import read_json

TOKYO = ZoneInfo("Asia/Tokyo")


def local_time(value: str | None = None) -> datetime:
    now = datetime.fromisoformat(value) if value else datetime.now(TOKYO)
    if now.tzinfo is None:
        raise ValueError("clock must include a timezone")
    return now.astimezone(TOKYO)


def latest_slot(now: datetime) -> datetime:
    now = now.astimezone(TOKYO)
    today = now.replace(minute=0, second=0, microsecond=0)
    if now.hour >= 20:
        return today.replace(hour=20)
    if now.hour >= 8:
        return today.replace(hour=8)
    return (today - timedelta(days=1)).replace(hour=20)


def next_slot(now: datetime) -> datetime:
    return latest_slot(now) + timedelta(hours=12)


class Schedule:
    """Schedule/run journal only; ownership is exclusively Store.parents.

    Every read/modify/write is called while holding the parent Store lock.
    The judgment lifetime lock is separate, so status and parent events remain
    available during a long read-only judgment.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory.absolute()
        self.path = self.directory / "schedule.json"

    def read(self) -> dict[str, Any]:
        try:
            return self._validate(object_value(read_json(read_state_text(self.path))))
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise DispatcherError(
                "invalid_schedule",
                "dispatcher/schedule.json contains invalid state.",
                RESTORE_STATE,
                state_path="dispatcher/schedule.json",
            ) from exc

    def read_optional(self) -> dict[str, Any] | None:
        try:
            return self.read()
        except FileNotFoundError:
            return None

    def _validate(self, value: dict[str, Any]) -> dict[str, Any]:
        if (
            set(value)
            != {"schema_version", "enabled", "cursor", "config", "runs", "active"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or type(value["enabled"]) is not bool
            or not isinstance(value["runs"], list)
            or not isinstance(value["config"], dict)
        ):
            raise ValueError("unknown or malformed dispatcher schedule")
        config = value["config"]
        if set(config) != {
            "repo",
            "cwd",
            "runner",
            "max_parents",
            "max_children",
            "judgement_timeout",
        }:
            raise ValueError("malformed dispatcher config")
        for key in ("max_parents", "max_children"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError("execution limits must be positive integers")
        timeout = config["judgement_timeout"]
        if (
            type(timeout) not in {int, float}
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("judgment timeout must be finite and positive")
        if not all(
            isinstance(config[k], str) and config[k] for k in ("repo", "cwd", "runner")
        ):
            raise ValueError("missing dispatcher paths/repository")
        for key in ("cwd", "runner"):
            if not Path(config[key]).is_absolute():
                raise ValueError("dispatcher paths must be absolute")
        if local_time(value["cursor"]).isoformat() != value["cursor"]:
            raise ValueError("invalid slot cursor")
        seen = set()
        for run in value["runs"]:
            if (
                not isinstance(run, dict)
                or set(run)
                != {
                    "id",
                    "slot",
                    "explicit",
                    "phase",
                    "controller",
                    "directory",
                    "diagnostic",
                    "candidates",
                }
                or run["phase"] not in {"running", "recovery_required", *RESULTS}
                or run["id"] in seen
                or not isinstance(run["candidates"], dict)
            ):
                raise ValueError("invalid dispatcher run journal")
            seen.add(run["id"])
        if value["active"] is not None and value["active"] not in seen:
            raise ValueError("active judgment is missing from journal")
        return value

    def write(self, value: dict[str, Any]) -> None:
        Store.write_json(self.path, value)

    def initialize(self, config: dict[str, Any], now: datetime) -> None:
        if self.read_optional() is not None:
            raise DispatcherError(
                "schedule_exists",
                "dispatcher/schedule.json already exists.",
                "Use `spin ops status dispatcher` to inspect the existing installation.",
                state_path="dispatcher/schedule.json",
            )
        self.write(
            {
                "schema_version": 1,
                "enabled": False,
                "cursor": latest_slot(now).isoformat(),
                "config": config,
                "runs": [],
                "active": None,
            }
        )
