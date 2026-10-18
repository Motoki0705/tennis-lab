"""Machine-local, explicitly initialized three-day schedule (08:00 JST).

The caller holds schedule.lock across a report run. Atomic state replacement lets
``schedule status`` inspect an in-progress run without waiting for the agent.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")
PERIOD = timedelta(days=3)


def state_dir() -> Path:
    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return root / "tennis-lab-agents" / "ops-report"


def aware_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timezone is required: {value!r}")
    return parsed.astimezone(JST)


def first_slot(now: datetime) -> datetime:
    now = now.astimezone(JST)
    candidate = now.replace(hour=8, minute=0, second=0, microsecond=0)
    return candidate if candidate > now else candidate + timedelta(days=1)


def latest_slot(anchor: datetime, now: datetime) -> datetime | None:
    if now < anchor:
        return None
    return anchor + ((now - anchor) // PERIOD) * PERIOD


def _slot(value: str) -> datetime:
    parsed = aware_time(value)
    if parsed.isoformat() != value or (
        parsed.hour,
        parsed.minute,
        parsed.second,
        parsed.microsecond,
    ) != (8, 0, 0, 0):
        raise ValueError(f"invalid 08:00 JST slot: {value!r}")
    return parsed


@dataclass
class Schedule:
    path: Path
    data: dict[str, Any]

    @property
    def anchor(self) -> datetime:
        return _slot(self.data["anchor"])

    @classmethod
    def initialize(cls, path: Path, now: datetime) -> Schedule:
        if path.exists() or path.is_symlink():
            raise ValueError(
                f"schedule already exists at {path}; init never resets the anchor"
            )
        state = cls(
            path,
            {
                "version": 1,
                "initialized_at": now.isoformat(),
                "anchor": first_slot(now).isoformat(),
                "last_attempt": None,
                "last_success": None,
                "skipped": [],
            },
        )
        state.save()
        return state

    @classmethod
    def load(cls, path: Path) -> Schedule:
        if not path.exists():
            raise ValueError(
                f"schedule is not initialized at {path}; run schedule init"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = cls(path, data)
            state._validate()
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(
                f"corrupt schedule at {path}: {exc}; repair explicitly, do not reinitialize"
            ) from exc
        return state

    def _validate(self) -> None:
        if (
            not isinstance(self.data, dict)
            or set(self.data)
            != {
                "version",
                "initialized_at",
                "anchor",
                "last_attempt",
                "last_success",
                "skipped",
            }
            or type(self.data["version"]) is not int
            or self.data["version"] != 1
        ):
            raise ValueError("invalid schedule schema/version")
        if first_slot(aware_time(self.data["initialized_at"])) != self.anchor:
            raise ValueError("anchor disagrees with initialization time")
        for key in ("last_attempt", "last_success"):
            entry = self.data[key]
            if entry is None:
                continue
            slot = _slot(entry["slot"])
            if slot < self.anchor or (slot - self.anchor) % PERIOD:
                raise ValueError(f"{key} is outside this schedule")
            aware_time(entry["started_at"])
            status = entry["status"]
            if status not in ("running", "failed", "succeeded") or (
                key == "last_success" and status != "succeeded"
            ):
                raise ValueError(f"invalid {key} status")
            if status != "running":
                if aware_time(entry["finished_at"]) < aware_time(entry["started_at"]):
                    raise ValueError("attempt finished before it started")
                if status == "succeeded" and not entry.get("url"):
                    raise ValueError("success has no report URL")
                if status == "failed" and not entry.get("error"):
                    raise ValueError("failure has no error")
        attempt, success = self.data["last_attempt"], self.data["last_success"]
        if success and (not attempt or success["slot"] > attempt["slot"]):
            raise ValueError("last success is after last attempt")
        if attempt and attempt["status"] == "succeeded" and attempt != success:
            raise ValueError("successful attempt disagrees with last success")
        if not isinstance(self.data["skipped"], list):
            raise ValueError("skipped must be a list of slot ranges")
        previous = self.anchor - PERIOD
        for skipped in self.data["skipped"]:
            start, end = _slot(skipped["from"]), _slot(skipped["through"])
            if (
                start <= previous
                or start < self.anchor
                or end < start
                or (start - self.anchor) % PERIOD
                or (end - start) % PERIOD
            ):
                raise ValueError("invalid skipped slot range")
            if (
                type(skipped["count"]) is not int
                or skipped["count"] != (end - start) // PERIOD + 1
            ):
                raise ValueError("invalid skipped slot count")
            if not attempt or end >= _slot(attempt["slot"]):
                raise ValueError("skipped range overlaps the last attempt")
            previous = end

    def save(self) -> None:
        self._validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as fh:
            tmp = Path(fh.name)
            try:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
                tmp.replace(self.path)
            finally:
                tmp.unlink(missing_ok=True)

    def begin(self, slot: datetime, now: datetime) -> None:
        previous = self.data["last_attempt"]
        if previous and _slot(previous["slot"]) > slot:
            raise ValueError("clock moved behind the last attempted slot")
        self.data["last_attempt"] = {
            "slot": slot.isoformat(),
            "started_at": now.isoformat(),
            "status": "running",
        }
        self.save()

    def reconcile(
        self, slot: datetime, now: datetime, published: dict[datetime, str]
    ) -> None:
        """Record missed ranges from GitHub, including a publication whose local
        success write was interrupted. Never classify such a report as skipped.
        """
        start = self.anchor
        skipped = []
        prior = []
        for done in sorted(published):
            if done < self.anchor:
                continue
            if (done - self.anchor) % PERIOD:
                raise ValueError(f"published report {done} is outside this schedule")
            if done >= slot:
                continue
            prior.append(done)
            if start < done:
                skipped.append(
                    {
                        "from": start.isoformat(),
                        "through": (done - PERIOD).isoformat(),
                        "count": (done - start) // PERIOD,
                    }
                )
            start = done + PERIOD
        if start < slot:
            skipped.append(
                {
                    "from": start.isoformat(),
                    "through": (slot - PERIOD).isoformat(),
                    "count": (slot - start) // PERIOD,
                }
            )
        changed = skipped != self.data["skipped"]
        self.data["skipped"] = skipped
        success = self.data["last_success"]
        if prior and (not success or _slot(success["slot"]) < prior[-1]):
            done = prior[-1]
            self.data["last_success"] = {
                "slot": done.isoformat(),
                "started_at": now.isoformat(),
                "finished_at": now.isoformat(),
                "status": "succeeded",
                "url": published[done],
            }
            changed = True
        if changed:
            self.save()

    def finish(
        self, now: datetime, *, url: str | None = None, error: str | None = None
    ) -> None:
        attempt = {**self.data["last_attempt"], "finished_at": now.isoformat()}
        if url:
            attempt.update(status="succeeded", url=url)
            self.data["last_success"] = attempt
        else:
            attempt.update(status="failed", error=error)
        self.data["last_attempt"] = attempt
        self.save()

    def status(self, now: datetime) -> dict[str, Any]:
        latest = latest_slot(self.anchor, now)
        next_slot = self.anchor if latest is None else latest + PERIOD
        success = self.data["last_success"]
        pending = (
            latest
            if latest and (not success or success["slot"] != latest.isoformat())
            else None
        )
        return {
            **self.data,
            "now": now.isoformat(),
            "latest_due": latest.isoformat() if latest else None,
            "pending_slot": pending.isoformat() if pending else None,
            "next_slot": next_slot.isoformat(),
        }
