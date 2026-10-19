"""Read-only training queue identities and terminal records (stdlib only).

Queue history policy and its on-disk contract are documented in the
training-queue SKILL.md. Readers never infer completion from file timestamps.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path
from typing import Any, cast

CEILING = 10**19 - 1
RETENTION_NS = 7 * 24 * 60 * 60 * 10**9
TERMINAL = ("done", "failed", "cancelled")
NEW_NAME = re.compile(r"^0-([0-9]{19})_[0-9T.Z-]+_[0-9]+_[A-Za-z0-9._-]+\.job$")


def registration_order(job: str) -> tuple[int, str]:
    match = NEW_NAME.fullmatch(job)
    # Legacy jobs keep their identity and embedded paths. Small numeric prefixes
    # also cover manually written legacy queue entries.
    timestamp = CEILING - int(match[1]) if match else int(job.split("_", 1)[0])
    return timestamp, job


def regular(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise ValueError(f"queue history path must be a regular file: {path}")
    return True


def completion_path(queue: Path, job: str) -> Path:
    if Path(job).name != job or not job.endswith(".job") or job.startswith("."):
        raise ValueError(f"invalid published job filename: {job}")
    return queue / "control/completions" / f"{job[:-4]}.json"


def read_completion(queue: Path, job: str) -> dict[str, Any] | None:
    path = completion_path(queue, job)
    if not regular(path):
        return None
    record = json.loads(path.read_text())
    if (
        not isinstance(record, dict)
        or type(record.get("version")) is not int
        or record.get("version") != 1
        or record.get("job") != job
        or record.get("state") not in TERMINAL
        or record.get("time_source") not in {"recorded", "legacy-observed"}
        or type(record.get("retention_started_at_ns")) is not int
        or record["retention_started_at_ns"] <= 0
        or (
            record.get("time_source") == "recorded"
            and (
                type(record.get("finished_at_ns")) is not int
                or record["finished_at_ns"] != record["retention_started_at_ns"]
            )
        )
        or (
            record.get("time_source") == "legacy-observed"
            and record.get("finished_at_ns") is not None
        )
        or (
            "expired_at_ns" in record
            and (
                type(record["expired_at_ns"]) is not int
                or record["expired_at_ns"] - record["retention_started_at_ns"]
                <= RETENTION_NS
            )
        )
    ):
        raise ValueError(f"invalid queue completion record: {path}")
    return cast(dict[str, Any], record)


def expired_completion(queue: Path, job: str) -> dict[str, Any] | None:
    record = read_completion(queue, job)
    if record is None or "expired_at_ns" not in record:
        return None
    if any((queue / state / job).exists() for state in ("jobs", "running")):
        raise ValueError(f"expired queue history conflicts with an active job: {job}")
    return record


def job_label(stem: str) -> str:
    """Read the sanitized name from a new or legacy immutable run id."""
    return (
        stem.split("_", 3)[-1]
        if NEW_NAME.fullmatch(stem + ".job")
        else stem.split("_", 2)[-1]
    )


def named_jobs(queue: Path, name: str) -> set[str]:
    """Find identities across live, expired and repro-only history."""
    jobs = set()
    for directory, suffix in (
        ("jobs", ".job"),
        ("running", ".job"),
        ("done", ".job"),
        ("failed", ".job"),
        ("cancelled", ".job"),
        ("logs", ".log"),
        ("control/completions", ".json"),
        ("repro", ""),
    ):
        for path in (queue / directory).glob(f"*_{name}{suffix}"):
            stem = path.name.removesuffix(suffix) if suffix else path.name
            jobs.add(stem + ".job")
    return jobs


def latest_named_job(queue: Path, name: str) -> str | None:
    """Choose the latest registration, independently of display filenames."""
    jobs = named_jobs(queue, name)
    return max(jobs, key=registration_order) if jobs else None
