"""Durable state of a manual worker's turns and CLI-owned follow-ups."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from dispatcher.runtime import alive, process_identity

from . import worker_result
from .policy import digest

ACTIVE = ("running", "waiting", "resuming")
RESUME_LIMIT = 3


def event_key(wait: dict[str, Any]) -> str:
    if wait["kind"] == "ci":
        return digest(["ci", wait["pr"], wait["head"]])
    return digest(["queue", sorted(wait["job_ids"])])


def current(task: dict[str, Any]) -> dict[str, Any] | None:
    record = task.get("manual_followup")
    if record is None:
        return None
    if not isinstance(record, dict):
        raise ValueError("invalid manual worker follow-up record")
    if record["delegation_id"] != task.get("delegation_id"):
        return None
    return cast(dict[str, Any], record)


def for_run(
    task: dict[str, Any], directories: list[Path], index: int
) -> dict[str, Any] | None:
    history = task.get("manual_followup_history", {})
    if not isinstance(history, dict):
        raise ValueError("invalid manual worker follow-up history")
    latest = current(task)
    matches = []
    first = directories[index]
    successors = {str(path) for path in directories[index:]}
    for record in [latest, *history.values()]:
        if record is None:
            continue
        if not isinstance(record, dict):
            raise ValueError("invalid saved manual worker follow-up")
        end = record.get("run_dir")
        if (
            record.get("delegation_id") != task["delegation_id"]
            or end not in successors
        ):
            continue
        members = record.get("run_dirs")
        # Older records only name the first and current directories. Automatic
        # resumes are contiguous while that follow-up owns the worker.
        if members is None:
            belongs = end == str(first) or any(
                path.name == record["id"] for path in directories[: index + 1]
            )
        else:
            if (
                not isinstance(members, list)
                or any(not isinstance(path, str) for path in members)
                or end not in members
            ):
                raise ValueError("invalid manual follow-up run directories")
            belongs = str(first) in members
        if belongs:
            matches.append(record)
    if len(matches) > 1:
        raise ValueError("ambiguous manual worker follow-up")
    return matches[0] if matches else None


def same_owner(row: dict[str, Any], record: dict[str, Any]) -> bool:
    return (
        row.get("status") == "active"
        and row.get("session") == record["owner"]
        and row.get("generation") == record["generation"]
    )


def check_admission(
    row: dict[str, Any], task: dict[str, Any], automatic: str | None
) -> None:
    record = current(task)
    if automatic is not None:
        if (
            record is None
            or record["id"] != automatic
            or record["state"] != "resuming"
            or record["process"] != process_identity(os.getpid())
            or not same_owner(row, record)
        ):
            raise ValueError("automatic worker resume no longer owns its reservation")
    elif record and record["state"] in ACTIVE and same_owner(row, record):
        raise ValueError("worker follow-up is active; use wait-worker or stop-worker")


def reserve(
    row: dict[str, Any],
    task: dict[str, Any],
    directory: Path,
    automatic: str | None,
    phase: str,
) -> None:
    record = current(task)
    if automatic is None:
        if record is not None:
            task.setdefault("manual_followup_history", {})[record["id"]] = record
        record = {
            "id": directory.name,
            "delegation_id": task["delegation_id"],
            "owner": row["session"],
            "generation": row["generation"],
            "resumptions": list(record["resumptions"]) if record else [],
            "wait": None,
            "monitor_started": False,
            "run_dirs": [],
        }
        task["manual_followup"] = record
    assert record is not None
    if automatic is not None:
        # Consume the observed event only when this transaction claims a slot
        # and saves the new turn. A full parent can leave it pending.
        record["resumptions"].append(event_key(record["wait"]))
    if "run_dirs" in record:
        record["run_dirs"].append(str(directory))
    record.update(
        state="running",
        run_dir=str(directory),
        process=process_identity(os.getpid()),
        phase=phase,
        reason=None,
    )


def finish(record: dict[str, Any], directory: Path, exit_code: int) -> int:
    if record["state"] != "running" or record["run_dir"] != str(directory):
        raise ValueError("worker follow-up changed before reading the final report")
    if exit_code:
        record.update(state="runtime_failed", reason="worker_runtime_failed", wait=None)
        return exit_code
    try:
        result = worker_result.parse_report(
            (directory / "final.md").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        record.update(state="error", reason=str(exc), wait=None)
        return 2
    record.update(state=result["state"], wait=result["wait"], reason=None)
    return 0


def stop(task: dict[str, Any]) -> None:
    record = current(task)
    if record and record["state"] in ACTIVE:
        record.update(state="needs_parent", reason="worker_stopped")


def public_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "state": record["state"],
        "auto_resumes": len(record["resumptions"]),
        "wait": record["wait"],
        "reason": record["reason"],
        "observation": record.get("observation"),
    }


def live(record: dict[str, Any]) -> bool:
    identity = record["process"]
    return identity is not None and alive(identity)
