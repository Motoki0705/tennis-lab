"""Collect a public, read-only observation. No agent launch, locks or state writes."""

from __future__ import annotations

import hashlib
import math
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dispatcher.schedule import Schedule
from parent.state import Store, state_directory

from .github import parent_snapshot
from .records import RecordError, inspect_run, timestamp
from .safety import Guard, StatusError

LABELS = {
    "running": "実行中",
    "turn_completed": "ターン終了",
    "failed": "失敗",
    "stopped": "停止中",
    "dependency_wait": "依存待ち",
    "not_started": "委任予約済み・未起動",
    "not_delegated": "未委任",
    "unknown": "状態不明",
}


def registry(store: Store) -> dict[str, Any]:
    if not store.path.exists() and not store.path.is_symlink():
        raise StatusError("registry_missing")
    value = store.read_data()
    if (
        set(value) != {"version", "parents"}
        or type(value["version"]) is not int
        or value["version"] not in {1, 2}
        or not isinstance(value["parents"], dict)
    ):
        raise StatusError("invalid_registry")
    parents: dict[str, Any] = {}
    owners: set[str] = set()
    for key, row in value["parents"].items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+#[1-9][0-9]*", key
            )
            or key.casefold() in parents
        ):
            raise StatusError("invalid_registry_key")
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("tasks"), dict)
            or not isinstance(row.get("reviews"), list)
            or not isinstance(row.get("session"), str)
            or not row["session"]
            or row.get("status")
            not in {"active", "handoff_pending", "released", "ended"}
            or type(row.get("generation")) is not int
            or row["generation"] < 1
        ):
            raise StatusError("invalid_registry_parent")
        if row["status"] in {"active", "handoff_pending"}:
            if row["session"] in owners:
                raise StatusError("ambiguous_registry_owner")
            owners.add(row["session"])
        for child, task in row["tasks"].items():
            if (
                not isinstance(child, str)
                or not re.fullmatch(r"[1-9][0-9]*", child)
                or not isinstance(task, dict)
            ):
                raise StatusError("invalid_registry_task")
        parents[key.casefold()] = row
    store.data = {"version": value["version"], "parents": parents}
    return parents


def evidence(source: str, fields: list[str], code: str) -> dict[str, Any]:
    return {"source": source, "fields": fields, "code": code}


def latest_run(
    task: dict[str, Any],
    row: dict[str, Any],
    child: int,
    state: Path,
    dispatch_enabled: bool,
) -> tuple[dict[str, Any] | None, str]:
    runs = task.get("worker_runs", [])
    if not isinstance(runs, list):
        raise ValueError("invalid worker runs")
    current = task.get("delegation_id")
    if current is not None and (
        not isinstance(current, str) or not re.fullmatch(r"[0-9a-f]{32}", current)
    ):
        raise ValueError("invalid delegation")
    found = []
    for index, run in enumerate(runs):
        if (
            not isinstance(run, dict)
            or set(run) != {"delegation_id", "run_dir", "kind", "recorded_at"}
            or not isinstance(run["delegation_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", run["delegation_id"])
            or not isinstance(run["run_dir"], str)
            or not run["run_dir"]
            or run["kind"] not in {"initial", "resume", "registered"}
        ):
            raise ValueError("invalid worker run")
        recorded = timestamp(run["recorded_at"])
        if run["delegation_id"] == current:
            path = Path(run["run_dir"])
            if not path.is_absolute():
                path = state / path
                if not path.resolve().is_relative_to(state.resolve()):
                    raise ValueError("relative run escapes parent state")
            found.append(
                {
                    "directory": path.resolve(),
                    "alias": f"run-{index + 1}",
                    "kind": run["kind"],
                    "recorded_at": recorded,
                    "source": "parent_registry.worker_runs",
                }
            )
    if found:
        return found[-1], "recorded"
    if dispatch_enabled and "dispatch" in row:
        dispatch = row["dispatch"]
        if (
            not isinstance(dispatch, dict)
            or type(dispatch.get("schema_version")) is not int
            or dispatch["schema_version"] != 1
            or not isinstance(dispatch.get("jobs"), dict)
        ):
            raise ValueError("invalid dispatcher assignment")
        jobs = []
        for job in dispatch["jobs"].values():
            if not isinstance(job, dict):
                raise ValueError("invalid dispatcher job")
            if (
                job.get("role") == "implementation"
                and job.get("child") == child
                and job.get("delegation_id") == current
            ):
                if (
                    type(job.get("sequence")) is not int
                    or job["sequence"] < 1
                    or not isinstance(job.get("directory"), str)
                    or not Path(job["directory"]).is_absolute()
                    or job.get("phase")
                    not in {"reserved", "starting", "ended", "recovery_required"}
                ):
                    raise ValueError("invalid dispatcher child")
                jobs.append(job)
        if jobs:
            job = max(jobs, key=lambda j: j["sequence"])
            directory = Path(job["directory"]) / "runtime"
            if job["phase"] == "reserved" and not directory.exists():
                return None, "reserved"
            return {
                "directory": directory,
                "alias": f"dispatcher-run-{job['sequence']}",
                "kind": "resume" if job.get("resume_from") else "initial",
                "recorded_at": None,
                "source": "parent_registry.dispatch.jobs",
            }, "recorded"
    if runs:
        return None, "no_current_delegation_run"
    return None, "reserved" if "worker_runs" in task else "no_run_record"


def local_dependencies(
    task: dict[str, Any], tasks: dict[str, Any]
) -> list[dict[str, Any]]:
    deps = task.get("dependencies", [])
    if not isinstance(deps, list) or any(type(n) is not int or n < 1 for n in deps):
        raise ValueError("invalid dependencies")
    return [
        {
            "child": n,
            "status": "stopped"
            if tasks.get(str(n), {}).get("stopped") is True
            else "unobserved",
        }
        for n in deps
    ]


def child_status(
    child: int,
    task: dict[str, Any],
    row: dict[str, Any],
    guard: Guard,
    *,
    state: Path,
    dispatch_enabled: bool,
    remote: dict[str, Any] | None,
    now: datetime,
    stale_after: float,
    tail_bytes: int,
    tool_limit: int,
) -> dict[str, Any]:
    stopped = remote["stopped"] if remote is not None else task.get("stopped") is True
    value: dict[str, Any] = {
        "child": child,
        "status": "unknown",
        "evidence": [],
        "delegation": (
            "delegation-"
            + hashlib.sha256(task["delegation_id"].encode()).hexdigest()[:12]
            if isinstance(task.get("delegation_id"), str)
            else f"child-{child}-delegation"
        )
        if task.get("delegated")
        else None,
        "run": None,
        "observed_at": now.isoformat(),
        "last_activity_at": None,
        "activity_time_status": "unavailable",
        "updates_stalled": None,
        "summary": {
            "message": {
                **guard.text(None),
                "received_at": None,
                "time_status": "unavailable",
            },
            "tools": [],
            "tail_truncated": False,
        },
        "dependencies": [],
        "stop_record": {},
        "github": remote or {"status": "not_requested"},
    }
    try:
        for key in ("delegated", "stopped"):
            if key in task and type(task[key]) is not bool:
                raise ValueError("invalid task flag")
        if stopped:
            notes = task.get("last_status", {})
            if not isinstance(notes, dict):
                raise ValueError("invalid stop record")
            value["stop_record"] = {
                key: guard.text(text) for key, text in notes.items()
            }
        deps = local_dependencies(task, row["tasks"])
        value["dependencies"] = deps
        run, availability = latest_run(task, row, child, state, dispatch_enabled)
        if run:
            details = inspect_run(
                run["directory"], guard, bound=tail_bytes, tool_limit=tool_limit
            )
            value["run"] = {k: v for k, v in run.items() if k != "directory"}
            for key in ("summary", "last_activity_at", "activity_time_status"):
                value[key] = details.pop(key)
            value["run"].update(details)
            value["evidence"].append(
                evidence(
                    run["source"],
                    ["delegation_id", "run_dir"],
                    "current_delegation_latest_run",
                )
            )
            if details["result"] == "running":
                value["status"] = (
                    "running" if details["identity_verified"] else "unknown"
                )
                value["evidence"].append(
                    evidence(
                        "runner.result",
                        [
                            "result",
                            "runner.pid",
                            "runner.boot_id",
                            "runner.start_ticks",
                        ],
                        details["identity_status"],
                    )
                )
                when = value["last_activity_at"]
                if value["status"] == "running" and when:
                    value["updates_stalled"] = (
                        now - datetime.fromisoformat(when)
                    ).total_seconds() > stale_after
            else:
                value["status"] = (
                    "turn_completed" if details["result"] == "completed" else "failed"
                )
                value["evidence"].append(
                    evidence(
                        "runner.result",
                        ["result", "turn_end_reason"],
                        "turn_result_only_not_task_acceptance",
                    )
                )
        elif task.get("delegated"):
            value["status"] = "not_started" if availability == "reserved" else "unknown"
            value["evidence"].append(
                evidence(
                    "parent_registry.task", ["delegated", "worker_runs"], availability
                )
            )
        else:
            value["status"] = "not_delegated"
            value["evidence"].append(
                evidence("parent_registry.task", ["delegated"], "not_delegated")
            )
        if not stopped and value["status"] in {"not_delegated", "not_started"}:
            waiting = any(d["status"] == "stopped" for d in deps)
            if remote:
                waiting = any(
                    r["code"].startswith("dependency_")
                    for r in remote["launch"]["reasons"]
                )
            if waiting:
                value["status"] = "dependency_wait"
                value["evidence"].append(
                    evidence(
                        "github.launch_plan" if remote else "parent_registry.task",
                        ["dependencies", "reasons"]
                        if remote
                        else ["dependencies", "stopped"],
                        "dependency_unavailable",
                    )
                )
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        value["status"] = "unknown"
        value["evidence"].append(
            evidence(
                exc.source if isinstance(exc, RecordError) else "runner.records",
                exc.fields
                if isinstance(exc, RecordError)
                else ["schema_version", "result", "session", "events", "worker_runs"],
                exc.code
                if isinstance(exc, RecordError)
                else "missing_corrupt_or_inconsistent_record",
            )
        )
    if stopped:
        value["status"] = "stopped"
        value["evidence"].append(
            evidence(
                "github.issue" if remote is not None else "parent_registry.task",
                ["body.parent-task-status"]
                if remote is not None
                else ["stopped", "last_status"],
                "explicit_stop_record",
            )
        )
        if remote and remote["stop_record"]:
            value["stop_record"] = remote["stop_record"]
    value["label"] = LABELS[value["status"]]
    if value["status"] != "running":
        value["updates_stalled"] = None
    return value


def collect_status(
    *,
    parent: int | None = None,
    repo: str | None = None,
    state_dir: Path | None = None,
    dispatcher_state_dir: Path | None = None,
    include_inactive: bool = False,
    github: bool = False,
    stale_after: float = 300,
    now: datetime | None = None,
    tail_bytes: int = 262144,
    tool_limit: int = 5,
) -> dict[str, Any]:
    """Return JSON-serializable parent/child observations; see README for fields.

    GitHub is opt-in. All sources are read without taking an operation/registry/
    session lock. A failed observation raises StatusError; no partial success.
    """
    if (
        type(stale_after) not in {int, float}
        or not math.isfinite(stale_after)
        or stale_after <= 0
        or type(tail_bytes) is not int
        or not 1024 <= tail_bytes <= 1048576
        or type(tool_limit) is not int
        or not 1 <= tool_limit <= 20
        or (parent is not None and (type(parent) is not int or parent < 1))
    ):
        raise StatusError("invalid_query")
    if repo is not None and (
        not isinstance(repo, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repo)
    ):
        raise StatusError("invalid_repository")
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise StatusError("clock_requires_timezone")
    now = now.astimezone(UTC)
    try:
        store = Store(Path(state_dir) if state_dir is not None else state_directory())
        parents = registry(store)
        guard = Guard(parents)
        directory = (
            Path(dispatcher_state_dir)
            if dispatcher_state_dir is not None
            else store.directory.parent / "dispatcher"
        )
        schedule = Schedule(directory)
        exists = schedule.path.exists() or schedule.path.is_symlink()
        dispatcher = {
            "status": "available" if exists else "unused",
            "label": "自律dispatcher記録あり" if exists else "自律dispatcher未使用",
        }
        if exists:
            schedule.read()  # validate only; never initialize, lock or save
        output: dict[str, Any] = {
            "schema_version": 1,
            "observed_at": now.isoformat(),
            "stale_after_seconds": stale_after,
            "dispatcher": dispatcher,
            "github": {"status": "not_requested", "fetched_at": None},
            "parents": [],
        }
        for key, row in sorted(parents.items()):
            repository, number = key.rsplit("#", 1)
            n = int(number)
            if (
                (repo and repository.casefold() != repo.casefold())
                or (parent is not None and parent != n)
                or (
                    not include_inactive
                    and row["status"] not in {"active", "handoff_pending"}
                )
            ):
                continue
            for attempt in range(2 if github else 1):
                # A preceding parent's GitHub observation may take seconds.
                # Start this parent's window from a fresh row, not the registry
                # captured at the beginning of the whole request.
                current = registry(store)
                guard.observe_parents(current)
                row = current.get(key)
                if row is None:
                    raise StatusError("registry_changed_retry")
                if not include_inactive and row["status"] not in {
                    "active",
                    "handoff_pending",
                }:
                    break
                remote = parent_snapshot(repository, n, store, guard) if github else {}
                if github:
                    fresh = registry(store)
                    guard.observe_parents(fresh)
                    if fresh.get(key) != row:
                        if attempt == 0:
                            continue
                        raise StatusError("registry_changed_retry")
                record = {
                    "repo": repository,
                    "parent": n,
                    "assignment": row["status"],
                    "generation": row["generation"],
                    "source": "parent_registry",
                    "children": [],
                }
                for child in sorted({int(k) for k in row["tasks"]} | set(remote)):
                    record["children"].append(
                        child_status(
                            child,
                            row["tasks"].get(str(child), {}),
                            row,
                            guard,
                            state=store.directory,
                            dispatch_enabled=exists,
                            remote=remote.get(child),
                            now=now,
                            stale_after=stale_after,
                            tail_bytes=tail_bytes,
                            tool_limit=tool_limit,
                        )
                    )
                if not github:
                    fresh = registry(store)
                    guard.observe_parents(fresh)
                    if fresh.get(key) != row:
                        raise StatusError("registry_changed_retry")
                output["parents"].append(record)
                break
        if github:
            output["github"] = {
                "status": "observed",
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        guard.check(output)
        return output
    except StatusError:
        raise
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        OverflowError,
        subprocess.SubprocessError,
    ) as exc:
        raise StatusError(
            "github_observation_failed" if github else "local_observation_failed"
        ) from exc
