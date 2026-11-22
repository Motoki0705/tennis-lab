"""Read-only waits for a pinned worker run or manual follow-up."""

from __future__ import annotations

import math
import re
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dispatcher.runtime import Runtime

from . import manual_workers, worker_followup, worker_slots
from .policy import object_value
from .state import Store

RECOVERY = ".agents/skills/parent-session/references/recovery.md"
LIVE_KEYS = ("runner_alive", "agent_alive", "agent_group_alive")


@dataclass
class Target:
    task: dict[str, Any]
    directory: Path | None
    job: dict[str, Any] | None = None
    worker: str = "primary"
    parent_binding: tuple[Any, ...] | None = None
    first_run: Path | None = None

    @property
    def followup(self) -> dict[str, Any] | None:
        record = worker_followup.current(self.task) if self.job is None else None
        return record if record and record["run_dir"] == str(self.directory) else None

    @property
    def delegation(self) -> tuple[Any, ...] | None:
        if not self.task:
            return None
        return (
            self.task["delegation_id"],
            self.task["worktree"],
            self.task["branch"],
        )

    @property
    def binding(self) -> tuple[Any, ...]:
        followup = self.followup
        return (
            self.delegation,
            self.task.get("model"),
            self.first_run,
            followup["id"] if followup else self.directory,
            self.job["id"] if self.job else None,
        )

    @property
    def worker_ended_recorded(self) -> bool:
        ended = self.task.get("worker_ended", {})
        return bool(
            self.directory is not None
            and isinstance(ended, dict)
            and ended.get("delegation_id") == self.task["delegation_id"]
        )


def target(
    store: Store, key: str, child: int, worker: str | None, after_run: str | None
) -> Target:
    # Atomic registry files are sufficient for observation. No operation,
    # registry or session locks, network calls, or state writes are needed.
    data = store.read_data()
    if (
        set(data) != {"version", "parents"}
        or type(data["version"]) is not int
        or data["version"] not in {1, 2}
    ):
        raise ValueError("unknown or malformed parent-session state")
    rows = [
        row
        for name, row in object_value(data["parents"]).items()
        if name.casefold() == key.casefold()
    ]
    if len(rows) != 1:
        raise ValueError("manual worker parent is missing or ambiguous")
    row = object_value(rows[0])
    name = worker_slots.select(row, child, worker, creating=after_run == "none")
    if row.get("dispatch"):
        if after_run is not None:
            raise ValueError("--after-run requires a manual worker")
        from .development import worker_jobs

        task = worker_slots.record(row, child, name)
        jobs = worker_jobs(row["dispatch"], child, name)
        if not jobs or jobs[-1]["delegation_id"] != task.get("delegation_id"):
            raise ValueError("no job for the current worker delegation")
        return Target(task, Path(jobs[-1]["directory"]) / "runtime", jobs[-1])
    parent_binding = tuple(row.get(k) for k in ("session", "generation", "status"))
    if after_run == "none" and not worker_slots.record(row, child, name).get(
        "delegated"
    ):
        return Target({}, None, worker=name, parent_binding=parent_binding)
    task = manual_workers.reserved_task(row, child, name)
    runs = [
        run
        for run in manual_workers.worker_runs(task)
        if run["delegation_id"] == task["delegation_id"]
    ]
    if after_run is None:
        if not runs:
            raise ValueError("no recorded run for the current manual worker delegation")
        return Target(task, manual_workers.run_directory(store, runs[-1]))

    directories = [manual_workers.run_directory(store, run) for run in runs]
    index = 0
    if after_run != "none":
        matches = [i for i, path in enumerate(directories) if str(path) == after_run]
        if len(matches) != 1:
            raise ValueError(
                "--after-run must identify a recorded run of the current delegation"
            )
        index = matches[0] + 1
    first = directory = directories[index] if index < len(directories) else None
    followup = worker_followup.current(task)
    if first is not None and followup and followup["run_dir"]:
        followup_dir = Path(followup["run_dir"])
        # The follow-up chain keeps the first manual directory's name as its ID
        # while automatic resumes advance its current run.
        if followup_dir in directories[index:] and any(
            path.name == followup["id"] for path in directories[: index + 1]
        ):
            directory = followup_dir
    return Target(
        task, directory, worker=name, parent_binding=parent_binding, first_run=first
    )


def validate_evidence(record: dict[str, Any]) -> None:
    """Reject incomplete/contradictory end evidence rather than guessing."""
    result = record.get("result")
    reason, code = record.get("turn_end_reason"), record.get("exit_code")
    if (
        not manual_workers.SUMMARY_KEYS.issubset(record)
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or not isinstance(record.get("run_id"), str)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", record["run_id"])
        or result not in {"running", "completed", "failed"}
        or "task_success" not in record
        or record["task_success"] is not None
        or type(record.get("cleanup_complete")) is not bool
    ):
        raise ValueError("invalid runner result evidence")
    if result == "running":
        if reason is not None or code is not None or record["cleanup_complete"]:
            raise ValueError("invalid running evidence")
    elif (
        type(code) is not int
        or not isinstance(reason, str)
        or not re.fullmatch(r"[a-z_]+", reason)
        or type(record.get("finished_at")) not in {int, float}
        or not math.isfinite(record["finished_at"])
        or (result == "completed" and (code != 0 or reason != "completed"))
        or (result == "failed" and (code == 0 or reason == "completed"))
    ):
        raise ValueError("invalid runner termination evidence")


def wait_worker(
    store: Store,
    key: str,
    child: int,
    *,
    worker: str | None = None,
    timeout: float | None = None,
    after_run: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any]]:
    if timeout is not None and (
        type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("--timeout must be a finite positive number of seconds")
    if after_run is not None and after_run != "none":
        anchor = Path(after_run)
        if not anchor.is_absolute():
            raise ValueError("--after-run requires an absolute run directory or none")
        after_run = str(anchor.resolve())
    deadline = None if timeout is None else clock() + timeout
    pinned = target(store, key, child, worker, after_run)
    runtime = Runtime(
        Path(pinned.job["runner"]) if pinned.job else manual_workers.RUNNER
    )
    run_id = None
    proof: dict[str, Any] | None = None
    current = pinned

    def read_bound() -> Target:
        observed = target(
            store,
            key,
            child,
            pinned.worker if after_run is not None else worker,
            after_run,
        )
        if (
            observed.parent_binding != pinned.parent_binding
            or pinned.delegation is not None
            and observed.delegation != pinned.delegation
            or pinned.directory is not None
            and observed.binding != pinned.binding
        ):
            raise ValueError("manual worker delegation/latest run changed during wait")
        return observed

    def response(state: str, error: str | None = None) -> dict[str, Any]:
        record = proof["record"] if proof is not None else {}
        directory = pinned.directory
        final = directory / "final.md" if directory is not None else None
        value = {
            "child": child,
            "delegation_id": pinned.task.get("delegation_id"),
            "run_id": run_id,
            "run_dir": str(directory) if directory is not None else None,
            "wait_state": state,
            "runner_summary": (
                {
                    **{k: record[k] for k in manual_workers.SUMMARY_KEYS},
                    **{k: proof["summary"][k] for k in LIVE_KEYS},
                }
                if proof is not None
                else None
            ),
            "final_file": str(final) if final is not None and final.is_file() else None,
            "termination_recorded": record.get("result") in {"completed", "failed"},
            "cleanup_complete": record.get("cleanup_complete"),
            "worker_ended_recorded": current.worker_ended_recorded,
        }
        if error is not None:
            value.update(error=error, recovery=RECOVERY)
        if current.followup:
            value["followup"] = worker_followup.public_view(current.followup)
        return value

    while True:
        current = read_bound()
        if (
            current.directory != pinned.directory
            or current.delegation != pinned.delegation
        ):
            pinned, run_id, proof = current, None, None
        remaining = None if deadline is None else deadline - clock()
        if remaining is not None and remaining <= 0:
            return 124, response("timeout", "wait_timeout")
        directory = pinned.directory
        if directory is None:
            sleep(1.0 if remaining is None else min(1.0, remaining))
            continue
        followup = current.followup
        if followup and not (directory / "result.json").exists():
            if followup["state"] != "running" or not worker_followup.live(followup):
                latest = read_bound()
                if latest.followup != followup or (directory / "result.json").exists():
                    continue
                return 2, response("liveness_unknown", "followup_runtime_unconfirmed")
            sleep(1.0 if remaining is None else min(1.0, remaining))
            continue
        # While the CLI monitors a job, the recovered runtime cannot change.
        # Inspect again only if the current turn is still live or a resume
        # created a new run directory.
        if proof is None or not runtime.ended(proof):
            try:
                settings = (
                    None
                    if pinned.job
                    else manual_workers.validate_run(pinned.task, directory)
                )
                remaining = None if deadline is None else deadline - clock()
                if remaining is not None and remaining <= 0:
                    return 124, response("timeout", "wait_timeout")
                observed = runtime.inspect(directory, timeout=remaining)
                record = observed["record"]
                validate_evidence(record)
                if pinned.job:
                    from dispatcher.supervisor import Supervisor

                    Supervisor._check_contract(pinned.job, record)
                elif manual_workers.model_settings(pinned.task, record) != settings:
                    raise ValueError("runner settings changed during observation")
                if run_id is not None and record["run_id"] != run_id:
                    raise ValueError("runner run ID changed during wait")
            except subprocess.TimeoutExpired:
                return 124, response("timeout", "wait_timeout")
            except (OSError, ValueError, KeyError, TypeError, OverflowError):
                proof = None
                return 2, response(
                    "liveness_unknown", "runner_records_unavailable_or_invalid"
                )
            proof, run_id = observed, record["run_id"]
        current = read_bound()
        if current.directory != directory:
            continue
        if runtime.ended(proof):
            followup = current.followup
            if not followup or followup["state"] not in worker_followup.ACTIVE:
                if followup and followup["state"] == "error":
                    return 2, response("followup_error", followup["reason"])
                return 0, response("ended")
            if not worker_followup.live(followup):
                # The controller writes its terminal state before exiting.
                # A negative liveness result may therefore race its final save.
                if read_bound().followup != followup:
                    continue
                return 2, response(
                    "liveness_unknown", "followup_controller_unconfirmed"
                )
        elif not proof["summary"]["runner_alive"]:
            state = (
                "cleanup_unconfirmed"
                if proof["record"]["result"] in {"completed", "failed"}
                else "liveness_unknown"
            )
            return 2, response(state, "runner_termination_or_cleanup_unconfirmed")
        remaining = None if deadline is None else deadline - clock()
        if remaining is not None and remaining <= 0:
            return 124, response("timeout", "wait_timeout")
        sleep(1.0 if remaining is None else min(1.0, remaining))
