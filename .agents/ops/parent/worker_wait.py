"""Read-only, explicitly requested waits for a pinned manual worker run."""

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

from . import manual_workers
from .policy import object_value
from .state import Store

RECOVERY = ".agents/ops/lib/README.md#状態確認停止中断からの復旧"
LIVE_KEYS = ("runner_alive", "agent_alive", "agent_group_alive")


@dataclass
class Target:
    task: dict[str, Any]
    directory: Path

    @property
    def binding(self) -> tuple[Any, ...]:
        return (
            self.task["delegation_id"],
            self.task["worktree"],
            self.task["branch"],
            self.task.get("model"),
            self.directory,
        )

    @property
    def worker_ended_recorded(self) -> bool:
        ended = self.task.get("worker_ended", {})
        return bool(
            isinstance(ended, dict)
            and ended.get("delegation_id") == self.task["delegation_id"]
        )


def target(store: Store, key: str, child: int) -> Target:
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
    task = manual_workers.reserved_task(object_value(rows[0]), child)
    runs = [
        run
        for run in manual_workers.worker_runs(task)
        if run["delegation_id"] == task["delegation_id"]
    ]
    if not runs:
        raise ValueError("no recorded run for the current manual worker delegation")
    return Target(task, manual_workers.run_directory(store, runs[-1]))


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
    timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, dict[str, Any]]:
    if timeout is not None and (
        type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0
    ):
        raise ValueError("--timeout must be a finite positive number of seconds")
    deadline = None if timeout is None else clock() + timeout
    pinned = target(store, key, child)
    runtime = Runtime(manual_workers.RUNNER)
    run_id = None
    proof: dict[str, Any] | None = None
    current = pinned

    def response(state: str, error: str | None = None) -> dict[str, Any]:
        record = proof["record"] if proof is not None else {}
        final = pinned.directory / "final.md"
        value = {
            "child": child,
            "delegation_id": pinned.task["delegation_id"],
            "run_id": run_id,
            "wait_state": state,
            "runner_summary": (
                {
                    **{k: record[k] for k in manual_workers.SUMMARY_KEYS},
                    **{k: proof["summary"][k] for k in LIVE_KEYS},
                }
                if proof is not None
                else None
            ),
            "final_file": str(final) if final.is_file() else None,
            "termination_recorded": record.get("result") in {"completed", "failed"},
            "cleanup_complete": record.get("cleanup_complete"),
            "worker_ended_recorded": current.worker_ended_recorded,
        }
        if error is not None:
            value.update(error=error, recovery=RECOVERY)
        return value

    while True:
        current = target(store, key, child)
        if current.binding != pinned.binding:
            raise ValueError("manual worker delegation/latest run changed during wait")
        remaining = None if deadline is None else deadline - clock()
        if remaining is not None and remaining <= 0:
            return 124, response("timeout", "wait_timeout")
        try:
            settings = manual_workers.validate_run(pinned.task, pinned.directory)
            remaining = None if deadline is None else deadline - clock()
            if remaining is not None and remaining <= 0:
                return 124, response("timeout", "wait_timeout")
            observed = runtime.inspect(pinned.directory, timeout=remaining)
            record = observed["record"]
            validate_evidence(record)
            if manual_workers.model_settings(pinned.task, record) != settings:
                raise ValueError("runner settings changed during observation")
            if run_id is not None and record["run_id"] != run_id:
                raise ValueError("runner run ID changed during wait")
        except subprocess.TimeoutExpired:
            return 124, response("timeout", "wait_timeout")
        except (OSError, ValueError, KeyError, TypeError, OverflowError):
            # In particular, a reservation without result.json is not proof
            # that a runtime started. Never wait on an unverified launch.
            proof = None
            return 2, response(
                "liveness_unknown", "runner_records_unavailable_or_invalid"
            )
        proof, run_id = observed, record["run_id"]
        current = target(store, key, child)
        if current.binding != pinned.binding:
            raise ValueError("manual worker delegation/latest run changed during wait")
        if runtime.ended(proof):
            return 0, response("ended")
        if not proof["summary"]["runner_alive"]:
            state = (
                "cleanup_unconfirmed"
                if record["result"] in {"completed", "failed"}
                else "liveness_unknown"
            )
            return 2, response(state, "runner_termination_or_cleanup_unconfirmed")
        remaining = None if deadline is None else deadline - clock()
        if remaining is not None and remaining <= 0:
            return 124, response("timeout", "wait_timeout")
        sleep(1.0 if remaining is None else min(1.0, remaining))
