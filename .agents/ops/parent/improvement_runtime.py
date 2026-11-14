"""Serialize the final spawn with decisions; reuse existing runner stop proof."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from decision_logs.capture import runtime_environment
from dispatcher.runtime import Runtime, alive, process_identity
from issues.improvement import human_state

from .execution import authorization
from .improvements import get, lock_key, require
from .session import ParentSession

if TYPE_CHECKING:
    from .github import GitHub
    from .state import Store


@dataclass(frozen=True)
class ManualExecution:
    completed: subprocess.CompletedProcess[bytes]
    spawned: bool


def execute_manual(
    gh: GitHub,
    store: Store,
    parent: int,
    identity: str,
    child: int,
    command: list[str],
    instruction: str,
    delegation: str,
    *,
    generation: int,
    reserved_run: dict[str, Any],
    reserved_turn: dict[str, Any],
) -> ManualExecution:
    from .manual_workers import current_task, latest_run, worker_runs

    directory = Path(command[command.index("--run-dir") + 1])
    process: subprocess.Popen[bytes] | None = None
    try:
        with (
            store.operation(lock_key(gh.repo)),
            store.operation(f"{gh.repo.casefold()}#{parent}"),
            store.locked(),
        ):
            session = ParentSession(gh, store, parent, identity)
            task = current_task(session, child)
            require(session, fresh=True)
            current = authorization(session)
            require(session, child, fresh=True)
            if (
                task["delegation_id"] != delegation
                or session.row["generation"] != generation
                or worker_runs(task)[-1] != reserved_run
                or task["preflight"]["turns"][-1] != reserved_turn
                or task["worktree"] != command[command.index("--cwd") + 1]
                or task["branch"] != command[command.index("--branch") + 1]
                or task.get("model") != reserved_run["model"]
                or latest_run(session, task) != directory
                or task.get("worker_ended")
                or task.get("stopped")
                or task.get("authorization") != current
            ):
                raise ValueError(
                    "manual reservation/authority changed before runtime admission"
                )
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=runtime_environment(
                    directory,
                    actual_operation="continue_ai"
                    if command[1] == "resume"
                    else "start_ai",
                ),
            )
            task["manual_launcher"] = {
                "directory": str(directory),
                "process": process_identity(process.pid),
            }
            session.row.pop("improvement_stop", None)
            store.save()
        stdout, stderr = process.communicate(instruction.encode("utf-8"))
        return ManualExecution(
            subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
            True,
        )
    except (Exception, KeyboardInterrupt) as exc:
        if process is not None:
            # A failed post-spawn persistence is not a pre-start rejection.
            if process.poll() is None:
                process.terminate()
            process.communicate()
            raise ValueError(
                "manual launcher started but admission persistence failed; inspect its runtime and cleanup before retrying"
            ) from exc
        return ManualExecution(
            subprocess.CompletedProcess(
                command, 127 if isinstance(exc, OSError) else 2, b"", str(exc).encode()
            ),
            False,
        )


def reconcile_stop(
    gh: GitHub,
    store: Store,
    source_parent: int,
    identity: str,
    candidate: str,
    dispatcher_directory: Path | None = None,
) -> dict[str, Any]:
    """Replay a decision's unfinished stop without claiming the target's owner.

    The explicit human decision authorizes this targeted stop. Source completion,
    target ownership, logs and the old execution evidence are never reset.
    """
    manual: list[tuple[int, str, Path, dict[str, Any] | None]] = []
    with store.operation(lock_key(gh.repo)), store.locked():
        source_session = ParentSession(gh, store, source_parent, identity)
        source, observed = get(source_session, candidate)
        publication = observed["extraction"] or observed["publication"]
        if publication is None or (
            observed["extraction"] is None and publication["data"]["mode"] != "created"
        ):
            return {"state": "reference_only"}
        target = publication["data"]["issue"]
        target_key = f"{gh.repo.casefold()}#{target}"
        row = store.parents.get(target_key)
        if row is None:
            return {
                "state": "not_started"
                if human_state(observed) in {"held", "declined", "revoked"}
                else "not_requested"
            }
        pending = row.get("improvement_stop", {})
        if (
            human_state(observed) not in {"held", "declined", "revoked"}
            and pending.get("state") != "stopping"
        ):
            return {"state": "not_requested"}
        # A later approval does not erase an already committed stop request.
        decision_id = pending.get("decision_id") or observed["decisions"][-1]["id"]
        # Focus on the row being saved; unrelated parents must retain their latest data.
        store.focus(target_key)
        row["improvement_stop"] = {
            "candidate_id": source.candidate_id,
            "decision_id": decision_id,
            "state": "stopping",
        }
        managed = row.get("dispatch") is not None
        if not managed:
            from .manual_workers import run_directory, worker_runs

            for n, task in row["tasks"].items():
                run_entries = [
                    run
                    for run in worker_runs(task)
                    if run["delegation_id"] == task.get("delegation_id")
                ]
                if run_entries and not task.get("worker_ended"):
                    run = run_entries[-1]
                    directory = run_directory(store, run)
                    launcher = task.get("manual_launcher", {})
                    process = (
                        launcher.get("process")
                        if launcher.get("directory") == str(directory)
                        else None
                    )
                    manual.append((int(n), task["delegation_id"], directory, process))
        store.save()
    errors: list[str] = []
    if managed:
        from dispatcher.schedule import Schedule
        from dispatcher.supervisor import Supervisor, children_pending

        control = Supervisor(
            gh,
            store,
            Schedule(dispatcher_directory or store.directory.parent / "dispatcher"),
        )
        try:
            control.reconcile(stop=True, parent=target)
        except (ValueError, OSError):
            errors.append("managed runtime stop remains unconfirmed")
        current_row = store.read_data()["parents"][target_key]
        complete = not children_pending(current_row) and all(
            job["phase"] == "ended" for job in current_row["dispatch"]["jobs"].values()
        )
    else:
        from .manual_workers import RUNNER, validate_run

        complete = True
        for n, delegation, directory, process in manual:
            try:
                if process is not None and alive(process):
                    os.kill(process["pid"], signal.SIGTERM)
                if not (directory / "result.json").exists():
                    # Prepared/startup is not end evidence. The admission gate
                    # prevents a later spawn; replay checks runner cleanup.
                    complete = False
                    continue
                current_row = store.read_data()["parents"][target_key]
                task = current_row["tasks"][str(n)]
                if task.get("delegation_id") != delegation:
                    raise ValueError("manual delegation changed during stop")
                validate_run(task, directory)
                runtime = Runtime(RUNNER)
                proof = runtime.inspect(directory, stop=True)
                if not runtime.ended(proof):
                    complete = False
                    continue
                with store.operation(target_key), store.locked():
                    store.focus(target_key)
                    task = store.parents[target_key]["tasks"][str(n)]
                    current_entries = [
                        entry
                        for entry in worker_runs(task)
                        if entry["delegation_id"] == delegation
                    ]
                    if (
                        task.get("delegation_id") != delegation
                        or not current_entries
                        or run_directory(store, current_entries[-1]) != directory
                    ):
                        raise ValueError(
                            "manual delegation changed during stop confirmation"
                        )
                    task["worker_ended"] = {
                        "delegation_id": delegation,
                        "evidence": "Human improvement decision: common runner confirmed stop and cleanup.",
                    }
                    store.save()
            except (ValueError, OSError):
                complete = False
                errors.append(f"worker #{n}: stop or cleanup unconfirmed")
    state = "stopped" if complete and not errors else "stopping"
    with store.operation(target_key), store.locked():
        store.focus(target_key)
        row = store.parents[target_key]
        saved = row.get("improvement_stop", {})
        if saved.get("decision_id") == decision_id:
            row["improvement_stop"] = {**saved, "state": state}
            store.save()
    return {"state": state, "errors": errors}
