"""Explicit manual-worker launches with durable, private runner provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dispatcher.runtime import Runtime, load, process_identity
from lib.agent_settings import (
    DEFAULT_CONFIG,
    check_saved_implementation_model,
    execution_contract,
    implementation_model,
    resolve,
    session_contract,
)
from lib.repair_turn import instruction as repair_instruction
from lib.repair_turn import validate_contract as validate_repair_contract
from lib.repair_turn import validate_request as validate_repair_request
from worker_logs import service as worker_logs
from worker_logs.ownership import publication_owner
from worker_logs.sources import original as original_text
from worker_logs.sources import read_original

from . import worker_followup, worker_preflight, worker_slots
from .execution import authorization
from .github import GitHub
from .launch import DEFAULT_LAUNCH_POLICY, active_reservations, launch_policy
from .policy import closing_issues, digest, nonempty, object_value
from .session import ParentSession
from .snapshot import git, validate_worktree
from .state import RegistryConflict, Store, unique_object
from .worker_instruction import prepare_instruction

RUNNER_LIB = Path(__file__).resolve().parents[1] / "lib"
RUNNER = RUNNER_LIB / "agent_exec.sh"
# agent_exec.sh uses the interpreter beside its own checkout, regardless of --cwd.
RUNNER_PYTHON = RUNNER_LIB.parents[2] / ".venv/bin/python"
SUMMARY_KEYS = {
    "schema_version",
    "run_id",
    "result",
    "turn_end_reason",
    "exit_code",
    "task_success",
}
RUN_KEYS = {"delegation_id", "run_dir", "kind", "recorded_at"}
MODEL_KEYS = {"model", "model_reason"}


class WorkerCapacityUnavailable(ValueError):
    """A later admission can succeed when another reservation ends."""


class RunnerEnvironmentUnavailable(ValueError):
    """A required runner file is missing or inaccessible."""


def require_runner_environment() -> None:
    for file in (RUNNER, RUNNER_PYTHON):
        if not file.is_file() or not os.access(file, os.X_OK):
            raise RunnerEnvironmentUnavailable(
                f"common runner requires an executable file: {file}"
            )
    entrypoint = RUNNER_LIB / "agent_runtime.py"
    if not entrypoint.is_file() or not os.access(entrypoint, os.R_OK):
        raise RunnerEnvironmentUnavailable(
            f"common runner requires a readable file: {entrypoint}"
        )


def current_task(session: ParentSession, child: int) -> dict[str, Any]:
    return reserved_task(session.owner(), child, session.worker)


def reserved_task(
    row: dict[str, Any], child: int, worker: str = "primary"
) -> dict[str, Any]:
    """Validate a manual reservation, also for readers without ownership."""
    if row.get("dispatch") is not None:
        raise ValueError("manual worker commands cannot change dispatcher workers")
    task: dict[str, Any] = object_value(worker_slots.record(row, child, worker))
    if (
        not task.get("delegated")
        or task.get("agent") != "codex"
        or not isinstance(task.get("delegation_id"), str)
        or not re.fullmatch(r"[0-9a-f]{32}", task["delegation_id"])
        or not isinstance(task.get("worktree"), str)
        or not task["worktree"]
        or not isinstance(task.get("branch"), str)
        or not task["branch"].startswith("codex/")
    ):
        raise ValueError(
            "manual worker requires the current Codex delegation reservation"
        )
    return task


def worker_runs(task: dict[str, Any]) -> list[dict[str, Any]]:
    runs = task.get("worker_runs", [])
    if not isinstance(runs, list):
        raise ValueError("malformed private worker run records")
    for run in runs:
        if (
            not isinstance(run, dict)
            or set(run)
            not in (
                RUN_KEYS,
                RUN_KEYS | MODEL_KEYS,
                RUN_KEYS | {"preflight"},
                RUN_KEYS | MODEL_KEYS | {"preflight"},
            )
            or not isinstance(run["delegation_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", run["delegation_id"])
            or not isinstance(run["run_dir"], str)
            or not run["run_dir"]
            or run["kind"] not in {"initial", "resume", "registered", "replacement"}
            or not isinstance(run["recorded_at"], str)
            or (
                "model" in run
                and (
                    not isinstance(run["model"], str)
                    or not run["model"]
                    or (
                        run["model_reason"] is not None
                        and (
                            not isinstance(run["model_reason"], str)
                            or not run["model_reason"].strip()
                        )
                    )
                )
            )
        ):
            raise ValueError("malformed private worker run records")
        if "preflight" in run:
            worker_preflight.validate_snapshot(run["preflight"])
        try:
            stamp = datetime.fromisoformat(run["recorded_at"])
            if stamp.utcoffset() != UTC.utcoffset(stamp):
                raise ValueError("non-UTC worker run timestamp")
        except ValueError:
            raise ValueError("malformed private worker run timestamp") from None
    return runs


def run_directory(store: Store, run: dict[str, Any]) -> Path:
    path = Path(run["run_dir"])
    resolved = (Path(store.directory) / path).resolve()
    if not path.is_absolute() and not resolved.is_relative_to(
        store.directory.resolve()
    ):
        raise ValueError("private worker run escapes the parent state directory")
    return resolved


def current_runs(session: ParentSession, task: dict[str, Any]) -> list[Path]:
    runs = [r for r in worker_runs(task) if r["delegation_id"] == task["delegation_id"]]
    if not runs:
        raise ValueError("no recorded run for the current manual worker delegation")
    return [run_directory(session.store, run) for run in runs]


def latest_run(session: ParentSession, task: dict[str, Any]) -> Path:
    return current_runs(session, task)[-1]


def matching_contract(task: dict[str, Any], record: dict[str, Any]) -> None:
    contract = object_value(record.get("contract"))
    cwd = contract.get("cwd")
    if (
        contract.get("task_id") != task["delegation_id"]
        or not isinstance(cwd, str)
        or not Path(cwd).is_absolute()
        or Path(cwd).resolve() != Path(task["worktree"]).resolve()
        or contract.get("branch") != task["branch"]
    ):
        raise ValueError("runner contract does not match the current delegation")


def model_settings(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Read evidence, never infer a legacy run's model from today's defaults."""
    matching_contract(task, record)
    contract = object_value(record.get("contract"))
    settings: dict[str, Any] = object_value(contract.get("settings"))
    selected = object_value(settings.get("selected"))
    overrides = object_value(settings.get("overrides"))
    model = nonempty(selected.get("model"), "saved worker model")
    check_saved_implementation_model(model)
    if "repair" in contract:
        validate_repair_contract(contract)
    if (
        contract.get("mode") != "write"
        or settings.get("agent") != "codex"
        or settings.get("role") != "implementation"
        or settings.get("profile") is not None
        or selected.get("effort") != ("high" if "repair" in contract else "max")
        or overrides.get("model", model) != model
        or (task.get("model") is not None and task["model"] != model)
    ):
        raise ValueError("saved worker model/settings do not match the delegation")
    return settings


def validate_run(task: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Check every available contract; invocation alone covers an in-flight run."""
    try:
        settings = None
        for name in ("result.json", "invocation.json"):
            path = directory / name
            if path.exists() or path.is_symlink():
                record = load(path)
                if (
                    type(record.get("schema_version")) is not int
                    or record["schema_version"] != 1
                ):
                    raise ValueError("unsupported runner record")
                observed = model_settings(task, record)
                if settings is not None and settings != observed:
                    raise ValueError("conflicting saved model settings")
                settings = observed
        if settings is None:
            raise ValueError("runner records missing")
        return settings
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(
            "runner records unavailable or do not match the current delegation"
        ) from None


def append_run(
    session: ParentSession,
    task: dict[str, Any],
    directory: Path,
    kind: str,
    model: str,
    model_reason: str | None,
    turn: dict[str, Any] | None = None,
) -> None:
    root = session.store.directory.resolve()
    path = directory.resolve()
    stored = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
    task["worker_runs"] = [
        *worker_runs(task),
        {
            "delegation_id": task["delegation_id"],
            "run_dir": stored,
            "kind": kind,
            "recorded_at": datetime.now(UTC).isoformat(),
            "model": model,
            "model_reason": model_reason,
            **({"preflight": turn} if turn is not None else {}),
        },
    ]
    task.update(model=model, model_reason=model_reason)
    session.store.save()


def record_worker_run(
    session: ParentSession, child: int, directory: Path
) -> dict[str, Any]:
    task = current_task(session, child)
    path = directory.resolve()
    settings = validate_run(task, path)
    model = settings["selected"]["model"]
    reason = task.get("model_reason")
    if not any(
        r["delegation_id"] == task["delegation_id"]
        and run_directory(session.store, r) == path
        for r in worker_runs(task)
    ):
        worker_followup.check_admission(session.row, task, None)
        append_run(session, task, path, "registered", model, reason)
    else:
        task.update(model=model, model_reason=reason)
        session.store.save()
    return {
        "child": child,
        "delegation_id": task["delegation_id"],
        "result": "recorded",
    }


def saved_session(record: dict[str, Any]) -> dict[str, Any]:
    """Read the runner's own latest-run/session receipt without repairing it."""
    try:
        root = Path(nonempty(record.get("session_root"), "session root"))
        if not root.is_absolute():
            raise ValueError("relative session root")
        value: dict[str, Any] = load(root / "session.json")
        if (
            type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or "session_id" not in value
            or (
                value["session_id"] is not None
                and (
                    not isinstance(value["session_id"], str)
                    or not value["session_id"].strip()
                )
            )
            or not Path(nonempty(value.get("latest_run"), "latest run")).is_absolute()
        ):
            raise ValueError("invalid session receipt")
        return value
    except (OSError, ValueError, TypeError):
        raise ValueError(
            "saved worker session metadata is unavailable or malformed"
        ) from None


def termination_reference(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    raw = (directory / "result.json").read_bytes()
    if json.loads(raw, object_pairs_hook=unique_object) != record:
        raise ValueError("worker termination evidence changed during inspection")
    return {
        "directory": str(directory),
        "run_id": record["run_id"],
        "result_sha256": hashlib.sha256(raw).hexdigest(),
        "termination": {
            key: record.get(key)
            for key in (
                "result",
                "turn_end_reason",
                "cleanup_complete",
                "finished_at",
                "runner_lost",
                "recovery_error",
                "cleanup_confirmation",
            )
        },
    }


def replacement_evidence(
    session: ParentSession, task: dict[str, Any], evidence: str | None
) -> dict[str, Any]:
    """Authorize a new session from terminal, unrecoverable lineage evidence.

    A separately registered takeover is retained as historical work. It cannot
    supply the saved instruction or make a resumable lineage replaceable.
    """
    evidence = nonempty(evidence, "session replacement restart evidence")
    entries = [
        r for r in worker_runs(task) if r["delegation_id"] == task["delegation_id"]
    ]
    directories = current_runs(session, task)
    value = worker_preflight.state(task)
    retry = None
    if (
        value["schema_version"] == 3
        and value["turns"]
        and all(t.get("cancelled") for t in value["turns"])
        and value["session_replacements"][-1].get("cancelled")
    ):
        retry = value["session_replacements"][-1]
        turn = None
        source = Path(
            nonempty(retry["from_run"].get("directory"), "saved replacement source")
        )
    else:
        turn = worker_preflight.result_turn(task)
        source = Path(turn["source"]) if turn is not None else directories[-1]
    if source not in directories:
        raise ValueError("saved worker turn has no registered run for replacement")
    source_index = directories.index(source)
    if any(
        r["kind"] != "registered" or "preflight" in r
        for r in entries[source_index + 1 :]
    ):
        raise ValueError("replacement source is not the latest saved worker turn")
    runtime = Runtime(RUNNER)
    records = []
    for directory in directories:
        validate_run(task, directory)
        proof = runtime.inspect(directory)
        if not runtime.ended(proof):
            raise ValueError(
                "worker termination and cleanup must be confirmed before session replacement"
            )
        records.append(proof["record"])
    record = records[source_index]
    source_reference = termination_reference(source, record)
    if retry is not None and source_reference != retry["from_run"]:
        raise ValueError("saved replacement source changed after rejected launch")
    saved = saved_session(record)
    if Path(saved["latest_run"]).resolve() != source:
        raise ValueError("replacement source is not the latest session run")
    if record.get("runner_lost"):
        from lib.agent_runtime_impl import validate_cleanup_confirmation

        confirmation = record.get("cleanup_confirmation")
        if not isinstance(confirmation, dict):
            raise ValueError("saved worker cleanup confirmation is missing")
        validate_cleanup_confirmation(confirmation, record)
    session_id = object_value(record.get("runtime")).get("session_id")
    if session_id is not None:
        session_id = nonempty(session_id, "saved worker session")
    if session_id is not None and saved["session_id"] not in (None, session_id):
        raise ValueError("saved worker session metadata conflicts with its runtime")
    binding = {
        "contract_sha256": digest(session_contract(record["contract"])),
        "session_id": session_id or saved["session_id"],
    }
    if binding["session_id"] is None and value["runtime"] is not None:
        binding["session_id"] = value["runtime"]["session_id"]
    legacy = {**binding, "contract_sha256": digest(record["contract"])}
    if value["runtime"] is not None and value["runtime"] not in (binding, legacy):
        raise ValueError("replacement source differs from the planning lineage")
    if turn is not None and record.get("prompt_sha256") != turn.get("prompt_sha256"):
        raise ValueError("replacement source does not match the saved worker turn")
    recovery_error = record.get("recovery_error")
    if recovery_error is not None:
        nonempty(recovery_error, "saved recovery error")
    if recovery_error:
        reason = "recovery_error"
    elif saved["session_id"] is None:
        reason = "missing_session_id"
    else:
        raise ValueError(
            "no saved non-resumable session evidence; inspect recovery or use --resume"
        )
    for registered in records[source_index + 1 :]:
        registered_session = saved_session(registered)
        latest = Path(registered_session["latest_run"]).resolve()
        if latest not in directories[source_index + 1 :]:
            raise ValueError("registered takeover is not the latest session run")
        last_record = records[directories.index(latest)]
        if (
            Path(last_record["session_root"]).resolve()
            != Path(registered["session_root"]).resolve()
        ):
            raise ValueError(
                "registered takeover session metadata conflicts with its latest run"
            )
        registered_id = object_value(registered.get("runtime")).get("session_id")
        if (
            registered_id is not None
            and registered_id != registered_session["session_id"]
        ):
            raise ValueError(
                "registered takeover session metadata conflicts with its runtime"
            )
        if session_contract(registered["contract"]) != session_contract(
            record["contract"]
        ):
            raise ValueError(
                "registered takeover contract differs from the planning lineage"
            )
        if binding["session_id"] is not None and registered_id == binding["session_id"]:
            raise ValueError(
                "registered continuation belongs to the old session; inspect its latest run"
            )
    return {
        "replacement_id": uuid.uuid4().hex,
        "recorded_at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "restart_evidence": evidence,
        "from_run": source_reference,
        "from_runtime": binding,
        "previous_runtime": copy.deepcopy(value["runtime"]),
        "registered_runs": [
            termination_reference(directory, registered)
            for directory, registered in zip(
                directories[source_index + 1 :],
                records[source_index + 1 :],
                strict=True,
            )
        ],
        "to_runtime": None,
    }


def prepare_run(
    session: ParentSession,
    child: int,
    prompt: Path | None,
    resume: bool,
    restart_evidence: str | None,
    model: str | None = None,
    model_reason: str | None = None,
    additional: str | None = None,
    planning: bool = False,
    replace_session: bool = False,
    repair: dict[str, Any] | None = None,
    automatic: str | None = None,
) -> tuple[list[str], str, str]:
    task = current_task(session, child)
    worker_followup.check_admission(session.row, task, automatic)
    if automatic is not None:
        followup = worker_followup.current(task)
        assert followup is not None
        if str(latest_run(session, task)) != followup["run_dir"]:
            raise ValueError("automatic resume source is no longer the latest run")
        if any(
            not d["invalidated"] and d["delivered_turn_id"] is None
            for d in worker_preflight.state(task)["decisions"]
        ):
            raise ValueError(
                "automatic resume cannot deliver a pending parent decision"
            )
        if task.get("stopped") or session.row["tasks"][str(child)].get("stopped"):
            raise ValueError("worker follow-up was stopped")
        if not task.get("worker_ended"):
            raise ValueError("automatic resume requires saved worker-end evidence")
        # The parent operation lock serializes admission and the slot claim.
        # Check local capacity before doing GitHub preflight for a busy parent.
        policy = launch_policy(session.row.get("launch_policy", DEFAULT_LAUNCH_POLICY))
        limit = policy["max_parallel_children"]
        if (
            limit is not None
            and len(active_reservations(session.row["tasks"])) >= limit
        ):
            raise WorkerCapacityUnavailable("implementation reservation limit reached")
    if repair is not None:
        validate_repair_request(repair)
        if (
            not resume
            or planning
            or replace_session
            or prompt is None
            or additional is not None
        ):
            raise ValueError("repair requires resume and a failure/gate prompt only")
        saved = task.get("repair_attempts", {}).get(repair["repair_id"])
        if saved is not None:
            if (
                saved["request"] != repair
                or saved["delegation_id"] != task["delegation_id"]
                or saved["details_sha256"]
                != hashlib.sha256(read_original(prompt).encode("utf-8")).hexdigest()
            ):
                raise ValueError(
                    "repair ID is already bound to another request/delegation"
                )
            implementation_model(model, previous=task["model"])
            return list(saved["command"]), task["delegation_id"], saved["instruction"]
    if replace_session and resume:
        raise ValueError("--replace-session and --resume are mutually exclusive")
    if replace_session:
        nonempty(restart_evidence, "session replacement restart evidence")
        if prompt is None:
            raise ValueError("session replacement requires --prompt-file")
    continuation = resume or replace_session
    worker_preflight.observe(session, child, persist=True)
    session.agreed()
    parent, items = session.work_items()
    if child not in items:
        raise ValueError(
            "manual worker requires a direct child or the standalone issue"
        )
    from .adopted_tasks import check_task_adoptions

    check_task_adoptions(session.gh, parent.body)
    check_task_adoptions(session.gh, items[child].body)
    try:
        instruction = read_original(prompt) if prompt is not None else None
    except OSError:
        raise ValueError("worker instruction file unavailable") from None
    instruction_digest = (
        hashlib.sha256(instruction.encode("utf-8")).hexdigest()
        if instruction is not None
        else None
    )
    ended = task.get("worker_ended")
    retry_unstarted = bool(task.get("launch_failures")) and not any(
        r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)
    )
    if ended and not continuation and not retry_unstarted:
        raise ValueError(
            "ended delegation requires an explicit resume with restart evidence"
        )
    plan = session.launch_plan(restart_child=child)
    if ended and automatic is None:
        limit = plan["policy"]["max_parallel_children"]
        if limit is not None and len(plan["active_reservations"]) >= limit:
            raise ValueError("implementation reservation limit reached")
    planned = next(t for t in plan["tasks"] if t["child"] == child)
    reasons = [
        r
        for r in planned["reasons"]
        if not (
            (
                (
                    session.worker != "primary"
                    or task.get("astra_handoff", {}).get("to") == task["delegation_id"]
                )
                and r["code"] == "existing_pr"
            )
            or continuation
            and r["code"] in {"existing_pr", "completed", "parent_closed"}
            or automatic is not None
            and r["code"] == "parallel_limit"
        )
    ]
    if reasons:
        raise ValueError(
            "manual worker is blocked: " + "; ".join(r["message"] for r in reasons)
        )
    migration = task.get("astra_handoff", {}).get("to") == task["delegation_id"]
    if (continuation or migration) and session.worker == "primary":
        prs = [
            p
            for p in session.gh.pulls()
            if child in closing_issues(p.get("body") or "", session.gh.repo)
        ]
        if len(prs) > 1 or any(
            (p["state"] != "open" and not p.get("merged_at"))
            or p["head"]["ref"] != task["branch"]
            for p in prs
        ):
            raise ValueError(
                "worker continuation requires a unique open or merged PR on the reserved branch"
            )
        if repair is not None and (
            len(prs) != 1
            or prs[0]["number"] != repair["pr"]
            or prs[0]["state"] != "open"
        ):
            raise ValueError("repair requires the creating worker's open PR")
    with session.gh.io_scope():
        validate_worktree(Path(task["worktree"]), task["branch"])
        for base in planned["stacked_on"]:
            git(
                "merge-base",
                "--is-ancestor",
                base["head_sha"],
                "HEAD",
                cwd=Path(task["worktree"]),
            )
    previous: Path | None = None
    previous_record = None
    replacement = None
    runner_model: str | None
    unstarted_migration = bool(
        worker_preflight.state(task).get("migration_unstarted")
    ) and not any(
        r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)
    )
    if not continuation or (unstarted_migration and not replace_session):
        if any(r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)):
            raise ValueError("worker already has a run; inspect it and use --resume")
        if restart_evidence is not None and not resume:
            raise ValueError("restart evidence requires --resume")
        model = implementation_model(model)
        if model_reason is None:
            model_reason = "通常の実装として agents.conf の既定モデルを選択。"
        runner_model = model
    else:
        previous = latest_run(session, task)
        with session.gh.io_scope():
            settings = validate_run(task, previous)
        model = implementation_model(model, previous=settings["selected"]["model"])
        if replace_session:
            # The delegation keeps its model, but a replacement starts a new session.
            implementation_model(model)
        # A legacy default run has no model override. Adding one would change
        # the runner's frozen selection_sources/overrides even for the same model.
        runner_model = settings["overrides"].get("model")
        if model_reason is None:
            model_reason = task.get("model_reason")
        with session.gh.io_scope():
            proof = Runtime(RUNNER).inspect(previous)
        previous_record = proof["record"]
        if model_settings(task, proof["record"]) != settings:
            raise ValueError("saved model settings changed during inspection")
        if not Runtime(RUNNER).ended(proof):
            raise ValueError(
                "worker termination and cleanup must be confirmed before resume"
            )
        if resume and previous_record.get("recovery_error"):
            raise ValueError(
                "previous turn evidence is unconfirmed; inspect stop recovery or use --replace-session"
            )
        if replace_session:
            with session.gh.io_scope():
                replacement = replacement_evidence(session, task, restart_evidence)
            current_settings = resolve(
                DEFAULT_CONFIG, "codex", None, runner_model, None, "write"
            )
            if (
                execution_contract({"settings": current_settings})["settings"]
                != session_contract(previous_record["contract"])["settings"]
            ):
                raise ValueError(
                    "current worker settings differ from the saved replacement contract"
                )
            previous = None
            previous_record = None
    directory = (
        session.store.directory
        / "worker-runs"
        / task["delegation_id"]
        / uuid.uuid4().hex
    )
    command = [str(RUNNER)]
    if previous is not None:
        command += ["resume", "--resume-from", str(previous)]
    command += [
        "--role",
        "implementation",
        "--mode",
        "write",
        "--task-id",
        task["delegation_id"],
        "--cwd",
        task["worktree"],
        "--branch",
        task["branch"],
        # The same standard instruction as managed workers, sent on stdin.
        "--prompt-file",
        "-",
        "--run-dir",
        str(directory),
    ]
    if runner_model is not None:
        command += ["--model", runner_model]
    if repair is not None:
        if previous is None:
            raise ValueError("repair requires the creating worker's existing runtime")
        command += [
            "--repair-id",
            repair["repair_id"],
            "--repair-pr",
            str(repair["pr"]),
            "--effort",
            "high",
        ]
        instruction = repair_instruction(repair, original_text(instruction))
    if model_reason is not None:
        model_reason = nonempty(model_reason, "model selection reason")
    if replacement is not None:
        replacement["to_run"] = str(directory.resolve())
    phase, instruction = worker_preflight.prepare_turn(
        session,
        child,
        instruction,
        resume=continuation,
        additional=additional,
        previous_record=previous_record,
        planning=planning,
        replacement=replacement,
        previous_directory=previous,
    )
    if repair is not None:
        if phase["phase"] != "implementing":
            raise ValueError("repair requires an implementation turn")
        phase["turns"][-1]["repair"] = dict(repair)
    with session.gh.io_scope():
        turn = worker_preflight.commit_turn(task, phase, directory)
    prepared_instruction = prepare_instruction(
        session.parent,
        child,
        Path(task["worktree"]),
        task["branch"],
        items[child].body,
        instruction,
        turn,
        worker=session.worker,
    )
    text = prepared_instruction.text
    # Log capture annotates legacy turns. Preserve replaced lineages on every turn.
    old_turns = (
        copy.deepcopy(phase["turns"][:-1])
        if phase.get("session_replacements")
        else None
    )
    with session.unlocked() as external:
        worker_logs.capture(external, child, items[child].body, prepared_instruction)
    if old_turns is not None:
        phase["turns"][:-1] = old_turns
    task["preflight"]["turns"][-1]["prompt_sha256"] = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    # Claim a released slot and save the run in one transaction, before execution.
    if ended:
        task.pop("worker_ended")
    if restart_evidence is not None:
        task["restart_evidence"] = nonempty(restart_evidence, "restart evidence")
    task.update(
        authorization=authorization(session),
        requirements_sha256=phase["context"]["requirements"],
    )
    if repair is not None:
        task.setdefault("repair_attempts", {})[repair["repair_id"]] = {
            "request": dict(repair),
            "delegation_id": task["delegation_id"],
            "command": command,
            "instruction": text,
            "details_sha256": instruction_digest,
            "launch_state": "reserved",
        }
    else:
        worker_followup.reserve(session.row, task, directory, automatic, turn["phase"])
    append_run(
        session,
        task,
        directory,
        "replacement"
        if replace_session
        else "resume"
        if previous is not None
        else "initial",
        model,
        model_reason,
        turn,
    )
    return command, task["delegation_id"], text


def rollback_launch(
    store: Store,
    task: dict[str, Any],
    directory: Path,
    *,
    code: int | None,
    stderr: bytes,
    reason: str,
    evidence: str,
    previous_ended: dict[str, Any] | None = None,
) -> None:
    runs = worker_runs(task)
    phase = worker_preflight.state(task)
    turn = phase["turns"][-1]
    turn["cancelled"] = True
    if runs[-1]["kind"] == "replacement":
        replacement = phase["session_replacements"][-1]
        replacement["cancelled"] = True
        phase["runtime"] = replacement["previous_runtime"]
    for decision in phase["decisions"]:
        if decision["delivered_turn_id"] == turn["snapshot"]["turn_id"]:
            decision["delivered_turn_id"] = None
    failure = directory.parent / "rejected" / directory.name
    failure.mkdir(parents=True, exist_ok=True, mode=0o700)
    (failure / "stderr.txt").write_bytes(stderr)
    task["worker_runs"] = runs[:-1]
    task.setdefault("launch_failures", []).append(
        {
            "attempt": directory.name,
            "exit_code": code,
            "stderr_file": str(failure / "stderr.txt"),
        }
    )
    if not task.get("worker_ended"):
        task["worker_ended"] = previous_ended or {
            "delegation_id": task["delegation_id"],
            "evidence": evidence,
        }
    if task.get("manual_launcher", {}).get("directory") == str(directory):
        task.pop("manual_launcher")
    followup = worker_followup.current(task)
    if followup and followup["run_dir"] == str(directory):
        automatic = followup["id"] != directory.name
        followup.update(
            state="needs_parent" if automatic else "error",
            reason=reason
            + (": " + stderr.decode(errors="replace") if automatic else ""),
            run_dir=str(run_directory(store, runs[-2]))
            if len(runs) > 1 and runs[-2]["delegation_id"] == task["delegation_id"]
            else None,
        )
    store.save()


def directory_absent(directory: Path) -> bool:
    try:
        directory.lstat()
    except FileNotFoundError:
        return True
    return False


def require_launcher_exited(launcher: dict[str, Any]) -> None:
    process = launcher.get("process")
    # A fast-exiting launcher may already be gone before /proc is sampled.
    reaped = type(launcher.get("exit_code")) is int
    unknown = "manual launcher liveness is unconfirmed"
    if process is None and reaped:
        return
    if (
        not isinstance(process, dict)
        or set(process) != {"pid", "start_ticks", "boot_id"}
        or type(process.get("pid")) is not int
        or process["pid"] <= 0
        or not isinstance(process.get("start_ticks"), str)
        or not process["start_ticks"].isascii()
        or not process["start_ticks"].isdecimal()
        or not isinstance(process.get("boot_id"), str)
        or not process["boot_id"]
    ):
        raise ValueError(unknown)
    pid = process["pid"]
    try:
        observed = process_identity(pid)
        if observed is None and not reaped:
            # A missing /proc observation alone can also mean a hidden PID.
            os.kill(pid, 0)
    except ProcessLookupError:
        return
    except (OSError, ValueError, IndexError, OverflowError):
        raise ValueError(unknown) from None
    if observed is None and not reaped:
        raise ValueError(unknown)
    if observed == process:
        raise ValueError("manual launcher is still alive")


def cancel_worker_launch(
    session: ParentSession, child: int, directory: Path
) -> dict[str, Any]:
    """Cancel the latest manual reservation while holding the parent operation lock."""
    task = current_task(session, child)
    directory = directory.absolute()
    runs = worker_runs(task)
    if (
        not runs
        or runs[-1]["delegation_id"] != task["delegation_id"]
        or runs[-1]["kind"] == "registered"
        or (session.store.directory / runs[-1]["run_dir"]) != directory
        or directory.parent
        != session.store.directory / "worker-runs" / task["delegation_id"]
        or directory.parent.resolve() != directory.parent
        or not re.fullmatch(r"[0-9a-f]{32}", directory.name)
    ):
        raise ValueError("--run-dir must match the latest manual launch reservation")
    phase = worker_preflight.state(task)
    turns = phase["turns"]
    if (
        not turns
        or turns[-1].get("source") != str(directory)
        or turns[-1].get("cancelled")
        or turns[-1]["snapshot"] != runs[-1].get("preflight")
        or "repair" in turns[-1]
    ):
        raise ValueError("no matching manual worker turn to cancel")
    launcher = object_value(task.get("manual_launcher", {}))
    if launcher.get("directory") != str(directory):
        raise ValueError("manual launcher liveness is unconfirmed")
    require_launcher_exited(launcher)
    # The common runner creates this directory before starting the worker CLI.
    # Check after confirming exit so the launcher cannot create it later.
    if not directory_absent(directory):
        raise ValueError("run directory exists; inspect runtime and cleanup")
    log_path = turns[-1].get("log", {}).get("journal")
    log_result = worker_logs.reject_unstarted(log_path, directory)
    if log_path is not None and log_result.get("stage") != "not_started":
        raise ValueError(
            "could not confirm unstarted worker log; inspect the journal before retrying"
        )
    evidence = "Manual launcher exited without creating the reserved run directory."
    rollback_launch(
        session.store,
        task,
        directory,
        code=launcher.get("exit_code"),
        stderr=evidence.encode(),
        reason="cancelled_unstarted",
        evidence=evidence,
    )
    return {
        "child": child,
        "worker": session.worker,
        "delegation_id": task["delegation_id"],
        "result": "cancelled_unstarted",
        "log_publication": log_result,
    }


def run_worker(
    gh: GitHub,
    store: Store,
    parent: int,
    identity: str,
    *,
    child: int,
    prompt: Path | None,
    resume: bool,
    base_ref: str | None = None,
    restart_evidence: str | None,
    model: str | None = None,
    model_reason: str | None = None,
    additional: str | None = None,
    planning: bool = False,
    replace_session: bool = False,
    repair: dict[str, Any] | None = None,
    worker: str | None = None,
    automatic: str | None = None,
) -> tuple[int, dict[str, Any]]:
    if automatic is not None and (not resume or repair or replace_session):
        raise ValueError("automatic follow-up requires a normal session resume")
    require_runner_environment()
    if repair is not None:
        gh.targeted_history = True
    with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
        session = ParentSession(gh, store, parent, identity)
        worker = worker_slots.select(
            session.row, child, worker, creating=base_ref is not None
        )
        session.worker = worker
        if automatic is None:
            session.agreed()
        if base_ref is not None:
            from .snapshot import worker_location

            if (
                session.row.get("dispatch")
                or resume
                or replace_session
                or repair
                or restart_evidence
            ):
                raise ValueError(
                    "--from is for a new manual worker; continue an existing session with --resume"
                )
            if prompt is None:
                raise ValueError("initial worker requires --prompt-file")
            nonempty(read_original(prompt), "worker instruction")
            implementation_model(model)
            branch, tree = worker_location(parent, child, worker)
            existing = session.worker_task(child)
            if not existing.get("delegated"):
                session.delegate(
                    child, "codex", branch, tree, None, None, base_ref=base_ref
                )
            elif (
                existing.get("worker_runs")
                or existing.get("worktree") != str(tree)
                or existing.get("branch") != branch
            ):
                raise ValueError(
                    "worker already reserved or started; inspect-worker before continuing"
                )
            else:
                from .snapshot import prepare_worker_tree

                with session.gh.io_scope():
                    prepare_worker_tree(
                        tree,
                        branch,
                        base_ref,
                        store.directory,
                    )
        previous_ended = copy.deepcopy(current_task(session, child).get("worker_ended"))
        command, delegation, instruction = prepare_run(
            session,
            child,
            prompt,
            resume,
            restart_evidence,
            model,
            model_reason,
            additional,
            planning,
            replace_session,
            repair,
            automatic,
        )
        if repair is not None:
            attempt = current_task(session, child)["repair_attempts"][
                repair["repair_id"]
            ]
            if attempt["launch_state"] != "reserved":
                return 0, repair_receipt(child, delegation, command, attempt)
        log_path = (
            session.worker_task(child)["preflight"]["turns"][-1]
            .get("log", {})
            .get("journal")
        )
        from decision_logs.observations import bind_worker_task

        bind_worker_task(session.worker_task(child))
        reserved_owner = {
            k: session.row[k] for k in ("session", "status", "generation")
        }
        reserved_task = session.worker_task(child)
        reserved_run = copy.deepcopy(worker_runs(reserved_task)[-1])
        reserved_turn = copy.deepcopy(reserved_task["preflight"]["turns"][-1])
        reserved_replacement = (
            copy.deepcopy(reserved_task["preflight"]["session_replacements"][-1])
            if replace_session
            else None
        )
    # A foreground runner must not hold a registry or parent operation lock.
    # Bytes in both directions: the runner reads UTF-8 and the locale may not be.
    from .improvement_runtime import execute_manual

    admission = execute_manual(
        gh,
        store,
        parent,
        identity,
        child,
        command,
        instruction,
        delegation,
        generation=reserved_owner["generation"],
        worker=worker,
        reserved_run=reserved_run,
        reserved_turn=reserved_turn,
    )
    proc = admission.completed
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    directory = Path(command[command.index("--run-dir") + 1])
    if repair is not None and admission.reused:
        with store.locked():
            session = ParentSession(gh, store, parent, identity)
            session.worker = worker
            attempt = current_task(session, child)["repair_attempts"][
                repair["repair_id"]
            ]
            return 0, repair_receipt(child, delegation, command, attempt)
    if not admission.spawned:
        if repair is not None:
            with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
                store.focus(session.key)
                row = store.parents.get(session.key, {})
                current = worker_slots.record(row, child, worker)
                attempt = current["repair_attempts"][repair["repair_id"]]
                if (
                    any(row.get(k) != reserved_owner[k] for k in reserved_owner)
                    or current.get("delegation_id") != delegation
                    or worker_runs(current)[-1] != reserved_run
                    or current["preflight"]["turns"][-1] != reserved_turn
                ):
                    raise RegistryConflict(
                        "repair reservation changed after rejected launch"
                    )
                if (
                    admission.repair_claim is not None
                    and attempt.get("launch_claim") == admission.repair_claim
                ):
                    attempt["launch_state"] = "reserved"
                    attempt.pop("launch_claim")
                elif attempt["launch_state"] != "reserved":
                    return 0, repair_receipt(child, delegation, command, attempt)
                store.save()
            failure = directory.parent / "rejected" / directory.name
            failure.mkdir(parents=True, exist_ok=True, mode=0o700)
            (failure / "stderr.txt").write_bytes(proc.stderr)
            return code, {
                **repair_receipt(child, delegation, command, attempt),
                "exit_code": code,
                "launch_state": "rejected_before_start",
                "next_action": "retry_same_operation",
                "reason": proc.stderr.decode("utf-8", errors="replace"),
            }
        with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
            # Roll back locally even when every GitHub request is failing. Never
            # construct ParentSession here: its login lookup is network I/O.
            store.focus(session.key)
            row = store.parents.get(session.key, {})
            current = worker_slots.record(row, child, worker)
            runs = worker_runs(current)
            turns = current.get("preflight", {}).get("turns", [])
            if (
                any(row.get(k) != reserved_owner[k] for k in reserved_owner)
                or current.get("delegation_id") != delegation
                or not runs
                or runs[-1] != reserved_run
                or not turns
                or turns[-1] != reserved_turn
                or (
                    replace_session
                    and (
                        current["preflight"]["session_replacements"][-1]
                        != reserved_replacement
                        or current["preflight"]["runtime"] is not None
                    )
                )
            ):
                raise RegistryConflict(
                    "worker reservation changed after rejected launch"
                )
            rollback_launch(
                store,
                current,
                directory,
                code=code,
                stderr=proc.stderr,
                reason="rejected_before_start",
                evidence="Runtime admission rejected this attempt before Popen; no worker started.",
                previous_ended=previous_ended,
            )
        log_result = worker_logs.reject_unstarted(log_path, directory)
        return code, {
            "child": child,
            "delegation_id": delegation,
            "exit_code": code,
            "runner_summary": None,
            "launch_state": "rejected_before_start",
            "next_action": "retry_same_operation",
            "log_publication": log_result,
            **(
                {"reason": admission.rejection_reason}
                if admission.rejection_reason
                else {}
            ),
        }
    if directory_absent(directory):
        # Retain the reservation. The parent may cancel after checking that the
        # exact launch has exited and that its directory is still absent.
        with store.operation(session.key), store.locked():
            store.focus(session.key)
            row = store.parents.get(session.key, {})
            launcher = worker_slots.record(row, child, worker).get(
                "manual_launcher", {}
            )
            if all(
                row.get(k) == reserved_owner[k] for k in reserved_owner
            ) and launcher.get("directory") == str(directory):
                launcher["exit_code"] = code
                store.save()
        return code or 2, {
            "child": child,
            "delegation_id": delegation,
            "exit_code": code or 2,
            "run_dir": str(directory),
            "runner_summary": None,
            "launch_state": "runtime_unconfirmed",
            "next_action": "inspect_or_cancel_worker_launch",
        }
    summary = None
    if proc.stdout.strip():
        try:
            summary = object_value(
                json.loads(proc.stdout, object_pairs_hook=unique_object)
            )
            if (
                set(summary) != SUMMARY_KEYS
                or type(summary["schema_version"]) is not int
                or summary["schema_version"] != 1
                or not isinstance(summary["run_id"], str)
                or not re.fullmatch(r"[A-Za-z0-9._-]+", summary["run_id"])
                or summary["result"] not in {"completed", "failed"}
                or not isinstance(summary["turn_end_reason"], str)
                or not re.fullmatch(r"[a-z_]+", summary["turn_end_reason"])
                or type(summary["exit_code"]) is not int
                or summary["exit_code"] != code
                or summary["task_success"] is not None
            ):
                raise ValueError("invalid runner summary")
        except (ValueError, TypeError, KeyError):
            raise ValueError(
                "common runner returned an invalid summary; inspect private records"
            ) from None
    elif code == 0:
        raise ValueError("common runner returned no summary; inspect private records")
    followup = None
    if summary is not None:
        from .review_handoff import runner_ended

        with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
            session = ParentSession(gh, store, parent, identity)
            session.worker = worker
            current = current_task(session, child)
            if (
                current["delegation_id"] != delegation
                or latest_run(session, current) != directory
            ):
                raise ValueError("worker reservation changed during execution")
            runner_ended(session, child)
            if replace_session:
                proof = Runtime(RUNNER).inspect(directory)
                if proof["record"].get("runtime", {}).get("session_id"):
                    worker_preflight.bind_runtime(
                        current["preflight"], proof["record"], directory=directory
                    )
            current.pop("manual_launcher", None)
            if repair is not None:
                current["repair_attempts"][repair["repair_id"]]["launch_state"] = (
                    "ended"
                )
            else:
                followup = worker_followup.current(current)
                if followup is None:
                    raise ValueError("worker follow-up reservation missing")
                if followup["reason"] != "worker_stopped":
                    code = worker_followup.finish(followup, directory, code)
            store.save()
    log_result = worker_logs.finish_manual(
        log_path,
        Path(command[command.index("--run-dir") + 1]),
        RUNNER,
        owner_check=publication_owner(
            store, gh.repo, parent, identity, reserved_owner["generation"]
        ),
    )
    if followup and followup["state"] == "waiting" and automatic is None:
        from .worker_monitor import Monitor, start

        try:
            start(store, gh.repo, parent, child, worker, followup["id"])
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            monitor = Monitor(store, gh.repo, parent, child, worker, followup["id"])
            # A lost launcher response can follow a successful service claim.
            if monitor.fail(str(exc), unclaimed_only=True):
                followup.update(state="error", reason=str(exc))
                code = 2
    return code, {
        "child": child,
        "delegation_id": delegation,
        "exit_code": code,
        "run_dir": str(directory),
        "runner_summary": summary,
        "log_publication": log_result,
        **({"followup": worker_followup.public_view(followup)} if followup else {}),
        **({"repair": dict(repair)} if repair is not None else {}),
        "final_file": str(directory / "final.md")
        if (directory / "final.md").exists()
        else None,
    }


def inspect_manual(session: ParentSession, child: int, *, stop: bool) -> dict[str, Any]:
    """The caller holds the parent operation lock while stopping a specific run."""
    with session.store.snapshot():
        task = current_task(session, child)
        directory = latest_run(session, task)
        delegation = task["delegation_id"]
        if not stop and directory_absent(directory):
            return {
                "child": child,
                "worker": session.worker,
                "delegation_id": delegation,
                "termination_confirmed": False,
                "result": "runtime_unconfirmed",
                "run_dir": str(directory),
                "final_file": None,
            }
        validate_run(task, directory)
    runtime = Runtime(RUNNER)
    proof = runtime.inspect(directory, stop=stop)
    with session.store.locked() if stop else session.store.snapshot():
        task = current_task(session, child)
        if (
            task["delegation_id"] != delegation
            or latest_run(session, task) != directory
        ):
            raise RegistryConflict("manual worker changed during inspection")
        matching_contract(task, proof["record"])
        terminated = runtime.ended(proof)
        if stop and terminated:
            worker_followup.stop(task)
            task["worker_ended"] = {
                "delegation_id": delegation,
                "evidence": "Runner confirmed termination and cleanup.",
            }
            session.store.save()
        return {
            "child": child,
            "worker": session.worker,
            "delegation_id": delegation,
            "termination_confirmed": terminated,
            "result": proof["record"]["result"],
            "run_dir": str(directory),
            **(
                {"followup": worker_followup.public_view(followup)}
                if (followup := worker_followup.current(task))
                and followup["run_dir"] == str(directory)
                else {}
            ),
            "final_file": str(directory / "final.md")
            if (directory / "final.md").is_file()
            else None,
        }


def repair_receipt(
    child: int, delegation: str, command: list[str], attempt: dict[str, Any]
) -> dict[str, Any]:
    directory = Path(command[command.index("--run-dir") + 1])
    return {
        "child": child,
        "delegation_id": delegation,
        "repair": dict(attempt["request"]),
        "run_dir": str(directory),
        "launch_state": attempt["launch_state"],
        "next_action": "read_repair_report"
        if attempt["launch_state"] == "ended"
        else "inspect_reserved_runtime",
        "final_file": str(directory / "final.md")
        if (directory / "final.md").is_file()
        else None,
    }
