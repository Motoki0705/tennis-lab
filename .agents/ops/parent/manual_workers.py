"""Explicit manual-worker launches with durable, private runner provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dispatcher.runtime import Runtime, load
from lib.agent_settings import implementation_model
from worker_logs import service as worker_logs
from worker_logs.sources import read_original

from . import worker_preflight
from .execution import authorization
from .github import GitHub
from .policy import closing_issues, nonempty, object_value
from .session import ParentSession
from .snapshot import git, validate_worktree
from .state import Store, unique_object
from .worker_instruction import prepare_instruction

RUNNER = Path(__file__).resolve().parents[1] / "lib/agent_exec.sh"
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


def current_task(session: ParentSession, child: int) -> dict[str, Any]:
    row = session.owner()
    if row.get("dispatch") is not None:
        raise ValueError("manual worker commands cannot change dispatcher workers")
    task: dict[str, Any] = object_value(row["tasks"].get(str(child), {}))
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
            or run["kind"] not in {"initial", "resume", "registered"}
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
    implementation_model(model)
    if (
        contract.get("mode") != "write"
        or settings.get("agent") != "codex"
        or settings.get("role") != "implementation"
        or settings.get("profile") is not None
        or selected.get("effort") != "max"
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
        append_run(session, task, path, "registered", model, reason)
    else:
        task.update(model=model, model_reason=reason)
        session.store.save()
    return {
        "child": child,
        "delegation_id": task["delegation_id"],
        "result": "recorded",
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
) -> tuple[list[str], str, str]:
    task = current_task(session, child)
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
    ended = task.get("worker_ended")
    retry_unstarted = bool(task.get("launch_failures")) and not any(
        r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)
    )
    if ended and not resume and not retry_unstarted:
        raise ValueError(
            "ended delegation requires an explicit resume with restart evidence"
        )
    plan = session.launch_plan(restart_child=child)
    if ended:
        limit = plan["policy"]["max_parallel_children"]
        if limit is not None and len(plan["active_reservations"]) >= limit:
            raise ValueError("implementation reservation limit reached")
    planned = next(t for t in plan["tasks"] if t["child"] == child)
    reasons = [
        r
        for r in planned["reasons"]
        if not (resume and r["code"] in {"existing_pr", "completed", "parent_closed"})
    ]
    if reasons:
        raise ValueError(
            "manual worker is blocked: " + "; ".join(r["message"] for r in reasons)
        )
    if resume:
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
    runner_model: str | None
    unstarted_migration = bool(
        worker_preflight.state(task).get("migration_unstarted")
    ) and not any(
        r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)
    )
    if not resume or unstarted_migration:
        if any(r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)):
            raise ValueError("worker already has a run; inspect it and use --resume")
        if restart_evidence is not None and not resume:
            raise ValueError("restart evidence requires --resume")
        model = implementation_model(model)
        if model_reason is None:
            if model != implementation_model(None):
                raise ValueError("initial alternate model requires --model-reason-file")
            model_reason = "通常の実装として agents.conf の既定モデルを選択。"
        runner_model = model
    else:
        previous = latest_run(session, task)
        settings = validate_run(task, previous)
        model = implementation_model(model, previous=settings["selected"]["model"])
        # A legacy default run has no model override. Adding one would change
        # the runner's frozen selection_sources/overrides even for the same model.
        runner_model = settings["overrides"].get("model")
        if model_reason is None:
            model_reason = task.get("model_reason")
        proof = Runtime(RUNNER).inspect(previous)
        previous_record = proof["record"]
        if model_settings(task, proof["record"]) != settings:
            raise ValueError("saved model settings changed during inspection")
        if not Runtime(RUNNER).ended(proof):
            raise ValueError(
                "worker termination and cleanup must be confirmed before resume"
            )
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
    if model_reason is not None:
        model_reason = nonempty(model_reason, "model selection reason")
    phase, instruction = worker_preflight.prepare_turn(
        session,
        child,
        instruction,
        resume=resume,
        additional=additional,
        previous_record=previous_record,
        planning=planning,
    )
    turn = worker_preflight.commit_turn(task, phase, directory)
    prepared_instruction = prepare_instruction(
        session.parent,
        child,
        Path(task["worktree"]),
        task["branch"],
        items[child].body,
        instruction,
        turn,
    )
    text = prepared_instruction.text
    worker_logs.capture(session, child, items[child].body, prepared_instruction)
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
    append_run(
        session,
        task,
        directory,
        "resume" if previous is not None else "initial",
        model,
        model_reason,
        turn,
    )
    return command, task["delegation_id"], text


def run_worker(
    gh: GitHub,
    store: Store,
    parent: int,
    identity: str,
    *,
    child: int,
    prompt: Path | None,
    resume: bool,
    restart_evidence: str | None,
    model: str | None = None,
    model_reason: str | None = None,
    additional: str | None = None,
    planning: bool = False,
) -> tuple[int, dict[str, Any]]:
    with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
        session = ParentSession(gh, store, parent, identity)
        session.agreed()
        previous_task = copy.deepcopy(current_task(session, child))
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
        )
        log_path = (
            session.row["tasks"][str(child)]["preflight"]["turns"][-1]
            .get("log", {})
            .get("journal")
        )
        from decision_logs.observations import bind_worker_task

        bind_worker_task(session.row["tasks"][str(child)])
    # A foreground runner must not hold a registry or parent operation lock.
    # Bytes in both directions: the runner reads UTF-8 and the locale may not be.
    from .improvement_runtime import execute_manual

    proc = execute_manual(
        gh, store, parent, identity, child, command, instruction, delegation
    )
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
    directory = Path(command[command.index("--run-dir") + 1])
    if code in {2, 127} and not directory.exists():
        with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
            session = ParentSession(gh, store, parent, identity)
            current = current_task(session, child)
            if (
                current["delegation_id"] != delegation
                or latest_run(session, current) != directory
            ):
                raise ValueError("worker reservation changed after rejected launch")
            failure = directory.parent / "rejected" / directory.name
            failure.mkdir(parents=True, mode=0o700)
            (failure / "stderr.txt").write_bytes(proc.stderr)
            previous_task.setdefault("launch_failures", []).append(
                {
                    "attempt": directory.name,
                    "exit_code": code,
                    "stderr_file": str(failure / "stderr.txt"),
                }
            )
            attempted = copy.deepcopy(current["preflight"])
            attempted_turn = attempted["turns"][-1]
            attempted_turn["cancelled"] = True
            for decision in attempted["decisions"]:
                if (
                    decision["delivered_turn_id"]
                    == attempted_turn["snapshot"]["turn_id"]
                ):
                    decision["delivered_turn_id"] = None
            previous_task["preflight"] = attempted
            previous_task["worker_ended"] = {
                "delegation_id": delegation,
                "evidence": "Common runner rejected this attempt before creating a runtime; no worker started.",
            }
            for field in ("stopped", "last_status", "last_operation"):
                if field in current:
                    previous_task[field] = current[field]
            session.row["tasks"][str(child)] = previous_task
            store.save()
        log_result = worker_logs.reject_unstarted(log_path)
        return code, {
            "child": child,
            "delegation_id": delegation,
            "exit_code": code,
            "runner_summary": None,
            "launch_state": "rejected_before_start",
            "next_action": "retry_same_operation",
            "log_publication": log_result,
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
    if summary is not None:
        from .review_handoff import runner_ended

        with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
            session = ParentSession(gh, store, parent, identity)
            current = current_task(session, child)
            if (
                current["delegation_id"] != delegation
                or latest_run(session, current) != directory
            ):
                raise ValueError("worker reservation changed during execution")
            runner_ended(session, child)
            current.pop("manual_launcher", None)
            store.save()
    log_result = worker_logs.finish_manual(
        log_path, Path(command[command.index("--run-dir") + 1]), RUNNER
    )
    return code, {
        "child": child,
        "delegation_id": delegation,
        "exit_code": code,
        "runner_summary": summary,
        "log_publication": log_result,
        "final_file": str(directory / "final.md")
        if (directory / "final.md").exists()
        else None,
    }
