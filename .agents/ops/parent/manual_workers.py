"""Explicit manual-worker launches with durable, private runner provenance."""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dispatcher.runtime import Runtime, load

from .execution import authorization, task_errors
from .github import GitHub
from .policy import closing_issues, nonempty, object_value
from .session import ParentSession
from .snapshot import git, validate_worktree
from .state import Store, unique_object

RUNNER = Path(__file__).resolve().parents[1] / "lib/agent_exec.sh"
SUMMARY_KEYS = {
    "schema_version",
    "run_id",
    "result",
    "turn_end_reason",
    "exit_code",
    "task_success",
}


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
            or set(run) != {"delegation_id", "run_dir", "kind", "recorded_at"}
            or not isinstance(run["delegation_id"], str)
            or not re.fullmatch(r"[0-9a-f]{32}", run["delegation_id"])
            or not isinstance(run["run_dir"], str)
            or not run["run_dir"]
            or run["kind"] not in {"initial", "resume", "registered"}
            or not isinstance(run["recorded_at"], str)
        ):
            raise ValueError("malformed private worker run records")
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


def validate_run(task: dict[str, Any], directory: Path) -> None:
    """Check every available contract; invocation alone covers an in-flight run."""
    try:
        found = False
        for name in ("result.json", "invocation.json"):
            path = directory / name
            if path.exists() or path.is_symlink():
                record = load(path)
                if (
                    type(record.get("schema_version")) is not int
                    or record["schema_version"] != 1
                ):
                    raise ValueError("unsupported runner record")
                matching_contract(task, record)
                found = True
        if not found:
            raise ValueError("runner records missing")
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError(
            "runner records unavailable or do not match the current delegation"
        ) from None


def append_run(
    session: ParentSession, task: dict[str, Any], directory: Path, kind: str
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
        },
    ]
    session.store.save()


def record_worker_run(
    session: ParentSession, child: int, directory: Path
) -> dict[str, Any]:
    task = current_task(session, child)
    path = directory.resolve()
    validate_run(task, path)
    if not any(
        r["delegation_id"] == task["delegation_id"]
        and run_directory(session.store, r) == path
        for r in worker_runs(task)
    ):
        append_run(session, task, path, "registered")
    return {
        "child": child,
        "delegation_id": task["delegation_id"],
        "result": "recorded",
    }


def prepare_run(
    session: ParentSession,
    child: int,
    prompt: Path,
    resume: bool,
    restart_evidence: str | None,
) -> tuple[list[str], str]:
    task = current_task(session, child)
    session.agreed()
    parent, items = session.work_items()
    if child not in items:
        raise ValueError(
            "manual worker requires a direct child or the standalone issue"
        )
    from .adopted_tasks import check_task_adoptions

    check_task_adoptions(session.gh, parent.body)
    check_task_adoptions(session.gh, items[child].body)
    if task_errors(task, authorization(session), items[child].body):
        raise ValueError(
            "worker delegation inputs changed; reconcile the current reservation"
        )
    try:
        nonempty(prompt.read_text(encoding="utf-8"), "worker instruction")
    except OSError:
        raise ValueError("worker instruction file unavailable") from None
    ended = task.get("worker_ended")
    if ended:
        if not resume:
            raise ValueError(
                "ended delegation requires an explicit resume with restart evidence"
            )
        nonempty(restart_evidence, "restart evidence for the ended delegation")
    plan = session.launch_plan(restart_child=child)
    if ended:
        limit = plan["policy"]["max_parallel_children"]
        if limit is not None and len(plan["active_reservations"]) >= limit:
            raise ValueError("implementation reservation limit reached")
    planned = next(t for t in plan["tasks"] if t["child"] == child)
    reasons = [
        r for r in planned["reasons"] if not (resume and r["code"] == "existing_pr")
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
            p["state"] != "open" or p["head"]["ref"] != task["branch"] for p in prs
        ):
            raise ValueError(
                "worker continuation requires a unique open PR on the reserved branch"
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
    if not resume:
        if any(r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)):
            raise ValueError("worker already has a run; inspect it and use --resume")
        if restart_evidence is not None:
            raise ValueError("restart evidence requires --resume")
    else:
        previous = latest_run(session, task)
        validate_run(task, previous)
        proof = Runtime(RUNNER).inspect(previous)
        matching_contract(task, proof["record"])
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
        "--prompt-file",
        str(prompt.resolve()),
        "--run-dir",
        str(directory),
    ]
    # Claim a released slot and save the run in one transaction, before execution.
    if ended:
        task.pop("worker_ended")
    if restart_evidence is not None:
        task["restart_evidence"] = nonempty(restart_evidence, "restart evidence")
    append_run(session, task, directory, "resume" if resume else "initial")
    return command, task["delegation_id"]


def run_worker(
    gh: GitHub,
    store: Store,
    parent: int,
    identity: str,
    *,
    child: int,
    prompt: Path,
    resume: bool,
    restart_evidence: str | None,
) -> tuple[int, dict[str, Any]]:
    with store.operation(f"{gh.repo.casefold()}#{parent}"), store.locked():
        session = ParentSession(gh, store, parent, identity)
        command, delegation = prepare_run(
            session, child, prompt, resume, restart_evidence
        )
    # A foreground runner must not hold a registry or parent operation lock.
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        raise ValueError(
            "common runner could not start; inspect the recorded private run"
        ) from None
    code = proc.returncode if proc.returncode >= 0 else 128 - proc.returncode
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
    return code, {
        "child": child,
        "delegation_id": delegation,
        "exit_code": code,
        "runner_summary": summary,
    }
