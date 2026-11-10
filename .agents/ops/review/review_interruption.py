"""Explicit local closure of an interrupted run; never a formal review result."""

from __future__ import annotations

import fcntl
import hashlib
import importlib
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from parent.policy import digest, object_value
from parent.snapshot import git
from parent_review_policy import execution_stages, hex_value, load_json

CLOSED = "interrupted_closed"


def agent_started(execution: Mapping[str, Any]) -> bool:
    """Whether launch could have happened; missing stages never prove absence."""
    stages = execution_stages(execution)
    not_invoked = stages is not None and (
        stages["agent"].get("status") == "not_started"
        or (
            execution.get("kind") == "adoption"
            and stages["agent"].get("status") == "not_invoked"
        )
    )
    return execution.get("agent_invocations") != 0 or not not_invoked


def supervisor_confirmation(value: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Only for a missing native result: native identities were not observed.

    Keep the common six-key format, bind to the known review ID, and require an
    explicit supervisor declaration covering the wrapper and all descendants.
    None identifies missing observations, never proof that no process existed.
    """
    confirmation: dict[str, Any] = object_value(value)
    expected = {"run_id": request["review_run_id"], "runner": None, "process": None}
    # The runner is also a standalone CLI; use the same module loading convention
    # as the parent CLI rather than giving its dependencies two package identities.
    runtime = importlib.import_module("agent_runtime")
    runtime.validate_cleanup_confirmation(confirmation, expected)
    return confirmation


def validate_cleanup_record(
    record: dict[str, Any],
    execution: Mapping[str, Any],
    request: Mapping[str, Any],
    timestamp: str,
) -> None:
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or execution.get("request_sha256") != digest(request)
        or record.get("request_sha256") != digest(request)
        or type(execution.get("agent_invocations")) is not int
        or execution["agent_invocations"] not in {0, 1}
        or record.get("workspace_cleanup_complete") is not True
        or record.get("agent_started") is not agent_started(execution)
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
        or not isinstance(record.get(timestamp), str)
        or not record[timestamp]
        or not isinstance(record.get("session"), str)
        or not record["session"]
        or not isinstance(record.get("operator_authorization"), dict)
    ):
        raise ValueError("invalid review cleanup confirmation record")
    runtime = record.get("runtime_sha256")
    confirmation = record.get("cleanup_confirmation")
    if record["agent_started"]:
        if runtime is not None:
            hex_value(runtime, 64)
            if confirmation is not None:
                raise ValueError(
                    "runtime confirmation has conflicting supervisor proof"
                )
        elif confirmation is None:
            raise ValueError(
                "cleanup confirmation requires runtime or supervisor evidence"
            )
        else:
            supervisor_confirmation(confirmation, request)
    elif runtime is not None or confirmation is not None:
        raise ValueError("pre-agent confirmation cannot claim runtime evidence")


def interruption_record(
    execution: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Validate a durable terminal record before allowing another run to launch."""
    if execution.get("status") != CLOSED:
        return None
    record: dict[str, Any] = object_value(execution.get("interruption"))
    validate_cleanup_record(record, execution, request, "closed_at")
    if (
        record.get("publication") != "not_published"
        or execution.get("finished_at") != record["closed_at"]
    ):
        raise ValueError("invalid interrupted closure record")
    return record


def exit_confirmation_record(
    execution: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Lease-only evidence; gate, coverage and resolution never consume it."""
    if "exit_confirmation" not in execution:
        return None
    record: dict[str, Any] = object_value(execution["exit_confirmation"])
    validate_cleanup_record(record, execution, request, "confirmed_at")
    if record.get("publication") != "unchanged":
        raise ValueError("exit confirmation cannot alter publication")
    return record


def confirmed_exit(execution: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    if execution["status"] == "running":
        return False
    if interruption_record(execution, request) is not None:
        return True
    if exit_confirmation_record(execution, request) is not None:
        return True
    stages = execution_stages(execution)
    if stages is None:
        return False
    agent = stages["agent"]
    return (
        agent.get("cleanup_complete") is True
        and agent.get("workspace_cleanup_complete") is True
        and (
            agent.get("status") == "finished"
            if agent_started(execution)
            else agent.get("status") in {"not_started", "not_invoked"}
        )
    )


@contextmanager
def runtime_lock(run_dir: Path) -> Iterator[None]:
    """The common runner owns this lock even before its initial result save."""
    runtime = run_dir / "agent-runtime"
    if not runtime.exists():
        yield
        return
    with (runtime / "session.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("common runner is active; cannot confirm cleanup") from exc
        yield


def require_not_closed(
    execution: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    if interruption_record(execution, request) is not None:
        raise ValueError("review run is explicitly closed; create a new review request")


def cleanup_evidence(
    run_dir: Path,
    request: Mapping[str, Any],
    execution: Mapping[str, Any],
    confirmation: Path | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Require runner cleanup, not a valid final or a successful agent turn.

    The caller holds PR leases and run.lock. Never infer exit from time or PID.
    A legacy/missing stage cannot establish that an agent was never started.
    """
    workspace_name = "review-" + request["review_run_id"]
    roots = {Path(git("rev-parse", "--show-toplevel").decode().strip())}
    config_path = run_dir / "retrieval-config.json"
    if config_path.exists():
        config = object_value(load_json(config_path.read_text(encoding="utf-8")))
        if (
            config.get("repository") != request["repo"]
            or config.get("run_id") != request["review_run_id"]
            or not isinstance(config.get("repo"), str)
            or not Path(config["repo"]).is_absolute()
        ):
            raise ValueError("workspace configuration belongs to another run")
        roots.add(Path(config["repo"]))
    paths = {root / ".claude/worktrees" / workspace_name for root in roots}
    runtime_sha256 = None
    supervisor = None
    if agent_started(execution):
        runtime_path = run_dir / "agent-runtime/result.json"
        instructions = (
            "runtime cleanup is unconfirmed; use agent_exec.sh stop --run-dir "
            f"{run_dir / 'agent-runtime'}; if the runner was lost, supply "
            "--cleanup-confirmation to agent_exec.sh stop after supervisor reconciliation; "
            "if runtime result is missing, use close-interrupted --cleanup-confirmation "
            "with a review-bound supervisor declaration covering all descendants"
        )
        if not runtime_path.exists():
            if confirmation is None:
                raise ValueError(instructions)
            supervisor = supervisor_confirmation(
                load_json(confirmation.read_text(encoding="utf-8")), request
            )
        elif confirmation is not None:
            raise ValueError(
                "native runtime exists; confirm cleanup via agent_exec.sh stop"
            )
        raw = runtime_path.read_bytes() if supervisor is None else None
        if raw is not None:
            runtime_sha256 = checked_runtime(raw, run_dir, paths, instructions)
    elif confirmation is not None:
        raise ValueError("pre-agent cleanup needs no supervisor confirmation")
    # Review PR checkouts share git's worktree registry. A missing directory with
    # a remaining registration is also unfinished cleanup, including another cwd.
    registered = set()
    for field in git("worktree", "list", "--porcelain", "-z").split(b"\0"):
        if field.startswith(b"worktree "):
            worktree = Path(field[len(b"worktree ") :].decode())
            for ancestor in (worktree, *worktree.parents):
                if ancestor.name == workspace_name:
                    registered.add(ancestor)
    remaining = registered | {p for p in paths if p.exists() or p.is_symlink()}
    if remaining:
        raise ValueError(
            "review workspace remains; after runtime cleanup, remove its PR "
            "worktrees with git worktree remove --force, then remove workspace "
            "metadata: " + ", ".join(str(p) for p in sorted(remaining))
        )
    return runtime_sha256, supervisor


def checked_runtime(
    raw: bytes, run_dir: Path, paths: set[Path], instructions: str
) -> str:
    runtime = object_value(load_json(raw.decode("utf-8")))
    if (
        runtime.get("cleanup_complete") is not True
        or runtime.get("result") not in {"completed", "failed"}
        or type(runtime.get("finished_at")) not in {int, float}
        or not math.isfinite(runtime["finished_at"])
        or runtime["finished_at"] <= 0
        or type(runtime.get("exit_code")) is not int
    ):
        raise ValueError(instructions)
    if any(
        not isinstance(runtime.get(key), str)
        or Path(runtime[key]).resolve() != expected.resolve()
        for key, expected in (
            ("run_dir", run_dir / "agent-runtime"),
            ("output_path", run_dir / "agent_output.md"),
            ("final_path", run_dir / "agent-runtime/final.md"),
        )
    ):
        raise ValueError("runtime cleanup evidence belongs to another run")
    cwd = object_value(runtime.get("contract")).get("cwd")
    if not isinstance(cwd, str) or Path(cwd).resolve() not in {
        p.resolve() for p in paths
    }:
        raise ValueError("runtime workspace belongs to another run")
    return hashlib.sha256(raw).hexdigest()
