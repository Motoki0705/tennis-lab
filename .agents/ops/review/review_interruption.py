"""Explicit local closure of an interrupted run; never a formal review result."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parent.policy import digest, object_value
from parent.snapshot import git
from parent_review_policy import execution_stages, hex_value, load_json

CLOSED = "interrupted_closed"


def agent_started(execution: Mapping[str, Any]) -> bool:
    stages = execution_stages(execution)
    return (
        execution.get("agent_invocations") == 1
        or stages is None
        or stages["agent"].get("status") != "not_started"
    )


def interruption_record(
    execution: Mapping[str, Any], request: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Validate a durable terminal record before allowing another run to launch."""
    if execution.get("status") != CLOSED:
        return None
    record: dict[str, Any] = object_value(execution.get("interruption"))
    if (
        type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or execution.get("request_sha256") != digest(request)
        or record.get("request_sha256") != digest(request)
        or type(execution.get("agent_invocations")) is not int
        or execution["agent_invocations"] not in {0, 1}
        or record.get("publication") != "not_published"
        or record.get("workspace_cleanup_complete") is not True
        or record.get("agent_started") is not agent_started(execution)
        or not isinstance(record.get("reason"), str)
        or not record["reason"].strip()
        or not isinstance(record.get("closed_at"), str)
        or not record["closed_at"]
        or execution.get("finished_at") != record["closed_at"]
        or not isinstance(record.get("session"), str)
        or not record["session"]
        or not isinstance(record.get("operator_authorization"), dict)
    ):
        raise ValueError("invalid interrupted closure record")
    if record["agent_started"]:
        hex_value(record.get("runtime_sha256"), 64)
    elif record.get("runtime_sha256") is not None:
        raise ValueError("pre-agent closure cannot claim runtime evidence")
    return record


def require_not_closed(
    execution: Mapping[str, Any], request: Mapping[str, Any]
) -> None:
    if interruption_record(execution, request) is not None:
        raise ValueError("review run is explicitly closed; create a new review request")


def cleanup_evidence(
    run_dir: Path, request: Mapping[str, Any], execution: Mapping[str, Any]
) -> str | None:
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
    if agent_started(execution):
        runtime_path = run_dir / "agent-runtime/result.json"
        instructions = (
            "runtime cleanup is unconfirmed; use agent_exec.sh stop --run-dir "
            f"{run_dir / 'agent-runtime'}; if the runner was lost, supply "
            "--cleanup-confirmation after supervisor reconciliation"
        )
        if not runtime_path.exists():
            raise ValueError(instructions)
        raw = runtime_path.read_bytes()
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
        runtime_sha256 = hashlib.sha256(raw).hexdigest()
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
    return runtime_sha256
