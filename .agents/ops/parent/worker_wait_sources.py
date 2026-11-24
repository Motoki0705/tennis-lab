"""Read CI, training-queue, and local job results for a manual follow-up."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dispatcher.runtime import alive

from .local_jobs import (
    log_excerpt,
    read_completion,
    read_log_tail,
    read_meta,
    read_running,
    validate_job_id,
)
from .policy import object_value

if TYPE_CHECKING:
    from lib.ops_review.progression_backend import LiveBackend

QUEUE_COMPLETION_GRACE = 60.0
LOCAL_COMPLETION_GRACE = 10.0


def require_ci_target(
    pr: dict[str, Any], repo: str, branch: str, request: dict[str, Any]
) -> None:
    head = object_value(pr.get("head"))
    if (
        pr.get("number") != request["pr"]
        or pr.get("state") != "open"
        or head.get("ref") != branch
        or str(object_value(head.get("repo")).get("full_name")).casefold()
        != repo.casefold()
        or head.get("sha") != request["head"]
    ):
        raise ValueError("CI target differs from the open PR on the reserved HEAD")


def _conflict_result(
    pr: int, head: str, ci: dict[str, Any], latest: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "ci",
        "pr": pr,
        "head": head,
        "status": "conflict",
        "reasons": ["pr_conflict", *ci["reasons"]],
    }
    base = latest.get("base")
    if base is not None:
        base_ref = object_value(base).get("ref")
        if isinstance(base_ref, str) and base_ref.strip():
            result["base_ref"] = base_ref
    mergeable_state = latest.get("mergeable_state")
    if isinstance(mergeable_state, str) and mergeable_state.strip():
        result["mergeable_state"] = mergeable_state
    return result


def ci_result(
    backend: LiveBackend, request: dict[str, Any], branch: str, *, root: Path
) -> dict[str, Any] | None:
    pr, head = request["pr"], request["head"]
    current = backend.pr(pr)
    require_ci_target(current, backend.repo, branch, request)
    ci = backend.ci(pr, head)
    if "ci_python_head_mismatch" in ci["reasons"]:
        raise ValueError("CI check was returned for a different HEAD")
    python_missing = "ci_python_missing" in ci["reasons"]
    if ci["reasons"] and set(ci["reasons"]) <= {
        "ci_python_missing",
        "ci_python_skipped",
    }:
        from lib.ops_review import knowledge_records

        _, comparison = knowledge_records.assess(root, backend.repo, current)
        if comparison is not None:
            ci = knowledge_records.observe_ci(backend, current, comparison)
    if ci["status"] == "pending":
        if python_missing and current.get("mergeable") is False:
            latest = backend.pr(pr)
            require_ci_target(latest, backend.repo, branch, request)
            if latest.get("mergeable") is False:
                return _conflict_result(pr, head, ci, latest)
        return None
    result = {
        "kind": "ci",
        "pr": pr,
        "head": head,
        "status": ci["status"],
        "reasons": ci["reasons"],
    }
    if "workflow" in ci:
        result.update(
            {
                key: ci[key]
                for key in ("workflow", "base_sha", "merge_sha", "run")
                if key in ci
            }
        )
    if ci["status"] == "failure":
        failure = backend.failure({"number": pr}, ci)
        result.update(
            failed_jobs=[
                {
                    **{key: job[key] for key in ("name", "run_id", "attempt", "url")},
                    "failed_steps": [
                        step["name"]
                        for step in job["steps"]
                        if step.get("conclusion") == "failure"
                    ],
                    "log_excerpt": _log_excerpt(job["log"]),
                    "log_unavailable_reason": job["log_unavailable_reason"],
                }
                for job in failure["failed_jobs"]
            ],
            missing_details=failure["missing_details"],
        )
    require_ci_target(backend.pr(pr), backend.repo, branch, request)
    return result


def _log_excerpt(log: str | None) -> str | None:
    excerpt = log_excerpt(log)
    return None if excerpt is None else str(excerpt)


def _format_local_completion(
    job_dir: Path, job: str, meta: dict[str, Any], completion: dict[str, Any]
) -> dict[str, Any]:
    log_path = job_dir / "output.log"
    tail = read_log_tail(log_path)
    return {
        "job_id": job,
        "name": completion.get("name") or meta["name"],
        "state": completion["state"],
        "exit_code": completion["exit_code"],
        "duration_seconds": completion.get("duration_seconds"),
        "log": str(log_path) if tail is not None else None,
        "log_excerpt": _log_excerpt(tail),
    }


def local_result(
    root: Path, job: str, *, now: float, missing_completion_since: dict[str, float]
) -> dict[str, Any] | None:
    job = validate_job_id(job)
    meta = read_meta(root, job)
    job_dir = root / job
    completion = read_completion(root, job)
    if completion is not None:
        missing_completion_since.pop(job, None)
        return _format_local_completion(job_dir, job, meta, completion)
    proc = read_running(root, job)
    if proc is not None and alive(proc):
        missing_completion_since.pop(job, None)
        return None
    # The wrapper writes completion.json right before exiting; re-check once
    # after a negative liveness probe before counting the grace window.
    completion = read_completion(root, job)
    if completion is not None:
        missing_completion_since.pop(job, None)
        return _format_local_completion(job_dir, job, meta, completion)
    since = missing_completion_since.setdefault(job, now)
    if now - since >= LOCAL_COMPLETION_GRACE:
        raise ValueError(f"local job terminated without completion record: {job}")
    return None


def queue_result(
    queue: Path, job: str, *, now: float, missing_completion_since: dict[str, float]
) -> dict[str, Any] | None:
    from src.utils.training_queue import TERMINAL, read_completion, regular

    # A stopped worker can requeue a job before it starts. Recheck jobs/ if
    # both active paths were absent: the rename may have crossed those reads.
    queued = regular(queue / "jobs" / job)
    running = regular(queue / "running" / job)
    active = queued or running or regular(queue / "jobs" / job)
    # Shell first moves the job, then publishes the record after verified
    # teardown. A terminal file alone starts a bounded publication grace.
    terminal = [state for state in TERMINAL if regular(queue / state / job)]
    record = read_completion(queue, job)
    if record is None:
        if terminal:
            since = missing_completion_since.setdefault(job, now)
            if now - since >= QUEUE_COMPLETION_GRACE:
                raise ValueError(f"missing training-queue completion record for {job}")
        elif not active:
            raise ValueError(f"unknown training-queue job_id: {job}")
        return None
    missing_completion_since.pop(job, None)
    if terminal and terminal != [record["state"]]:
        raise ValueError(f"training-queue terminal state disagrees for {job}")
    if (
        not terminal
        and "expired_at_ns" not in record
        and not regular(queue / record["state"] / job)
    ):
        # Collection can remove the terminal file while we read it.
        current = read_completion(queue, job)
        if current is None or "expired_at_ns" not in current:
            raise ValueError(f"missing terminal job evidence for {job}")
    stem = job.removesuffix(".job")
    log_path = queue / "logs" / f"{stem}.log"
    repro = queue / "repro" / stem
    pointer = repro / "output_dir.txt"
    exit_code = None
    log = None
    if regular(log_path):
        with log_path.open("rb") as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - 16384))
            codes = re.findall(rb"^exit_code=(\d+)$", stream.read(), re.M)
        exit_code = int(codes[-1]) if codes else None
        log = str(log_path)
    return {
        "job_id": job,
        "state": record["state"],
        "exit_code": exit_code,
        "log": log,
        "repro": str(repro) if repro.is_dir() else None,
        "output_dir": pointer.read_text().strip() or None if regular(pointer) else None,
    }
