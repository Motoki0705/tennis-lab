"""One explicit handoff: runner completion, results, fixed targets and instruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy import closing_issues, object_value
from .session import ParentSession
from .snapshot import prepare, transition


def runner_ended(session: ParentSession, child: int) -> None:
    task = session.row["tasks"].get(str(child), {})
    if not task.get("delegated"):
        return
    from .development import Runtime, managed, worker_jobs

    dispatch = managed(session.row)
    expected_task: str
    if dispatch:
        if task.get("worker_ended", {}).get("delegation_id") == task.get(
            "delegation_id"
        ):
            return
        jobs = worker_jobs(dispatch, child)
        if not jobs or jobs[-1]["delegation_id"] != task["delegation_id"]:
            raise ValueError("no runtime for the current delegation")
        job = jobs[-1]
        directories = [Path(job["directory"]) / "runtime"]
        runner = Path(job["runner"])
        expected_task = job["task_id"]
    else:
        from .manual_workers import RUNNER, current_runs, validate_run

        directories = current_runs(session, task)
        for directory in directories:
            validate_run(task, directory)
        runner = RUNNER
        expected_task = task["delegation_id"]
    runtime = Runtime(runner)
    for directory in directories:
        proof = runtime.inspect(directory)
        contract = object_value(proof["record"].get("contract"))
        if (
            not runtime.ended(proof)
            or contract.get("task_id") != expected_task
            or Path(contract.get("cwd", "")).resolve()
            != Path(task["worktree"]).resolve()
            or contract.get("branch") != task["branch"]
        ):
            raise ValueError(
                f"#{child}: runner completion does not match the current delegation"
            )
    if task.get("worker_ended", {}).get("delegation_id") == task["delegation_id"]:
        return
    task["worker_ended"] = {
        "delegation_id": task["delegation_id"],
        "evidence": "Common runner confirmed termination and cleanup.",
    }
    session.store.save()


def request(
    session: ParentSession,
    prs: list[int],
    instruction: str,
    evidence: str,
    reason: str | None,
) -> dict[str, Any]:
    session.agreed()
    parent, children = session.work_items()
    targets = set(children) or {parent.number}
    linked: set[int] = set()
    for n in prs:
        pr = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
        relevant = closing_issues(pr.get("body") or "", session.gh.repo) & targets
        if pr.get("number") != n or not relevant or pr.get("state") != "open":
            raise ValueError("review requires related open PRs")
        linked.update(relevant)
    for child in sorted(linked):
        runner_ended(session, child)
        session.record_result(child, evidence)
    fixed = prepare(session, prs, reason)
    return transition(session, fixed["review_run_id"], "instructed", instruction)
