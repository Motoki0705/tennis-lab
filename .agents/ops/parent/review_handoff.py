"""One explicit handoff: runner completion, results, fixed targets and instruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .policy import closing_issues, object_value
from .session import ParentSession


def runner_ended(session: ParentSession, child: int, *, persist: bool = True) -> None:
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
    with session.gh.io_scope():
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
    if not persist:
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
    *,
    review_selection: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    session.gh.targeted_history = True
    session.agreed()
    parent, children = session.work_items()
    targets = set(children)
    linked: set[int] = set()
    for n in prs:
        pr = object_value(session.gh.get(f"repos/{session.gh.repo}/pulls/{n}"))
        from .work_targets import require_separate_self_prs

        require_separate_self_prs(parent, [pr], session.gh.repo)
        relevant = closing_issues(pr.get("body") or "", session.gh.repo) & targets
        if pr.get("number") != n or not relevant or pr.get("state") != "open":
            raise ValueError("review requires related open PRs")
        linked.update(relevant)
    for child in sorted(linked):
        runner_ended(session, child)
    session.record_results(sorted(linked), evidence)
    from review.merge_gate import register

    return register(session, prs, instruction, reason, review_selection, options or {})
