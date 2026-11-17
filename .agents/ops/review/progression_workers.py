"""Creating-worker provenance and explicit continuation after a parent's fixes."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any

from lib.agent_settings import session_contract
from parent.policy import nonempty
from parent.session import ParentSession


def source(session: ParentSession, child: int) -> dict[str, Any]:
    """Inspect the latest ended creating runtime, for either worker owner."""
    from dispatcher.runtime import Runtime

    task = session.row["tasks"][str(child)]
    delegation = task["delegation_id"]
    dispatch = session.row.get("dispatch")
    if dispatch:
        from dispatcher.supervisor import Supervisor

        jobs = [j for j in dispatch["jobs"].values() if j["child"] == child]
        if not jobs or any(j["phase"] != "ended" for j in jobs):
            raise ValueError("creating worker termination is unconfirmed")
        job = max(jobs, key=lambda j: j["sequence"])
        if job.get("cancelled_before_start") or job["delegation_id"] != delegation:
            raise ValueError("creating worker has no completed current turn")
        directory = Path(job["directory"]) / "runtime"
        runner = Path(job["runner"])
    else:
        from parent.manual_workers import RUNNER, latest_run, validate_run

        directory = latest_run(session, task)
        validate_run(task, directory)
        runner = RUNNER
    with session.gh.io_scope():
        proof = Runtime(runner).inspect(directory)
    record = proof["record"]
    if (
        not Runtime(runner).ended(proof)
        or record.get("result") != "completed"
        or record.get("recovery_error")
    ):
        raise ValueError("creating worker termination/completion is unconfirmed")
    if dispatch:
        Supervisor._check_contract(job, record)
    if session.row["tasks"][str(child)]["delegation_id"] != delegation:
        raise ValueError("creating worker changed during inspection")
    return {
        "session_id": nonempty(
            record.get("runtime", {}).get("session_id"), "creating worker session"
        ),
        "contract": session_contract(record["contract"]),
        "directory": str(directory),
    }


def refresh(
    backend: Any, state: dict[str, Any], prs: list[int] | None, evidence: str
) -> None:
    """The owner's explicit continuation may follow a proven same-worker chain."""
    from dispatcher.repairs import register_progression
    from lib.ops_review.progression_backend import ProgressionGitHub
    from parent.review_handoff import runner_ended

    selected = [
        t
        for t in state["targets"].values()
        if t["stage"] != "completed" and (prs is None or t["number"] in prs)
    ]
    gh = ProgressionGitHub(backend.repo, store=backend.store)
    with backend.store.locked():
        backend.store.focus(backend.key)
        session = ParentSession(gh, backend.store, backend.parent, state["owner"])
        session.agreed()
        replacements = []
        for target in selected:
            worker = target.get("worker")
            if not worker:
                continue
            task = session.row["tasks"].get(str(worker["child"]), {})
            if (
                not task.get("delegated")
                or task.get("agent") != "codex"
                or task.get("branch") != target["branch"]
            ):
                raise ValueError("creating worker identity changed")
            original = worker.get("source")
            if not original:
                raise ValueError("creating worker session provenance is missing")
            old, current = worker["delegation_id"], task["delegation_id"]
            cursor = old
            for step in task.get("preflight", {}).get("continuations", []):
                if step["from"] == cursor:
                    cursor = step["to"]
            if cursor != current:
                raise ValueError("creating worker continuation lineage is unconfirmed")
            runner_ended(session, worker["child"], persist=False)
            actual = source(session, worker["child"])
            if (
                actual["session_id"] != original["session_id"]
                or actual["contract"] != original["contract"]
            ):
                raise ValueError("creating worker session/model/contract changed")
            replacements.append((target, worker, current, actual))
        # All targets passed before recording acceptance or new repair capabilities.
        for target, worker, current, actual in replacements:
            runner_ended(session, worker["child"])
            if current != worker["delegation_id"]:
                updated = {**worker, "delegation_id": current, "source": actual}
                if worker["kind"] == "dispatcher":
                    updated["progression_id"] = uuid.uuid4().hex
                    register_progression(
                        session,
                        updated["progression_id"],
                        child=worker["child"],
                        pr=target["number"],
                    )
                state.setdefault("worker_continuations", []).append(
                    {
                        "pr": target["number"],
                        "from": copy.deepcopy(worker),
                        "to": copy.deepcopy(updated),
                    }
                )
                target["worker"] = updated
        if (
            state["review"]["stage"] == "parent_wait"
            and state["review"].get("result", {}).get("verdict") == "major"
        ):
            session.record_results(
                sorted({n for t in selected for n in t["issues"]}), evidence
            )
