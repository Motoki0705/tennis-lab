"""Owner-registered repair capability for a detached automatic progression."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lib.agent_settings import implementation_model, session_contract
from lib.repair_turn import instruction, request
from parent import worker_preflight
from parent.execution import authorization
from parent.policy import digest, nonempty
from parent.session import ParentSession
from parent.worker_instruction import prepare_instruction
from worker_logs import service as worker_logs

from .runtime import load

if TYPE_CHECKING:
    from .supervisor import Supervisor


def source(
    control: Supervisor, row: dict[str, Any], child: int
) -> tuple[str, dict[str, Any]]:
    jobs = [
        j
        for j in row["dispatch"]["jobs"].values()
        if j["child"] == child and j.get("worker_name", "primary") == "primary"
    ]
    if not jobs or any(j["phase"] != "ended" for j in jobs):
        raise ValueError("repair requires the creating worker to have ended")
    previous = max(jobs, key=lambda j: j["sequence"])
    directory = control.resume_source(row, previous)
    if directory is None:
        raise ValueError("repair requires a resumable creating worker")
    record = load(Path(directory) / "result.json")
    control._check_contract(previous, record)
    # Runtime inspection is supplied by the configured runner, as for normal resumes.
    from .runtime import Runtime

    runtime = Runtime(Path(previous["runner"]))
    proof = runtime.inspect(Path(directory))
    if (
        not runtime.ended(proof)
        or proof["record"] != record
        or record.get("recovery_error")
    ):
        raise ValueError("repair source termination/evidence is unconfirmed")
    settings = record["contract"]["settings"]
    if "repair" not in record["contract"] and settings["selected"]["effort"] != "max":
        raise ValueError("normal worker effort must be max")
    return directory, record


def register_progression(
    session: ParentSession, progression_id: str, *, child: int, pr: int
) -> dict[str, Any]:
    """Called by the actual owner while registering automatic progression."""
    request(progression_id, pr)
    row = session.owner()
    dispatch = row.get("dispatch")
    if (
        not dispatch
        or dispatch["session"] != row["session"]
        or dispatch["phase"] not in {"running", "waiting"}
    ):
        raise ValueError(
            "progression registration requires an accepted dispatcher owner"
        )
    session.agreed()
    task = row["tasks"].get(str(child), {})
    if (
        not task.get("delegated")
        or task.get("agent") != "codex"
        or not task.get("worker_ended")
        or task.get("stopped")
    ):
        raise ValueError("progression requires the ended creating Codex worker")
    jobs = [
        j
        for j in dispatch["jobs"].values()
        if j["child"] == child and j.get("worker_name", "primary") == "primary"
    ]
    if not jobs:
        raise ValueError("creating worker runtime is missing")
    previous = max(jobs, key=lambda j: j["sequence"])
    if (
        previous["phase"] != "ended"
        or previous["delegation_id"] != task["delegation_id"]
    ):
        raise ValueError("creating worker runtime differs from the delegation")
    record = load(Path(previous["directory"]) / "runtime/result.json")
    value = {
        "owner": row["session"],
        "generation": row["generation"],
        "assignment_id": dispatch["assignment_id"],
        "authorization": authorization(session),
        "child": child,
        "pr": pr,
        "delegation_id": task["delegation_id"],
        "source_contract": session_contract(record["contract"]),
        "source_session": nonempty(
            record.get("runtime", {}).get("session_id"), "creating worker session"
        ),
    }
    registrations = row.setdefault("repair_progressions", {})
    if progression_id in registrations and registrations[progression_id] != value:
        raise ValueError("progression ID is already registered to another target/owner")
    registrations[progression_id] = copy.deepcopy(value)
    session.store.save()
    return copy.deepcopy(value)


def require_registration(
    control: Supervisor, key: str, row: dict[str, Any], progression_id: str
) -> dict[str, Any]:
    saved = row.get("repair_progressions", {}).get(progression_id)
    dispatch = row["dispatch"]
    if not isinstance(saved, dict) or dispatch["phase"] not in {"running", "waiting"}:
        raise ValueError("repair requires an owner-registered automatic progression")
    control._authorized(key, row)
    task = row["tasks"].get(str(saved["child"]), {})
    if (
        saved["owner"] != row["session"]
        or saved["generation"] != row["generation"]
        or saved["assignment_id"] != dispatch["assignment_id"]
        or saved["authorization"] != authorization(control._session(key, row))
        or task.get("authorization") != saved["authorization"]
        or task.get("delegation_id") != saved["delegation_id"]
        or task.get("stopped")
    ):
        raise ValueError("repair progression owner/authority/delegation changed")
    return dict(saved)


def reserve(
    control: Supervisor, parent: int, progression_id: str, repair_id: str, details: str
) -> str:
    """Reserve once without claiming or impersonating the parent runtime."""
    with control.store.locked():
        key = f"{control.gh.repo.casefold()}#{parent}"
        row = control.store.parents[key]
        saved = require_registration(control, key, row, progression_id)
        repair = request(repair_id, saved["pr"])
        details = nonempty(details, "repair failure/gate details")
        dispatch = row["dispatch"]
        for job in dispatch["jobs"].values():
            if job.get("repair", {}).get("repair_id") == repair_id:
                if (
                    job["repair"] != repair
                    or job.get("progression_id") != progression_id
                    or job.get("repair_details_sha256") != digest(details)
                ):
                    raise ValueError("repair ID is already bound to another request")
                return str(job["id"])
        child = saved["child"]
        task = row["tasks"][str(child)]
        session = control._session(key, row)
        session.agreed()
        _, items = session.work_items()
        if child not in items:
            raise ValueError("repair target is no longer a child of this parent")
        prs = [p for p in control.gh.pulls({child}) if p["number"] == saved["pr"]]
        if (
            len(prs) != 1
            or prs[0]["state"] != "open"
            or prs[0]["head"]["ref"] != task["branch"]
        ):
            raise ValueError("repair requires the creating worker's open PR")
        if (
            sum(
                j["role"] == "implementation" and j["phase"] != "ended"
                for j in dispatch["jobs"].values()
            )
            >= dispatch["child_limit"]
        ):
            raise ValueError("implementation runtime limit reached")
        directory, record = source(control, row, child)
        implementation_model(
            None, previous=record["contract"]["settings"]["selected"]["model"]
        )
        if (
            session_contract(record["contract"]) != saved["source_contract"]
            or record["runtime"]["session_id"] != saved["source_session"]
        ):
            raise ValueError("repair source model/session/contract changed")
        phase, specific = worker_preflight.prepare_turn(
            session,
            child,
            instruction(repair, details),
            resume=True,
            previous_record=record,
            previous_directory=Path(directory),
        )
        if phase["phase"] != "implementing":
            raise ValueError("repair requires an implementation turn")
        phase["turns"][-1]["repair"] = repair
        turn = worker_preflight.commit_turn(task, phase)
        prepared = prepare_instruction(
            parent,
            child,
            Path(task["worktree"]),
            task["branch"],
            items[child].body,
            specific,
            turn,
        )
        worker_logs.capture(session, child, items[child].body, prepared)
        task.pop("worker_ended", None)
        task["restart_evidence"] = f"Automatic repair {repair_id} for PR {saved['pr']}"
        job = control._job(
            key,
            row,
            child=child,
            delegation_id=task["delegation_id"],
            prompt=prepared.text,
            model=record["contract"]["settings"]["selected"]["model"],
            resume_from=directory,
        )
        job.update(
            repair=repair,
            progression_id=progression_id,
            repair_details_sha256=digest(details),
        )
        control.store.save()
        return str(job["id"])
