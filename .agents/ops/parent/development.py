"""Explicit development actions connecting parent claims to managed workers."""

from __future__ import annotations

import importlib
import os
import signal
import sys
from pathlib import Path
from typing import Any, cast

from dispatcher.runtime import Runtime, alive, gate, load
from dispatcher.schedule import Schedule
from dispatcher.state import DispatchStore
from dispatcher.supervisor import Supervisor, children_pending, managed, unstarted
from issues import Issue, task_status

from .adopted_tasks import check_task_adoptions
from .execution import authorization, require_request_authorization, task_errors
from .github import GitHub
from .policy import (
    closing_issues,
    digest,
    nonempty,
    object_value,
    raw_section,
    replace_section,
)
from .session import ParentSession
from .state import Store

OPS = Path(__file__).resolve().parents[1]


def workflow_store(directory: Path, dispatcher_directory: Path | None) -> DispatchStore:
    return DispatchStore(
        directory, Schedule(dispatcher_directory or directory.parent / "dispatcher")
    )


def controller(session: ParentSession, directory: Path | None) -> Supervisor:
    schedule = Schedule(directory or session.store.directory.parent / "dispatcher")
    config = schedule.read()["config"]
    if config["repo"].casefold() != session.gh.repo.casefold():
        raise ValueError("dispatcher repository differs from the parent")
    return Supervisor(session.gh, session.store, schedule)


def accepted(session: ParentSession, env: dict[str, str]) -> dict[str, Any]:
    dispatch: dict[str, Any] | None = managed(session.owner())
    if not dispatch or dispatch["phase"] != "running":
        raise ValueError("accept the running dispatcher parent before worker actions")
    job = dispatch["jobs"][dispatch["active_job"]]
    if (
        env.get("AGENT_RUNTIME_RUN_DIR") != str(Path(job["directory"]) / "runtime")
        or env.get("AGENT_RUNTIME_TASK_ID") != job["task_id"]
    ):
        raise ValueError("worker action belongs to another parent runtime")
    return dispatch


def worker_jobs(dispatch: dict[str, Any], child: int) -> list[dict[str, Any]]:
    return sorted(
        (j for j in dispatch["jobs"].values() if j["child"] == child),
        key=lambda j: j["sequence"],
    )


def check_dependencies(session: ParentSession, dispatch: dict[str, Any]) -> None:
    for n in dispatch["candidate"]["dependencies"]:
        raw = session.gh.issue(n)
        if raw["state"] != "closed" or raw.get("state_reason") != "completed":
            raise ValueError(f"candidate dependency #{n} is no longer completed")


def ended(job: dict[str, Any]) -> dict[str, Any] | None:
    directory = Path(job["directory"])
    if job.get("cancelled_before_start"):
        with gate(directory / "lifetime.lock") as fd:
            if job["phase"] != "ended" or fd is None or not unstarted(job):
                raise ValueError(
                    "cancelled reservation has unconfirmed launch evidence"
                )
        return None
    evidence = Runtime(Path(job["runner"])).inspect(directory / "runtime")
    with gate(directory / "lifetime.lock") as fd:
        if (
            job["phase"] != "ended"
            or fd is None
            or alive(job["worker"])
            or not Runtime(Path(job["runner"])).ended(evidence)
        ):
            raise ValueError(
                "worker termination must be reconciled before continuation/completion"
            )
    return cast(dict[str, Any], evidence["record"])


def start_worker(
    session: ParentSession,
    directory: Path | None,
    env: dict[str, str],
    *,
    child: int,
    prompt: str,
    model: str | None,
    branch: str | None = None,
    worktree: Path | None = None,
    continuation: str | None = None,
) -> dict[str, Any]:
    """Reserve via parent.delegate, then use the dispatcher start-child contract.

    The caller must not hold Store.locked(): the dispatcher takes the same lock.
    A failure between the two claims intentionally leaves the delegation intact.
    """
    nonempty(prompt, "worker instruction")
    if model not in {None, "gpt-6.1-sol", "gpt-6-astra"}:
        raise ValueError("unknown implementation model")
    with session.store.locked():
        dispatch = accepted(session, env)
        control = controller(session, directory)
        if authorization(session) != dispatch["authorization"]:
            raise ValueError("dispatcher execution basis changed")
        check_dependencies(session, dispatch)
        jobs = worker_jobs(dispatch, child)
        if continuation is not None:
            nonempty(continuation, "continuation evidence")
            if not jobs:
                raise ValueError(
                    "no existing worker session; never create a replacement"
                )
            for job in jobs:
                ended(job)
            source = control.resume_source(session.row, jobs[-1])
            if source is None:
                if any(not job.get("cancelled_before_start") for job in jobs):
                    raise ValueError(
                        "cancelled reservation lost its saved resume source"
                    )
                if any(
                    child in closing_issues(p.get("body") or "", session.gh.repo)
                    for p in session.gh.pulls()
                ):
                    raise ValueError(
                        "existing PR has no implementation session to resume"
                    )
                if model is not None and model != jobs[-1]["model"]:
                    raise ValueError("continuation cannot change the worker model")
                model = jobs[-1]["model"]
            else:
                record = load(Path(source) / "result.json")
                if (
                    model
                    and model != record["contract"]["settings"]["selected"]["model"]
                ):
                    raise ValueError("continuation cannot change the worker model")
            task = session.row["tasks"][str(child)]
            branch, worktree = task["branch"], Path(task["worktree"])
        elif jobs:
            raise ValueError("existing worker: inspect it and use continue-worker")
        if branch is None or worktree is None:
            raise ValueError("a dedicated branch and worktree are required")
        if len(children_pending(session.row)) >= dispatch["child_limit"]:
            raise ValueError(
                "implementation slots are occupied; reconcile workers first"
            )
        claim = session.delegate(
            child,
            "codex",
            branch,
            worktree,
            None,
            continuation,
            continue_existing=continuation is not None,
        )
        parent, items = session.work_items()
        check_task_adoptions(session.gh, parent.body)
        check_task_adoptions(session.gh, items[child].body)
        check_dependencies(session, dispatch)
        if authorization(session) != dispatch["authorization"] or task_errors(
            claim["assignment"], dispatch["authorization"], items[child].body
        ):
            raise ValueError(
                "worker inputs changed after delegation; reconcile the saved reservation"
            )
        text = Path(__file__).with_name("worker_prompt.md").read_text(encoding="utf-8")
        text += (
            f"\n\n親issue: #{session.parent}\n担当issue: #{child}\n"
            f"worktree: {worktree.resolve()}\nbranch: {branch}\n"
            f"PYTEST_LOCK: {env.get('PYTEST_LOCK') or session.store.directory.parent / 'pytest.lock'}\n\n"
            f"## 担当issueの原文\n\n{items[child].body}\n\n"
            f"## 親の具体的な指示\n\n{prompt}\n"
        )
        delegation = claim["assignment"]["delegation_id"]
    job_id = control.start_child(session.parent, child, delegation, text, env, model)
    return {
        "child": child,
        "delegation_id": delegation,
        "job_id": job_id,
        "result": "reserved",
    }


def inspect_worker(session: ParentSession, child: int) -> dict[str, Any]:
    dispatch = managed(session.owner())
    if not dispatch:
        raise ValueError("worker inspection requires a managed parent")
    jobs = worker_jobs(dispatch, child)
    if not jobs:
        raise ValueError("no managed runtime for this issue")
    job = jobs[-1]
    directory = Path(job["directory"]) / "runtime"
    result: dict[str, Any] = {
        "child": child,
        "job_id": job["id"],
        "phase": job["phase"],
        "delegation_id": job["delegation_id"],
        "task_success": None,
        "termination_confirmed": False,
        "cancelled_before_start": bool(job.get("cancelled_before_start")),
    }
    if job.get("cancelled_before_start"):
        ended(job)
        result["termination_confirmed"] = True
    elif (directory / "result.json").exists():
        evidence = Runtime(Path(job["runner"])).inspect(directory)
        record = evidence["record"]
        result.update(
            runtime_result=record["result"],
            turn_end_reason=record["turn_end_reason"],
            termination_confirmed=job["phase"] == "ended"
            and Runtime(Path(job["runner"])).ended(evidence),
            settings=record["contract"]["settings"],
            final_file=str(directory / "final.md"),
        )
    return result


def stop_worker(
    session: ParentSession, child: int, detail: dict[str, Any]
) -> dict[str, Any]:
    dispatch = managed(session.owner())
    if not dispatch:
        raise ValueError("worker stop requires a managed parent")
    session.stop_or_resume(child, detail, False)
    plan = session.launch_plan()
    targets = {child} | {t["child"] for t in plan["tasks"] if t["state"] == "held"}
    for job in dispatch["jobs"].values():
        if job["role"] != "implementation" or job["child"] not in targets:
            continue
        if job["phase"] == "ended":
            continue
        if job["phase"] == "reserved":
            Supervisor.cancel_reservation(
                session.row, job, "cancelled by parent before launch"
            )
        elif (Path(job["directory"]) / "runtime/result.json").exists():
            Runtime(Path(job["runner"])).inspect(
                Path(job["directory"]) / "runtime", stop=True
            )
        elif alive(job["worker"]):
            os.kill(job["worker"]["pid"], signal.SIGTERM)
    session.store.save()
    return {
        "child": child,
        "result": "stop_requested",
        "launch_plan": session.launch_plan(),
    }


def cancellation_workers_ended(session: ParentSession, child: int) -> None:
    if child in children_pending(session.row):
        raise ValueError(f"#{child}: cancelled worker termination is still pending")
    dispatch = managed(session.row)
    if dispatch:
        for job in worker_jobs(dispatch, child):
            ended(job)


def record_cancellation(
    session: ParentSession, child: int, spec: dict[str, Any]
) -> dict[str, Any]:
    """Record the parent's explicit cancellation/handoff check, never success."""
    session.owner()
    auth = authorization(session)
    _, items = session.work_items()
    if child not in items or task_status(items[child]) != "cancelled":
        raise ValueError("cancellation evidence requires a cancelled direct child")
    text = cancellation_text(spec)
    session.public(spec)
    cancellation_workers_ended(session, child)
    if any(
        p["state"] == "open"
        and child in closing_issues(p.get("body") or "", session.gh.repo)
        for p in session.gh.pulls()
    ):
        raise ValueError(f"#{child}: cancelled task still has an open PR")
    item = items[child]
    current = raw_section(item.body, "作業状況・停止事項")
    body = replace_section(
        item.body,
        "作業状況・停止事項",
        current if current.endswith(text) else current + "\n\n" + text,
    )
    session.update_body(child, body, item.body)
    session.row["tasks"].setdefault(str(child), {})["cancellation"] = {
        "schema_version": 1,
        "authorization": auth,
        "state_reason": item.state_reason,
        "body_sha256": digest(body),
        "confirmation": dict(spec),
    }
    session.store.save()
    return {"child": child, "state": "cancelled", "progress": session.sync_progress()}


def cancellation_text(spec: dict[str, Any]) -> str:
    if (
        set(spec) != {"schema_version", "reason", "handoff", "evidence"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
    ):
        raise ValueError("invalid cancellation evidence schema")
    for key in ("reason", "handoff", "evidence"):
        nonempty(spec[key], key)
    return "取消の確認\n\n" + "\n\n".join(
        f"{key}: {spec[key]}" for key in ("reason", "handoff", "evidence")
    )


def verify_cancellation(
    session: ParentSession, item: Issue, auth: dict[str, Any]
) -> None:
    record = session.row["tasks"].get(str(item.number), {}).get("cancellation", {})
    if (
        record.get("schema_version") != 1
        or record.get("authorization") != auth
        or record.get("state_reason") != item.state_reason
        or record.get("body_sha256") != digest(item.body)
    ):
        raise ValueError(
            f"#{item.number}: cancellation reason/handoff evidence missing or stale; "
            "inspect the issue and use record-cancellation"
        )
    confirmation = object_value(record.get("confirmation"))
    if not raw_section(item.body, "作業状況・停止事項").endswith(
        cancellation_text(confirmation)
    ):
        raise ValueError(
            f"#{item.number}: cancellation confirmation differs from the issue"
        )
    cancellation_workers_ended(session, item.number)
    if "needs-human" in item.labels or any(
        p["state"] == "open"
        and item.number in closing_issues(p.get("body") or "", session.gh.repo)
        for p in session.gh.pulls()
    ):
        raise ValueError(
            f"#{item.number}: cancellation still has unresolved work or PR"
        )


def verify_completion(session: ParentSession) -> list[int]:
    """Confirm gate journals against remote merges, closes and parent evidence."""
    auth = authorization(session)  # Also works after a standalone issue was closed.
    parent, items = session.work_items()
    if not items:
        raise ValueError(
            "no implementation targets; acceptance and merged PR evidence are required"
        )
    if "needs-human" in parent.labels or children_pending(session.row):
        raise ValueError("human decision or worker termination is still pending")
    dispatch = managed(session.row)
    if dispatch:
        check_dependencies(session, dispatch)
        for job in dispatch["jobs"].values():
            if job["role"] == "implementation":
                ended(job)
    check_task_adoptions(session.gh, parent.body)
    sys.path.insert(0, str(OPS / "review"))
    merge_gate = importlib.import_module("merge_gate")
    merged: set[int] = set()
    pulls = session.gh.pulls()
    for n, item in items.items():
        if task_status(item) == "cancelled":
            verify_cancellation(session, item, auth)
            continue
        task = session.row["tasks"].get(str(n), {})
        verification = task.get("verification", {})
        if (
            task_status(item) != "completed"
            or task.get("stopped")
            or "needs-human" in item.labels
            or task_errors(task, auth, item.body)
            or verification.get("authorization") != auth
            or not verification.get("evidence")
        ):
            raise ValueError(
                f"#{n}: close, acceptance evidence or current result verification missing"
            )
        check_task_adoptions(session.gh, item.body)
        related = [
            p
            for p in pulls
            if n in closing_issues(p.get("body") or "", session.gh.repo)
        ]
        candidates = [
            p
            for p in related
            if p.get("merged_at") and p["number"] in verification["prs"]
        ]
        if len(candidates) != 1 or any(p["state"] == "open" for p in related):
            raise ValueError(f"#{n}: merged PR is not unique or an open PR remains")
        pr = object_value(
            session.gh.get(f"repos/{session.gh.repo}/pulls/{candidates[0]['number']}")
        )
        if (
            pr.get("number") != candidates[0]["number"]
            or n not in closing_issues(pr.get("body") or "", session.gh.repo)
            or pr.get("state") != "closed"
            or task.get("branch", pr["head"]["ref"]) != pr["head"]["ref"]
            or pr.get("merged") is not True
            or not pr.get("merged_at")
            or merge_gate.completion(session, pr, write=False)
        ):
            raise ValueError(f"#{n}: GitHub merge/close is not confirmed")
        journal = load(
            merge_gate.attempt_path(session, pr["number"], pr["head"]["sha"])
        )
        snapshot = journal["snapshot"]
        if (
            journal["pr"] != pr["number"]
            or journal["head_sha"] != pr["head"]["sha"]
            or snapshot["head_sha"] != pr["head"]["sha"]
            or snapshot["branch"] != pr["head"]["ref"]
            or snapshot["base_ref"] != pr["base"]["ref"]
            or pr["head"]["repo"]["full_name"].casefold() != session.gh.repo.casefold()
            or pr["base"]["repo"]["full_name"].casefold() != session.gh.repo.casefold()
            or journal["status"] not in {"merged", "attempting", "unconfirmed"}
        ):
            raise ValueError("merged PR differs from the gate's checked snapshot")
        request = load(
            session.store.directory
            / "reviews"
            / journal["review_run_id"]
            / "request.json"
        )
        require_request_authorization(session, request, required_issues={n})
        view = merge_gate.collect(session, request)
        _, _, state, errors, _ = merge_gate.load_evidence(session, request, view)
        if (
            errors
            or view.errors[pr["number"]]
            or state not in {"fixes_verified", "complete"}
        ):
            raise ValueError(
                "merged PR's parent review is missing, stale or unconfirmed"
            )
        merged.add(pr["number"])
    if not merged:
        raise ValueError(
            "no completed implementation targets; cancellation is not success"
        )
    return sorted(merged)


def report(session: ParentSession, spec: dict[str, Any]) -> dict[str, Any]:
    if (
        set(spec)
        != {"schema_version", "state", "reason", "resume_condition", "evidence"}
        or type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["state"] not in {"waiting", "completed", "failed"}
    ):
        raise ValueError("invalid workflow report schema")
    session.owner()
    for key in ("reason", "resume_condition", "evidence"):
        nonempty(spec[key], key)
    # Validate public text before any result/state mutation.
    session.public(spec)
    prs = verify_completion(session) if spec["state"] == "completed" else []
    session.note(
        f"開発workflow: {spec['state']}\n\n理由: {spec['reason']}\n\n"
        f"再開条件: {spec['resume_condition']}\n\n受入・検証の証拠:\n{spec['evidence']}"
    )
    session.row["development_result"] = {
        "schema_version": 1,
        "report_sha256": digest(spec),
        "prs": prs,
        "cancelled": sorted(
            n
            for n, item in session.work_items()[1].items()
            if task_status(item) == "cancelled"
        )
        if spec["state"] == "completed"
        else [],
    }
    session.store.save()
    return {
        k: spec[k] for k in ("schema_version", "state", "reason", "resume_condition")
    }


def worker_command(
    args: Any, gh: GitHub, parent: int, identity: str, store: Store
) -> dict[str, Any]:
    with store.locked():
        session = ParentSession(gh, store, parent, identity)
        if args.command == "inspect-worker":
            return inspect_worker(session, args.child)
        if args.command == "stop-worker":
            return stop_worker(session, args.child, load(Path(args.spec)))
    return start_worker(
        session,
        args.dispatcher_state_dir,
        dict(os.environ),
        child=args.child,
        prompt=Path(args.prompt_file).read_text(encoding="utf-8"),
        model=args.model,
        branch=getattr(args, "branch", None),
        worktree=getattr(args, "worktree", None),
        continuation=Path(args.evidence_file).read_text(encoding="utf-8")
        if args.command == "continue-worker"
        else None,
    )
