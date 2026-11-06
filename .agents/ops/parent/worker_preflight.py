"""Planning decisions and turn provenance, independent of runner permissions.

The task is authoritative. Run/job metadata is an immutable launch snapshot;
neither a successful turn nor a discovered PR grants implementation approval.
"""

from __future__ import annotations

import copy
import hashlib
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues import Issue

from .execution import authorization, requirement_body
from .policy import closing_issues, dependencies, digest, nonempty

if TYPE_CHECKING:
    from .session import ParentSession

HEADINGS = (
    "確認した根拠",
    "指示との不一致",
    "実装案",
    "テスト設計",
    "制約・未確認事項",
    "親に決めてほしい点",
)
SNAPSHOT_KEYS = {
    "schema_version",
    "lineage_id",
    "turn_id",
    "previous_turn_id",
    "phase",
    "decision_id",
    "context_sha256",
}


def new(delegation: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lineage_id": uuid.uuid4().hex,
        "delegation_id": delegation,
        "phase": "planning",
        "context": None,
        "runtime": None,
        "turns": [],
        "decisions": [],
        "continuations": [],
        "invalidations": [],
        "skip_reason": None,
        "refresh_required": False,
        "accepted_turn_id": None,
    }


def state(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("preflight")
    if value is None:
        raise ValueError("worker phase is unknown; migrate-worker-phase explicitly")
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("phase") not in {"planning", "implementing"}
        or value.get("delegation_id") != task.get("delegation_id")
        or not isinstance(value.get("lineage_id"), str)
        or not value["lineage_id"]
        or any(
            not isinstance(value.get(k), list)
            for k in ("turns", "decisions", "continuations", "invalidations")
        )
        or type(value.get("refresh_required")) is not bool
    ):
        raise ValueError("malformed worker preflight or delegation binding")
    for turn in value["turns"]:
        validate_snapshot(turn.get("snapshot"))
        if turn["snapshot"]["lineage_id"] != value["lineage_id"]:
            raise ValueError("turn belongs to another worker lineage")
    return value


def validate_snapshot(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != SNAPSHOT_KEYS
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["phase"] not in {"planning", "implementing"}
        or not all(
            isinstance(value[k], str) and value[k]
            for k in ("lineage_id", "turn_id", "context_sha256")
        )
        or any(
            value[k] is not None and not isinstance(value[k], str)
            for k in ("previous_turn_id", "decision_id")
        )
    ):
        raise ValueError("malformed worker turn snapshot")


def context(
    session: ParentSession,
    child: int,
    *,
    authorized: dict[str, Any] | None = None,
    work_items: tuple[Issue, dict[int, Issue]] | None = None,
) -> dict[str, Any]:
    """Freeze material inputs, including heads of completed dependencies."""
    auth = authorization(session) if authorized is None else authorized
    parent, items = session.work_items() if work_items is None else work_items
    if child not in items:
        raise ValueError("worker requires a direct child or the standalone issue")
    task = session.row["tasks"].get(str(child), {})
    pulls = session.gh.pulls()
    found: dict[int, dict[str, Any]] = {}

    def visit(n: int, body: str) -> None:
        deps = dependencies(body) if re.search(r"(?m)^## 依存関係\s*$", body) else []
        for dep in deps:
            if dep == child or dep in found:
                continue
            if dep in items:
                item = items[dep]
                raw = {
                    "state": item.state,
                    "state_reason": item.state_reason,
                    "body": item.body,
                }
            else:
                raw = session.gh.issue(dep)
            found[dep] = {
                "issue": dep,
                "state": raw["state"],
                "state_reason": raw.get("state_reason"),
                "requirements_sha256": digest(requirement_body(raw.get("body") or "")),
                "prs": sorted(
                    [
                        {
                            "number": p["number"],
                            "state": p["state"],
                            "merged": p.get("merged_at") is not None,
                            "branch": p["head"]["ref"],
                            "head": p["head"]["sha"],
                            "repo": p["head"]["repo"]["full_name"].casefold(),
                        }
                        for p in pulls
                        if dep in closing_issues(p.get("body") or "", session.gh.repo)
                    ],
                    key=lambda p: p["number"],
                ),
            }
            if dep in items:
                visit(dep, raw.get("body") or "")

    visit(child, items[child].body)
    return {
        "repo": session.gh.repo.casefold(),
        "parent": session.parent,
        "child": child,
        "authorization": auth,
        "parent_requirements": digest(requirement_body(parent.body)),
        "requirements": digest(requirement_body(items[child].body)),
        "dependencies": [found[n] for n in sorted(found)],
        "branch": task.get("branch"),
        "worktree": task.get("worktree"),
    }


def invalidate(value: dict[str, Any], reason: str) -> None:
    if not value["refresh_required"]:
        value["invalidations"].append(
            {
                "context_sha256": digest(value["context"]),
                "reason": reason,
            }
        )
    value.update(phase="planning", refresh_required=True, skip_reason=None)
    for decision in value["decisions"]:
        decision["invalidated"] = True


def observe_context(
    session: ParentSession,
    child: int,
    value: dict[str, Any],
    *,
    persist: bool,
    current: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    reason = None
    fresh = None
    try:
        fresh = context(session, child) if current is None else current
        if value["context"] is not None and value["context"] != fresh:
            reason = "worker requirements/authorization/generation/dependencies changed"
    except ValueError:
        reason = "current execution basis cannot be verified"
    if reason and persist:
        invalidate(value, reason)
        session.store.save()
    return fresh, reason is not None


def observe(
    session: ParentSession, child: int, *, persist: bool = False
) -> dict[str, Any]:
    task = session.row.get("tasks", {}).get(str(child), {})
    if not task.get("delegated"):
        return {
            "phase": "not_delegated",
            "status": "not_delegated",
            "next_action": "delegate",
        }
    if "preflight" not in task:
        return {
            "phase": "unknown",
            "status": "migration_required",
            "next_action": "migrate-worker-phase",
        }
    value = state(task)
    _, stale = observe_context(session, child, value, persist=persist)
    return summary(task, stale=stale)


def summary(task: dict[str, Any], *, stale: bool = False) -> dict[str, Any]:
    if "preflight" not in task:
        return {
            "phase": "unknown",
            "status": "migration_required",
            "next_action": "migrate-worker-phase",
        }
    value = state(task)
    refresh = stale or value["refresh_required"]
    turns = value["turns"]
    pending = [
        d
        for d in value["decisions"]
        if not d["invalidated"] and d["delivered_turn_id"] is None
    ]
    if refresh:
        status, action = (
            "refresh_required",
            "resume with revised instructions; managed basis/owner changes require refresh-worker-plan",
        )
    elif pending:
        status, action = (
            "decision_pending_delivery",
            "run-worker --resume / continue-worker",
        )
    elif not turns:
        status, action = "not_started", "run-worker / start-worker"
    elif value["phase"] == "planning":
        confirmed = task.get("worker_ended", {}).get("delegation_id") == task.get(
            "delegation_id"
        )
        status, action = (
            "awaiting_decision" if confirmed else "planning_turn",
            "confirm termination, then decide-worker-plan",
        )
    else:
        status, action = (
            "implementing",
            "confirm implementation turn and verify results",
        )
    return {
        "phase": "planning" if refresh else value["phase"],
        "status": status,
        "next_action": action,
        "turn_count": len(turns),
        "turn_id": turns[-1]["snapshot"]["turn_id"] if turns else None,
        "decision_pending": len(pending),
        "planning_skipped": bool(value["skip_reason"]),
    }


def report_missing(text: str) -> tuple[str, ...]:
    # Fenced examples are not report sections.
    clean = re.sub(r"(?ms)^```[^\n]*\n.*?^```[^\n]*$", "", text)
    sections = list(re.finditer(r"(?m)^## ([^\n]+)\s*$", clean))
    contents: dict[str, list[str]] = {}
    for i, match in enumerate(sections):
        end = sections[i + 1].start() if i + 1 < len(sections) else len(clean)
        contents.setdefault(match[1].strip(), []).append(
            clean[match.end() : end].strip()
        )
    return tuple(
        h for h in HEADINGS if len(contents.get(h, [])) != 1 or not contents[h][0]
    )


def runtime_binding(record: dict[str, Any]) -> dict[str, str]:
    contract = record["contract"]
    return {
        "contract_sha256": digest(contract),
        "session_id": nonempty(
            record.get("runtime", {}).get("session_id"), "saved worker session"
        ),
    }


def bind_runtime(value: dict[str, Any], record: dict[str, Any]) -> None:
    observed = runtime_binding(record)
    if value["runtime"] is not None and value["runtime"] != observed:
        raise ValueError("worker session/contract differs from the planning lineage")
    value["runtime"] = observed


def latest_evidence(session: ParentSession, child: int) -> tuple[Path, dict[str, Any]]:
    """Reuse the launcher's completion checks; never infer end from a final."""
    task = session.row["tasks"][str(child)]
    if session.row.get("dispatch") is not None:
        from .development import Supervisor, ended, worker_jobs

        jobs = worker_jobs(session.row["dispatch"], child)
        if not jobs or jobs[-1]["delegation_id"] != task["delegation_id"]:
            raise ValueError("no runtime for the current worker delegation")
        record = ended(jobs[-1])
        if record is None:
            raise ValueError("cancelled reservation has no planning final")
        Supervisor._check_contract(jobs[-1], record)
        if (
            record["contract"].get("branch") != task["branch"]
            or record["contract"].get("cwd") != task["worktree"]
        ):
            raise ValueError(
                "planning runtime worktree/branch does not match the delegation"
            )
        return Path(jobs[-1]["directory"]) / "runtime", record
    from dispatcher.runtime import Runtime

    from .manual_workers import RUNNER, latest_run, validate_run
    from .review_handoff import runner_ended

    runner_ended(session, child)
    directory = latest_run(session, task)
    validate_run(task, directory)
    proof = Runtime(RUNNER).inspect(directory)
    if not Runtime(RUNNER).ended(proof):
        raise ValueError("worker termination and cleanup must be confirmed")
    return directory, proof["record"]


def decide(
    session: ParentSession, child: int, action: str, answer: str
) -> dict[str, Any]:
    session.owner()
    nonempty(answer, "parent planning answer")
    if action not in {"approve", "investigate", "reject"}:
        raise ValueError("unknown worker planning decision")
    task = session.row["tasks"][str(child)]
    value = state(task)
    fresh, stale = observe_context(session, child, value, persist=True)
    if fresh is None or stale:
        raise ValueError("planning inputs changed; resume a new planning turn")
    if value["refresh_required"] or value["context"] != fresh:
        raise ValueError("planning inputs changed; resume a new planning turn")
    if not value["turns"] or value["turns"][-1]["snapshot"]["phase"] != "planning":
        raise ValueError("decision requires the latest planning turn")
    turn = value["turns"][-1]
    directory, record = latest_evidence(session, child)
    if turn.get("source") != str(directory) or record["result"] != "completed":
        raise ValueError("latest completed planning runtime does not match the turn")
    if (
        turn.get("prompt_sha256") is not None
        and record.get("prompt_sha256") != turn["prompt_sha256"]
    ):
        raise ValueError("saved planning instruction does not match the runtime")
    bind_runtime(value, record)
    raw = (directory / "final.md").read_bytes()
    text = raw.decode("utf-8")
    nonempty(text, "planning final")
    missing = report_missing(text)
    if action == "approve" and (missing or turn.get("reply_only")):
        raise ValueError(
            "planning final missing required report headings: "
            + ", ".join(missing or HEADINGS)
        )
    final_sha = hashlib.sha256(raw).hexdigest()
    previous = [
        d
        for d in value["decisions"]
        if not d["invalidated"] and d["delivered_turn_id"] is None
    ]
    if previous:
        if len(previous) == 1 and all(
            previous[0][k] == v
            for k, v in {
                "action": action,
                "answer": answer,
                "final_sha256": final_sha,
                "turn_id": turn["snapshot"]["turn_id"],
            }.items()
        ):
            return {
                "child": child,
                "decision_id": previous[0]["decision_id"],
                "action": action,
            }
        raise ValueError("an undelivered planning decision already exists")
    decision = {
        "decision_id": uuid.uuid4().hex,
        "lineage_id": value["lineage_id"],
        "delegation_id": task["delegation_id"],
        "turn_id": turn["snapshot"]["turn_id"],
        "context_sha256": digest(fresh),
        "final_sha256": final_sha,
        "runtime": copy.deepcopy(value["runtime"]),
        "action": action,
        "answer": answer,
        "delivered_turn_id": None,
        "invalidated": False,
    }
    turn["final_sha256"] = final_sha
    value["decisions"].append(decision)
    session.store.save()
    return {
        "child": child,
        "decision_id": decision["decision_id"],
        "action": action,
        "worker_phase": summary(task),
    }


def decision_matches(value: dict[str, Any], decision: dict[str, Any]) -> bool:
    turns = {t["snapshot"]["turn_id"]: t for t in value["turns"]}
    turn = turns.get(decision.get("turn_id"), {})
    allowed = {value["delegation_id"]} | {c["from"] for c in value["continuations"]}
    return bool(
        not decision.get("invalidated")
        and decision.get("lineage_id") == value["lineage_id"]
        and decision.get("delegation_id") in allowed
        and turn.get("delegation_id") == decision.get("delegation_id")
        and turn.get("final_sha256") == decision.get("final_sha256")
        and decision.get("context_sha256") == digest(value["context"])
        and decision.get("runtime") == value["runtime"]
    )


def prepare_turn(
    session: ParentSession,
    child: int,
    instruction: str | None,
    *,
    resume: bool,
    skip_reason: str | None = None,
    additional: str | None = None,
    previous_record: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    task = session.row["tasks"][str(child)]
    saved = state(task)
    fresh, _ = observe_context(session, child, saved, persist=True)
    if fresh is None:
        raise ValueError("worker execution basis cannot be verified; refresh_required")
    value = copy.deepcopy(saved)
    if skip_reason is not None:
        nonempty(skip_reason, "planning skip reason")
        if resume or value["turns"]:
            raise ValueError("--skip-planning-reason-file is initial-only")
    if previous_record is not None:
        bind_runtime(value, previous_record)
    pending = [
        d
        for d in value["decisions"]
        if not d["invalidated"] and d["delivered_turn_id"] is None
    ]
    if len(pending) > 1:
        raise ValueError(
            "multiple undelivered planning decisions; reconcile explicitly"
        )
    decision = None
    reply_only = False
    if value["refresh_required"]:
        if not resume:
            raise ValueError(
                "changed inputs require the same session's planning resume"
            )
        text = nonempty(instruction, "revised instructions for planning refresh")
        value.update(context=fresh, refresh_required=False, phase="planning")
    elif pending:
        if not resume or instruction is not None:
            raise ValueError(
                "pending planning answer cannot be replaced by --prompt-file; use --additional-instruction-file"
            )
        decision = pending[0]
        if not decision_matches(value, decision):
            raise ValueError(
                "planning decision belongs to another delegation/plan/context"
            )
        original = next(
            t for t in value["turns"] if t["snapshot"]["turn_id"] == decision["turn_id"]
        )
        raw = (Path(original["source"]) / "final.md").read_bytes()
        if hashlib.sha256(raw).hexdigest() != decision["final_sha256"]:
            raise ValueError("planning final changed after the parent's decision")
        text = "## 親の計画判断\n\n" + decision["action"] + "\n\n" + decision["answer"]
        value["phase"] = (
            "implementing" if decision["action"] == "approve" else "planning"
        )
        reply_only = decision["action"] == "reject"
    else:
        if (
            resume
            and value["phase"] == "planning"
            and value["turns"]
            and not value["turns"][-1].get("cancelled")
            and (previous_record is None or previous_record.get("result") != "failed")
        ):
            raise ValueError(
                "planning is awaiting decide-worker-plan; no undelivered decision"
            )
        text = nonempty(instruction, "worker instruction")
        if value["context"] is None:
            value["context"] = fresh
        if skip_reason is not None:
            value.update(phase="implementing", skip_reason=skip_reason)
    if additional is not None:
        if decision is None:
            raise ValueError(
                "additional instructions require an undelivered planning decision"
            )
        text += "\n\n## 親の追加指示\n\n" + nonempty(
            additional, "additional instruction"
        )
    turn_id = uuid.uuid4().hex
    snapshot = {
        "schema_version": 1,
        "lineage_id": value["lineage_id"],
        "turn_id": turn_id,
        "previous_turn_id": value["turns"][-1]["snapshot"]["turn_id"]
        if value["turns"]
        else None,
        "phase": value["phase"],
        "decision_id": decision["decision_id"] if decision else None,
        "context_sha256": digest(value["context"]),
    }
    if decision:
        decision["delivered_turn_id"] = turn_id
    value["turns"].append(
        {
            "snapshot": snapshot,
            "delegation_id": task["delegation_id"],
            "source": None,
            "head_at_start": None,
            "reply_only": reply_only,
        }
    )
    if value["phase"] == "implementing" and not implementation_authorized(value):
        raise ValueError(
            "implementation requires the current parent's planning approval"
        )
    return value, text


def commit_turn(
    task: dict[str, Any], value: dict[str, Any], directory: Path | None = None
) -> dict[str, Any]:
    if value["delegation_id"] != task["delegation_id"]:
        value["continuations"].append(
            {"from": value["delegation_id"], "to": task["delegation_id"]}
        )
        value["delegation_id"] = task["delegation_id"]
        value["turns"][-1]["delegation_id"] = task["delegation_id"]
    if directory is not None:
        value["turns"][-1]["source"] = str(directory.resolve())
    from .snapshot import git

    value["turns"][-1]["head_at_start"] = (
        git("rev-parse", "HEAD", cwd=Path(task["worktree"])).decode().strip()
    )
    task["preflight"] = value
    return copy.deepcopy(value["turns"][-1]["snapshot"])


def check_admission(
    session: ParentSession,
    child: int,
    snapshot: Any,
    *,
    current: dict[str, Any] | None = None,
) -> None:
    value = state(session.row["tasks"][str(child)])
    fresh, stale = observe_context(session, child, value, persist=True, current=current)
    validate_snapshot(snapshot)
    if (
        stale
        or fresh is None
        or value["refresh_required"]
        or not value["turns"]
        or value["turns"][-1]["snapshot"] != snapshot
        or digest(fresh) != snapshot["context_sha256"]
        or value["phase"] != snapshot["phase"]
    ):
        raise ValueError("worker planning turn/context changed; refresh_required")
    if snapshot["phase"] == "implementing" and not implementation_authorized(value):
        raise ValueError(
            "implementation requires the current parent's planning approval"
        )


def implementation_authorized(value: dict[str, Any]) -> bool:
    return bool(
        value["phase"] == "implementing"
        and not value["refresh_required"]
        and (
            value["skip_reason"]
            or any(
                d["action"] == "approve"
                and d["delivered_turn_id"] is not None
                and decision_matches(value, d)
                for d in value["decisions"]
            )
        )
    )


def require_implementation(session: ParentSession, child: int) -> None:
    task = session.row["tasks"].get(str(child), {})
    if not task.get("delegated"):
        return
    observe(session, child, persist=True)
    value = state(task)
    if not implementation_authorized(value):
        raise ValueError(
            "worker planning approval required; implementation artifacts are not accepted"
        )
    if not value["turns"] or value["turns"][-1]["snapshot"]["phase"] != "implementing":
        raise ValueError(
            "completed implementation turn required after planning approval"
        )
    directory, record = latest_evidence(session, child)
    if (
        value["turns"][-1]["source"] != str(directory)
        or record["result"] != "completed"
    ):
        raise ValueError(
            "completed implementation turn required after planning approval"
        )
    if record.get("prompt_sha256") != value["turns"][-1].get("prompt_sha256"):
        raise ValueError("implementation instruction does not match the saved turn")
    bind_runtime(value, record)


def migrate(session: ParentSession, child: int, evidence: str) -> dict[str, Any]:
    session.agreed()
    nonempty(evidence, "worker phase migration evidence")
    task = session.row["tasks"].get(str(child), {})
    if not task.get("delegated") or "preflight" in task:
        raise ValueError("migration requires an existing worker without a phase")
    unstarted = False
    if session.row.get("dispatch") is None:
        from .manual_workers import worker_runs

        unstarted = "worker_runs" in task and not any(
            r["delegation_id"] == task["delegation_id"] for r in worker_runs(task)
        )
    else:
        from .development import ended, worker_jobs

        jobs = worker_jobs(session.row["dispatch"], child)
        unstarted = bool(jobs) and all(
            j.get("cancelled_before_start")
            and ended(j) is None
            and j["resume_from"] is None
            for j in jobs
        )
    value = new(task["delegation_id"])
    value.update(
        context=context(session, child),
        refresh_required=not unstarted,
        migration_evidence=evidence,
        migration_unstarted=unstarted,
    )
    if unstarted:
        from .snapshot import validate_worktree

        validate_worktree(Path(task["worktree"]), task["branch"])
    else:
        _, record = latest_evidence(session, child)
        bind_runtime(value, record)
    task["preflight"] = value
    session.store.save()
    return {"child": child, "worker_phase": summary(task)}
