"""Optional parent-directed investigation and immutable worker-turn provenance."""

from __future__ import annotations

import copy
import hashlib
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from issues import Issue
from lib.agent_settings import execution_contract
from worker_logs.sources import original as original_text

from .execution import ExecutionBasisStale, observed_authorization, requirement_body
from .policy import closing_issues, dependencies, digest, nonempty

if TYPE_CHECKING:
    from .session import ParentSession

SNAPSHOT_KEYS = {
    "schema_version",
    "lineage_id",
    "turn_id",
    "previous_turn_id",
    "phase",
    "decision_id",
    "context_sha256",
}
REPLACEMENT_KEYS = {
    "replacement_id",
    "recorded_at",
    "reason",
    "restart_evidence",
    "from_run",
    "from_runtime",
    "previous_runtime",
    "registered_runs",
    "to_run",
    "to_runtime",
    "turn_id",
}


def new(delegation: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "lineage_id": uuid.uuid4().hex,
        "public_key": uuid.uuid4().hex,
        "delegation_id": delegation,
        "phase": "implementing",
        "context": None,
        "runtime": None,
        "turns": [],
        "decisions": [],
        "continuations": [],
        "invalidations": [],
        "refresh_required": False,
        "accepted_turn_id": None,
    }


def state(task: dict[str, Any]) -> dict[str, Any]:
    value = task.get("preflight")
    if value is None:
        return new(task["delegation_id"])
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] not in {1, 2, 3}
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
    if value["schema_version"] == 3:
        replacements = value.get("session_replacements")
        if not isinstance(replacements, list) or not replacements:
            raise ValueError("malformed worker session replacement history")
        for replacement in replacements:
            if (
                not isinstance(replacement, dict)
                or set(replacement)
                not in (REPLACEMENT_KEYS, REPLACEMENT_KEYS | {"cancelled"})
                or replacement.get("reason")
                not in {"recovery_error", "missing_session_id"}
                or not isinstance(replacement.get("from_run"), dict)
                or not isinstance(replacement.get("from_runtime"), dict)
                or not isinstance(replacement.get("registered_runs"), list)
                or (
                    replacement["to_runtime"] is not None
                    and not isinstance(replacement["to_runtime"], dict)
                )
                or (
                    "cancelled" in replacement
                    and type(replacement["cancelled"]) is not bool
                )
                or any(
                    not isinstance(replacement.get(key), str)
                    or not replacement[key].strip()
                    for key in (
                        "replacement_id",
                        "turn_id",
                        "to_run",
                        "restart_evidence",
                        "recorded_at",
                    )
                )
                or not any(
                    t["snapshot"]["turn_id"] == replacement["turn_id"]
                    and t.get("source") == replacement["to_run"]
                    and bool(t.get("cancelled")) == bool(replacement.get("cancelled"))
                    for t in value["turns"]
                )
            ):
                raise ValueError("malformed worker session replacement history")
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
    auth = observed_authorization(session) if authorized is None else authorized
    parent, items = session.work_items() if work_items is None else work_items
    if child not in items:
        raise ValueError("worker requires a direct child or the standalone issue")
    task = session.row["tasks"].get(str(child), {})
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
            }
            if dep in items:
                visit(dep, raw.get("body") or "")

    visit(child, items[child].body)
    # The default query only covers the parent's family. Include external and
    # completed dependency issues explicitly, through their own timelines.
    pulls = session.gh.pulls(set(found))
    for dep, entry in found.items():
        entry["prs"] = sorted(
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
        )
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


def observe_context(
    session: ParentSession,
    child: int,
    value: dict[str, Any],
    *,
    persist: bool,
    current: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    try:
        fresh = context(session, child) if current is None else current
    except ExecutionBasisStale:
        return None, True
    return fresh, value["context"] is not None and value["context"] != fresh


def review_errors(task: dict[str, Any]) -> tuple[str, ...]:
    # An explicit task stop is a decision. Planning provenance is not evidence
    # that the reviewed code is invalid, and may predate this recording format.
    return ("worker_stopped",) if task.get("stopped") else ()


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
        return summary(task)
    value = state(task)
    _, changed = observe_context(session, child, value, persist=False)
    return {**summary(task), "inputs_changed": changed}


def summary(task: dict[str, Any], *, stale: bool = False) -> dict[str, Any]:
    if "preflight" not in task:
        return {
            "phase": "unrecorded",
            "status": "unrecorded",
            "next_action": "continue_existing_worker",
            "refresh_required": False,
        }
    value = state(task)
    result: dict[str, Any] = {
        "phase": value["phase"],
        "status": value["phase"],
        "next_action": "parent_judgement",
        "refresh_required": False,
        "legacy_result_migrated": "legacy_result" in value,
    }
    replacements = value.get("session_replacements", [])
    if replacements:
        latest = replacements[-1]
        result["session_replacement"] = {
            "count": len(replacements),
            "state": "cancelled"
            if latest.get("cancelled")
            else "bound"
            if latest["to_runtime"] is not None
            else "reserved",
            "reason": latest["reason"],
        }
    return result


def runtime_binding(record: dict[str, Any]) -> dict[str, str]:
    contract = record["contract"]
    return {
        "contract_sha256": digest(execution_contract(contract)),
        "session_id": nonempty(
            record.get("runtime", {}).get("session_id"), "saved worker session"
        ),
    }


def bind_runtime(
    value: dict[str, Any], record: dict[str, Any], *, directory: Path | None = None
) -> None:
    observed = runtime_binding(record)
    replacement = next(
        (
            r
            for r in reversed(value.get("session_replacements", []))
            if not r.get("cancelled")
        ),
        None,
    )
    if replacement is not None and replacement["to_runtime"] is None:
        location = (
            str(directory.resolve()) if directory is not None else record.get("run_dir")
        )
        if (
            location != replacement["to_run"]
            or observed["contract_sha256"]
            != replacement["from_runtime"]["contract_sha256"]
            or observed["session_id"] == replacement["from_runtime"]["session_id"]
        ):
            raise ValueError(
                "runtime does not match the explicit worker session replacement"
            )
        replacement["to_runtime"] = copy.deepcopy(observed)
        value["runtime"] = observed
        return
    legacy = {**observed, "contract_sha256": digest(record["contract"])}
    if value["runtime"] is not None and value["runtime"] not in (observed, legacy):
        raise ValueError("worker session/contract differs from the planning lineage")
    value["runtime"] = observed


def latest_evidence(
    session: ParentSession, child: int, *, persist_ended: bool = True
) -> tuple[Path, dict[str, Any]]:
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

    runner_ended(session, child, persist=persist_ended)
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
    fresh, _ = observe_context(session, child, value, persist=False)
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
    bind_runtime(value, record, directory=directory)
    raw = (directory / "final.md").read_bytes()
    text = raw.decode("utf-8")
    nonempty(text, "planning final")
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


def prepare_turn(
    session: ParentSession,
    child: int,
    instruction: str | None,
    *,
    resume: bool,
    additional: str | None = None,
    previous_record: dict[str, Any] | None = None,
    planning: bool = False,
    replacement: dict[str, Any] | None = None,
    previous_directory: Path | None = None,
) -> tuple[dict[str, Any], str]:
    task = session.row["tasks"][str(child)]
    value = copy.deepcopy(state(task))
    fresh = context(session, child)
    if previous_record is not None:
        bind_runtime(value, previous_record, directory=previous_directory)
    pending = [
        d
        for d in value["decisions"]
        if not d["invalidated"] and d["delivered_turn_id"] is None
    ]
    if len(pending) > 1:
        raise ValueError("multiple undelivered planning answers")
    decision = pending[0] if pending else None
    if decision:
        if not resume:
            raise ValueError("a saved answer requires the existing worker session")
        source = next(
            t for t in value["turns"] if t["snapshot"]["turn_id"] == decision["turn_id"]
        )
        if (
            hashlib.sha256(
                (Path(source["source"]) / "final.md").read_bytes()
            ).hexdigest()
            != decision["final_sha256"]
        ):
            raise ValueError("planning final changed after the parent answer")
        text = "## 親の判断\n\n" + decision["answer"]
        if instruction is not None:
            text += "\n\n" + original_text(instruction)
        selected_phase = (
            "implementing" if decision["action"] == "approve" else "planning"
        )
    else:
        text = original_text(instruction, "worker instruction")
        selected_phase = "planning" if planning else "implementing"
    if additional is not None:
        text += "\n\n## 親の追加指示\n\n" + original_text(additional)
    value.update(
        schema_version=max(2, value["schema_version"]),
        context=fresh,
        refresh_required=False,
        phase=selected_phase,
    )
    turn_id = uuid.uuid4().hex
    previous_turn = next(
        (t for t in reversed(value["turns"]) if not t.get("cancelled")), None
    )
    snapshot = {
        "schema_version": 1,
        "lineage_id": value["lineage_id"],
        "turn_id": turn_id,
        "previous_turn_id": previous_turn["snapshot"]["turn_id"]
        if previous_turn
        else None,
        "phase": selected_phase,
        "decision_id": decision["decision_id"] if decision else None,
        "context_sha256": digest(fresh),
    }
    if decision:
        decision["delivered_turn_id"] = turn_id
    value["turns"].append(
        {
            "snapshot": snapshot,
            "delegation_id": task["delegation_id"],
            "source": None,
            "head_at_start": None,
            "reply_only": False,
            "instruction_source": {
                "kind": "answer" if decision else "task",
                "text": decision["answer"] if decision else original_text(instruction),
                "additional": (
                    [instruction] if decision and instruction is not None else []
                )
                + ([additional] if additional is not None else []),
            },
        }
    )
    if replacement is not None:
        value["schema_version"] = 3
        value.setdefault("session_replacements", []).append(
            {**copy.deepcopy(replacement), "turn_id": turn_id}
        )
        value["runtime"] = None
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
    validate_snapshot(snapshot)
    # Admit exactly the turn reserved by the current parent. Changed prose is
    # supplied as an observation; it does not fabricate a different instruction.
    if not value["turns"] or value["turns"][-1]["snapshot"] != snapshot:
        raise ValueError("worker turn differs from the reserved instruction")


def result_turn(task: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the current result instruction, preserving empty legacy history."""
    value = state(task)
    turn = next((t for t in reversed(value["turns"]) if not t.get("cancelled")), None)
    if value["turns"] and turn is None:
        raise ValueError("no non-cancelled worker turn to verify")
    return turn


def require_implementation(session: ParentSession, child: int) -> str | None:
    """Verify the actual runtime and return the turn ID that was verified."""
    task = session.row["tasks"].get(str(child), {})
    if not task.get("delegated"):
        return None
    turn = result_turn(task)
    directory, record = latest_evidence(session, child)
    value = state(task)
    if turn is not None and (
        turn["source"] != str(directory)
        or record.get("prompt_sha256") != turn.get("prompt_sha256")
    ):
        raise ValueError("worker result does not match the saved instruction")
    bind_runtime(value, record, directory=directory)
    return turn["snapshot"]["turn_id"] if turn is not None else None
