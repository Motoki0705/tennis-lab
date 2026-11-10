"""The explicit parent decision entry and observation context for parent CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from worker_logs.journal import decode, locked, require
from worker_logs.schema import json_bytes, sha

from .capture import (
    CaptureInputError,
    Operation,
    checked_link,
    comparison,
    current,
    initialization_failed,
    observe,
    record,
    source,
    warn,
)
from .graph import validate_graph
from .journal import Journal
from .schema import new_id, validate_record

COMMANDS = {
    "improvement": "record",
    "improvement-publish": "publish",
    "workflow-report": "verify",
    "agree": "record",
    "create-child": "record",
    "basis-record": "record",
    "run-worker": "start_ai",
    "start-worker": "start_ai",
    "continue-worker": "continue_ai",
    "record-worker-run": "record",
    "record-worker-ended": "verify",
    "record-result": "verify",
    "decide-worker-plan": "record",
    "note": "record",
    "delegate": "record",
    "stop": "stop",
    "resume": "resume",
    "stop-worker": "stop",
    "handoff": "handoff",
    "release": "handoff",
    "review-prepare": "record",
    "review-request": "review",
    "review-state": "verify",
    "inspect-worker": "fetch",
    "publish-worker-log": "publish",
    "worker-log-status": "fetch",
}


def add_parsers(commands: Any) -> None:
    commands.add_parser("decide").add_argument("--spec", type=Path, required=True)
    commands.add_parser("decision-log-status")
    commands.add_parser("retry-decision-capture")
    for name in COMMANDS:
        sub = commands.choices[name]
        sub.add_argument("--decision-id")
        sub.add_argument("--plan-id")
        sub.add_argument(
            "--retry-of",
            help="previous captured result ID; never implies an AI restart",
        )
        sub.add_argument(
            "--changes-file",
            type=Path,
            help="explicit code/requirements/instruction/conditions/verified_scope comparison",
        )
        if name in {"record-result", "review-state"}:
            sub.add_argument(
                "--verifies",
                nargs="+",
                help="captured results explicitly checked by this parent",
            )


def capture_for_runtime(
    directory: Path, repo: str, parent: int, runtime: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    journal = Journal(directory, repo, parent)
    for path in (journal.path.parent / "captures").glob("*.json"):
        op = Operation.load(path)
        if op.row["runtime_directory"] == runtime:
            rows.extend(
                r
                for r in op.records("result")
                if "runner.result:" in r["data"]["summary"]
            )
    return sorted(rows, key=lambda r: r["time"]["recorded_at"])


def begin(
    args: argparse.Namespace, store: Any, repo: str, parent: int
) -> Operation | None:
    try:
        return _begin(args, store, repo, parent)
    except CaptureInputError:
        raise
    except Exception:
        initialization_failed(store.directory, repo, parent, "parent." + args.command)
        return None


def _begin(
    args: argparse.Namespace, store: Any, repo: str, parent: int
) -> Operation | None:
    if args.command not in COMMANDS:
        return None
    operation = COMMANDS[args.command]
    if args.command == "run-worker" and args.resume:
        operation = "continue_ai"
    if args.command == "publish-worker-log" and args.retry:
        operation = "retry_publication"
    if getattr(args, "retry_of", None):
        if operation == "fetch":
            operation = "retry_fetch"
        elif operation == "publish":
            operation = "retry_publication"
    # Explicit link validation is part of this new API. A bad supplied ID is an
    # input error; storage failures without supplied links remain best effort.
    journal = Journal(store.directory, repo, parent)
    previous = args.retry_of
    changes = comparison(args.changes_file)
    verifies = list(getattr(args, "verifies", None) or [])
    task: dict[str, Any] = {}
    specified_by = "parent_choice" if getattr(args, "model", None) else "settings"
    try:
        data = store.read_data()
        task = (
            data.get("parents", {})
            .get(f"{repo.casefold()}#{parent}", {})
            .get("tasks", {})
            .get(str(getattr(args, "child", None)), {})
        )
        runtime = None
        if task.get("worker_runs"):
            raw = Path(task["worker_runs"][-1]["run_dir"])
            runtime = str((store.directory / raw).resolve())
        if not runtime:
            turns = task.get("preflight", {}).get("turns", [])
            runtime = turns[-1].get("source") if turns else None
        if args.command == "review-state":
            runtime = str(store.directory / "reviews" / args.run_id / "agent-runtime")
        predecessors = (
            capture_for_runtime(store.directory, repo, parent, runtime)
            if runtime
            else []
        )
        if operation == "continue_ai" and previous is None and predecessors:
            previous = predecessors[-1]["id"]
        if operation == "continue_ai" and predecessors:
            selected = predecessors[0]["data"]["auxiliary"]["selected_model"]
            if selected["status"] == "observed":
                specified_by = selected["value"]["specified_by"]
        if (
            args.command in {"record-result", "record-worker-ended", "review-state"}
            and not verifies
        ):
            verifies = [
                p["id"] for p in predecessors if p["data"]["status"] != "unknown"
            ]
        if args.command == "review-state" and not verifies:
            from .review_capture import previous_run

            mapped = previous_run(store.directory, repo, parent, args.run_id)
            if mapped:
                verifies = [
                    r["id"]
                    for r in mapped.records("result")
                    if r["data"]["status"] != "unknown"
                ]
    except Exception:
        warn()
    scope = None
    phase = task.get("preflight", {})
    if (
        args.command == "run-worker"
        and args.resume
        and phase.get("migration_unstarted")
        and not task.get("worker_runs")
    ):
        operation = "start_ai"
    public_turn = len(phase.get("turns", [])) + (
        1 if args.command in {"run-worker", "start-worker", "continue-worker"} else 0
    )
    if phase.get("public_key") and public_turn:
        scope = {
            "repo": repo,
            "parent": parent,
            "issue": getattr(args, "child", None),
            "activity_key": new_id(),
            "delegation": {
                "key": "d-" + phase["public_key"],
                "turn": public_turn,
            },
            "review_key": None,
        }
    if args.command.startswith("review-"):
        scope = {
            "repo": repo,
            "parent": parent,
            "issue": None,
            "activity_key": new_id(),
            "delegation": None,
            "review_key": new_id(),
        }
        if getattr(args, "run_id", None):
            from .review_capture import previous_run

            try:
                mapped = previous_run(store.directory, repo, parent, args.run_id)
            except Exception:
                mapped = None
                warn()
            if mapped:
                scope = mapped.row["scope"]
    if args.command in {"publish-worker-log", "worker-log-status"}:
        # A command may address several historical turns. Per-log observations
        # retain the actual delegation/sequence instead of claiming the last turn.
        scope = None
    try:
        plan = checked_link(journal, args.decision_id, args.plan_id, operation)
    except CaptureInputError:
        try:
            rejected = Operation.begin(
                store.directory,
                repo,
                parent,
                "parent." + args.command,
                operation,
                issue=getattr(args, "child", None),
            )
            with rejected.transaction():
                rejected.finish(
                    {"decision_id": args.decision_id, "plan_id": args.plan_id},
                    status="rejected",
                    state="rejected_before_start",
                    summary="Explicit decision/plan link rejected before execution",
                )
        except Exception:
            initialization_failed(
                store.directory, repo, parent, "parent." + args.command
            )
        # Failure to record a rejection must not turn invalid input into work.
        raise
    op = Operation.begin(
        store.directory,
        repo,
        parent,
        "parent." + args.command,
        operation,
        issue=getattr(args, "child", None),
        decision_id=args.decision_id,
        plan_id=plan,
        previous_result_id=previous,
        changes=changes,
        scope=scope,
    )
    try:
        with op.transaction():
            op.row["verification_targets"] = verifies
            op.row["model_specified_by"] = specified_by
            if args.command.startswith("review-"):
                op.row["review_run_id"] = getattr(args, "run_id", None)
            if operation not in {"start_ai", "continue_ai", "review"}:
                op.action("running", "parent handler entered; no AI launch implied")
            op._save()
    except Exception:
        op.mark_failed(lost_source="parent.context")
    return op


def complete(args: argparse.Namespace, value: Any, *, exit_code: int = 0) -> Any:
    try:
        return _complete(args, value, exit_code=exit_code)
    except Exception:
        op = current()
        if op is not None:
            op.mark_failed(lost_source="command.response")
        return value


def _complete(args: argparse.Namespace, value: Any, *, exit_code: int = 0) -> Any:
    op = current()
    if op is None:
        return value
    try:
        with op.transaction():
            identifiers = response_identifiers(value)
            if args.command == "improvement" and isinstance(value, dict):
                # This is a public decision journal ID, not a private worker-plan ID.
                identifiers.discard(value.get("decision_id"))
            op.row["private"]["identifiers"] = sorted(
                set(op.row["private"]["identifiers"]) | identifiers
            )
            op._save()
    except Exception:
        op.mark_failed(lost_source="parent.response-context")
    if isinstance(value, dict) and value.get("review_run_id"):
        try:
            with op.transaction():
                op.row["review_run_id"] = value["review_run_id"]
                op._save()
        except Exception:
            op.mark_failed(lost_source="review.binding")
    is_worker = args.command in {
        "run-worker",
        "start-worker",
        "continue-worker",
        "record-worker-run",
    }
    if is_worker:
        # Real runners deliver at durable writes; this also imports existing or
        # fake runner records after the exact binding has been verified.
        try:
            from .observations import runtime_record

            if args.command == "record-worker-run":
                from .capture import runtime_environment

                runtime_environment(args.run_dir, operation=op)

            if op.path.exists():
                op.row = Operation.load(op.path).row
            runtime = op.row["runtime_directory"]
            if runtime and (Path(runtime) / "result.json").is_file():
                path = Path(runtime) / "result.json"
                runtime_record(path, decode(path.read_bytes()), operation=op)
        except Exception:
            warn()
    state: str | None = "ended"
    status = "success" if exit_code == 0 else "failure"
    if args.command == "run-worker" and exit_code == 0:
        runtime_results = [
            r for r in op.records("result") if "runner.result:" in r["data"]["summary"]
        ]
        if not runtime_results:
            status = "unknown"
        elif runtime_results[-1]["data"]["final_report"]["status"] == "unavailable":
            status = "partial"
    if isinstance(value, dict) and value.get("launch_state") == "rejected_before_start":
        state, status = "rejected_before_start", "rejected"
    elif args.command in {"start-worker", "continue-worker", "review-request"}:
        state, status = None, "unknown"
    elif op.records("action")[-1]["data"]["state"] == "rejected_before_start":
        state, status = "rejected_before_start", "rejected"
    verification = (
        args.command
        in {
            "record-result",
            "record-worker-ended",
            "review-state",
        }
        or (args.command == "release" and bool(getattr(args, "spec", None)))
        or (
            args.command == "workflow-report"
            and isinstance(value, dict)
            and value.get("state") == "completed"
        )
    )
    summary = f"parent.{args.command}: {status}"
    if isinstance(value, dict):
        selected = {
            k: value[k]
            for k in (
                "action",
                "child",
                "parent",
                "result",
                "review_state",
                "launch_state",
                "exit_code",
                "phase",
                "parent_action_applied",
                "runtime_result",
                "turn_end_reason",
                "termination_confirmed",
                "cancelled_before_start",
                "candidate_id",
                "revision",
                "issue",
                "publication",
                "state",
            )
            if k in value and type(value[k]) in {str, int, bool}
        }
        assignment = value.get("assignment")
        if isinstance(assignment, dict):
            selected.update(
                {
                    k: assignment[k]
                    for k in ("agent", "branch", "delegated")
                    if k in assignment and type(assignment[k]) in {str, bool}
                }
            )
        if isinstance(value.get("prs"), list):
            selected["prs"] = [
                p["number"] if isinstance(p, dict) else p
                for p in value["prs"]
                if type(p) is int
                or isinstance(p, dict)
                and type(p.get("number")) is int
            ]
        if isinstance(value.get("progress"), dict):
            selected["progress"] = {
                str(k): v
                for k, v in value["progress"].items()
                if str(k).isdigit() and isinstance(v, str)
            }
        if isinstance(value.get("completion"), dict):
            selected["acceptance"] = value["completion"].get("acceptance", "unverified")
        if selected:
            summary += "; " + json_bytes(selected).decode("utf-8").strip()
    observe(
        value,
        status=status,
        state=state,
        summary=summary,
        observer="parent_verification"
        if verification and exit_code == 0
        else "machine",
        verifies=op.row.get("verification_targets", [])
        if verification and exit_code == 0
        else [],
    )
    # Review manifests and runner responses are strict downstream contracts.
    # Discover record IDs through decision-log-status, without changing them.
    return value


def response_identifiers(value: Any) -> set[str]:
    """Register private provenance before a response becomes a captured result."""
    if isinstance(value, list):
        return set().union(*(response_identifiers(v) for v in value))
    if not isinstance(value, dict):
        return set()
    found = {
        v
        for k, v in value.items()
        if k
        in {
            "delegation_id",
            "lineage_id",
            "turn_id",
            "decision_id",  # worker planning response, not parent decide
            "session_id",
            "job_id",
            "run_id",
            "review_run_id",
        }
        and isinstance(v, str)
        and v.strip()
    }
    return found.union(*(response_identifiers(v) for v in value.values()))


def failed(exc: BaseException) -> None:
    op = current()
    if op is None:
        return
    try:
        if op.path.exists():
            op.row = Operation.load(op.path).row
        started = any(
            r["data"]["state"] in {"running", "ended"} for r in op.records("action")
        )
        launch_unknown = op.row["runtime_directory"] is not None and not started
        # A post-start validation/publication exception cannot become pre-start rejection.
        observe(
            {"error_type": type(exc).__name__, "error": str(exc)},
            status="unknown"
            if launch_unknown
            else "failure"
            if started
            else "rejected",
            state="unknown"
            if launch_unknown
            else "ended"
            if started
            else "rejected_before_start",
            summary=f"{op.row['entrypoint']}: {'execution status unknown' if launch_unknown else 'post-start failure' if started else 'rejected before execution'}; {type(exc).__name__}; details in private evidence",
        )
    except Exception:
        warn()


def reason_reference(
    session: Any, spec: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    if "reason_ref" not in spec:
        return spec["reason"], []
    require("reason" not in spec, "record_decision")
    reference = spec["reason_ref"]
    kind = reference["kind"]
    reference_fields = {
        "note": {"kind", "body_sha256"},
        "worker_plan": {"kind", "child", "decision_id"},
        "model_reason": {"kind", "child"},
        "review_reason": {"kind", "run_id"},
        "improvement": {"kind", "key", "event_id"},
    }
    require(
        kind in reference_fields and set(reference) == reference_fields[kind],
        "record_decision",
    )
    if kind == "improvement":
        from issues.improvement import render_event
        from parent.improvements import get

        _, candidate = get(session, reference["key"])
        original = candidate["candidate"]
        require(original["id"] == reference["event_id"], "reference_unavailable")
        evidence = {
            "kind": "public",
            "url": f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{candidate['comments'][original['id']]}",
            "revision": None,
            "content_sha256": sha(render_event(original).encode()),
        }
    elif kind == "note":
        notes = {
            e["body"]
            for e in session.history()
            if e.get("event") == "note"
            and e.get("body_sha256") == reference["body_sha256"]
        }
        matches = [
            c
            for c in session.gh.comments(session.parent)
            if c["body"] in notes and c.get("user", {}).get("login") == session.login
        ]
        require(bool(matches), "reference_unavailable")
        comment = min(matches, key=lambda c: c["id"])
        evidence = {
            "kind": "public",
            "url": f"https://github.com/{session.gh.repo}/issues/{session.parent}#issuecomment-{comment['id']}",
            "revision": None,
            "content_sha256": sha(comment["body"].encode()),
        }
    elif kind == "worker_plan":
        decisions = session.row["tasks"][str(reference["child"])]["preflight"][
            "decisions"
        ]
        original = next(
            d for d in decisions if d["decision_id"] == reference["decision_id"]
        )
        evidence = source(
            "parent.decide-worker-plan", "saved planning answer", original
        )
    elif kind == "model_reason":
        original = session.row["tasks"][str(reference["child"])]
        require(bool(original.get("model_reason")), "reference_unavailable")
        evidence = source(
            "parent.run-worker", "saved model_reason", original["model_reason"]
        )
    elif kind == "review_reason":
        request = next(
            r["request"]
            for r in session.row["reviews"]
            if r["request"]["review_run_id"] == reference["run_id"]
        )
        evidence = source("parent.review-prepare", "request.json reason", request)
    else:
        raise ValueError("unknown reason_ref kind")
    return f"See existing {kind} reason in referenced evidence", [evidence]


def decide(session: Any, path: Path) -> dict[str, Any]:
    return decide_value(session, decode(path.read_bytes()))


def decide_value(session: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """One input path for explicit decisions and canonical domain reason references."""
    session.owner()
    require(
        set(spec)
        <= {
            "inputs",
            "basis",
            "evaluation",
            "reason",
            "reason_ref",
            "next_actions",
            "reuse_scope",
            "recheck_scope",
            "unresolved",
            "resume_conditions",
            "source_kind",
        },
        "record_schema",
    )
    reason, references = reason_reference(session, spec)
    journal = Journal(session.store.directory, session.gh.repo, session.parent)
    key = "parent-decide:" + sha(
        json_bytes(
            {"spec": spec, "reason_evidence": references, "session": session.session}
        )
    )
    # Serialize explicit decision creation so a repeated spec returns the exact
    # frozen record, including generated plan IDs and times.
    with locked(journal.path.parent / "decision-input.lock"):
        frozen = []
        for receipt in (journal.path.parent / "captures").glob("*.json"):
            candidate = Operation.load(receipt)
            for entry in candidate.row["entries"]:
                if entry["key"] == key and entry["record"]["kind"] == "decision":
                    frozen.append((candidate, entry["record"]))
        require(len(frozen) <= 1, "source_changed")
        if frozen:
            pending, row = frozen[0]
            with pending.transaction():
                pending._deliver()
            return {
                "decision_id": row["id"],
                "plan_ids": [p["id"] for p in row["data"]["next_actions"]],
            }
        existing = next(
            (
                e
                for e in journal.read()["entries"]
                if e["event"] == "capture" and e["key"] == key
            ),
            None,
        )
        if existing:
            row = existing["record"]
        else:
            plans = []
            for plan in spec.get("next_actions", []):
                require(
                    isinstance(plan, dict)
                    and set(plan)
                    == {
                        "operation",
                        "summary",
                        "actor",
                        "target",
                        "instruction",
                        "conditions",
                        "expected_result",
                    },
                    "record_schema",
                )
                plans.append({**plan, "id": new_id()})
            evidence = references or [source("parent.decide", "decision spec", spec)]
            scope = {
                "repo": session.gh.repo,
                "parent": session.parent,
                "issue": None,
                "activity_key": new_id(),
                "delegation": None,
                "review_key": None,
            }
            session_id = session.session
            actor = next(
                (
                    e["record"]["data"]["source"]["actor_key"]
                    for e in journal.read()["entries"]
                    if e["event"] == "capture"
                    and e["record"]["kind"] == "decision"
                    and session_id in e["private"]["sessions"]
                ),
                new_id(),
            )
            row = validate_record(
                record(
                    "decision",
                    scope,
                    {
                        "inputs": spec.get("inputs", []),
                        "basis": spec.get("basis", []),
                        "evaluation": spec["evaluation"],
                        "reason": reason,
                        "next_actions": plans,
                        **{
                            k: spec.get(k, [])
                            for k in (
                                "reuse_scope",
                                "recheck_scope",
                                "unresolved",
                                "resume_conditions",
                            )
                        },
                        "source": {
                            "kind": spec.get("source_kind", "parent_choice"),
                            "actor_key": actor,
                            "evidence": evidence,
                        },
                    },
                )
            )
            validate_graph(
                [
                    e["record"]
                    for e in journal.read()["entries"]
                    if e["event"] == "capture"
                ]
                + [row]
            )
            pending = Operation.begin(
                session.store.directory,
                session.gh.repo,
                session.parent,
                "parent.decide",
                "record",
                scope=scope,
            )
            with pending.transaction():
                pending.row["private"]["sessions"] = list(
                    dict.fromkeys([session_id, session_id.partition(":")[2]])
                )
                pending.emit(
                    row, key, payload={"spec": spec, "reason_evidence": references}
                )
                pending.finish(
                    {"decision_id": row["id"]},
                    summary="parent.decide: explicit decision recorded",
                )
            require(not pending.failed, "journal_unavailable")
    return {
        "decision_id": row["id"],
        "plan_ids": [p["id"] for p in row["data"]["next_actions"]],
    }
