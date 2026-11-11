"""Observe the current review policy without changing reuse or recovery rules."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from worker_logs.journal import decode
from worker_logs.schema import sha

from .capture import (
    RETRIEVAL_ENV,
    CaptureInputError,
    Operation,
    checked_link,
    comparison,
    current,
    initialization_failed,
    observe,
    runtime_environment,
)
from .journal import Journal
from .schema import new_id

T = TypeVar("T")


def initialize(
    args: Any,
    request: dict[str, Any],
    entrypoint: str,
    call: Callable[[], T],
) -> T | None:
    """Reject explicit bad input; isolate media failures from the review itself."""
    try:
        return call()
    except CaptureInputError:
        raise
    except Exception:
        initialization_failed(
            args.state_dir, request["repo"], request["parent_issue"], entrypoint
        )
        return None


def previous_run(
    directory: Path, repo: str, parent: int, run_id: str
) -> Operation | None:
    candidates = []
    for path in (Journal(directory, repo, parent).path.parent / "captures").glob(
        "*.json"
    ):
        op = Operation.load(path)
        if op.row.get("review_run_id") == run_id:
            candidates.append(op)
    # This is a mapping of an exact canonical run, not a time-based causal guess.
    return (
        max(
            candidates,
            key=lambda op: (
                bool(op.row["runtime_directory"]) and op.row["operation"] == "review",
                bool(op.row.get("review_execution")),
                op.row["entries"][-1]["record"]["time"]["recorded_at"],
            ),
        )
        if candidates
        else None
    )


def begin(args: Any, request: dict[str, Any]) -> Operation:
    directory, repo, parent = args.state_dir, request["repo"], request["parent_issue"]
    mapped = previous_run(directory, repo, parent, request["review_run_id"])
    execution_path = directory / "reviews" / request["review_run_id"] / "execution.json"
    execution = decode(execution_path.read_bytes()) if execution_path.exists() else {}
    operation = (
        "fetch"
        if execution.get("agent_invocations", 0) or execution.get("kind") == "adoption"
        else "review"
    )
    if getattr(args, "retry_of", None) and operation == "fetch":
        operation = "retry_fetch"
    if (
        mapped
        and operation == "review"
        and mapped.row["operation"] == "review"
        and not mapped.row["closed"]
        and not getattr(args, "retry_of", None)
        and not getattr(args, "changes_file", None)
        and (
            getattr(args, "decision_id", None) is None
            or args.decision_id == mapped.row["decision_id"]
        )
        and (
            getattr(args, "plan_id", None) is None
            or args.plan_id == mapped.row["plan_id"]
        )
    ):
        with mapped.transaction():
            if not mapped.row.get("review_execution"):
                mapped.row["review_execution"] = True
                mapped._save()
                return mapped
    scope = (
        dict(mapped.row["scope"])
        if mapped
        else {
            "repo": repo,
            "parent": parent,
            "issue": None,
            "activity_key": new_id(),
            "delegation": None,
            "review_key": new_id(),
        }
    )
    plan = checked_link(
        Journal(directory, repo, parent),
        getattr(args, "decision_id", None),
        getattr(args, "plan_id", None),
        operation,
    )
    previous = getattr(args, "retry_of", None)
    if previous is None and mapped and mapped.records("result"):
        previous = mapped.records("result")[-1]["id"]
    op = Operation.begin(
        directory,
        repo,
        parent,
        "review.run",
        operation,
        scope=scope,
        decision_id=getattr(args, "decision_id", None),
        plan_id=plan,
        previous_result_id=previous,
        changes=comparison(getattr(args, "changes_file", None)),
    )
    with op.transaction():
        op.row["review_run_id"] = request["review_run_id"]
        if request.get("review_selection", {}).get("kind") == "run":
            op.row["model_specified_by"] = request["review_selection"]["specified_by"]
        op.row["review_execution"] = operation == "review"
        op._save()
    return op


def observation_source(op: Operation) -> str:
    return (
        "review.run"
        if op.row["entrypoint"] == "parent.review-request"
        else str(op.row["entrypoint"])
    )


def begin_verification(args: Any, request: dict[str, Any]) -> Operation:
    directory, repo, parent = args.state_dir, request["repo"], request["parent_issue"]
    mapped = previous_run(directory, repo, parent, request["review_run_id"])
    plan = checked_link(
        Journal(directory, repo, parent),
        getattr(args, "decision_id", None),
        getattr(args, "plan_id", None),
        "verify",
    )
    op = Operation.begin(
        directory,
        repo,
        parent,
        "review.resolve",
        "verify",
        decision_id=getattr(args, "decision_id", None),
        plan_id=plan,
        previous_result_id=getattr(args, "retry_of", None),
        changes=comparison(getattr(args, "changes_file", None)),
        scope=mapped.row["scope"] if mapped else None,
    )
    with op.transaction():
        op.row["verification_targets"] = (
            [r["id"] for r in mapped.records("result")] if mapped else []
        )
        op.action("running", "review.resolve: explicit parent verification entered")
    return op


def begin_control(
    args: Any, request: dict[str, Any]
) -> tuple[Operation, Operation | None]:
    directory, repo, parent = args.state_dir, request["repo"], request["parent_issue"]
    source_run = getattr(args, "source_run", None) or request["review_run_id"]
    source_op = previous_run(directory, repo, parent, source_run)
    target_op = previous_run(directory, repo, parent, request["review_run_id"])
    operation = {
        "recover": "retry_fetch",
        "resume-publication": "retry_publication",
        "complete-evidence": "verify",
    }[args.command]
    plan = checked_link(
        Journal(directory, repo, parent), args.decision_id, args.plan_id, operation
    )
    previous = args.retry_of or (
        source_op.records("result")[-1]["id"]
        if source_op and source_op.records("result")
        else None
    )
    scope = (
        target_op.row["scope"]
        if target_op
        else {
            "repo": repo,
            "parent": parent,
            "issue": None,
            "activity_key": new_id(),
            "delegation": None,
            "review_key": new_id(),
        }
    )
    op = Operation.begin(
        directory,
        repo,
        parent,
        "review." + args.command,
        operation,
        decision_id=args.decision_id,
        plan_id=plan,
        previous_result_id=previous,
        changes=comparison(args.changes_file),
        scope=scope,
    )
    with op.transaction():
        op.row["review_run_id"] = request["review_run_id"]
        op.row["private"]["runs"] = list({source_run, request["review_run_id"]})
        op.action("running", "review control entered; no AI requested")
    return op, source_op


def control(
    args: Any, request: dict[str, Any], call: Callable[[Any], dict[str, Any]]
) -> dict[str, Any]:
    """Observe current-main recovery/publication/evidence paths without an AI."""
    from .capture import CURRENT
    from .parent_commands import failed

    initialized = initialize(
        args, request, "review." + args.command, lambda: begin_control(args, request)
    )
    op, source_op = initialized if initialized is not None else (None, None)
    token = CURRENT.set(op)
    try:
        value = call(args)
        if op is None:
            return value
        operation = op.row["operation"]
        if value.get("result"):
            result(value["result"], artifact="control result.json")
        observe(
            value,
            status="success"
            if value.get("publication", "published") == "published"
            and value.get("result", {}).get("status", "succeeded") == "succeeded"
            else "failure",
            observer="parent_verification" if operation == "verify" else "machine",
            verifies=[r["id"] for r in source_op.records("result")]
            if operation == "verify" and source_op
            else [],
            summary=f"review.{args.command}: agent_invocations=0; publication={value.get('publication')}; source_review_key={source_op.row['scope']['review_key'] if source_op else None}",
        )
        return value
    except BaseException as exc:
        failed(exc)
        raise
    finally:
        CURRENT.reset(token)


def result(value: dict[str, Any], *, artifact: str) -> None:
    op = current()
    if op is None:
        return
    try:
        with op.transaction():
            if value.get("review_run_id"):
                op.row["private"]["runs"] = sorted(
                    set(op.row["private"]["runs"])
                    | {value["review_run_id"]}
                    | set((value.get("exception") or {}).get("previous_runs", []))
                )
            exception = value.get("exception")
            prior = []
            if exception:
                for run_id in exception["previous_runs"]:
                    mapped = previous_run(
                        Path(op.row["directory"]),
                        op.row["repo"],
                        op.row["parent"],
                        run_id,
                    )
                    if mapped:
                        prior.append(
                            mapped.row["scope"].get("review_key")
                            or mapped.row["scope"]["activity_key"]
                        )
            state = value.get("status") or value.get("publication") or "unknown"
            op.result(
                value,
                artifact=artifact,
                status="success"
                if state in {"succeeded", "published"}
                else "failure"
                if state in {"failed", "publication_failed", "preparation_failed"}
                else "unknown",
                summary=f"review.{artifact}: status={state}; verdict={value.get('verdict')}; exception={bool(exception)}; previous_review_keys={prior}; "
                + str(value.get("summary", "")),
                entrypoint=observation_source(op),
            )
    except Exception:
        op.mark_failed(lost_source="review.result")


def runtime_env(directory: Path) -> dict[str, str]:
    return runtime_environment(directory)


def published(url: str, body: str) -> None:
    op = current()
    if op is None:
        return
    try:
        with op.transaction():
            op.result(
                {"publication": "published", "content_sha256": sha(body.encode())},
                artifact="published review comment",
                status="success",
                summary="review.publication: all PR projections and the parent result published",
                evidence=[
                    {
                        "kind": "public",
                        "url": url,
                        "revision": None,
                        "content_sha256": sha(body.encode()),
                    }
                ],
                entrypoint=observation_source(op),
            )
    except Exception:
        op.mark_failed(lost_source="review.publication")


def retrieval_env() -> dict[str, str]:
    op = current()
    return {RETRIEVAL_ENV: str(op.path)} if op else {}


def complete(value: dict[str, Any]) -> None:
    try:
        _complete(value)
    except Exception:
        op = current()
        if op is not None:
            op.mark_failed(lost_source="review.response")


def _complete(value: dict[str, Any]) -> None:
    op = current()
    if op is None:
        return
    result({"publication": value["publication"]}, artifact="publication")
    state = (
        "rejected_before_start"
        if op.records("action")[-1]["data"]["state"] == "rejected_before_start"
        or value["publication"] == "preparation_failed"
        and not any(a["data"]["state"] == "running" for a in op.records("action"))
        else "ended"
    )
    observe(
        value,
        status="success"
        if value["result"]["status"] == "succeeded"
        and value["publication"] == "published"
        else "failure",
        summary=f"review.run: reused={value['reused']}; publication={value['publication']}; status={value['result']['status']}; verdict={value['result']['verdict']}",
        state=state,
        entrypoint="review.run",
    )


def gate_result(value: dict[str, Any], artifact: str) -> None:
    op = current()
    if op is None:
        return
    try:
        with op.transaction():
            op.result(
                value,
                artifact=artifact,
                status="unknown"
                if value.get("status") in {"attempting", "unconfirmed"}
                else "success"
                if value.get("decision") == "merge" or value.get("status") == "merged"
                else "rejected",
                summary=f"merge_gate.{artifact}: pr={value.get('pr')}; decision={value.get('decision')}; action={value.get('action')}; status={value.get('status')}; reasons={value.get('reasons', [])}",
            )
    except Exception:
        op.mark_failed(lost_source="merge_gate.result")
