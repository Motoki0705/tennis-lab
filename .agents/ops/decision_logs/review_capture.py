"""Machine observations of the automatic progression, separate from parent decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .capture import Operation, current, runtime_environment, warn
from .journal import Journal


def bind_request(identifier: str) -> None:
    capture = current()
    if capture is not None:
        try:
            with capture.transaction():
                capture.row["review_run_id"] = identifier
                capture._save()
        except Exception:
            capture.mark_failed(lost_source="review.binding")


def start_review(
    directory: Path, state: dict[str, Any], runtime: Path
) -> tuple[Operation | None, dict[str, str]]:
    op = None
    try:
        previous = previous_run(
            directory,
            state["request"]["repo"],
            state["request"]["parent_issue"],
            state["id"],
        )
        op = Operation.begin(
            directory,
            state["request"]["repo"],
            state["request"]["parent_issue"],
            "review.run",
            "review",
            scope=previous.row["scope"] if previous else None,
        )
        with op.transaction():
            op.row["review_run_id"] = state["id"]
            op.row["private"]["runs"] = [state["id"]]
            op.row["model_specified_by"] = state["review"].get(
                "selection", state["request"]["review_selection"]
            )["specified_by"]
            op.action("running", "One broad Codex review")
        return op, runtime_environment(runtime, operation=op)
    except Exception:
        if op:
            op.mark_failed(lost_source="review.initialization")
        else:
            warn()
        return None, runtime_environment(runtime)


def reviewed(op: Operation | None, value: dict[str, Any]) -> None:
    if op is None:
        return
    try:
        with op.transaction():
            op.finish(
                value, status="success", summary="Broad review: " + value["verdict"]
            )
    except Exception:
        op.mark_failed(lost_source="review.result")


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


def progression_event(
    directory: Path, state: dict[str, Any], event: dict[str, Any]
) -> None:
    """Capture a persisted notification once; never call it parent verification."""
    path = directory / "progressions" / state["id"] / "decision-capture.json"
    from lib.ops_review.progression import KNOWLEDGE_MERGE, read, save

    op = None
    kind = state["request"].get("kind")
    confirmation = kind == "post_merge_confirmation"
    summary = {
        "post_merge_confirmation": "Confirm existing merged artifacts",
        KNOWLEDGE_MERGE: "Verify knowledge records CI and merge",
    }.get(kind, "CLI automatic review, CI, repair and merge")
    try:
        if path.exists():
            op = Operation.load(Path(read(path)["operation"]))
        else:
            previous = previous_run(
                directory,
                state["request"]["repo"],
                state["request"]["parent_issue"],
                state["id"],
            )
            op = Operation.begin(
                directory,
                state["request"]["repo"],
                state["request"]["parent_issue"],
                "merge.confirmation" if confirmation else "merge.progression",
                "verify" if confirmation else "merge",
                scope=previous.row["scope"] if previous else None,
            )
            with op.transaction():
                op.row["review_run_id"] = state["id"]
                op.row["private"]["runs"] = [state["id"]]
                op.action("running", summary)
            save(path, {"operation": str(op.path)})
        with op.transaction():
            op.result(
                {
                    "event": event,
                    "review": state["review"],
                    "targets": state["targets"],
                },
                artifact="progression state",
                key=f"progression-event:{event['id']}",
                status="success" if event["kind"] == "completed" else "partial",
                observer="machine",
                summary=f"Automatic progression: {event['kind']}; {event.get('reason')}",
            )
            if event["kind"] == "completed":
                op.action(
                    "ended",
                    "Post-merge confirmation completed"
                    if confirmation
                    else "Automatic merge and post-merge confirmation completed",
                    key="completed",
                )
    except Exception:
        if op is not None:
            op.mark_failed(lost_source="merge.progression")
        else:
            warn()
