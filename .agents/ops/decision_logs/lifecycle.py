"""Bind canonical parent outcomes to one pre-publication termination snapshot."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worker_logs.reconciliation import decision_journal

from .capture import Operation, source, warn
from .trace_capture import capture_termination


def canonical_sources(
    checkpoint: dict[str, Any], *, state_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    """Read fixed structured comment receipts, never interpret their summaries."""
    log = decision_journal(
        checkpoint["repo"], checkpoint["parent"], state_dir=state_dir
    )
    end_events = {
        "ended": {"ended", "released"},
        "handoff_pending": {"handoff_pending"},
        "handoff_accepted": {"handoff"},
    }[checkpoint["event"]]
    comments = []
    for path in sorted((log.path.parent / "captures").glob("*.json")):
        op = Operation.load(path)
        for entry in op.row["entries"]:
            payload = entry["observation"]
            if not isinstance(payload, dict) or set(payload) != {
                "event",
                "body_sha256",
            }:
                continue
            event = payload["event"]
            if (
                not isinstance(event, dict)
                or event.get("parent") != checkpoint["parent"]
                or type(event.get("generation")) is not int
                or event.get("generation") != checkpoint["generation"]
            ):
                continue
            row = entry["record"]
            if (
                row["kind"] == "result"
                and op.row["entrypoint"].startswith("parent.")
                and source(op.row["entrypoint"], "parent issue comment", payload)
                in row["data"]["evidence"]
            ):
                comments.append(
                    (
                        row["time"]["recorded_at"],
                        row["id"],
                        event,
                        row["data"]["evidence"],
                    )
                )
    evidence: list[dict[str, Any]] = []
    acceptance = None
    previous = None
    for _, record_id, event, proof in sorted(comments):
        if event["event"] in end_events:
            evidence = proof
            previous = record_id
        if event["event"] == "completion" and event.get("state") in {
            "completed",
            "incomplete",
        }:
            acceptance = {"state": event["state"], "evidence": proof}
    if previous is None and checkpoint["event"] == "ended":
        for entry in log.read()["entries"]:
            if entry.get("key") == f"supervisor-ended:{checkpoint['generation']}":
                previous = entry["record"]["id"]
                evidence = entry["record"]["data"]["evidence"]
    return evidence, acceptance, previous


def capture_checkpoint(checkpoint: dict[str, Any], *, state_dir: Path) -> bool:
    """Capture once at the exit boundary; failure never vetoes the lifecycle."""
    op = None
    try:
        proof, acceptance, previous = canonical_sources(checkpoint, state_dir=state_dir)
        log = decision_journal(
            checkpoint["repo"], checkpoint["parent"], state_dir=state_dir
        )
        op = Operation.begin(
            state_dir,
            log.repo,
            checkpoint["parent"],
            "parent.termination",
            "record",
            previous_result_id=previous,
        )
        capture_termination(
            op, checkpoint, evidence=proof or None, acceptance=acceptance
        )
        with op.transaction():
            op.row["closed"] = True
            op.action(
                "ended",
                "Parent termination snapshot captured before decision publication",
            )
        return not op.failed
    except Exception:
        if op is not None:
            op.mark_failed(lost_source="parent.termination")
        else:
            # Preserve an inspectable failure even when the source scan failed.
            try:
                op = Operation.begin(
                    state_dir,
                    checkpoint["repo"],
                    checkpoint["parent"],
                    "parent.termination",
                    "record",
                )
                op.mark_failed(lost_source="parent.termination")
            except Exception:
                warn()
        return False
