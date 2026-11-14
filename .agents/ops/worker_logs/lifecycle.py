"""Complete a fixed publication boundary and retain private exit checkpoints."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .journal import PublicationError, atomic, locked
from .reconciliation import (
    continue_workers,
    decision_journal,
    remaining_publications,
    require_batch_captures,
)
from .schema import sha


def unavailable_exit(
    repo: str,
    parent: int,
    generation: int | None,
    event: str,
    *,
    publication_actor: int | str = "unassigned",
) -> dict[str, Any]:
    """Retain the exit outcome even when private publication storage fails."""
    return {
        "repo": repo,
        "parent": parent,
        "generation": generation,
        "event": event,
        "observed_at": datetime.now(UTC).isoformat(),
        "main_revision": None,
        "remaining": [],
        "remaining_complete": False,
        "publication_actor": publication_actor,
        "resume_command": f"parent --repo {repo} --parent {parent} publish-worker-log --all-parents --source-parent {parent}",
        "decision_resume_command": f"parent --repo {repo} --parent {parent} publish-decision-log",
        "error": "journal_unavailable",
    }


def publish_decisions(
    repo: str,
    parent: int,
    *,
    state_dir: Path,
    root: Path,
    owner_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Finish an old batch and then the captured boundary, without polling.

    At most one old batch and one new batch are needed. A pending/blocked
    response ends this attempt. Recheck ownership before each batch so a
    completed handoff prevents another publication. Publication bookkeeping
    creates no captures.
    """
    from decision_logs.capture import status as capture_status
    from decision_logs.publication import Publisher
    from decision_logs.publication_state import read

    log = decision_journal(repo, parent, state_dir=state_dir)
    captured = capture_status(state_dir, log.repo, parent, retry=True)
    capture_error = "input_capture_failed" if captured["status"] != "captured" else None
    existing = read(log.path.with_name("publication.json"), log.repo, parent)
    require_batch_captures(log, existing)
    limit = {r["id"] for r in log.status() if r["publication_status"] != "published"}
    if not limit:
        return {
            "saved_to_main": capture_error is None,
            "unpublished": None if capture_error else 0,
            "error": capture_error,
        }
    had_batch = any(b["state"] != "merged" for b in existing["batches"])
    value: dict[str, Any] = {}
    for _ in range(2 if had_batch else 1):
        if owner_check is not None and not owner_check():
            return {**value, "saved_to_main": False, "error": "log_owner_mismatch"}
        value = Publisher(root, log, record_ids=limit).publish()
        if value["state"] != "merged":
            break
        pending = {
            r["id"] for r in log.status() if r["publication_status"] != "published"
        }
        if not limit.intersection(pending):
            break
    if capture_error:
        value = {
            **value,
            "saved_to_main": False,
            "error": value.get("error") or capture_error,
        }
    return value


def record_exit(
    repo: str,
    parent: int,
    generation: int,
    event: str,
    *,
    state_dir: Path,
    root: Path,
    publication_actor: int | str = "unassigned",
    publication_error: str | None = None,
) -> dict[str, Any]:
    """Persist/return only public identifiers; never changes the assignment."""
    if event not in {"ended", "handoff_pending", "handoff_accepted"}:
        raise ValueError("invalid publication exit event")
    observed = datetime.now(UTC).isoformat()
    try:
        revision = (
            subprocess.run(
                ["git", "rev-parse", "origin/main"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if root.is_dir()
            else None
        )
    except (OSError, subprocess.SubprocessError):
        revision = None
    remaining_complete = True
    try:
        remaining = remaining_publications(repo, parent, state_dir=state_dir)
    except (ValueError, OSError, KeyError, TypeError):
        remaining = []
        remaining_complete = False
        publication_error = "journal_unavailable"
    checkpoint = {
        **unavailable_exit(
            repo, parent, generation, event, publication_actor=publication_actor
        ),
        "observed_at": observed,
        "main_revision": revision.stdout.strip()
        if revision is not None and revision.returncode == 0
        else None,
        "remaining": remaining,
        "remaining_complete": remaining_complete,
        "error": publication_error,
    }
    path = (
        state_dir
        / "publication-exits"
        / sha(repo.casefold().encode())
        / str(parent)
        / f"{generation}-{event}.json"
    )
    with locked(path):
        atomic(path, checkpoint)
    return checkpoint


def finish_supervised(
    repo: str,
    parent: int,
    generation: int,
    *,
    state_dir: Path,
    root: Path,
    owner_check: Callable[[], tuple[bool, int | str]] | None = None,
) -> dict[str, Any]:
    """Called after the supervisor has saved ended and released its locks."""
    error = None
    actor: int | str = "unassigned"
    try:
        current, actor = owner_check() if owner_check else (True, actor)
        if current:
            workers = continue_workers(
                repo,
                parent,
                state_dir=state_dir,
                owner_check=lambda: owner_check()[0] if owner_check else True,
            )
            if any(row.get("error") for row in workers):
                error = "publication_failed"
        else:
            error = "log_owner_mismatch"
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        error = str(exc) if isinstance(exc, PublicationError) else "publication_failed"
    try:
        capture_supervised_exit(repo, parent, generation, state_dir=state_dir)
    except (ValueError, OSError, KeyError, TypeError):
        error = error or "journal_unavailable"
    try:
        current, actor = owner_check() if owner_check else (True, actor)
        if current:
            decision = publish_decisions(
                repo,
                parent,
                state_dir=state_dir,
                root=root,
                owner_check=lambda: owner_check()[0] if owner_check else True,
            )
            error = error or decision.get("error")
        else:
            error = "log_owner_mismatch"
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        error = str(exc) if isinstance(exc, PublicationError) else "publication_failed"
    try:
        current, actor = owner_check() if owner_check else (True, actor)
        if not current:
            error = "log_owner_mismatch"
        return record_exit(
            repo,
            parent,
            generation,
            "ended",
            state_dir=state_dir,
            root=root,
            publication_error=error,
            publication_actor=actor,
        )
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError):
        return unavailable_exit(repo, parent, generation, "ended")


def capture_supervised_exit(
    repo: str, parent: int, generation: int, *, state_dir: Path
) -> None:
    """Capture the saved assignment outcome once per generation, before publishing.

    This is a machine observation about the ended assignment. It does not
    reconstruct a missing runtime transcript or infer a new parent decision.
    """
    from decision_logs.capture import missing, record, source
    from decision_logs.schema import CHANGE_FIELDS, new_id

    log = decision_journal(repo, parent, state_dir=state_dir)
    key = f"supervisor-ended:{generation}"
    with locked(log.path.parent / "supervisor-exit-record.json"):
        if any(e["key"] == key for e in log.read()["entries"]):
            return
        payload = {"parent": parent, "generation": generation, "assignment": "ended"}
        row = record(
            "result",
            {
                "repo": log.repo,
                "parent": parent,
                "issue": None,
                "activity_key": new_id(),
                "delegation": None,
                "review_key": None,
            },
            {
                "action_id": None,
                "previous_result_id": None,
                "verifies": [],
                "status": "success",
                "summary": f"dispatcher.supervisor: parent #{parent} generation {generation} assignment ended; matching workflow receipt and runtime cleanup verified",
                "changes": dict.fromkeys(CHANGE_FIELDS, "unconfirmed"),
                "evidence": [
                    source("dispatcher.supervisor", "saved ended assignment", payload)
                ],
                "observer": "machine",
                "final_report": missing(
                    "assignment observation; see existing runtime and workflow records"
                ),
                "auxiliary": {
                    k: missing("assignment observation")
                    for k in (
                        "selected_model",
                        "effective_model",
                        "elapsed_seconds",
                        "wait_seconds",
                        "usage",
                    )
                },
                "repeat": None,
            },
        )
        log.capture(key, row)
