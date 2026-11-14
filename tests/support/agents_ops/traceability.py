"""Explicit public facts, independent of free-form summaries and private IDs."""

from __future__ import annotations

from typing import Any

from tests.support.agents_ops.decision_logs import (
    AT,
    evidence,
    missing,
    observed,
    result,
)


def worker_observation(
    execution: str = "unknown", final: str = "unknown", publication: str = "unknown"
) -> dict[str, Any]:
    proof = [evidence()]
    return {
        "version": 1,
        "kind": "worker_turn",
        "execution": {"state": execution, "evidence": proof},
        "final": {"state": final, "evidence": proof},
        "publication": {
            "state": publication,
            "stage": None,
            "error": None,
            "pr_url": "https://github.com/o/r/pull/42"
            if publication == "public_pr"
            else None,
            "pr_head": "d" * 40 if publication == "public_pr" else None,
            "main_revision": "e" * 40 if publication == "main" else None,
            "evidence": proof,
        },
    }


def remaining_publication(kind: str, **updates: Any) -> dict[str, Any]:
    """The full RemainingPublication contract from PR #1984 reconciliation.py."""
    worker = kind == "worker_log"
    return {
        "kind": kind,
        "parent": 1,
        "issue": 10 if worker else None,
        "public_key": "d-" + "a" * 32 if worker else None,
        "sequence": 2 if worker else None,
        "state": "pending" if worker else "unprepared",
        "stage": "pr_open" if worker else "awaiting_publication",
        "error": None,
        "pr_number": 42 if worker else None,
        "pr_head": "d" * 40 if worker else None,
        "unpublished_count": None if worker else 2,
        "batch_key": None,
        **updates,
    }


def lifecycle_receipt(**updates: Any) -> dict[str, Any]:
    """record_exit's public identifiers, without depending on its sibling branch."""
    return {
        "repo": "o/r",
        "parent": 1,
        "generation": 1,
        "event": "ended",
        "observed_at": AT,
        "main_revision": "c" * 40,
        "remaining": [],
        "remaining_complete": True,
        "publication_actor": 1,
        "resume_command": "parent --repo o/r --parent 1 publish-worker-log --all-parents --source-parent 1",
        "decision_resume_command": "parent --repo o/r --parent 1 publish-decision-log",
        "error": None,
        **updates,
    }


def termination_observation(**updates: Any) -> dict[str, Any]:
    receipt = lifecycle_receipt()
    return {
        **{k: v for k, v in receipt.items() if k not in {"repo", "parent"}},
        "version": 2,
        "kind": "termination",
        "acceptance": {"state": "completed", "evidence": [evidence()]},
        "evidence": [evidence()],
        **updates,
    }


def observation_record(observation: dict[str, Any], **scope: Any) -> dict[str, Any]:
    row: dict[str, Any] = result()
    row["schema_version"] = 2
    row["scope"].update(scope)
    row["data"].update(
        observation=observation,
        summary="Explicit source observation; no inferred acceptance",
        final_report=(
            observed(evidence())
            if observation["kind"] == "worker_turn"
            and observation["final"]["state"] == "present"
            else missing("missing_final")
            if observation["kind"] == "worker_turn"
            and observation["final"]["state"] == "absent"
            else missing("not a final-report observation")
        ),
    )
    return row
