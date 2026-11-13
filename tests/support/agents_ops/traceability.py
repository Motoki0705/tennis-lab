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


def termination_observation(**updates: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": "termination",
        "generation": 1,
        "event": "ended",
        "observed_at": AT,
        "main_revision": "c" * 40,
        "acceptance": {"state": "completed", "evidence": [evidence()]},
        "evidence": [evidence()],
        "remaining": [],
        "remaining_status": "observed",
        "publication_actor": "parent #1",
        "resume_command": None,
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
