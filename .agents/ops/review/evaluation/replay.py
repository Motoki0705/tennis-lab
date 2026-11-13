"""Read-only assessment of historical retries using the current evidence validator."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from .catalog import DATA, read, timestamp


def assess_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    reviewed = all(pr["coverage"] == "reviewed" for pr in candidate["prs"])
    waits = sum(len(pr.get("evidence_waits", [])) for pr in candidate["prs"])
    return {
        "can_avoid_full_ai_rereview": reviewed,
        "reason": "validated_code_review_retained"
        if reviewed
        else "not_reviewed_cannot_be_promoted_or_given_invented_evidence_waits",
        "candidate_verdict": candidate["verdict"],
        "coverage": [
            {"pr": pr["number"], "coverage": pr["coverage"]} for pr in candidate["prs"]
        ],
        "evidence_waits": waits,
        "findings_retained": len(candidate["cross_findings"])
        + sum(len(pr["findings"]) for pr in candidate["prs"]),
        "requires_parent_verification": True,
        "current_formal_recovery_authorized": False,
    }


def assess_source(parent_state: Path, identifier: str) -> dict[str, Any]:
    ops = Path(__file__).resolve().parents[2]
    for directory in (ops / "review", ops / "lib"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from review_recovery import source_evidence

    try:
        candidate, validation = source_evidence(parent_state, identifier)
    except (ValueError, OSError) as exc:
        return {
            "can_avoid_full_ai_rereview": None,
            "reason": "saved_evidence_could_not_be_validated",
            "error_type": type(exc).__name__,
            "current_formal_recovery_authorized": False,
        }
    return {**assess_candidate(candidate), "validation": validation}


def historical_reuse(parent_state: Path, *, data: Path = DATA) -> dict[str, Any]:
    specification = read(data / "reuse_pairs.json")
    # A repeated source may legitimately feed several different attempts. Each
    # physical repeat can appear only once in the extra-work total.
    identifiers = [pair["repeat_run"] for pair in specification["pairs"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("the same historical repeat cannot be charged twice")
    rows = []
    totals = {
        key: 0 for key in ("input_tokens", "cached_input_tokens", "output_tokens")
    }
    missing = {key: 0 for key in totals}
    elapsed = 0.0
    wall = 0.0
    for pair in specification["pairs"]:
        source = parent_state / "reviews" / pair["source_run"]
        repeat = parent_state / "reviews" / pair["repeat_run"]
        before, after = read(source / "request.json"), read(repeat / "request.json")
        if (
            before["prs"] != after["prs"]
            or len(before["prs"]) != 1
            or before["prs"][0]["number"] != pair["pr"]
        ):
            raise ValueError(
                "historical retry does not have the identical fixed snapshot"
            )
        if any(
            pr["findings"]
            for directory in (source, repeat)
            for pr in read(directory / "result.json")["prs"]
        ):
            raise ValueError("the historical zero-finding retry contains findings")
        runtime = read(repeat / "agent-runtime/result.json")
        usage = runtime["runtime"]["usage"]
        turn = usage.get("status") == "observed" and (
            usage.get("basis") == "turn" or usage.get("source") == "turn.completed"
        )
        observed = {}
        for key in totals:
            value = usage["value"].get(key) if turn else None
            if value is None:
                missing[key] += 1
            elif type(value) is not int or value < 0:
                raise ValueError("invalid historical usage")
            else:
                totals[key] += value
            observed[key] = value
        duration = runtime["finished_at"] - runtime["started_at"]
        execution = read(repeat / "execution.json")
        total_duration = (
            timestamp(execution["finished_at"]) - timestamp(execution["started_at"])
        ).total_seconds()
        if duration < 0 or total_duration < 0:
            raise ValueError("negative historical elapsed time")
        elapsed += duration
        wall += total_duration
        assessment = assess_source(parent_state, pair["source_run"])
        rows.append(
            {
                **pair,
                "snapshot": before["prs"][0],
                "assessment": assessment,
                "repeat_usage": observed,
                "agent_seconds": duration,
                "review_seconds": total_duration,
                "source_request_file_sha256": hashlib.sha256(
                    (source / "request.json").read_bytes()
                ).hexdigest(),
                "repeat_runtime_file_sha256": hashlib.sha256(
                    (repeat / "agent-runtime/result.json").read_bytes()
                ).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "audit_url": specification["audit_url"],
        "assessment_scope": "saved-evidence eligibility at the historical predecessor boundary; excludes live authorization, publication and superseding-request checks",
        "actual_ai_invocations_in_this_assessment": 0,
        "historical_extra_invocations": len(rows),
        "avoidable_with_current_saved_evidence": sum(
            row["assessment"]["can_avoid_full_ai_rereview"] is True for row in rows
        ),
        "not_avoidable_without_more_code_review": sum(
            row["assessment"]["can_avoid_full_ai_rereview"] is False for row in rows
        ),
        "unconfirmed": sum(
            row["assessment"]["can_avoid_full_ai_rereview"] is None for row in rows
        ),
        "historical_extra_usage": {
            key: {"observed_sum": totals[key], "missing": missing[key]}
            for key in totals
        },
        "historical_extra_agent_seconds": elapsed,
        "historical_extra_review_seconds": wall,
        "monetary_or_quota_savings_claimed": False,
        "pairs": rows,
    }
