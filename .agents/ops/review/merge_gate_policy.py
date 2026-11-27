"""Light gate CI policy. No public-copy, digest or review coverage checks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from lib.ops_review.review_policy import check_started_at
from parent.policy import nonempty, number

CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
REQUIRED_CHECK = "Python"


def latest_push_ci_run(
    runs: Sequence[dict[str, Any]], head: str, branch: str
) -> dict[str, Any] | None:
    return max(
        (
            run
            for run in runs
            if nonempty(run.get("path"), "Actions workflow path").partition("@")[0]
            == CI_WORKFLOW_PATH
            and run.get("event") == "push"
            and run.get("head_sha") == head
            and run.get("head_branch") == branch
        ),
        key=lambda run: (number(run.get("run_number")), number(run.get("run_attempt"))),
        default=None,
    )


def python_ci_errors(runs: Sequence[Mapping[str, Any]], head: str) -> tuple[str, ...]:
    """Require actual success from the latest Python runs at this HEAD."""
    required = [run for run in runs if run.get("name") == REQUIRED_CHECK]
    if not required:
        return ("ci_python_missing",)
    reasons = []
    dated_errors = []
    for run in required:
        if not isinstance(run.get("head_sha"), str) or not re.fullmatch(
            r"[0-9a-f]{40}", run["head_sha"]
        ):
            raise ValueError("Python check run has no valid head_sha")
        if run["head_sha"] != head:
            reasons.append("ci_python_head_mismatch")
        errors = []
        status = run.get("status")
        if status not in {
            "queued",
            "in_progress",
            "completed",
            "requested",
            "waiting",
            "pending",
        }:
            raise ValueError(f"unknown Python check run status: {status!r}")
        if status != "completed":
            errors.append("ci_python_pending")
        else:
            conclusion = run.get("conclusion")
            if conclusion not in {
                "success",
                "failure",
                "neutral",
                "cancelled",
                "skipped",
                "timed_out",
                "action_required",
                "startup_failure",
                "stale",
            }:
                raise ValueError(f"unknown Python check run conclusion: {conclusion!r}")
            if conclusion != "success":
                errors.append(f"ci_python_{conclusion}")
        dated_errors.append(
            (check_started_at({"startedAt": run.get("started_at")}), errors)
        )
    dated = [started for started, _ in dated_errors if started is not None]
    latest = max(dated) if dated else None
    reasons.extend(
        error
        for started, errors in dated_errors
        if started is None or started == latest
        for error in errors
    )
    return tuple(dict.fromkeys(reasons))


def auto_merge(value: str | None) -> bool:
    if value not in {None, "0", "1"}:
        raise ValueError("AGENT_AUTO_MERGE must be unset, 0 or 1")
    return value == "1"
