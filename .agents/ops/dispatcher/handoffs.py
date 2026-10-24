"""Recognize explicit publication-to-review handoffs, without granting authority."""

from __future__ import annotations

import re
from typing import Any

TASK_PATTERN = r"\A<!-- ops-report-memory-task:v1 slot=(\d{8}) publication=[a-f0-9]{64} report=([1-9]\d*) -->\n"


def memory_report(body: str) -> int | None:
    marker = re.match(TASK_PATTERN, body)
    return int(marker[2]) if marker else None


def memory_review_handoff(
    snapshot: dict[str, Any], n: int, pull: dict[str, Any]
) -> bool:
    """A publisher's unowned PR can enter judgment for review, not implementation.

    This only removes the PR-exists veto. The agent must still decide startability;
    existing ownership, holds and dependencies above remain vetoes. Content and
    publication are revalidated by the ordinary merge gate from the local journal.
    """
    issue = snapshot["issues"][str(n)]
    marker = re.match(
        TASK_PATTERN,
        issue["body"],
    )
    pr_marker = re.match(
        r"\A<!-- ops-report-memory:v1 slot=(\d{8}) base=[a-f0-9]{40} head=([a-f0-9]{40}) -->\n",
        pull["body"],
    )
    related = [p for p in snapshot["pulls"] if n in p["issues"]]
    return bool(
        marker
        and pr_marker
        and marker[1] == pr_marker[1]
        and not issue["children"]
        and issue["parent"] is None
        and related == [pull]
        and pull["issues"] == [n]
        and pull["state"] == "open"
        and pull["merged_at"] is None
        and pull["head"]["ref"] == f"ops-report-memory/{marker[1]}"
        and pull["head"]["sha"] == pr_marker[2]
        and f"HEAD `{pr_marker[2]}`" in issue["body"]
    )
