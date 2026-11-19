"""Evidence containers are never work issues, even when reopened or mislabelled."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .format import PROPOSAL_LABEL

EVIDENCE_LABEL = "parent-evidence"
EVIDENCE_PREFIX = "<!-- parent-evidence:v1 "


def is_evidence_issue(issue: Mapping[str, Any]) -> bool:
    labels = {
        label if isinstance(label, str) else label["name"]
        for label in issue.get("labels", ())
    }
    body = issue.get("body") or ""
    return EVIDENCE_LABEL in labels or body.startswith("<!-- parent-evidence:")


def is_proposal_issue(issue: Mapping[str, Any]) -> bool:
    labels = {
        label if isinstance(label, str) else label["name"]
        for label in issue.get("labels", ())
    }
    return PROPOSAL_LABEL in labels or (issue.get("body") or "").startswith(
        "<!-- parent-improvement-proposal:"
    )


def require_work_issue(
    issue: Mapping[str, Any], *, allow_proposal: bool = False
) -> None:
    if is_evidence_issue(issue):
        raise ValueError("parent evidence issue cannot be used as a work issue")

    if not allow_proposal and is_proposal_issue(issue):
        raise ValueError("improvement proposal cannot be used as a work issue")
