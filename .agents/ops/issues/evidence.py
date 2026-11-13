"""Evidence containers are never work issues, even when reopened or mislabelled."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EVIDENCE_LABEL = "parent-evidence"
EVIDENCE_PREFIX = "<!-- parent-evidence:v1 "


def is_evidence_issue(issue: Mapping[str, Any]) -> bool:
    labels = {
        label if isinstance(label, str) else label["name"]
        for label in issue.get("labels", ())
    }
    body = issue.get("body") or ""
    return EVIDENCE_LABEL in labels or body.startswith("<!-- parent-evidence:")


def require_work_issue(issue: Mapping[str, Any]) -> None:
    if is_evidence_issue(issue):
        raise ValueError("parent evidence issue cannot be used as a work issue")
