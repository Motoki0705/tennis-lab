"""GitHub-only supplement state shared by workflow, parent CLI and dispatcher.

The workflow, parent CLI and dispatcher use the same public names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal, Protocol

from .format import PARENT_LABEL

SUPPLEMENT_DONE_LABEL = "supplement:done"
SUPPLEMENT_FAILED_LABEL = "supplement:failed"
SUPPLEMENT_SKIPPED_LABEL = "supplement:skipped"
SUPPLEMENT_ENABLED_SINCE_VARIABLE = "ISSUE_SUPPLEMENT_ENABLED_SINCE"
SUPPLEMENT_FINAL_LABELS = frozenset(
    {SUPPLEMENT_DONE_LABEL, SUPPLEMENT_FAILED_LABEL, SUPPLEMENT_SKIPPED_LABEL}
)
SupplementState = Literal["not_target", "pending", "done", "failed", "skipped"]


class GitHubReader(Protocol):
    repo: str

    def get(self, endpoint: str) -> object: ...


def _utc_timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value
    ):
        raise ValueError(f"{field} must be UTC YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"invalid {field}: {value}") from exc


def read_enabled_since(gh: GitHubReader) -> str | None:
    """Read the repository variable through gh api; only HTTP 404 disables it."""
    from .cli import GitHubHTTPError

    try:
        raw = gh.get(
            f"repos/{gh.repo}/actions/variables/{SUPPLEMENT_ENABLED_SINCE_VARIABLE}"
        )
    except GitHubHTTPError as exc:
        if exc.status == 404:
            return None
        raise
    if (
        not isinstance(raw, dict)
        or raw.get("name") != SUPPLEMENT_ENABLED_SINCE_VARIABLE
    ):
        raise ValueError("invalid issue supplement repository variable response")
    value = raw.get("value")
    _utc_timestamp(value, SUPPLEMENT_ENABLED_SINCE_VARIABLE)
    return str(value)


def supplement_state(
    labels: Iterable[str], created_at: str, enabled_since: str | None
) -> SupplementState:
    """Classify current GitHub label names and timestamps, without local state."""
    if isinstance(labels, str):
        raise ValueError("labels must be an iterable of label names")
    names = set(labels)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("labels must contain nonempty names")
    final = names & SUPPLEMENT_FINAL_LABELS
    if len(final) > 1:
        raise ValueError(f"conflicting issue supplement labels: {sorted(final)}")
    created = _utc_timestamp(created_at, "created_at")
    enabled = (
        _utc_timestamp(enabled_since, SUPPLEMENT_ENABLED_SINCE_VARIABLE)
        if enabled_since is not None
        else None
    )
    if enabled is None or PARENT_LABEL not in names or created < enabled:
        return "not_target"
    if any(name.startswith("rewrite:") for name in names):
        raise ValueError("legacy rewrite labels must be migrated to supplement labels")
    if SUPPLEMENT_DONE_LABEL in final:
        return "done"
    if SUPPLEMENT_FAILED_LABEL in final:
        return "failed"
    if SUPPLEMENT_SKIPPED_LABEL in final:
        return "skipped"
    return "pending"
