"""GitHub-only state shared by rewrite workflows, parent CLI and dispatcher."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Literal, Protocol

from .format import PARENT_LABEL

REWRITE_DONE_LABEL = "rewrite:done"
REWRITE_FAILED_LABEL = "rewrite:failed"
REWRITE_SKIPPED_LABEL = "rewrite:skipped"
REWRITE_ENABLED_SINCE_VARIABLE = "ISSUE_REWRITE_ENABLED_SINCE"
REWRITE_FINAL_LABELS = frozenset(
    {REWRITE_DONE_LABEL, REWRITE_FAILED_LABEL, REWRITE_SKIPPED_LABEL}
)
RewriteState = Literal["not_target", "pending", "done", "failed", "skipped"]


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
            f"repos/{gh.repo}/actions/variables/{REWRITE_ENABLED_SINCE_VARIABLE}"
        )
    except GitHubHTTPError as exc:
        if exc.status == 404:
            return None
        raise
    if not isinstance(raw, dict) or raw.get("name") != REWRITE_ENABLED_SINCE_VARIABLE:
        raise ValueError("invalid issue rewrite repository variable response")
    value = raw.get("value")
    _utc_timestamp(value, REWRITE_ENABLED_SINCE_VARIABLE)
    return str(value)


def rewrite_state(
    labels: Iterable[str], created_at: str, enabled_since: str | None
) -> RewriteState:
    """Classify current GitHub label names and timestamps, without local state."""
    if isinstance(labels, str):
        raise ValueError("labels must be an iterable of label names")
    names = set(labels)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("labels must contain nonempty names")
    final = names & REWRITE_FINAL_LABELS
    if len(final) > 1:
        raise ValueError(f"conflicting issue rewrite labels: {sorted(final)}")
    created = _utc_timestamp(created_at, "created_at")
    enabled = (
        _utc_timestamp(enabled_since, REWRITE_ENABLED_SINCE_VARIABLE)
        if enabled_since is not None
        else None
    )
    if enabled is None or PARENT_LABEL not in names or created < enabled:
        return "not_target"
    if REWRITE_DONE_LABEL in final:
        return "done"
    if REWRITE_FAILED_LABEL in final:
        return "failed"
    if REWRITE_SKIPPED_LABEL in final:
        return "skipped"
    return "pending"
