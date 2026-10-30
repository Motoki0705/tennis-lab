"""Reusable review parsing and merge policy (no I/O).

The rules implemented here are documented in ``README.md`` next to this file, which
is the canonical description. Keep both in sync when changing behaviour.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

AGENTS = ("claude", "codex")
BRANCH_PREFIXES: Mapping[str, str] = {"claude/": "claude", "codex/": "codex"}
AGENT_LABELS: Mapping[str, str] = {"agent:claude": "claude", "agent:codex": "codex"}

# Explicit parent runs always use the review profile with Codex.
REVIEWER = "codex"

NEEDS_HUMAN_LABEL = "needs-human"


# --------------------------------------------------------------------------- authorship


class AuthorConflictError(ValueError):
    """The PR carries contradicting authorship signals."""


def detect_author_agent(head_ref: str, labels: Iterable[str]) -> str | None:
    """Return the agent that authored a PR, or ``None`` when it cannot be determined.

    The branch prefix is the primary signal and ``agent:<name>`` labels are auxiliary.
    Contradicting signals raise instead of silently picking one side.
    """
    by_branch = next(
        (
            agent
            for prefix, agent in BRANCH_PREFIXES.items()
            if head_ref.startswith(prefix)
        ),
        None,
    )
    by_label = sorted(
        {AGENT_LABELS[label] for label in labels if label in AGENT_LABELS}
    )
    if len(by_label) > 1:
        raise AuthorConflictError(f"multiple agent labels on the PR: {by_label}")
    label_agent = by_label[0] if by_label else None
    if by_branch is not None and label_agent is not None and by_branch != label_agent:
        raise AuthorConflictError(
            f"branch {head_ref!r} says {by_branch} but label says {label_agent}"
        )
    return by_branch if by_branch is not None else label_agent


# --------------------------------------------------------------------------- path lists


def load_list_file(path: Path) -> tuple[str, ...]:
    """Read a one-entry-per-line file; ``#`` starts a comment. Empty lists are errors."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            entries.append(entry)
    if not entries:
        raise ValueError(f"{path} contains no entries")
    return tuple(entries)


# --------------------------------------------------------------------------- review output

VERDICTS = ("approve", "request_changes")
# Only merge-blocking findings are reported (#1178); minor / nit are not part of the
# schema, so nothing below merge-blocking can pile up as issues.
SEVERITIES = ("blocker", "major")
_FENCE = re.compile(r"```(?:json)?[ \t]*\n(.*)\n```", re.DOTALL)


class ReviewParseError(ValueError):
    """The reviewer agent's final message does not follow the required schema."""


@dataclass(frozen=True)
class Finding:
    severity: str
    path: str | None
    line: int | None
    rule: str
    detail: str


@dataclass(frozen=True)
class ReviewOutcome:
    """What the gate needs from a review: the verdict and its finding count.

    ``approve`` has no findings and passes; ``request_changes`` has at least one.
    """

    verdict: str
    blocking: int

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown review verdict: {self.verdict!r}")
        if self.blocking < 0:
            raise ValueError(f"negative blocking count: {self.blocking}")
        if self.verdict == "approve" and self.blocking:
            raise ValueError("approve cannot carry findings")
        if self.verdict == "request_changes" and not self.blocking:
            raise ValueError("request_changes needs findings")

    @property
    def passed(self) -> bool:
        return self.verdict == "approve"


@dataclass(frozen=True)
class Review:
    verdict: str
    summary: str
    findings: tuple[Finding, ...]

    @property
    def outcome(self) -> ReviewOutcome:
        return ReviewOutcome(self.verdict, len(self.findings))


def _require_str(obj: Mapping[str, Any], key: str, *, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReviewParseError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def parse_finding(raw: Any, index: int) -> Finding:
    where = f"findings[{index}]"
    if not isinstance(raw, dict):
        raise ReviewParseError(f"{where} must be an object")
    expected = {"severity", "path", "line", "rule", "detail"}
    if set(raw) != expected:
        raise ReviewParseError(
            f"{where} keys must be exactly {sorted(expected)}, got {sorted(raw)}"
        )
    severity = raw["severity"]
    if severity not in SEVERITIES:
        raise ReviewParseError(
            f"{where}.severity must be one of {SEVERITIES}, got {severity!r}"
        )
    path = raw["path"]
    if path is not None and (not isinstance(path, str) or not path.strip()):
        raise ReviewParseError(f"{where}.path must be null or a non-empty string")
    line = raw["line"]
    if line is not None and (
        isinstance(line, bool) or not isinstance(line, int) or line < 1
    ):
        raise ReviewParseError(f"{where}.line must be null or a positive integer")
    return Finding(
        severity=severity,
        path=path,
        line=line,
        rule=_require_str(raw, "rule", where=where),
        detail=_require_str(raw, "detail", where=where),
    )


def parse_review(text: str) -> Review:
    """Parse the reviewer's final message: one JSON object, optionally in one json fence."""
    stripped = text.strip()
    fence = _FENCE.fullmatch(stripped)
    payload = fence.group(1) if fence else stripped
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReviewParseError(f"final message is not a JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewParseError("final message must be a JSON object")
    expected = {"verdict", "summary", "findings"}
    if set(data) != expected:
        raise ReviewParseError(
            f"top-level keys must be exactly {sorted(expected)}, got {sorted(data)}"
        )
    verdict = data["verdict"]
    if verdict not in VERDICTS:
        raise ReviewParseError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if not isinstance(data["findings"], list):
        raise ReviewParseError("findings must be a list")
    findings = tuple(parse_finding(raw, i) for i, raw in enumerate(data["findings"]))
    # Every reported finding blocks the merge, so the verdict must agree with them.
    if verdict == "approve" and findings:
        raise ReviewParseError("verdict approve must come with no findings")
    if verdict == "request_changes" and not findings:
        raise ReviewParseError("verdict request_changes needs at least one finding")
    return Review(
        verdict=verdict,
        summary=_require_str(data, "summary", where="root"),
        findings=findings,
    )


def neutralize_markers(text: str) -> str:
    """Prevent untrusted agent output and paths from forming HTML comments."""
    return text.replace("<!--", "&lt;!--").replace("-->", "--&gt;")


# --------------------------------------------------------------------------- CI state


class CheckState(StrEnum):
    SUCCESS = "success"
    PENDING = "pending"
    FAILURE = "failure"


@dataclass(frozen=True)
class Check:
    name: str
    state: CheckState


_RUN_OK = {"SUCCESS", "NEUTRAL", "SKIPPED"}
_RUN_BAD = {
    "FAILURE",
    "CANCELLED",
    "TIMED_OUT",
    "ACTION_REQUIRED",
    "STARTUP_FAILURE",
    "STALE",
}
_CONTEXT_STATES = {
    "SUCCESS": CheckState.SUCCESS,
    "PENDING": CheckState.PENDING,
    "EXPECTED": CheckState.PENDING,
    "FAILURE": CheckState.FAILURE,
    "ERROR": CheckState.FAILURE,
}


# gh prints an unset GraphQL DateTime (e.g. a queued CheckRun's startedAt) as Go's zero
# time instead of null.
_ZERO_TIME = datetime(1, 1, 1, tzinfo=UTC)
# Pessimism order used when the latest run of a check cannot be determined.
_SEVERITY = {CheckState.SUCCESS: 0, CheckState.PENDING: 1, CheckState.FAILURE: 2}


def check_started_at(item: Mapping[str, Any]) -> datetime | None:
    """``startedAt`` as an aware datetime, or ``None`` when unset (null, "" or zero time)."""
    raw = item.get("startedAt")
    if raw is None or raw == "":
        return None
    started = datetime.fromisoformat(str(raw))
    if started.tzinfo is None:
        raise ValueError(f"statusCheckRollup startedAt without timezone: {raw!r}")
    return None if started == _ZERO_TIME else started


def _check_from_rollup(item: Mapping[str, Any]) -> tuple[tuple[str, str], Check]:
    kind = item.get("__typename")
    if kind == "CheckRun":
        name = str(item["name"])
        status = item.get("status")
        if status != "COMPLETED":
            state = CheckState.PENDING
        elif item.get("conclusion") in _RUN_OK:
            state = CheckState.SUCCESS
        elif item.get("conclusion") in _RUN_BAD:
            state = CheckState.FAILURE
        else:
            raise ValueError(f"unknown CheckRun conclusion: {item.get('conclusion')!r}")
        return (str(item.get("workflowName") or ""), name), Check(name, state)
    if kind == "StatusContext":
        name = str(item["context"])
        raw_state = item.get("state")
        if raw_state not in _CONTEXT_STATES:
            raise ValueError(f"unknown StatusContext state: {raw_state!r}")
        return ("", name), Check(name, _CONTEXT_STATES[raw_state])
    raise ValueError(f"unknown statusCheckRollup entry type: {kind!r}")


def normalize_rollup(items: Iterable[Mapping[str, Any]]) -> tuple[Check, ...]:
    """Collapse a ``statusCheckRollup`` to the latest run per (workflow, check name).

    When the latest run cannot be determined (runs without ``startedAt``, or several runs
    sharing the latest time) the most pessimistic state among the candidates is used
    (failure > pending > success), preserving an unordered queued re-run as pending.
    """
    runs: dict[tuple[str, str], list[tuple[datetime | None, Check]]] = {}
    for item in items:
        key, check = _check_from_rollup(item)
        runs.setdefault(key, []).append((check_started_at(item), check))
    result: list[Check] = []
    for entries in runs.values():
        dated = [started for started, _ in entries if started is not None]
        newest = max(dated) if dated else None
        candidates = [
            check for started, check in entries if started is None or started == newest
        ]
        result.append(max(candidates, key=lambda check: _SEVERITY[check.state]))
    return tuple(result)


# --------------------------------------------------------------------------- merge gate


class GateDecision(StrEnum):
    MERGE = "merge"
    NEEDS_HUMAN = "needs_human"
    BLOCKED = "blocked"


# Reasons worth a gate comment; the caller supplies verified parent-run evidence.
COMMENT_REASONS = frozenset({"conflict"})


@dataclass(frozen=True)
class GateInput:
    is_draft: bool
    is_cross_repository: bool
    mergeable: str
    review: ReviewOutcome | None


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    reasons: tuple[str, ...]


def evaluate_gate(gate: GateInput) -> GateResult:
    """Decide from review, repository identity and mergeability."""
    reasons: list[str] = []
    if gate.is_draft:
        reasons.append("draft")
    if gate.is_cross_repository:
        reasons.append("fork")
    if gate.mergeable == "CONFLICTING":
        reasons.append("conflict")
    elif gate.mergeable == "UNKNOWN":
        reasons.append("mergeable_unknown")
    elif gate.mergeable != "MERGEABLE":
        raise ValueError(f"unknown mergeable state: {gate.mergeable!r}")

    if gate.review is None:
        reasons.append("review_missing")
    elif not gate.review.passed:
        reasons.append("review_request_changes")

    if reasons:
        return GateResult(GateDecision.BLOCKED, tuple(reasons))
    return GateResult(GateDecision.MERGE, ())


def gate_should_comment(result: GateResult) -> bool:
    if result.decision is GateDecision.BLOCKED:
        return bool(set(result.reasons) & COMMENT_REASONS)
    return True
