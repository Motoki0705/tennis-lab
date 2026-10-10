"""Pure policy of the cross-review / merge-gate loop (no I/O).

The rules implemented here are documented in ``README.md`` next to this file, which
is the canonical description. Keep both in sync when changing behaviour.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

AGENTS = ("claude", "codex")
BRANCH_PREFIXES: Mapping[str, str] = {"claude/": "claude", "codex/": "codex"}
AGENT_LABELS: Mapping[str, str] = {"agent:claude": "claude", "agent:codex": "codex"}

NEEDS_HUMAN_LABEL = "needs-human"
CHANGES_REQUESTED_LABEL = "agent-review:changes-requested"


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


def reviewer_for(author: str) -> str:
    """Return the agent that must review a PR written by ``author``."""
    if author not in AGENTS:
        raise ValueError(f"unknown author agent: {author!r}")
    return "codex" if author == "claude" else "claude"


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


def load_shared_patterns(path: Path) -> tuple[str, ...]:
    patterns = load_list_file(path)
    for pattern in patterns:
        if pattern.startswith("/") or "\\" in pattern or pattern.startswith("./"):
            raise ValueError(
                f"{path}: patterns must be repo-relative POSIX globs: {pattern!r}"
            )
    return patterns


def _glob_regex(pattern: str) -> re.Pattern[str]:
    body = pattern.rstrip("/")
    parts: list[str] = []
    index = 0
    while index < len(body):
        if body.startswith("**/", index):
            parts.append("(?:.*/)?")
            index += 3
        elif body.startswith("**", index):
            parts.append(".*")
            index += 2
        elif body[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif body[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(body[index]))
            index += 1
    # A pattern matches the path itself or anything below it.
    return re.compile("".join(parts) + "(?:/.*)?")


def shared_path_hits(
    paths: Iterable[str], patterns: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """Return ``(path, first matching pattern)`` for every path under shared foundations."""
    compiled = [(pattern, _glob_regex(pattern)) for pattern in patterns]
    hits: list[tuple[str, str]] = []
    for path in paths:
        for pattern, regex in compiled:
            if regex.fullmatch(path):
                hits.append((path, pattern))
                break
    return tuple(hits)


# --------------------------------------------------------------------------- review output

VERDICTS = ("approve", "request_changes", "comment")
SEVERITIES = ("blocker", "major", "minor", "nit")
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
class Review:
    verdict: str
    summary: str
    findings: tuple[Finding, ...]


def _require_str(obj: Mapping[str, Any], key: str, *, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReviewParseError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _parse_finding(raw: Any, index: int) -> Finding:
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
    findings = tuple(_parse_finding(raw, i) for i, raw in enumerate(data["findings"]))
    severities = {finding.severity for finding in findings}
    if verdict == "approve" and "blocker" in severities:
        raise ReviewParseError("verdict approve contradicts a blocker finding")
    if verdict == "request_changes" and not severities & {"blocker", "major"}:
        raise ReviewParseError(
            "verdict request_changes needs at least one blocker/major finding"
        )
    return Review(
        verdict=verdict,
        summary=_require_str(data, "summary", where="root"),
        findings=findings,
    )


# --------------------------------------------------------------------------- comment markers

_SHA = r"[0-9a-f]{40}"
REVIEW_MARKER_RE = re.compile(
    rf"<!-- agent-cross-review:v1 sha=({_SHA}) reviewer=(claude|codex) "
    r"verdict=(approve|request_changes|comment) -->"
)
GATE_MARKER_RE = re.compile(
    rf"<!-- agent-merge-gate:v1 sha=({_SHA}) decision=(merge|needs_human|blocked) "
    r"reasons=([a-z_,]*) -->"
)


@dataclass(frozen=True)
class Comment:
    author: str
    body: str


@dataclass(frozen=True)
class ReviewRecord:
    sha: str
    reviewer: str
    verdict: str


@dataclass(frozen=True)
class GateRecord:
    sha: str
    decision: str
    reasons: tuple[str, ...]


def review_marker(sha: str, reviewer: str, verdict: str) -> str:
    marker = f"<!-- agent-cross-review:v1 sha={sha} reviewer={reviewer} verdict={verdict} -->"
    if REVIEW_MARKER_RE.fullmatch(marker) is None:
        raise ValueError(
            f"invalid review marker fields: {sha!r} {reviewer!r} {verdict!r}"
        )
    return marker


def gate_marker(sha: str, decision: str, reasons: Sequence[str]) -> str:
    marker = f"<!-- agent-merge-gate:v1 sha={sha} decision={decision} reasons={','.join(reasons)} -->"
    if GATE_MARKER_RE.fullmatch(marker) is None:
        raise ValueError(
            f"invalid gate marker fields: {sha!r} {decision!r} {reasons!r}"
        )
    return marker


def review_records(
    comments: Iterable[Comment], *, trusted_login: str
) -> tuple[ReviewRecord, ...]:
    """Review markers posted by ``trusted_login`` (markers from anyone else are ignored)."""
    records: list[ReviewRecord] = []
    for comment in comments:
        if comment.author != trusted_login:
            continue
        for sha, reviewer, verdict in REVIEW_MARKER_RE.findall(comment.body):
            records.append(ReviewRecord(sha=sha, reviewer=reviewer, verdict=verdict))
    return tuple(records)


def gate_records(
    comments: Iterable[Comment], *, trusted_login: str
) -> tuple[GateRecord, ...]:
    records: list[GateRecord] = []
    for comment in comments:
        if comment.author != trusted_login:
            continue
        for sha, decision, reasons in GATE_MARKER_RE.findall(comment.body):
            records.append(
                GateRecord(
                    sha=sha,
                    decision=decision,
                    reasons=tuple(r for r in reasons.split(",") if r),
                )
            )
    return tuple(records)


def latest_review_for(
    records: Sequence[ReviewRecord], *, sha: str, reviewer: str
) -> ReviewRecord | None:
    matching = [r for r in records if r.sha == sha and r.reviewer == reviewer]
    return matching[-1] if matching else None


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


def _check_from_rollup(item: Mapping[str, Any]) -> tuple[tuple[str, str], str, Check]:
    kind = item.get("__typename")
    started = item.get("startedAt") or "9999"  # queued: newer than anything started
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
        return (
            (str(item.get("workflowName") or ""), name),
            str(started),
            Check(name, state),
        )
    if kind == "StatusContext":
        name = str(item["context"])
        raw_state = item.get("state")
        if raw_state not in _CONTEXT_STATES:
            raise ValueError(f"unknown StatusContext state: {raw_state!r}")
        return ("", name), str(started), Check(name, _CONTEXT_STATES[raw_state])
    raise ValueError(f"unknown statusCheckRollup entry type: {kind!r}")


def normalize_rollup(items: Iterable[Mapping[str, Any]]) -> tuple[Check, ...]:
    """Collapse a ``statusCheckRollup`` to the latest run per (workflow, check name)."""
    latest: dict[tuple[str, str], tuple[str, Check]] = {}
    for item in items:
        key, started, check = _check_from_rollup(item)
        if key not in latest or started >= latest[key][0]:
            latest[key] = (started, check)
    return tuple(check for _, check in latest.values())


# --------------------------------------------------------------------------- merge gate


class GateDecision(StrEnum):
    MERGE = "merge"
    NEEDS_HUMAN = "needs_human"
    BLOCKED = "blocked"


# Blocked reasons worth a gate comment. Others either resolve on a later poll (CI pending,
# mergeability unknown, review missing) or already have their own review comment.
COMMENT_REASONS = frozenset({"ci_failed", "conflict"})


@dataclass(frozen=True)
class GateInput:
    is_draft: bool
    is_cross_repository: bool
    mergeable: str
    checks: tuple[Check, ...]
    required_checks: tuple[str, ...]
    review_verdict: str | None
    changed_paths: tuple[str, ...]
    shared_patterns: tuple[str, ...]


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    reasons: tuple[str, ...]
    shared_hits: tuple[tuple[str, str], ...]


def evaluate_gate(gate: GateInput) -> GateResult:
    """Decide merge / needs_human / blocked. Shared-foundation changes need a human."""
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

    states = {check.state for check in gate.checks}
    if CheckState.FAILURE in states:
        reasons.append("ci_failed")
    if CheckState.PENDING in states:
        reasons.append("ci_pending")
    present = {check.name for check in gate.checks}
    if any(name not in present for name in gate.required_checks):
        reasons.append("ci_missing_required")

    if gate.review_verdict is None:
        reasons.append("review_missing")
    elif gate.review_verdict == "request_changes":
        reasons.append("review_request_changes")
    elif gate.review_verdict == "comment":
        reasons.append("review_not_approved")
    elif gate.review_verdict != "approve":
        raise ValueError(f"unknown review verdict: {gate.review_verdict!r}")

    hits = shared_path_hits(gate.changed_paths, gate.shared_patterns)
    if reasons:
        return GateResult(GateDecision.BLOCKED, tuple(reasons), hits)
    if hits:
        return GateResult(GateDecision.NEEDS_HUMAN, ("touches_shared",), hits)
    return GateResult(GateDecision.MERGE, (), hits)


def gate_should_comment(result: GateResult) -> bool:
    if result.decision is GateDecision.BLOCKED:
        return bool(set(result.reasons) & COMMENT_REASONS)
    return True
