"""Pure policy of the cross-review / merge-gate loop (no I/O).

The rules implemented here are documented in ``README.md`` next to this file, which
is the canonical description. Keep both in sync when changing behaviour.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

AGENTS = ("claude", "codex")
BRANCH_PREFIXES: Mapping[str, str] = {"claude/": "claude", "codex/": "codex"}
AGENT_LABELS: Mapping[str, str] = {"agent:claude": "claude", "agent:codex": "codex"}

NEEDS_HUMAN_LABEL = "needs-human"
CHANGES_REQUESTED_LABEL = "agent-review:changes-requested"
FOLLOWUP_LABEL = "ai-proposed"
# Consecutive request_changes by the cross reviewer that hand the PR to a human.
ESCALATION_STREAK = 3


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


EXCLUDE_PREFIX = "!"


def load_shared_patterns(path: Path) -> tuple[str, ...]:
    """Read shared-foundation patterns; ``!pattern`` lines exclude paths from the set."""
    patterns = load_list_file(path)
    for pattern in patterns:
        body = pattern.removeprefix(EXCLUDE_PREFIX)
        if (
            not body
            or body.startswith(EXCLUDE_PREFIX)
            or body.startswith("/")
            or "\\" in body
            or body.startswith("./")
        ):
            raise ValueError(
                f"{path}: patterns must be repo-relative POSIX globs: {pattern!r}"
            )
    includes = [p for p in patterns if not p.startswith(EXCLUDE_PREFIX)]
    if not includes:
        raise ValueError(f"{path} contains only exclusion patterns")
    for pattern in patterns:
        body = pattern.removeprefix(EXCLUDE_PREFIX)
        if pattern.startswith(EXCLUDE_PREFIX) and not shared_path_hits(
            [body], includes
        ):
            raise ValueError(
                f"{path}: exclusion {pattern!r} is not under any included pattern"
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
    # A pattern matches the path itself or anything below it. DOTALL: file names may
    # contain newlines, and `.*` must still cover them.
    return re.compile("".join(parts) + "(?:/.*)?", re.DOTALL)


def shared_path_hits(
    paths: Iterable[str], patterns: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """Return ``(path, first matching include)`` for every path under shared foundations.

    ``!pattern`` entries exclude matching paths regardless of their position in the list.
    """
    includes: list[tuple[str, re.Pattern[str]]] = []
    excludes: list[re.Pattern[str]] = []
    for pattern in patterns:
        if pattern.startswith(EXCLUDE_PREFIX):
            excludes.append(_glob_regex(pattern.removeprefix(EXCLUDE_PREFIX)))
        else:
            includes.append((pattern, _glob_regex(pattern)))
    hits: list[tuple[str, str]] = []
    for path in paths:
        if any(regex.fullmatch(path) for regex in excludes):
            continue
        for pattern, regex in includes:
            if regex.fullmatch(path):
                hits.append((path, pattern))
                break
    return tuple(hits)


# --------------------------------------------------------------------------- review output

VERDICTS = ("approve", "request_changes", "comment")
SEVERITIES = ("blocker", "major", "minor", "nit")
BLOCKING_SEVERITIES = frozenset({"blocker", "major"})
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
    """What the gate needs from a review: the verdict and its blocker/major count.

    A review passes when it is ``approve`` (never with blocker/major findings), or
    ``comment`` without blocker/major findings; minor/nit findings do not block.
    """

    verdict: str
    blocking: int

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown review verdict: {self.verdict!r}")
        if self.blocking < 0:
            raise ValueError(f"negative blocking count: {self.blocking}")
        if self.verdict == "approve" and self.blocking:
            raise ValueError("approve cannot carry blocker/major findings")
        if self.verdict == "request_changes" and not self.blocking:
            raise ValueError("request_changes needs blocker/major findings")

    @property
    def passed(self) -> bool:
        return self.verdict != "request_changes" and self.blocking == 0


@dataclass(frozen=True)
class Review:
    verdict: str
    summary: str
    findings: tuple[Finding, ...]

    @property
    def outcome(self) -> ReviewOutcome:
        blocking = sum(f.severity in BLOCKING_SEVERITIES for f in self.findings)
        return ReviewOutcome(self.verdict, blocking)

    @property
    def light_findings(self) -> tuple[Finding, ...]:
        """minor / nit findings: they never block, and become a follow-up issue."""
        return tuple(f for f in self.findings if f.severity not in BLOCKING_SEVERITIES)


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
    if verdict == "approve" and severities & {"blocker", "major"}:
        # The gate trusts the verdict alone, so an approve must not carry merge-blocking
        # findings.
        raise ReviewParseError("verdict approve contradicts a blocker/major finding")
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
_DIGEST = r"[0-9a-f]{64}"
_TARGET = rf"sha=({_SHA}) base=({_SHA}) diff=({_DIGEST})"
REVIEW_MARKER_RE = re.compile(
    rf"<!-- agent-cross-review:v3 {_TARGET} reviewer=(claude|codex) "
    r"verdict=(approve|request_changes|comment) blocking=(0|[1-9][0-9]*) -->"
)
# Second line of a review comment: the minor/nit findings as JSON, so a later poll can
# file the follow-up issue from the comment alone.
LIGHT_FINDINGS_RE = re.compile(r"<!-- agent-cross-review-light:v1 (\[.*\]) -->")
GATE_MARKER_RE = re.compile(
    rf"<!-- agent-merge-gate:v2 {_TARGET} decision=(merge|needs_human|blocked) "
    r"reasons=([a-z_,]*) -->"
)
ESCALATION_MARKER_RE = re.compile(rf"<!-- agent-review-escalation:v1 {_TARGET} -->")
FOLLOWUP_MARKER_RE = re.compile(
    rf"<!-- agent-review-followup:v1 pr=([1-9][0-9]*) sha=({_SHA}) -->"
)


@dataclass(frozen=True)
class Comment:
    author: str
    body: str
    url: str


@dataclass(frozen=True)
class Issue:
    author: str
    body: str
    url: str


@dataclass(frozen=True)
class ReviewTarget:
    """What a review judged: the PR head, its merge-base with the base branch, and the
    SHA-256 of the ``merge_base..head`` diff given to the reviewer.

    A retargeted base (e.g. from an unmerged parent branch to main) keeps the head but
    changes the merge-base and the diff, so the earlier review no longer applies.
    """

    head: str
    merge_base: str
    diff_sha256: str

    def marker_fields(self) -> str:
        return f"sha={self.head} base={self.merge_base} diff={self.diff_sha256}"


@dataclass(frozen=True)
class ReviewRecord:
    target: ReviewTarget
    reviewer: str
    outcome: ReviewOutcome
    light_findings: tuple[Finding, ...]
    url: str | None  # the review comment; None for a dry-run review that was not posted


@dataclass(frozen=True)
class GateRecord:
    target: ReviewTarget
    decision: str
    reasons: tuple[str, ...]


def review_header(target: ReviewTarget, reviewer: str, review: Review) -> str:
    """The two marker lines that start a review comment (state for later polls)."""
    outcome = review.outcome
    marker = (
        f"<!-- agent-cross-review:v3 {target.marker_fields()} "
        f"reviewer={reviewer} verdict={outcome.verdict} blocking={outcome.blocking} -->"
    )
    if REVIEW_MARKER_RE.fullmatch(marker) is None:
        raise ValueError(
            f"invalid review marker fields: {target!r} {reviewer!r} {outcome!r}"
        )
    payload = json.dumps([vars(f) for f in review.light_findings], ensure_ascii=False)
    # JSON escapes keep the payload on one line and unable to close the HTML comment.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"{marker}\n<!-- agent-cross-review-light:v1 {payload} -->"


def gate_marker(target: ReviewTarget, decision: str, reasons: Sequence[str]) -> str:
    marker = (
        f"<!-- agent-merge-gate:v2 {target.marker_fields()} "
        f"decision={decision} reasons={','.join(reasons)} -->"
    )
    if GATE_MARKER_RE.fullmatch(marker) is None:
        raise ValueError(
            f"invalid gate marker fields: {target!r} {decision!r} {reasons!r}"
        )
    return marker


def escalation_marker(streak_start: ReviewTarget) -> str:
    return f"<!-- agent-review-escalation:v1 {streak_start.marker_fields()} -->"


def followup_marker(pr_number: int, head: str) -> str:
    marker = f"<!-- agent-review-followup:v1 pr={pr_number} sha={head} -->"
    if FOLLOWUP_MARKER_RE.fullmatch(marker) is None:
        raise ValueError(f"invalid follow-up marker fields: {pr_number!r} {head!r}")
    return marker


def _header_line(body: str, index: int = 0) -> str:
    """Line ``index`` of a comment; markers live in the first lines the caller writes."""
    lines = body.split("\n", index + 1)
    return lines[index].rstrip("\r") if len(lines) > index else ""


def _light_findings(line: str, *, where: str) -> tuple[Finding, ...]:
    match = LIGHT_FINDINGS_RE.fullmatch(line)
    if match is None:
        raise ValueError(f"{where}: review marker without the findings line")
    raw = json.loads(match.group(1))
    if not isinstance(raw, list):
        raise ValueError(f"{where}: findings line is not a JSON list")
    findings = tuple(_parse_finding(item, i) for i, item in enumerate(raw))
    if any(f.severity in BLOCKING_SEVERITIES for f in findings):
        raise ValueError(f"{where}: findings line contains blocker/major findings")
    return findings


def neutralize_markers(text: str) -> str:
    """Make untrusted text (agent output, file names) unable to form an HTML comment.

    Markers are only read from a comment's first line, which the caller writes; this is
    defence in depth so quoted text can neither look like a marker nor hide content.
    """
    return text.replace("<!--", "&lt;!--").replace("-->", "--&gt;")


def review_records(
    comments: Iterable[Comment], *, trusted_login: str
) -> tuple[ReviewRecord, ...]:
    """Review markers on the first line of comments by ``trusted_login``.

    Markers from other accounts, and markers anywhere but the first line (e.g. quoted
    in a review summary), are ignored.
    """
    records: list[ReviewRecord] = []
    for comment in comments:
        if comment.author != trusted_login:
            continue
        match = REVIEW_MARKER_RE.fullmatch(_header_line(comment.body))
        if match is not None:
            head, base, diff, reviewer, verdict, blocking = match.groups()
            records.append(
                ReviewRecord(
                    target=ReviewTarget(head, base, diff),
                    reviewer=reviewer,
                    outcome=ReviewOutcome(verdict, int(blocking)),
                    light_findings=_light_findings(
                        _header_line(comment.body, 1), where=comment.url
                    ),
                    url=comment.url,
                )
            )
    return tuple(records)


def gate_records(
    comments: Iterable[Comment], *, trusted_login: str
) -> tuple[GateRecord, ...]:
    """Gate markers on the first line of comments by ``trusted_login``."""
    records: list[GateRecord] = []
    for comment in comments:
        if comment.author != trusted_login:
            continue
        match = GATE_MARKER_RE.fullmatch(_header_line(comment.body))
        if match is not None:
            head, base, diff, decision, reasons = match.groups()
            records.append(
                GateRecord(
                    target=ReviewTarget(head, base, diff),
                    decision=decision,
                    reasons=tuple(r for r in reasons.split(",") if r),
                )
            )
    return tuple(records)


def escalated_streaks(
    comments: Iterable[Comment], *, trusted_login: str
) -> frozenset[ReviewTarget]:
    """Streak starts (first request_changes target) that were already escalated."""
    started: set[ReviewTarget] = set()
    for comment in comments:
        if comment.author != trusted_login:
            continue
        match = ESCALATION_MARKER_RE.fullmatch(_header_line(comment.body))
        if match is not None:
            started.add(ReviewTarget(*match.groups()))
    return frozenset(started)


def followup_issues(
    issues: Iterable[Issue], *, trusted_login: str
) -> dict[tuple[int, str], str]:
    """``(PR number, head) -> issue URL`` for follow-up issues filed by ``trusted_login``."""
    filed: dict[tuple[int, str], str] = {}
    for issue in issues:
        if issue.author != trusted_login:
            continue
        match = FOLLOWUP_MARKER_RE.fullmatch(_header_line(issue.body))
        if match is not None:
            filed.setdefault((int(match.group(1)), match.group(2)), issue.url)
    return filed


def request_changes_streak(
    records: Sequence[ReviewRecord], *, reviewer: str
) -> tuple[ReviewRecord, ...]:
    """The current run of consecutive request_changes reviews by ``reviewer``.

    Across head SHAs, in comment order. A passing review resets the run; a ``comment``
    with blocker/major findings neither extends nor resets it.
    """
    streak: list[ReviewRecord] = []
    for record in records:
        if record.reviewer != reviewer:
            continue
        if record.outcome.passed:
            streak = []
        elif record.outcome.verdict == "request_changes":
            streak.append(record)
    return tuple(streak)


def latest_review_for(
    records: Sequence[ReviewRecord], *, target: ReviewTarget, reviewer: str
) -> ReviewRecord | None:
    """The last review of exactly ``target`` (head, merge-base and diff) by ``reviewer``."""
    matching = [r for r in records if r.target == target and r.reviewer == reviewer]
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


# gh prints an unset GraphQL DateTime (e.g. a queued CheckRun's startedAt) as Go's zero
# time instead of null.
_ZERO_TIME = datetime(1, 1, 1, tzinfo=UTC)
# Pessimism order used when the latest run of a check cannot be determined.
_SEVERITY = {CheckState.SUCCESS: 0, CheckState.PENDING: 1, CheckState.FAILURE: 2}


def _started_at(item: Mapping[str, Any]) -> datetime | None:
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
    (failure > pending > success), so an unordered queued re-run blocks the merge.
    """
    runs: dict[tuple[str, str], list[tuple[datetime | None, Check]]] = {}
    for item in items:
        key, check = _check_from_rollup(item)
        runs.setdefault(key, []).append((_started_at(item), check))
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
    review: ReviewOutcome | None
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

    if gate.review is None:
        reasons.append("review_missing")
    elif gate.review.verdict == "request_changes":
        reasons.append("review_request_changes")
    elif not gate.review.passed:
        reasons.append("review_comment_blocking")

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
