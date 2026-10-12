"""Pure policy of the follow-up issue resolver (no I/O).

The rules are documented in ``README.md`` next to this file, which is the canonical
description. Keep both in sync when changing behaviour.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "review"))

from review_policy import neutralize_markers  # noqa: E402

DISPOSITIONS = ("fixed", "declined", "already_resolved", "needs_human")
# Comment authors whose words reach the agent prompt (the repo is public).
TRUSTED_ASSOCIATIONS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})
BRANCH_PREFIX = "codex/followup-i"

ROUND_MARKER_RE = re.compile(
    r"<!-- agent-followup-round:v1 issue=(\d+) round=(\d+) base=([0-9a-f]{40}) "
    r"pr=(\d+|none) -->"
)
REJECTED_MARKER_RE = re.compile(
    r"<!-- agent-followup-rejected:v1 issue=(\d+) round=(\d+) pr=(\d+) -->"
)
_FENCE_RE = re.compile(r"\A```(?:json)?\n(.*)\n```\Z", re.DOTALL)


# --------------------------------------------------------------------------- source PR


class SourceAction(StrEnum):
    """What to do with a follow-up issue given the state of the PR it came from."""

    DEFER = "defer"  # source PR still open: its findings target an unmerged head
    PROCESS = "process"  # merged: resolve against the base branch
    INVALIDATE = "invalidate"  # closed without merge: the findings no longer apply


def source_action(state: str) -> SourceAction:
    actions = {
        "OPEN": SourceAction.DEFER,
        "MERGED": SourceAction.PROCESS,
        "CLOSED": SourceAction.INVALIDATE,
    }
    if state not in actions:
        raise ValueError(f"unknown PR state: {state!r}")
    return actions[state]


def is_followup_branch(head_ref: str) -> bool:
    return head_ref.startswith(BRANCH_PREFIX)


def branch_name(issue: int, round_no: int) -> str:
    return f"{BRANCH_PREFIX}{issue}-r{round_no}"


# --------------------------------------------------------------------------- rounds


@dataclass(frozen=True)
class Round:
    """One resolver pass over an issue, recorded by its comment's first line."""

    issue: int
    number: int
    base: str
    pr: int | None


@dataclass(frozen=True)
class IssueComment:
    author: str
    association: str
    body: str
    url: str


def _first_line(body: str) -> str:
    return body.split("\n", 1)[0].rstrip("\r")


def round_marker(r: Round) -> str:
    pr = "none" if r.pr is None else str(r.pr)
    marker = (
        f"<!-- agent-followup-round:v1 issue={r.issue} round={r.number} "
        f"base={r.base} pr={pr} -->"
    )
    if ROUND_MARKER_RE.fullmatch(marker) is None:
        raise ValueError(f"invalid round marker fields: {r!r}")
    return marker


def rejected_marker(issue: int, round_no: int, pr: int) -> str:
    return f"<!-- agent-followup-rejected:v1 issue={issue} round={round_no} pr={pr} -->"


def recorded_rounds(
    issue: int, comments: Iterable[IssueComment], *, trusted_login: str
) -> tuple[Round, ...]:
    """Rounds recorded on ``issue`` by ``trusted_login``, numbered 1..n without gaps."""
    rounds: list[Round] = []
    for c in comments:
        match = ROUND_MARKER_RE.fullmatch(_first_line(c.body))
        if c.author != trusted_login or match is None:
            continue
        if int(match.group(1)) != issue:
            raise ValueError(f"{c.url}: round marker names issue #{match.group(1)}")
        pr = None if match.group(4) == "none" else int(match.group(4))
        rounds.append(Round(issue, int(match.group(2)), match.group(3), pr))
    numbers = [r.number for r in rounds]
    if numbers != list(range(1, len(rounds) + 1)):
        raise ValueError(f"issue #{issue}: round markers out of sequence: {numbers}")
    return tuple(rounds)


def rejection_noted(
    issue: int, r: Round, comments: Iterable[IssueComment], *, trusted_login: str
) -> bool:
    for c in comments:
        match = REJECTED_MARKER_RE.fullmatch(_first_line(c.body))
        if (
            c.author == trusted_login
            and match is not None
            and (int(match.group(1)), int(match.group(2))) == (issue, r.number)
        ):
            return True
    return False


class NextStep(StrEnum):
    RUN = "run"  # start a new round
    WAIT_REVIEW = "wait_review"  # the latest round's PR is still open
    NOTE_REJECTION = "note_rejection"  # its PR was closed unmerged: hand to a human


def next_step(
    latest: Round | None, latest_pr_state: str | None, *, rejection_noted: bool
) -> NextStep:
    """Decide from the latest round (and its PR's state) whether a new round may start.

    A merged PR or a rejection a human has since acknowledged (by removing
    ``needs-human``) lets a new round start; the issue being open at all means some
    findings are still unresolved.
    """
    if latest is None or latest.pr is None:
        if latest_pr_state is not None:
            raise ValueError("a PR state was given for a round without a PR")
        return NextStep.RUN
    if latest_pr_state == "OPEN":
        return NextStep.WAIT_REVIEW
    if latest_pr_state == "MERGED":
        return NextStep.RUN
    if latest_pr_state == "CLOSED":
        return NextStep.RUN if rejection_noted else NextStep.NOTE_REJECTION
    raise ValueError(f"unknown PR state for round {latest.number}: {latest_pr_state!r}")


# --------------------------------------------------------------------------- agent output


class ResolutionParseError(ValueError):
    """The agent's final message does not follow the required schema."""


@dataclass(frozen=True)
class Disposition:
    index: int
    disposition: str
    reason: str


class CloseReason(StrEnum):
    COMPLETED = "completed"
    NOT_PLANNED = "not planned"


@dataclass(frozen=True)
class Resolution:
    summary: str
    dispositions: tuple[Disposition, ...]
    pr_title: str | None
    pr_body: str | None
    verification: str

    def count(self, disposition: str) -> int:
        return sum(d.disposition == disposition for d in self.dispositions)

    @property
    def opens_pr(self) -> bool:
        return self.count("fixed") > 0

    @property
    def needs_human(self) -> bool:
        return self.count("needs_human") > 0

    @property
    def closes_issue(self) -> bool:
        """Whether the issue is done once this round lands (PR merge or direct close)."""
        return not self.needs_human

    @property
    def close_reason(self) -> CloseReason | None:
        """How to close the issue directly (no PR, nothing left for a human)."""
        if self.opens_pr or self.needs_human:
            return None
        if all(d.disposition == "already_resolved" for d in self.dispositions):
            return CloseReason.COMPLETED
        return CloseReason.NOT_PLANNED


def _require_text(obj: Mapping[str, Any], key: str, *, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResolutionParseError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _optional_text(obj: Mapping[str, Any], key: str) -> str | None:
    value = obj[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ResolutionParseError(f"{key} must be null or a non-empty string")
    return value.strip()


def parse_resolution(text: str, n_findings: int) -> Resolution:
    """Parse the agent's final message: one JSON object, optionally in one json fence.

    Every finding 1..``n_findings`` must get exactly one disposition, and the PR fields
    must be present exactly when something was fixed.
    """
    stripped = text.strip()
    fenced = _FENCE_RE.fullmatch(stripped)
    try:
        raw = json.loads(fenced.group(1) if fenced else stripped)
    except json.JSONDecodeError as exc:
        raise ResolutionParseError(f"final message is not JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ResolutionParseError("final message must be a JSON object")
    expected = {"summary", "findings", "pr_title", "pr_body", "verification"}
    if set(raw) != expected:
        raise ResolutionParseError(
            f"keys must be exactly {sorted(expected)}, got {sorted(raw)}"
        )
    items = raw["findings"]
    if not isinstance(items, list):
        raise ResolutionParseError("findings must be a list")
    dispositions: list[Disposition] = []
    for i, item in enumerate(items):
        where = f"findings[{i}]"
        if not isinstance(item, dict) or set(item) != {
            "index",
            "disposition",
            "reason",
        }:
            raise ResolutionParseError(
                f"{where} must be an object with keys index, disposition, reason"
            )
        index = item["index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise ResolutionParseError(f"{where}.index must be an integer")
        if item["disposition"] not in DISPOSITIONS:
            raise ResolutionParseError(
                f"{where}.disposition must be one of {DISPOSITIONS}, "
                f"got {item['disposition']!r}"
            )
        dispositions.append(
            Disposition(
                index, item["disposition"], _require_text(item, "reason", where=where)
            )
        )
    indices = sorted(d.index for d in dispositions)
    if indices != list(range(1, n_findings + 1)):
        raise ResolutionParseError(
            f"findings must cover indices 1..{n_findings} exactly once, got {indices}"
        )
    resolution = Resolution(
        summary=_require_text(raw, "summary", where="output"),
        dispositions=tuple(sorted(dispositions, key=lambda d: d.index)),
        pr_title=_optional_text(raw, "pr_title"),
        pr_body=_optional_text(raw, "pr_body"),
        verification=_require_text(raw, "verification", where="output"),
    )
    has_pr_fields = resolution.pr_title is not None and resolution.pr_body is not None
    if (resolution.pr_title is None) != (resolution.pr_body is None):
        raise ResolutionParseError("pr_title and pr_body must both be set or both null")
    if resolution.opens_pr != has_pr_fields:
        raise ResolutionParseError(
            "pr_title/pr_body must be set exactly when some finding is fixed"
        )
    return resolution


def check_commits(resolution: Resolution, n_commits: int) -> None:
    """The branch must carry commits exactly when the agent reports a fix."""
    if resolution.opens_pr and n_commits == 0:
        raise ResolutionParseError(
            "findings marked fixed but the branch has no commits"
        )
    if not resolution.opens_pr and n_commits:
        raise ResolutionParseError(
            f"no finding marked fixed but the branch has {n_commits} commit(s)"
        )


# --------------------------------------------------------------------------- rendering

DISPOSITION_LABELS: Mapping[str, str] = {
    "fixed": "対応した",
    "declined": "対応しない",
    "already_resolved": "解決済み",
    "needs_human": "人間の判断が必要",
}


def _cell(text: str) -> str:
    return neutralize_markers(text).replace("|", "\\|").replace("\n", "<br>")


def disposition_table(resolution: Resolution, findings: Sequence[str]) -> str:
    """Markdown table of each finding's disposition; ``findings`` are one-line labels."""
    if len(findings) != len(resolution.dispositions):
        raise ValueError("findings and dispositions differ in length")
    lines = ["| # | 指摘 | 処置 | 理由 |", "|---|---|---|---|"]
    for d, label in zip(resolution.dispositions, findings, strict=True):
        lines.append(
            f"| {d.index} | {_cell(label)} | {DISPOSITION_LABELS[d.disposition]} "
            f"| {_cell(d.reason)} |"
        )
    return "\n".join(lines)
