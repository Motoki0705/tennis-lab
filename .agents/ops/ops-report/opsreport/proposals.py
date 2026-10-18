"""Proposal model, stable IDs, report rendering and parsing of checked items.

The report issue body is the single source of truth for triage. Its first line is
a metadata marker (slot, collection boundary, number of proposals), and proposals
live only between the ``PROPOSALS_BEGIN`` / ``PROPOSALS_END`` marker lines. A
proposal is a list item whose head line carries a bold stable ID
(``**R-20261019-03**``). It is selected either by its own checkbox, or -- for
proposals with options -- by checking exactly one option checkbox. Triage appends
``→ #<child>`` to the head line once the child issue exists.

Agent-written text is neutralized (no checkbox syntax, no HTML comments) so it can
neither create selectable items nor forge the markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

ID_PATTERN = r"R-\d{8}-\d{2,3}"
CHILD_TITLE_RE = re.compile(rf"^\[({ID_PATTERN})\]")

# Agent proposal categories, rendered in this order after the stale-PR section.
CATEGORIES: dict[str, str] = {
    "code-debt": "コード負債",
    "doc-drift": "ドキュメント / AI環境のドリフト",
    "architecture": "アーキテクチャの抜本的改善",
    "inefficiency": "非効率",
    "memory": "memoryの陳腐化・削除候補",
    "control-rules": "制御ルール（AGENTS.md・skill・ops・このレポート自身）の改訂",
    "ai-proposed": "ai-proposed issueの整理",
    "cleanup": "掃除候補",
    "research": "研究の次の一手",
    "other": "その他の助言",
}
STALE_PR_SECTION = "放置PR"
STALE_PR_OPTIONS = ("継続", "rebaseして仕上げ", "close")
PRIORITIES: dict[str, str] = {"high": "高", "medium": "中", "low": "低"}

_HEAD_RE = re.compile(
    rf"^- (?:\[(?P<check>[ xX])\] )?\*\*(?P<id>{ID_PATTERN})\*\*(?P<rest>.*)$"
)
_OPTION_RE = re.compile(r"^  - \[(?P<check>[ xX])\] (?P<label>.+?)\s*$")
_CHILD_RE = re.compile(r"\s→ #(?P<num>\d+)\s*$")
# Text that would read as a child annotation (``→ #123``); escaped as ``→ \#123``,
# which GitHub renders identically but _CHILD_RE does not match.
_ANNOTATION_LIKE_RE = re.compile(r"→(\s*)#(?=\d)")
_CHECKBOX_IN_TEXT_RE = re.compile(r"^(\s*(?:[-*+]|\d+[.)])) \[[ xX]\] ")

REPORT_MARKER_PREFIX = "<!-- ops-report:v2 "
_REPORT_MARKER_RE = re.compile(r"^<!-- ops-report:v2 (?P<fields>[^<>]*?) -->$")
_REQUIRED_META = ("slot", "since", "until", "proposals", "status")
PROPOSALS_BEGIN = "<!-- ops-report:proposals:begin -->"
PROPOSALS_END = "<!-- ops-report:proposals:end -->"


class SelectionError(ValueError):
    """A proposal's checkboxes are in an ambiguous state (e.g. two options checked)."""


def slot_tag(day: date) -> str:
    """Scheduled date used in IDs, e.g. ``20261019``."""
    return day.strftime("%Y%m%d")


def proposal_id(tag: str, index: int) -> str:
    return f"R-{tag}-{index:02d}"


@dataclass(frozen=True)
class Proposal:
    """A proposal before rendering (no ID yet)."""

    section: str
    title: str
    detail: str = ""
    evidence: str = ""
    priority: str | None = None
    options: tuple[str, ...] = ()


def neutralize(text: str) -> str:
    """Make agent-written text inert: drop checkbox syntax at line starts and escape
    HTML comment openers, so it can neither be selected nor forge a marker line."""
    escaped = text.replace("<!--", "&lt;!--")
    return "\n".join(
        _CHECKBOX_IN_TEXT_RE.sub(r"\1 ", line) for line in escaped.splitlines()
    )


def _plain_lines(text: str) -> list[str]:
    """Free text as list-item continuation lines (neutralized, blank lines dropped)."""
    return [
        line.rstrip() for line in neutralize(text.strip()).splitlines() if line.strip()
    ]


def _one_line(text: str) -> str:
    return " ".join(neutralize(text).split())


def _head_text(text: str) -> str:
    """One-line text for a proposal head line, where only triage may append
    ``→ #N``; title-derived text (e.g. a PR titled "移管 → #123") is escaped."""
    return _ANNOTATION_LIKE_RE.sub(r"→\1\\#", _one_line(text))


def unescape_head_text(text: str) -> str:
    """Inverse of the annotation escape, for plain-text uses (issue titles)."""
    return re.sub(r"→(\s*)\\#(?=\d)", r"→\1#", text)


def render_proposals(proposals: list[Proposal], tag: str) -> tuple[str, list[str]]:
    """Render proposals grouped by section, assigning IDs in rendered order.

    Returns the markdown and the IDs in order.
    """
    sections: list[str] = [STALE_PR_SECTION, *CATEGORIES.values()]
    unknown = {p.section for p in proposals} - set(sections)
    if unknown:
        raise ValueError(f"unknown proposal sections: {sorted(unknown)}")
    out: list[str] = []
    ids: list[str] = []
    index = 0
    for section in sections:
        items = [p for p in proposals if p.section == section]
        if not items:
            continue
        out.append(f"### {section}")
        out.append("")
        for p in items:
            index += 1
            pid = proposal_id(tag, index)
            ids.append(pid)
            prio = f"（優先度: {PRIORITIES[p.priority]}）" if p.priority else ""
            box = "" if p.options else "[ ] "
            out.append(f"- {box}**{pid}** {_head_text(p.title)}{prio}")
            out.extend(f"  {line}" for line in _plain_lines(p.detail))
            if p.evidence.strip():
                out.append(f"  - 根拠: {_one_line(p.evidence)}")
            out.extend(f"  - [ ] {_one_line(opt)}" for opt in p.options)
        out.append("")
    return "\n".join(out).rstrip() + "\n", ids


@dataclass
class ParsedProposal:
    pid: str
    head_index: int
    head_checked: bool
    title: str
    child: int | None
    options: list[tuple[str, bool]] = field(default_factory=list)
    block: list[str] = field(default_factory=list)

    def selection(self) -> tuple[bool, str | None]:
        """Return ``(selected, chosen_option)``; raise on ambiguous checkboxes."""
        if not self.options:
            return self.head_checked, None
        chosen = [label for label, checked in self.options if checked]
        if len(chosen) > 1:
            raise SelectionError(
                f"{self.pid}: {len(chosen)} options checked ({', '.join(chosen)}); check exactly one"
            )
        if self.head_checked and not chosen:
            raise SelectionError(f"{self.pid}: head checked but no option chosen")
        return (True, chosen[0]) if chosen else (False, None)


@dataclass(frozen=True)
class ReportMeta:
    """Metadata from a report body's first line."""

    slot: str
    since: datetime
    until: datetime
    proposal_count: int
    status: str
    base: str | None
    audit: str | None

    def proposal_ids(self) -> set[str]:
        return {proposal_id(self.slot, i) for i in range(1, self.proposal_count + 1)}


def report_marker(
    *,
    slot: str,
    since: datetime,
    until: datetime,
    proposals: int,
    status: str = "complete",
    **extra: str,
) -> str:
    fields = {
        "slot": slot,
        "since": since.isoformat(timespec="seconds"),
        "until": until.isoformat(timespec="seconds"),
        "proposals": str(proposals),
        "status": status,
        **extra,
    }
    for key, value in fields.items():
        if not value or any(c.isspace() or c in "<>" for c in value):
            raise ValueError(f"report marker field {key}={value!r} is not a token")
    return (
        REPORT_MARKER_PREFIX + " ".join(f"{k}={v}" for k, v in fields.items()) + " -->"
    )


def parse_report_meta(body: str) -> ReportMeta:
    first = body.split("\n", 1)[0].rstrip("\r")
    if first.startswith("<!-- ops-report:v1 "):
        raise ValueError(
            "unsupported ops-report:v1 week format; resolve old reports explicitly before using v2 slots"
        )
    m = _REPORT_MARKER_RE.match(first)
    if not m:
        raise ValueError("report body does not start with an ops-report:v2 marker")
    fields: dict[str, str] = {}
    for part in m.group("fields").split():
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"malformed report marker field {part!r}")
        if key in fields:
            raise ValueError(f"duplicate report marker field {key!r}")
        fields[key] = value
    missing = [k for k in _REQUIRED_META if k not in fields]
    if missing:
        raise ValueError(f"report marker lacks {missing}")
    if "week" in fields or not re.fullmatch(r"[0-9]{8}", fields["slot"]):
        raise ValueError(f"report marker has invalid slot {fields['slot']!r}")
    date.fromisoformat(fields["slot"])
    if not fields["proposals"].isdigit():
        raise ValueError(f"report marker has invalid proposals={fields['proposals']!r}")
    if fields["status"] not in {"complete", "incomplete"}:
        raise ValueError("report marker has invalid status")
    since = datetime.fromisoformat(fields["since"])
    until = datetime.fromisoformat(fields["until"])
    if since.tzinfo is None or until.tzinfo is None or since > until:
        raise ValueError(f"report marker has invalid period {since}..{until}")
    return ReportMeta(
        slot=fields["slot"],
        since=since,
        until=until,
        proposal_count=int(fields["proposals"]),
        status=fields["status"],
        base=fields.get("base"),
        audit=fields.get("audit"),
    )


def parse_report(body: str) -> tuple[ReportMeta, list[ParsedProposal]]:
    """Parse a report body: metadata plus the proposals inside the marked region.

    Only the region between ``PROPOSALS_BEGIN`` and ``PROPOSALS_END`` is scanned,
    and every ID found there must be one the report generated for its slot.
    """
    meta = parse_report_meta(body)
    lines = body.splitlines()
    begins = [i for i, line in enumerate(lines) if line.rstrip() == PROPOSALS_BEGIN]
    ends = [i for i, line in enumerate(lines) if line.rstrip() == PROPOSALS_END]
    if len(begins) != 1 or len(ends) != 1 or begins[0] > ends[0]:
        raise ValueError(
            f"report body must contain exactly one proposal region "
            f"(found {len(begins)} begin / {len(ends)} end markers)"
        )
    start = begins[0] + 1
    found = _parse_region(lines[start : ends[0]], offset=start)
    allowed = meta.proposal_ids()
    unknown = [p.pid for p in found if p.pid not in allowed]
    if unknown:
        raise ValueError(
            f"proposal IDs {unknown} were not generated by the {meta.slot} report "
            f"({meta.proposal_count} proposals)"
        )
    return meta, found


def _parse_region(lines: list[str], *, offset: int) -> list[ParsedProposal]:
    found: list[ParsedProposal] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        m = _HEAD_RE.match(lines[i])
        if not m:
            if re.match(r"^- (?:\[[ xX]\] )?\*\*R-", lines[i]):
                raise ValueError(f"invalid proposal ID format: {lines[i]!r}")
            i += 1
            continue
        pid = m.group("id")
        if pid in seen:
            raise ValueError(f"duplicate proposal ID in report body: {pid}")
        seen.add(pid)
        rest = m.group("rest")
        child_m = _CHILD_RE.search(rest)
        title = _CHILD_RE.sub("", rest).strip()
        prop = ParsedProposal(
            pid=pid,
            head_index=offset + i,
            head_checked=(m.group("check") or " ").lower() == "x",
            title=title,
            child=int(child_m.group("num")) if child_m else None,
            block=[lines[i]],
        )
        j = i + 1
        while j < len(lines) and lines[j].startswith("  "):
            prop.block.append(lines[j])
            opt = _OPTION_RE.match(lines[j])
            if opt:
                prop.options.append(
                    (opt.group("label"), opt.group("check").lower() == "x")
                )
            j += 1
        found.append(prop)
        i = j
    return found


def annotate_children(body: str, children: dict[str, int]) -> str:
    """Append ``→ #N`` to each proposal head line in ``children`` lacking one."""
    lines = body.splitlines()
    _, parsed = parse_report(body)
    for prop in parsed:
        num = children.get(prop.pid)
        if num is None:
            continue
        if prop.child is not None:
            if prop.child != num:
                raise ValueError(
                    f"{prop.pid} already links #{prop.child}, refusing to relink to #{num}"
                )
            continue
        lines[prop.head_index] = f"{lines[prop.head_index].rstrip()} → #{num}"
    trailing = "\n" if body.endswith("\n") else ""
    return "\n".join(lines) + trailing
