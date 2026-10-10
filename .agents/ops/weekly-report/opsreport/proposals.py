"""Proposal model, stable IDs, report rendering and parsing of checked items.

The report issue body is the single source of truth for triage: a proposal is a
list item whose head line carries a bold stable ID (``**R-2026W41-03**``). It is
selected either by its own checkbox, or -- for proposals with options -- by
checking exactly one option checkbox. Triage appends ``→ #<child>`` to the head
line once the child issue exists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

ID_PATTERN = r"R-\d{4}W\d{2}-\d{2,3}"
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
_CHECKBOX_IN_TEXT_RE = re.compile(r"^(\s*[-*+]) \[[ xX]\] ")


class SelectionError(ValueError):
    """A proposal's checkboxes are in an ambiguous state (e.g. two options checked)."""


def week_tag(day: date) -> str:
    """ISO week tag used in IDs and titles, e.g. ``2026W41``."""
    iso = day.isocalendar()
    return f"{iso.year}W{iso.week:02d}"


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


def _plain_lines(text: str) -> list[str]:
    """Free text as list-item continuation lines; strips checkbox syntax so agent
    prose can never be mistaken for a selectable option."""
    lines = []
    for raw in text.strip().splitlines():
        if not raw.strip():
            continue
        lines.append(_CHECKBOX_IN_TEXT_RE.sub(r"\1 ", raw.rstrip()))
    return lines


def _one_line(text: str) -> str:
    return " ".join(text.split())


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
            out.append(f"- {box}**{pid}** {_one_line(p.title)}{prio}")
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


def parse_proposals(body: str) -> list[ParsedProposal]:
    lines = body.splitlines()
    found: list[ParsedProposal] = []
    seen: set[str] = set()
    i = 0
    while i < len(lines):
        m = _HEAD_RE.match(lines[i])
        if not m:
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
            head_index=i,
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
    for prop in parse_proposals(body):
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
