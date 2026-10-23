"""Pure, deterministic issue titles, bodies and proposal identities.

The public contract and lifecycle rules live in README.md beside this module.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import quote, unquote

PARENT_LABEL = "issue:parent"
CHILD_LABEL = "issue:child"
TASKS_HEADING = "子タスクのチェックリスト"
NO_CHILDREN = "未分割（子issue未作成）"
PARENT_REQUIRED = (
    "目的・背景",
    "全体のスコープ",
    "全体方針・合意事項",
    "全体の完了条件",
    TASKS_HEADING,
    "進捗概要",
    "依存関係・進め方",
)
CHILD_REQUIRED = (
    "このタスクの目的",
    "具体的な変更内容・対象範囲",
    "対象外",
    "依存関係",
    "作業状況・停止事項",
    "Acceptance checklist",
)
ACCEPTANCE_FIELDS = ("文脈", "下流への影響", "検証方法・証拠")
_SOURCE_PREFIX = "<!-- ops-report-proposal:"


def _line(value: str, field: str) -> str:
    if (
        not value
        or value != value.strip()
        or len(value.splitlines()) != 1
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(f"{field}: expected a nonempty, trimmed single line")
    return value


def _number(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("issue number must be a positive integer")
    return value


def _identifier(value: str) -> str:
    _line(value, "identifier")
    if any(c in value for c in "[]#") or value.isdecimal():
        raise ValueError(
            "identifier must be meaningful text without [], # or only digits"
        )
    return value


@dataclass(frozen=True)
class ParentRef:
    identifier: str
    number: int

    def __post_init__(self) -> None:
        _identifier(self.identifier)
        _number(self.number)


@dataclass(frozen=True)
class ParsedTitle:
    summary: str
    parent: ParentRef | None = None
    identifier: str | None = None


def _summary(value: str) -> str:
    _line(value, "summary")
    if value.startswith("["):
        raise ValueError("summary must start with task text, not another prefix")
    return value


def parent_title(
    identifier: str, summary: str, *, parent: ParentRef | None = None
) -> str:
    """A nested parent has both its ancestor prefix and its own identifier."""
    title = f"[{_identifier(identifier)}] {_summary(summary)}"
    return f"[{parent.identifier} #{parent.number}] {title}" if parent else title


def child_title(parent: ParentRef, summary: str) -> str:
    return f"[{parent.identifier} #{parent.number}] {_summary(summary)}"


def parse_title(title: str) -> ParsedTitle:
    _line(title, "title")
    child = re.fullmatch(r"\[([^\[\]#]+) #([1-9][0-9]*)\] (.+)", title)
    if child:
        parent = ParentRef(child[1], int(child[2]))
        rest = child[3]
        own = re.fullmatch(r"\[([^\[\]#]+)\] (.+)", rest)
        if own:
            return ParsedTitle(_summary(own[2]), parent, _identifier(own[1]))
        return ParsedTitle(_summary(rest), parent)
    root = re.fullmatch(r"\[([^\[\]#]+)\] (.+)", title)
    if root:
        return ParsedTitle(_summary(root[2]), identifier=_identifier(root[1]))
    raise ValueError("title must be '[識別名] 要約' or '[親識別名 #番号] 要約'")


def _scan_markdown(
    body: str, *, prefix: str = _SOURCE_PREFIX
) -> tuple[list[str], list[tuple[int, str]]]:
    """Separate visible lines and reserved metadata, respecting comments/fences."""
    lines: list[str] = []
    metadata: list[tuple[int, str]] = []
    fence: str | None = None
    comment = False
    for index, line in enumerate(body.splitlines()):
        if fence is not None:
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*", line
            ):
                fence = None
            continue
        opening = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if not comment and opening:
            fence = opening[1]
            continue
        if not comment and line.lstrip().startswith(prefix):
            metadata.append((index, line))
        remaining = line
        visible = ""
        while remaining:
            if comment:
                _, separator, remaining = remaining.partition("-->")
                if not separator:
                    break
                comment = False
            else:
                before, separator, remaining = remaining.partition("<!--")
                visible += before
                if separator:
                    comment = True
        lines.append(visible)
    if fence is not None:
        raise ValueError("unclosed code fence")
    if comment:
        raise ValueError("unclosed HTML comment")
    return lines, metadata


def visible_lines(body: str) -> list[str]:
    """Ignore HTML comments and fenced examples, without accepting unclosed blocks."""
    return _scan_markdown(body)[0]


def parse_sections(body: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in visible_lines(body):
        heading = re.fullmatch(r"## (\S.*?)\s*", line)
        if heading:
            name = heading[1]
            if name in sections:
                raise ValueError(f"duplicate section: {name}")
            current = sections[name] = []
        elif current is not None:
            current.append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parent_reference(body: str) -> int | None:
    lines = visible_lines(body)
    references = [
        (i, line) for i, line in enumerate(lines) if line.startswith("親issue:")
    ]
    if not references:
        return None
    if len(references) != 1:
        raise ValueError("body must have exactly one parent reference")
    index, line = references[0]
    match = re.fullmatch(r"親issue: #([1-9][0-9]*)", line)
    if not match or any(previous.strip() for previous in lines[:index]):
        raise ValueError(
            "parent reference must be '親issue: #番号' before the sections"
        )
    return int(match[1])


TaskStatus = Literal["open", "completed", "cancelled"]


@dataclass(frozen=True)
class ChildTask:
    number: int
    title: str
    status: TaskStatus = "open"

    def __post_init__(self) -> None:
        _number(self.number)
        _line(self.title, "child task title")
        if self.status not in ("open", "completed", "cancelled"):
            raise ValueError(f"unknown task status: {self.status}")
        if self.title.endswith("（取消）"):
            raise ValueError(
                "child task title must not include the cancellation suffix"
            )


def render_child_tasks(tasks: Sequence[ChildTask]) -> str:
    if len({task.number for task in tasks}) != len(tasks):
        raise ValueError("duplicate child in checklist")
    if not tasks:
        return NO_CHILDREN
    return "\n".join(
        f"- [{'x' if task.status == 'completed' else ' '}] #{task.number} {task.title}"
        + ("（取消）" if task.status == "cancelled" else "")
        for task in tasks
    )


def parse_child_tasks(text: str) -> tuple[ChildTask, ...]:
    if text.strip() == NO_CHILDREN:
        return ()
    tasks: list[ChildTask] = []
    for line in visible_lines(text):
        if not line.strip():
            continue
        match = re.fullmatch(r"- \[([ xX])\] #([1-9][0-9]*) (\S.*)", line)
        if not match:
            raise ValueError(f"invalid child checklist row: {line}")
        checked, number, title = match.groups()
        cancelled = title.endswith("（取消）")
        if cancelled and checked != " ":
            raise ValueError("cancelled child must remain unchecked")
        status: TaskStatus = (
            "cancelled" if cancelled else "completed" if checked != " " else "open"
        )
        tasks.append(ChildTask(int(number), title.removesuffix("（取消）"), status))
    if not tasks:
        raise ValueError(
            f"empty child checklist; use '{NO_CHILDREN}' before subdivision"
        )
    if len({task.number for task in tasks}) != len(tasks):
        raise ValueError("duplicate child in checklist")
    return tuple(tasks)


def _has_content(text: str) -> bool:
    content = text.strip().strip("- ").strip()
    return (
        bool(content)
        and content not in {"TODO", "TBD", "未記入"}
        and not re.fullmatch(r"<[^>]+>", content)
    )


def validate_acceptance(text: str) -> None:
    """Each criterion has contiguous, indented context, impact and evidence."""
    lines = visible_lines(text)
    count = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip():
            continue
        match = re.fullmatch(r"- \[[ xX]\] (.+)", line)
        if not match or not _has_content(match[1]):
            raise ValueError(f"invalid Acceptance checklist row: {line}")
        count += 1
        fields: dict[str, str] = {}
        while index < len(lines) and lines[index].startswith("  "):
            detail = re.fullmatch(r"  ([^:]+): (.+)", lines[index])
            if not detail or detail[1] not in ACCEPTANCE_FIELDS or detail[1] in fields:
                raise ValueError(f"invalid acceptance detail: {lines[index]}")
            fields[detail[1]] = detail[2]
            index += 1
        if any(not _has_content(fields.get(key, "")) for key in ACCEPTANCE_FIELDS):
            raise ValueError(
                f"criterion requires contiguous indented details: {ACCEPTANCE_FIELDS}"
            )
    if not count:
        raise ValueError("Acceptance checklist must contain at least one criterion")


@dataclass(frozen=True)
class ProposalSource:
    """An opaque proposal ID: no assumptions about report cadence or ID spelling."""

    proposal_id: str
    report_number: int
    option: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", self.proposal_id):
            raise ValueError("proposal ID must be a nonempty ASCII token")
        _number(self.report_number)
        if self.option is not None:
            _line(self.option, "selected option")


def proposal_marker(source: ProposalSource) -> str:
    encoded = quote(source.option or "", safe="")
    return f"<!-- ops-report-proposal: {source.proposal_id} report=#{source.report_number} option={encoded} -->"


def parse_proposal_source(body: str) -> ProposalSource | None:
    # A metadata marker is authoritative only as the first line, never in examples.
    if _SOURCE_PREFIX not in body:
        return None
    _, markers = _scan_markdown(body)
    if not markers:
        return None
    index, marker = markers[0]
    match = re.fullmatch(
        r"<!-- ops-report-proposal: ([^ ]+) report=#([1-9][0-9]*) option=([^ ]*) -->",
        marker,
    )
    if len(markers) != 1 or index != 0 or not match:
        raise ValueError("proposal metadata must occur exactly once, on the first line")
    source = ProposalSource(
        match[1], int(match[2]), unquote(match[3], errors="strict") or None
    )
    if proposal_marker(source) != marker:
        raise ValueError("proposal metadata is not canonically percent-encoded")
    return source


def index_proposals(bodies: Iterable[tuple[int, str]]) -> dict[tuple[int, str], int]:
    """Index open AND closed issues by (report, proposal), never by display title.

    Callers must also compare the recorded ``option`` before reusing an issue.
    """
    result: dict[tuple[int, str], int] = {}
    for number, body in bodies:
        source = parse_proposal_source(body)
        if source is None:
            continue
        key = (source.report_number, source.proposal_id)
        if key in result:
            raise ValueError(f"duplicate proposal {key}: #{result[key]} and #{number}")
        result[key] = _number(number)
    return result


def validate_body(body: str, *, is_parent: bool, is_child: bool) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        parse_proposal_source(body)
        reference = parent_reference(body)
        if (reference is not None) != is_child:
            errors.append("parent reference must be present only for a child")
        sections = parse_sections(body)
    except ValueError as exc:
        return (*errors, str(exc))
    required = (
        *(PARENT_REQUIRED if is_parent else ()),
        *(CHILD_REQUIRED if is_child else ()),
    )
    for name in required:
        if name not in sections or not _has_content(sections[name]):
            errors.append(f"missing or empty section: {name}")
    if is_parent and TASKS_HEADING in sections:
        try:
            parse_child_tasks(sections[TASKS_HEADING])
        except ValueError as exc:
            errors.append(str(exc))
    if is_child and "Acceptance checklist" in sections:
        try:
            validate_acceptance(sections["Acceptance checklist"])
        except ValueError as exc:
            errors.append(str(exc))
    return tuple(errors)


def _render_body(
    sections: Mapping[str, str],
    *,
    is_parent: bool,
    parent_number: int | None,
    source: ProposalSource | None,
) -> str:
    for name, content in sections.items():
        _line(name, "section heading")
        if parse_sections(content):
            raise ValueError(
                "section values must not embed other ## sections; use separate keys"
            )
    prefix = [proposal_marker(source)] if source else []
    if parent_number is not None:
        prefix.append(f"親issue: #{_number(parent_number)}")
    order = dict.fromkeys(
        (
            *(PARENT_REQUIRED if is_parent else ()),
            *(CHILD_REQUIRED if parent_number else ()),
            *sections,
        )
    )
    parts = [
        *prefix,
        *(
            f"## {name}\n\n{sections[name].strip()}"
            for name in order
            if name in sections
        ),
    ]
    body = "\n\n".join(parts) + "\n"
    errors = validate_body(
        body, is_parent=is_parent, is_child=parent_number is not None
    )
    if errors:
        raise ValueError("; ".join(errors))
    return body


def render_parent_body(
    sections: Mapping[str, str],
    children: Sequence[ChildTask] = (),
    *,
    parent_number: int | None = None,
    source: ProposalSource | None = None,
) -> str:
    if TASKS_HEADING in sections:
        raise ValueError("pass checklist entries via children, not sections")
    return _render_body(
        {**sections, TASKS_HEADING: render_child_tasks(children)},
        is_parent=True,
        parent_number=parent_number,
        source=source,
    )


def render_child_body(
    parent_number: int,
    sections: Mapping[str, str],
    *,
    source: ProposalSource | None = None,
) -> str:
    return _render_body(
        sections, is_parent=False, parent_number=_number(parent_number), source=source
    )
