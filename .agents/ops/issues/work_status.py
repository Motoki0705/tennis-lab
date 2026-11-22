"""Read the exact Markdown records written by parent stop and resume."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from html import escape, unescape

from .format import _scan_markdown

DETAIL_FIELDS = ("reason", "facts", "impact", "resume_condition", "next_action")
BLOCK_START = "<!-- parent-task-status-block:v1 -->"
STATUS_MARKER = re.compile(
    r"<!-- parent-task-status:v1 state=(stopped|resumed) detail=[0-9a-f]{64} -->"
)
ACTIONS = {"stopped": "停止", "resumed": "再開"}
LEGACY_FIELD_BOUNDARY = re.compile(r"\n\n(?=(?:" + "|".join(DETAIL_FIELDS) + r"): )")


@dataclass(frozen=True)
class Record:
    start: int
    end: int
    state: str
    detail: dict[str, str]


def detail_sha256(detail: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            detail, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _quote(value: str) -> str:
    return "\n".join(f"> {escape(line, quote=False)}" for line in value.split("\n"))


def _render(detail: dict[str, str], state: str, *, quoted: bool) -> str:
    parts = [ACTIONS[state]]
    if quoted:
        parts.insert(0, BLOCK_START)
        parts.extend(f"{key}:\n{_quote(detail[key])}" for key in DETAIL_FIELDS)
    else:
        parts.extend(f"{key}: {value}" for key, value in detail.items())
    parts.append(
        f"<!-- parent-task-status:v1 state={state} detail={detail_sha256(detail)} -->"
    )
    return "\n\n".join(parts)


def render(detail: dict[str, str], state: str) -> str:
    return _render(detail, state, quoted=True)


def _verified_detail(block: str, state: str, *, quoted: bool) -> dict[str, str] | None:
    start = (f"{BLOCK_START}\n\n" if quoted else "") + ACTIONS[state] + "\n\n"
    content, _, _ = block.rpartition("\n\n")
    if not content.startswith(start):
        return None
    text = content.removeprefix(start)
    if not quoted and re.search(r"(?m)^ {0,3}#{1,6}(?:[ \t]+|$)", text):
        # Bare Markdown headings in the old unframed format remain material.
        return None
    detail = {}
    fields = text.split("\n\n") if quoted else LEGACY_FIELD_BOUNDARY.split(text)
    for field in fields:
        key, separator, value = field.partition(":\n" if quoted else ": ")
        if not separator or key not in DETAIL_FIELDS or key in detail:
            return None
        if quoted:
            lines = value.split("\n")
            if any(not line.startswith("> ") for line in lines):
                return None
            value = unescape("\n".join(line[2:] for line in lines))
        if not value or value != value.strip():
            return None
        detail[key] = value
    # Check the shape, boundaries, and hash together against the writer.
    if (
        len(detail) != len(DETAIL_FIELDS)
        or _render(detail, state, quoted=quoted) != block
    ):
        return None
    return detail


def records(section: str) -> Iterator[Record]:
    lines = section.splitlines()
    _, markers = _scan_markdown(section, prefix="<!-- parent-task-status")
    boundary = 0
    begin = None
    for end, line in markers:
        if line == BLOCK_START:
            begin = end
            continue
        match = STATUS_MARKER.fullmatch(line)
        if match is None:
            continue
        state = match[1]
        start = begin
        if start is None:
            start = next(
                (
                    i
                    for i in range(end - 1, boundary - 1, -1)
                    if lines[i] == ACTIONS[state]
                ),
                None,
            )
        if start is not None:
            block = "\n".join(lines[start : end + 1])
            detail = _verified_detail(block, state, quoted=begin is not None)
            if detail is not None:
                yield Record(start, end + 1, state, detail)
        begin, boundary = None, end + 1


def without_records(section: str) -> str:
    lines = section.splitlines()
    kept = []
    previous = 0
    for record in records(section):
        kept.extend(lines[previous : record.start])
        previous = record.end
    return "\n".join([*kept, *lines[previous:]])


def last_record(section: str) -> Record | None:
    return next(iter(reversed(list(records(section)))), None)
