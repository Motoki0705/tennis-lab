"""Deterministic content preservation checks, independent of model judgement."""

from __future__ import annotations

import json
import re
from collections import Counter

from issues.format import CHILD_REQUIRED, PARENT_REQUIRED, parse_sections, validate_body
from parent.execution import PROGRESS_SECTIONS
from parent.policy import section_span

from .markdown import markdown_content

REFERENCE = re.compile(r"#[0-9]+")
MENTION = re.compile(
    r"(?<![\w./])@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:/[\w-]+)?"
)
NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?")
LIST_NUMBER = re.compile(r"(?m)^([ \t]*)[0-9]+[.)](?=[ \t])")
CHECKBOX = re.compile(r"(?m)^\s*(?:[-*+]|[0-9]+[.)])\s+\[([ xX])\](?=\s|$)")


def protected(body: str) -> dict[str, list[str]]:
    """Keep raw bytes (including whitespace) rather than rendered Markdown."""
    comments = list(re.finditer(r"<!--[\s\S]*?-->", body))
    fences: list[str] = []
    fence: str | None = None
    start = 0
    offset = 0
    masked = list(body)
    for match in comments:
        masked[match.start() : match.end()] = " " * len(match[0])
    for line in body.splitlines(keepends=True):
        if fence is not None:
            if re.fullmatch(
                rf" {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}\s*", line
            ):
                end = offset + len(line)
                fences.append(body[start:end])
                masked[start:end] = " " * (end - start)
                fence = None
        elif not any(m.start() <= offset < m.end() for m in comments):
            opening = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if opening:
                start, fence = offset, opening[1]
        offset += len(line)
    inline = [
        match[0]
        for match in re.finditer(
            r"(?<!`)(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)", "".join(masked)
        )
    ]
    return {
        "quotes": markdown_content(body)[0],
        "html_comments": [match[0] for match in comments],
        "fenced_code": fences,
        "inline_code": inline,
    }


def _delta(before: Counter[str], after: Counter[str]) -> str:
    return json.dumps(
        {"removed": dict(before - after), "added": dict(after - before)},
        ensure_ascii=False,
        sort_keys=True,
    )


def check_invariants(
    original: str, candidate: str, *, is_child: bool = False
) -> list[dict[str, str]]:
    problems: list[dict[str, str]] = []

    def add(rule: str, detail: str) -> None:
        problems.append({"rule": rule, "detail": detail})

    for error in validate_body(candidate, is_parent=True, is_child=is_child):
        add("validate_body", error)
    try:
        before = parse_sections(original)
        after = parse_sections(candidate)
    except ValueError as exc:
        add("markdown", str(exc))
        return problems
    required = {*PARENT_REQUIRED, *(CHILD_REQUIRED if is_child else ())}
    order_before = [name for name in before if name in required]
    order_after = [name for name in after if name in required]
    if order_before != order_after:
        add(
            "required_sections",
            f"required section order/set changed: {order_before!r} -> {order_after!r}",
        )
    try:
        for name in sorted(PROGRESS_SECTIONS):
            left, right = section_span(original, name), section_span(candidate, name)
            section_before = original[left[0] : left[1]] if left else None
            section_after = candidate[right[0] : right[1]] if right else None
            if section_before != section_after:
                add("managed_section", f"machine-managed section changed: {name}")
    except ValueError as exc:
        add("markdown", str(exc))
    raw_before, raw_after = protected(original), protected(candidate)
    for kind in raw_before:
        if raw_before[kind] != raw_after[kind]:
            left_fragments, right_fragments = raw_before[kind], raw_after[kind]
            index = next(
                (
                    i
                    for i, (left, right) in enumerate(
                        zip(left_fragments, right_fragments, strict=False)
                    )
                    if left != right
                ),
                min(len(left_fragments), len(right_fragments)),
            )
            left = left_fragments[index] if index < len(left_fragments) else "[missing]"
            right = (
                right_fragments[index] if index < len(right_fragments) else "[missing]"
            )
            add(
                kind,
                f"{kind}: fragment {index + 1} changed (including order/whitespace): {left!r} -> {right!r}",
            )
    for kind, pattern in (
        ("references", REFERENCE),
        ("mentions", MENTION),
        ("numbers", NUMBER),
    ):
        left_text, right_text = original, candidate
        if kind == "numbers":
            left_text = LIST_NUMBER.sub(r"\1", left_text)
            right_text = LIST_NUMBER.sub(r"\1", right_text)
        left_values = Counter(m[0] for m in pattern.finditer(left_text))
        right_values = Counter(m[0] for m in pattern.finditer(right_text))
        if left_values != right_values:
            add(kind, f"{kind} multiset changed: {_delta(left_values, right_values)}")
    before_urls, after_urls = (
        markdown_content(original)[1],
        markdown_content(candidate)[1],
    )
    if before_urls != after_urls:
        add("urls", f"urls multiset changed: {_delta(before_urls, after_urls)}")
    for name in dict.fromkeys([*before, *after]):
        left_boxes = Counter(
            "checked" if x.lower() == "x" else "unchecked"
            for x in CHECKBOX.findall(before.get(name, ""))
        )
        right_boxes = Counter(
            "checked" if x.lower() == "x" else "unchecked"
            for x in CHECKBOX.findall(after.get(name, ""))
        )
        if left_boxes != right_boxes:
            add(
                "checkboxes",
                f"checkbox states/count changed in {name}: {_delta(left_boxes, right_boxes)}",
            )
    return problems
