"""Insert validated supplements without giving the writer ownership of source text."""

from __future__ import annotations

from typing import TypedDict

from issues.format import validate_body
from parent.execution import PROGRESS_SECTIONS

from .markdown import insertion_points, validate_supplement_markdown

SUMMARY = "Geminiによる補足・背景情報"


class Supplement(TypedDict):
    after: str
    content: str


def render_supplements(original: str, supplements: list[Supplement]) -> str:
    """Resolve unique source anchors once; insert at complete block boundaries.

    All anchors refer to the original, never to an earlier insertion. Slicing the
    source preserves every character, including CRLF, trailing spaces and EOF.
    """
    points = insertion_points(original)
    insertions: dict[int, str] = {}
    for supplement in supplements:
        anchor, content = supplement["after"], supplement["content"]
        start = original.find(anchor)
        if not anchor or start < 0 or original.find(anchor, start + 1) >= 0:
            raise ValueError("supplement_anchor_must_be_unique_in_original")
        position = start + len(anchor)
        if position not in points:
            raise ValueError("supplement_anchor_must_end_at_a_complete_markdown_block")
        if points[position] in PROGRESS_SECTIONS:
            raise ValueError("supplement_in_machine_managed_section")
        if position in insertions:
            raise ValueError("duplicate_supplement_position")
        validate_supplement_markdown(content)
        insertions[position] = (
            f"\n\n<details>\n<summary>{SUMMARY}</summary>\n\n{content}\n\n</details>\n"
        )
    parts: list[str] = []
    previous = 0
    for position, addition in sorted(insertions.items()):
        parts.extend((original[previous:position], addition))
        previous = position
    parts.append(original[previous:])
    candidate = "".join(parts)
    if len(candidate) > 60000:
        raise ValueError("supplemented_body_too_long")
    return candidate


def check_invariants(
    original: str,
    candidate: str,
    *,
    supplements: list[Supplement],
    is_child: bool = False,
) -> list[dict[str, str]]:
    """Rebuild from trusted source and compare the entire candidate exactly.

    A substring/subsequence test or stripping arbitrary details would allow text
    to move into a collapsed region. Declared insertions must explain every added
    character; existing details, comments and code remain source text.
    """
    problems = [
        {"rule": "invalid_source", "detail": error}
        for error in validate_body(original, is_parent=True, is_child=is_child)
    ]
    try:
        expected = render_supplements(original, supplements)
    except ValueError as exc:
        problems.append({"rule": "invalid_supplement", "detail": str(exc)})
        return problems
    if candidate != expected:
        problems.append(
            {
                "rule": "original_preservation",
                "detail": "body must equal the original with only the declared details insertions",
            }
        )
    problems.extend(
        {"rule": "validate_body", "detail": error}
        for error in validate_body(candidate, is_parent=True, is_child=is_child)
    )
    return problems
