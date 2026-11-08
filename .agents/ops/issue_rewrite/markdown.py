"""CommonMark boundaries and self-contained Markdown for details insertions."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token


class _HTMLScope(HTMLParser):
    """Keep insertions outside existing HTML containers, including details."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.invalid = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        pass

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack.pop() != tag:
            self.invalid = True

    def consume(self, token: Token) -> None:
        if token.type in {"html_block", "html_inline"}:
            self.feed(token.content)
        for child in token.children or []:
            self.consume(child)


def _source_lines(source: str) -> list[str]:
    # str.splitlines also splits Unicode separators and would corrupt offsets.
    return re.findall(r"[^\r\n]*(?:\r\n?|\n)|[^\r\n]+$", source)


def insertion_points(source: str) -> dict[int, str | None]:
    """End offsets and issue sections of complete prose blocks outside raw HTML."""
    lines = _source_lines(source)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    scope = _HTMLScope()
    points: dict[int, str | None] = {}
    end: int | None = None
    section = None
    tokens = MarkdownIt("commonmark").parse(source)
    for index, token in enumerate(tokens):
        if token.level == 0 and token.type == "heading_open" and token.tag == "h2":
            section = tokens[index + 1].content
        if token.level == 0 and token.map:
            end = (
                token.map[1]
                if token.type
                in {
                    "paragraph_open",
                    "heading_open",
                    "bullet_list_open",
                    "ordered_list_open",
                    "blockquote_open",
                }
                else None
            )
        scope.consume(token)
        if token.level == 0 and token.nesting == -1 and end is not None:
            if not scope.stack and not scope.invalid:
                points[offsets[end - 1] + len(lines[end - 1].rstrip("\r\n"))] = section
            end = None
    return points


def validate_supplement_markdown(content: str) -> None:
    if not content.strip():
        raise ValueError("empty_supplement")
    # The wrapper belongs to Python. Reject delimiters even in code examples.
    if re.search(r"</?\s*(?:details|summary)\b", content, re.IGNORECASE):
        raise ValueError("supplement_contains_details_delimiter")
    env: dict[str, Any] = {}
    tokens = MarkdownIt("commonmark").parse(content, env)
    if env.get("references"):
        raise ValueError("supplement_reference_definition_can_change_original_links")

    def check(token: Token) -> None:
        if token.type in {"html_block", "html_inline"}:
            raise ValueError("supplement_raw_html")
        if token.type == "heading_open" and token.tag in {"h1", "h2"}:
            raise ValueError("supplement_cannot_add_issue_sections")
        if token.type == "inline" and re.match(r"\[[ xX]\](?:\s|$)", token.content):
            raise ValueError("supplement_cannot_add_task_checkboxes")
        if token.type == "fence":
            assert token.map is not None
            lines = _source_lines(content)
            closing_line = lines[token.map[1] - 1]
            last = re.sub(r"^(?:[ \t]*>[ \t]?)+", "", closing_line).strip()
            if token.map[1] - token.map[0] < 2 or not re.fullmatch(
                re.escape(token.markup[0]) + "{" + str(len(token.markup)) + ",}", last
            ):
                raise ValueError("supplement_unclosed_code_fence")
        for child in token.children or []:
            check(child)

    for token in tokens:
        check(token)
