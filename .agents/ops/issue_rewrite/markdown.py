"""CommonMark source ranges and link destinations used by preservation checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.rules_inline.backticks import backtick
from markdown_it.token import Token

BARE_URL = re.compile(r'https?://[^\s<>\[\]`"。、，；！？「」『』（）]+')


@dataclass
class MarkdownContent:
    protected: dict[str, list[str]] = field(
        default_factory=lambda: {
            "quotes": [],
            "html_comments": [],
            "fenced_code": [],
            "indented_code": [],
            "inline_code": [],
        }
    )
    urls: Counter[str] = field(default_factory=Counter)


def _raw_backtick(state: StateInline, silent: bool) -> bool:
    """Attach source bytes before CommonMark normalizes code span whitespace."""
    start, count = state.pos, len(state.tokens)
    matched = backtick(state, silent)
    if (
        not silent
        and len(state.tokens) > count
        and state.tokens[-1].type == "code_inline"
    ):
        state.tokens[-1].meta["raw_code"] = state.src[start : state.pos]
    return bool(matched)


def _inline_code(tokens: Sequence[Token]) -> list[str]:
    spans: list[str] = []
    for token in tokens:
        if token.type == "code_inline":
            spans.append(token.meta["raw_code"])
        spans.extend(_inline_code(token.children or []))
    return spans


class _HTMLURLs(HTMLParser):
    def __init__(self, urls: Counter[str]) -> None:
        super().__init__()
        self.urls = urls

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.urls.update(
            value
            for _, value in attrs
            if value and value.startswith(("http://", "https://"))
        )

    def handle_data(self, data: str) -> None:
        self.urls.update(_bare_url(match[0]) for match in BARE_URL.finditer(data))


def _bare_url(value: str) -> str:
    value = value.rstrip(".,;!?:")
    while value.endswith(")") and value.count(")") > value.count("("):
        value = value[:-1].rstrip(".,;!?:")
    return value


def _inline_urls(tokens: Sequence[Token], urls: Counter[str]) -> None:
    autolinks: list[bool] = []
    for token in tokens:
        if token.type in {"link_open", "image"}:
            href = token.attrGet("src" if token.type == "image" else "href")
            if (
                href
                and href.startswith(("http://", "https://"))
                and "label" not in token.meta
            ):
                urls[href] += 1
            if token.type == "image":
                _inline_urls(token.children or [], urls)
            else:
                autolinks.append(token.markup == "autolink")
        elif token.type == "link_close":
            autolinks.pop()
        elif token.type == "text" and not any(autolinks):
            # Explicit labels/alt text are source occurrences. Only an
            # autolink's generated display text duplicates its destination.
            urls.update(
                _bare_url(match[0]) for match in BARE_URL.finditer(token.content)
            )
        elif token.type == "html_inline":
            _HTMLURLs(urls).feed(token.content)


def _source_lines(source: str) -> list[str]:
    """Retain terminators; only CRLF, CR and LF advance CommonMark line maps."""
    lines: list[str] = []
    start = 0
    for ending in re.finditer(r"\r\n?|\n", source):
        lines.append(source[start : ending.end()])
        start = ending.end()
    if start < len(source):
        lines.append(source[start:])
    return lines


def markdown_content(body: str) -> MarkdownContent:
    """Preserve raw quote blocks, and count URL occurrences rather than references.

    Block maps preserve raw quote/code lines, including containers and EOF.
    Inline rule metadata preserves source before code whitespace normalization.
    Link labels prevent reference uses from duplicating definition URLs.
    """
    env: dict[str, Any] = {}
    parser = MarkdownIt("commonmark", {"store_labels": True}).enable(
        ["table", "strikethrough"]
    )
    lines = _source_lines(body)
    expected_lines = [
        line.replace("\r\n", "\n").replace("\r", "\n").replace("\0", "\ufffd")
        for line in lines
    ]

    def check_source_lines(state: StateCore) -> None:
        # Check the parser's actual normalized source before it computes maps.
        # Preserve original bytes for slicing; unknown normalization must fail.
        if _source_lines(state.src) != expected_lines:
            raise ValueError("markdown_source_line_mapping_changed")

    parser.core.ruler.after("normalize", "check_source_lines", check_source_lines)
    parser.inline.ruler.at("backticks", _raw_backtick)
    tokens = parser.parse(body, env)
    content = MarkdownContent()
    content.protected["html_comments"] = re.findall(r"<!--[\s\S]*?-->", body)
    quote_end = 0
    urls = content.urls
    for token in tokens:
        if token.type == "html_block":
            _HTMLURLs(urls).feed(token.content)
        if token.type == "blockquote_open" and token.map:
            start, end = token.map
            if end > quote_end:
                content.protected["quotes"].append("".join(lines[start:end]))
                quote_end = end
        if token.type in {"fence", "code_block"} and token.map:
            start, end = token.map
            kind = "fenced_code" if token.type == "fence" else "indented_code"
            content.protected[kind].append("".join(lines[start:end]))
        content.protected["inline_code"].extend(_inline_code(token.children or []))
        _inline_urls(token.children or [], urls)
    references = [*env.get("references", {}).values(), *env.get("duplicate_refs", [])]
    for definition in references:
        href = definition["href"]
        if href.startswith(("http://", "https://")):
            urls[href] += 1
    return content
