"""CommonMark source ranges and link destinations used by preservation checks."""

from __future__ import annotations

import re
from collections import Counter
from html.parser import HTMLParser
from typing import Any

from markdown_it import MarkdownIt

BARE_URL = re.compile(r'https?://[^\s<>\[\]`"。、，；！？「」『』（）]+')


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


def markdown_content(body: str) -> tuple[list[str], Counter[str]]:
    """Preserve raw quote blocks, and count URL occurrences rather than references.

    The parser supplies complete destinations and quote line maps. Link labels
    are stored so reference uses do not duplicate the source definition's URL.
    Inline/fenced code and HTML comments are already protected byte for byte.
    """
    env: dict[str, Any] = {}
    parser = MarkdownIt("commonmark", {"store_labels": True}).enable(
        ["table", "strikethrough"]
    )
    tokens = parser.parse(body, env)
    lines = body.splitlines(keepends=True)
    quotes: list[str] = []
    quote_end = 0
    urls: Counter[str] = Counter()
    for token in tokens:
        if token.type == "html_block":
            _HTMLURLs(urls).feed(token.content)
        if token.type == "blockquote_open" and token.map:
            start, end = token.map
            if end > quote_end:
                quotes.append("".join(lines[start:end]))
                quote_end = end
        link_depth = 0
        for child in token.children or []:
            if child.type == "link_open":
                href = child.attrGet("href")
                if (
                    href
                    and href.startswith(("http://", "https://"))
                    and "label" not in child.meta
                ):
                    urls[href] += 1
                link_depth += 1
            elif child.type == "link_close":
                link_depth -= 1
            elif child.type == "image":
                href = child.attrGet("src")
                if (
                    href
                    and href.startswith(("http://", "https://"))
                    and "label" not in child.meta
                ):
                    urls[href] += 1
            elif child.type == "text" and link_depth == 0:
                urls.update(
                    _bare_url(match[0]) for match in BARE_URL.finditer(child.content)
                )
            elif child.type == "html_inline":
                _HTMLURLs(urls).feed(child.content)
    references = [*env.get("references", {}).values(), *env.get("duplicate_refs", [])]
    for definition in references:
        href = definition["href"]
        if href.startswith(("http://", "https://")):
            urls[href] += 1
    return quotes, urls
