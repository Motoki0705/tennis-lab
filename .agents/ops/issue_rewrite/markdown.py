"""CommonMark source ranges and link destinations used by preservation checks."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Any

from markdown_it import MarkdownIt
from markdown_it.token import Token

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
        _inline_urls(token.children or [], urls)
    references = [*env.get("references", {}).values(), *env.get("duplicate_refs", [])]
    for definition in references:
        href = definition["href"]
        if href.startswith(("http://", "https://")):
            urls[href] += 1
    return quotes, urls
