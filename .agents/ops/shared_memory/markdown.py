"""Local Markdown references and GitHub-style section anchors."""

from __future__ import annotations

from html.parser import HTMLParser
from unicodedata import category

from markdown_it import MarkdownIt
from markdown_it.token import Token

_MARKDOWN = MarkdownIt("commonmark").enable(["strikethrough", "table"])


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        for name in ("id", "name") if tag == "a" else ("id",):
            if value := attributes.get(name):
                self.anchors.add(value)
        target = attributes.get({"a": "href", "img": "src"}.get(tag, ""))
        if target is not None:
            self.links.append(target)

    def consume(self, token: Token) -> None:
        if token.type in {"html_block", "html_inline"}:
            self.feed(token.content)
        elif token.type in {"link_open", "image"}:
            attribute = "href" if token.type == "link_open" else "src"
            if target := token.attrGet(attribute):
                self.links.append(target)
        for child in token.children or []:
            self.consume(child)


def _slug(tokens: list[Token]) -> str:
    title = "".join(
        token.content
        for token in tokens
        if token.type in {"text", "code_inline", "image"}
    )
    return "".join(
        char
        for char in title.strip().lower()
        if char in " _-" or category(char)[0] in "LMN"
    ).replace(" ", "-")


def references(text: str) -> tuple[tuple[str, ...], set[str]]:
    """Return rendered link targets and available fragments, excluding code blocks."""
    tokens = _MARKDOWN.parse(text)
    raw = _References()
    headings: set[str] = set()
    suffixes: dict[str, int] = {}
    for index, token in enumerate(tokens):
        raw.consume(token)
        if token.type == "heading_open":
            base = slug = _slug(tokens[index + 1].children or [])
            suffix = suffixes.get(base, 0)
            while slug in headings:
                suffix += 1
                slug = f"{base}-{suffix}"
            suffixes[base] = suffix
            headings.add(slug)
    return tuple(raw.links), headings | raw.anchors
