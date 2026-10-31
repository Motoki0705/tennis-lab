"""Structural contracts for Markdown and embedded HTML documentation."""

from __future__ import annotations

import re
from collections.abc import Collection
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class _HTMLReferences(HTMLParser):
    def __init__(self, text: str) -> None:
        super().__init__()
        self.targets: set[str] = set()
        self.ids: set[str] = set()
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name == "id":
                self.ids.add(value)
            if (tag == "a" and name == "href") or (tag == "img" and name == "src"):
                self.targets.add(value)


def links(text: str) -> set[str]:
    """Read Markdown links/images and HTML href/src without layout assumptions."""
    return _HTMLReferences(text).targets | set(re.findall(r"\]\(([^)\s]+)\)", text))


def anchors(text: str) -> set[str]:
    headings = re.findall(r"^#+ (.+)$", text, re.MULTILINE)
    return _HTMLReferences(text).ids | {
        re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        for heading in headings
    }


def assert_sections(text: str, required: Collection[str], *, context: str) -> None:
    """Require unique public anchors with headings, independent of wording/order."""
    sections = list(re.finditer(
        r"<a\b[^>]*\sid\s*=\s*[\"']([^\"']+)[\"'][^>]*>", text, re.IGNORECASE
    ))
    for section_id in required:
        matches = [i for i, section in enumerate(sections) if section[1] == section_id]
        assert len(matches) == 1, f"{context}: missing or duplicate section {section_id}"
        index = matches[0]
        end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        body = text[sections[index].end():end]
        assert re.search(r"^#{1,6} \S", body, re.MULTILINE), (
            f"{context}: missing heading for {section_id}"
        )


def assert_local_links(document: Path, text: str) -> None:
    """Resolve every local destination and fragment, including image assets."""
    for target in links(text):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        destination = document.parent / unquote(parsed.path) if parsed.path else document
        assert destination.is_file(), f"{document.name}: missing file for {target}"
        if parsed.fragment:
            assert unquote(parsed.fragment) in anchors(destination.read_text(encoding="utf-8")), (
                f"{document.name}: missing anchor for {target}"
            )
