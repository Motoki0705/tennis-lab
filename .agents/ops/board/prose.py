"""Remove code while tracking CommonMark quote and list containers.

This is a block scanner for board addressing, not a Markdown renderer. Container
continuation must be checked before a fence: a missing quote marker or list
indentation ends even an unclosed fence in that container.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

QUOTE = re.compile(r"^ {0,3}> ?")
LIST = re.compile(r"^( {0,3})([-+*]|[0-9]{1,9}[.)])( +|$)")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
THEMATIC = re.compile(r"^ {0,3}(?:\*(?: *\*){2,}|-(?: *-){2,}|_(?: *_){2,}) *$")
HEADING = re.compile(r"^ {0,3}#{1,6}(?: |$)")
SETEXT = re.compile(r"^ {0,3}(?:=+|-+) *$")
INLINE_CODE = re.compile(r"(?<![\\`])(?:\\\\)*(`+)(?!`).*?(?<!`)\1(?!`)", re.DOTALL)


@dataclass(frozen=True)
class Container:
    # Zero indentation denotes a quote; otherwise this is a list item's content
    # indentation, including the marker's indentation and its following padding.
    indent: int = 0
    empty: bool = False
    # Sibling items share a bullet character or an ordered list's delimiter.
    marker: str = ""


def _list_start(line: str, *, paragraph: bool) -> tuple[Container, str] | None:
    marker = LIST.match(line)
    if marker is None or THEMATIC.fullmatch(line):
        return None
    content = line[marker.end() :]
    # Only a nonempty bullet or an ordered item starting at 1 can interrupt a
    # paragraph. Other numbers can still begin sibling items in a list.
    if paragraph and (
        not content or (marker[2][0].isdigit() and int(marker[2][:-1]) != 1)
    ):
        return None
    padding = len(marker[3])
    if not content or padding > 4:
        padding = 1
    width = len(marker[1]) + len(marker[2]) + padding
    return Container(width, not content, marker[2][-1]), line[width:]


def _opening_fence(line: str) -> re.Match[str] | None:
    marker = FENCE.fullmatch(line)
    if marker is not None and (marker[1][0] == "~" or "`" not in marker[2]):
        return marker
    return None


def _interrupts(line: str) -> bool:
    return bool(
        not line.strip(" ")
        or QUOTE.match(line)
        or _opening_fence(line)
        or THEMATIC.fullmatch(line)
        or HEADING.match(line)
        or SETEXT.fullmatch(line)
        or _list_start(line, paragraph=True)
    )


def _blocks(body: str) -> Iterator[str]:
    containers: list[Container] = []
    paragraph: list[str] = []
    fence: str | None = None
    # A tab contributes columns at stops of four when matching container
    # indentation. Only the scan uses expanded text; the saved body is untouched.
    for raw in re.split(r"\r\n?|\n", body):
        line = raw.expandtabs(4)
        matched = 0
        for container in containers:
            if container.indent == 0:
                quote = QUOTE.match(line)
                if quote is None:
                    break
                line = line[quote.end() :]
            elif container.empty and not line.strip(" "):
                break
            elif line.startswith(" " * container.indent):
                line = line[container.indent :]
                if container.empty and line.strip(" "):
                    containers[matched] = Container(
                        container.indent, marker=container.marker
                    )
            elif not line.strip(" "):
                line = ""
            else:
                break
            matched += 1
        continued = matched == len(containers)
        if fence is not None:
            if continued:
                marker = FENCE.fullmatch(line)
                if (
                    marker is not None
                    and marker[1][0] == fence[0]
                    and len(marker[1]) >= len(fence)
                    and not marker[2].strip(" ")
                ):
                    fence = None
                # Container-looking text inside a fence is literal code.
                continue
            fence = None

        if not continued and paragraph and not _interrupts(line):
            item = _list_start(line, paragraph=False)
            # A sibling can end the current item without interrupting a
            # paragraph, but only when its enclosing containers all matched.
            # A missing quote or a different list marker stays lazy prose.
            if item is None or item[0].marker != containers[matched].marker:
                # Lazy continuation never continues a code block.
                paragraph.append(line.lstrip(" "))
                continue
        if not continued:
            yield "\n".join(paragraph)
            paragraph.clear()
            del containers[matched:]

        while True:
            quote = QUOTE.match(line)
            item = _list_start(line, paragraph=bool(paragraph))
            if quote is None and item is None:
                break
            yield "\n".join(paragraph)
            paragraph.clear()
            if quote is not None:
                containers.append(Container())
                line = line[quote.end() :]
            else:
                assert item is not None
                container, line = item
                containers.append(container)

        marker = _opening_fence(line)
        if marker is not None:
            yield "\n".join(paragraph)
            paragraph.clear()
            fence = marker[1]
        elif not line.strip(" ") or THEMATIC.fullmatch(line):
            yield "\n".join(paragraph)
            paragraph.clear()
        elif HEADING.match(line) or (paragraph and SETEXT.fullmatch(line)):
            yield "\n".join(paragraph)
            paragraph.clear()
            yield line
        elif not paragraph and line.startswith("    "):
            # Four remaining columns start indented code, rather than a fence
            # or a new container. Indented code cannot interrupt a paragraph.
            continue
        else:
            paragraph.append(line.lstrip(" "))
    yield "\n".join(paragraph)


def without_code(body: str) -> str:
    """Return text outside code, keeping inline spans within their own block."""
    return "\n".join(INLINE_CODE.sub(" ", block) for block in _blocks(body))
