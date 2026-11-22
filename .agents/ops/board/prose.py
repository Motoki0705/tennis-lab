"""Find code while tracking CommonMark quote and list containers.

Container continuation must be checked before a fence. A missing quote marker
or list indentation ends even an unclosed fence in that container.
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
SOURCE_LINE = re.compile(r"[^\r\n]*(?:\r\n?|\n)|[^\r\n]+$")
BACKTICKS = re.compile(r"(?<!\\)(\\*)(`+)")
HTML_MARKUP = re.compile(r"<(?:[!?]|/?[A-Za-z][A-Za-z0-9-]*(?=[\s/>]|$))")
AMBIGUOUS_INDENT = re.compile(
    r"^(?:[ >]|(?:[-+*]|[0-9]{1,9}[.)])(?=[ \t]))*(?: {4,}|\t)"
)


@dataclass(frozen=True)
class _Line:
    offset: int
    raw: str
    content: str


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
    # This is called when the container markers change. A setext underline
    # only applies to a paragraph in the same container.
    return bool(
        not line.strip(" ")
        or QUOTE.match(line)
        or _opening_fence(line)
        or THEMATIC.fullmatch(line)
        or HEADING.match(line)
        or _list_start(line, paragraph=True)
    )


def _blocks(body: str, *, check_layout: bool = False) -> Iterator[list[_Line] | None]:
    containers: list[Container] = []
    paragraph: list[_Line] = []
    fence: str | None = None
    # A tab contributes columns at stops of four when matching container
    # indentation. Only the scan uses expanded text; the saved body is untouched.
    for source in SOURCE_LINE.finditer(body):
        raw = source[0].rstrip("\r\n")
        line = raw.expandtabs(4)
        if check_layout and AMBIGUOUS_INDENT.match(raw):
            root_code = not containers and (
                fence is not None or (not paragraph and line.startswith("    "))
            )
            if not root_code:
                # Deep indentation and tabs can change whether a container
                # continues or whether a later line belongs to a code block.
                yield None
                return
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
                paragraph.append(_Line(source.start(), raw, line.lstrip(" ")))
                continue
        if not continued:
            yield paragraph
            paragraph = []
            del containers[matched:]

        while True:
            quote = QUOTE.match(line)
            item = _list_start(line, paragraph=bool(paragraph))
            if quote is None and item is None:
                break
            yield paragraph
            paragraph = []
            if quote is not None:
                containers.append(Container())
                line = line[quote.end() :]
            else:
                assert item is not None
                container, line = item
                containers.append(container)

        marker = _opening_fence(line)
        if marker is not None:
            yield paragraph
            paragraph = []
            fence = marker[1]
        elif not line.strip(" ") or THEMATIC.fullmatch(line):
            yield paragraph
            paragraph = []
        elif HEADING.match(line) or (paragraph and SETEXT.fullmatch(line)):
            yield paragraph
            paragraph = []
            yield [_Line(source.start(), raw, line)]
        elif not paragraph and line.startswith("    "):
            # Four remaining columns start indented code, rather than a fence
            # or a new container. Indented code cannot interrupt a paragraph.
            continue
        else:
            paragraph.append(_Line(source.start(), raw, line.lstrip(" ")))
    yield paragraph


def _inline_spans(block: str) -> Iterator[tuple[int, int, int]]:
    """Yield offsets and delimiter widths for the code spans in one block."""
    runs = [
        (run.start(2), len(run[2]), len(run[1]) % 2)
        for run in BACKTICKS.finditer(block)
    ]
    closers: list[int | None] = [None] * len(runs)
    next_run: dict[int, int] = {}
    for index in reversed(range(len(runs))):
        _, width, escaped = runs[index]
        # An odd backslash escapes the first opening tick in prose. Within
        # code, a closing run counts every tick regardless of backslashes.
        closers[index] = next_run.get(width - escaped)
        next_run[width] = index

    index = 0
    while index < len(runs):
        closer = closers[index]
        if closer is None:
            index += 1
            continue
        start, width, escaped = runs[index]
        end, closing_width, _ = runs[closer]
        yield start + escaped, end + closing_width, width - escaped
        index = closer + 1


def single_backtick_openers(body: str) -> set[int]:
    """Find source offsets where a single-backtick code span is established."""
    # Raw HTML blocks can extend across blank lines. Keep path checks on the
    # full text whenever HTML makes the inline context uncertain.
    if HTML_MARKUP.search(body):
        return set()
    openers: set[int] = set()
    for lines in _blocks(body, check_layout=True):
        if lines is None:
            return set()
        block = "\n".join(line.content for line in lines)
        # Link targets, titles and reference labels can contain literal ticks.
        link = block.find("](")
        reference = block.find("][")
        if (
            "]:" in block
            or (link >= 0 and "`" in block[link : block.rfind(")") + 1])
            or (reference >= 0 and "`" in block[reference : block.rfind("]") + 1])
        ):
            continue
        starts = {start for start, _, width in _inline_spans(block) if width == 1}
        if not starts:
            continue
        offset = 0
        for line in lines:
            # Container prefix removal and tab expansion preserve the order of
            # the backticks on this line. Use that order to recover raw offsets.
            for raw, scanned in zip(
                re.finditer("`", line.raw),
                re.finditer("`", line.content),
                strict=True,
            ):
                if offset + scanned.start() in starts:
                    openers.add(line.offset + raw.start())
            offset += len(line.content) + 1
    return openers


def without_code(body: str) -> str:
    """Return text outside code, keeping inline spans within their own block."""
    visible: list[str] = []
    for lines in _blocks(body):
        assert lines is not None
        block = "\n".join(line.content for line in lines)
        offset = 0
        for start, end, _ in _inline_spans(block):
            visible.extend((block[offset:start], " "))
            offset = end
        visible.extend((block[offset:], "\n"))
    return "".join(visible).removesuffix("\n")
