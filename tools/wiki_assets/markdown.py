"""Render copyable tables and validate Markdown as it will be embedded in the Wiki."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .pages import Asset, asset_data
from .publication import scan_public_text
from .sources import REPOSITORY, Sources, require
from .table_spec import SLUG, escape_markdown, markdown_table

PARSER = MarkdownIt("commonmark", {"html": True}).enable("table")
DETAILS = re.compile(
    r"<details(?:\s+[^>]*)?>\s*(<summary>.*?</summary>)?", re.IGNORECASE
)
SUMMARY = re.compile(r"<summary>.+?</summary>", re.IGNORECASE)
METADATA = re.compile(r"^(?:想定読者|検索キーワード|正本ファイル|主な正本パス)$")
MARKER = re.compile(
    rf"<!-- wiki-assets:(begin|end) ({SLUG.pattern})/({SLUG.pattern}) -->"
)


def snippet(page: str, sources: Sources, assets: tuple[Asset, ...]) -> str:
    sections = []
    for asset in assets:
        data = asset_data(asset)
        output = f"assets/{page}/{data.slug}"
        title = escape_markdown(data.title)
        note = escape_markdown(data.note)
        alt = f"{title}。{note} 数値と状態は直下のテキスト表を参照。"
        sections.append(
            f"<!-- wiki-assets:begin {page}/{data.slug} -->\n\n"
            f"### {title}\n\n"
            f"![{alt}]({output}.svg)\n\n"
            f"[SVG 原寸]({output}.svg) · [互換 PNG]({output}.png)\n\n"
            f"{note}\n\n"
            f"<details>\n<summary>{title}のテキスト表</summary>\n\n"
            f"{markdown_table(data)}\n\n</details>\n\n"
            f"<!-- wiki-assets:end {page}/{data.slug} -->"
        )
    links = [
        f"- [`{p.removeprefix('knowledge/runs/')}`]({REPOSITORY}/blob/{sources.revision}/{p})"
        for p in sorted(sources.digests)
    ]
    sections.append("### 図表の正本\n\n" + "\n".join(links))
    return "\n\n".join(sections) + "\n"


def embedded_sections(source: str, name: str) -> dict[tuple[str, str], str]:
    """Read whole, uniquely identified blocks from standalone HTML comments."""
    sections: dict[tuple[str, str], str] = {}
    opened: tuple[tuple[str, str], int] | None = None
    lines = source.split("\n")
    for token in PARSER.parse(source):
        if token.type != "html_block" or token.map is None or token.level != 0:
            continue
        marker = MARKER.fullmatch(token.content.rstrip("\n"))
        if marker is None:
            continue
        start, end = token.map
        key = marker[2], marker[3]
        if marker[1] == "begin":
            require(
                opened is None and key not in sections,
                f"{name}:{start + 1}: nested or duplicate wiki-assets block",
            )
            require(
                start == 0 or not lines[start - 1].strip(),
                f"{name}:{start + 1}: blank line before wiki-assets block required",
            )
            opened = key, start
        else:
            if opened is None or opened[0] != key:
                raise ValueError(
                    f"{name}:{start + 1}: unmatched wiki-assets end marker"
                )
            require(
                end == len(lines) or not lines[end].strip(),
                f"{name}:{start + 1}: blank line after wiki-assets block required",
            )
            sections[key] = "\n".join(lines[opened[1] : end])
            opened = None
    require(opened is None, f"{name}: unclosed wiki-assets block")
    return sections


class References(HTMLParser):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.filename = name
        self.links: set[str] = set()
        self.images: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "img":
            require(
                bool((values.get("alt") or "").strip()),
                f"{self.filename}: image alt is empty",
            )
            source = values.get("src")
            require(bool(source), f"{self.filename}: image source is empty")
            if source:
                self.images.add(source)
        if tag in {"img", "source"} and values.get("srcset"):
            entries = (values["srcset"] or "").split(",")
            self.images.update(v.split()[0] for v in entries if v.strip())
        href = values.get("href")
        if tag == "a" and href:
            self.links.add(href)


def _opening_metadata(lines: list[str], tokens: list[Token], name: str) -> None:
    introduction_end = len(lines)
    for token in tokens:
        if token.type == "heading_open" and token.tag != "h1" and token.map is not None:
            introduction_end = token.map[0]
            break
    for token in tokens:
        if token.map is None or token.map[0] >= introduction_end:
            continue
        start, end = token.map
        if token.type == "table_open":
            labels = [
                re.split(r"(?<!\\)\|", line.lstrip().removeprefix("|"), maxsplit=1)[0]
                for line in lines[start:end]
            ]
        elif token.type == "html_block":
            labels = re.findall(
                r"<t[dh]\b[^>]*>(.*?)</t[dh]>",
                "\n".join(lines[start:end]),
                re.IGNORECASE | re.DOTALL,
            )
        else:
            continue
        for label in labels:
            plain = re.sub(r"<[^>]+>|[*_`]", "", label).strip()
            require(
                METADATA.fullmatch(plain) is None,
                f"{name}:{start + 1}: opening metadata table",
            )


def _details_spacing(lines: list[str], tokens: list[Token], filename: str) -> None:
    code_lines = {
        line
        for token in tokens
        if token.type in {"fence", "code_block"} and token.map is not None
        for line in range(*token.map)
    }
    stack = []

    def blank_before(line: int) -> bool:
        return line == 0 or not lines[line - 1].strip()

    def blank_after(line: int) -> bool:
        return line == len(lines) - 1 or not lines[line + 1].strip()

    for line, value in enumerate(lines):
        if line in code_lines:
            continue
        value = value.strip()
        opening = DETAILS.fullmatch(value)
        if opening:
            require(
                blank_before(line),
                f"{filename}:{line + 1}: blank line before <details> required",
            )
            summary = line if opening[1] else line + 1
            require(
                summary < len(lines)
                and (
                    opening[1] is not None
                    or SUMMARY.fullmatch(lines[summary].strip()) is not None
                ),
                f"{filename}:{line + 1}: <summary> required",
            )
            require(
                blank_after(summary),
                f"{filename}:{summary + 1}: blank line after <summary> required",
            )
            stack.append(line)
        elif value.lower() == "</details>":
            require(
                bool(stack),
                f"{filename}:{line + 1}: closing </details> has no opening tag",
            )
            require(
                blank_before(line) and blank_after(line),
                f"{filename}:{line + 1}: blank line before and after </details> required",
            )
            stack.pop()
        elif re.match(r"</?details\b", value, re.IGNORECASE):
            raise ValueError(f"{filename}:{line + 1}: use a standalone <details> block")
    require(not stack, f"{filename}: unclosed <details>")


def _asset_path(url: str, root: Path, *, image: bool) -> Path | None:
    parsed = urlsplit(url)
    relative = unquote(parsed.path).removeprefix("./")
    if "/wiki/assets/" in relative or relative.startswith(("/assets/", "../assets/")):
        raise ValueError("Wiki assets must use a relative assets/<page-slug>/ path")
    if not relative.startswith("assets/"):
        if (
            not parsed.netloc
            and not parsed.scheme
            and (image or PurePosixPath(relative).suffix.lower() in {".svg", ".png"})
        ):
            raise ValueError("Wiki images must use a relative assets/<page-slug>/ path")
        return None
    parts = PurePosixPath(relative).parts
    require(
        not (
            parsed.netloc
            or parsed.scheme
            or parsed.query
            or "\\" in relative
            or ".." in parts
        )
        and len(parts) >= 3,
        "Invalid relative Wiki asset",
    )
    target = root.joinpath(*parts)
    require(
        target.resolve().is_relative_to(root.resolve())
        and target.is_file()
        and not target.is_symlink(),
        f"Missing relative Wiki asset: {parts[-1]}",
    )
    return target


def verify_markdown(path: Path, *, source: str | None = None) -> set[Path]:
    """Check a Wiki page or snippet and return all referenced relative SVGs and PNGs."""
    if source is None:
        source = path.read_text(encoding="utf-8")
    scan_public_text(source, path.name)
    tokens = PARSER.parse(source)
    lines = source.splitlines()
    _opening_metadata(lines, tokens, path.name)
    _details_spacing(lines, tokens, path.name)
    refs = References(path.name)
    refs.feed(PARSER.render(source))
    refs.close()
    for url in refs.images | refs.links:
        scan_public_text(url, path.name)
    assets = {
        item: _asset_path(item, path.parent, image=item in refs.images)
        for item in refs.images | refs.links
    }
    links = {assets[item] for item in refs.links if assets[item] is not None}
    images = {assets[item] for item in refs.images if assets[item] is not None}
    svg_images = {
        item for item in images if item is not None and item.suffix.lower() == ".svg"
    }
    for svg in svg_images:
        require(
            svg.with_suffix(".png") in links,
            f"{svg.name}: corresponding PNG link missing",
        )
    return {
        item
        for item in images | links
        if item is not None and item.suffix.lower() in {".svg", ".png"}
    }
