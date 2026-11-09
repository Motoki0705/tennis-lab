"""Layout-independent identity, reference and change accounting checks.

This is the structural boundary. HTML/SVG sanitization, renderer policy, semantic
verification and browser rendering checks belong to the downstream validators.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from html import unescape
from urllib.parse import unquote, urlsplit

from .parsing import StrictHTMLParser, reject_css_resources
from .paths import asset_path, relative_path
from .schema import (
    IDENTIFIER_PATTERN,
    Candidate,
    ContentChange,
    ContractError,
    Document,
    DocumentTarget,
    EditRequest,
    ErrorCode,
    EvidenceChange,
    Revision,
    SelectionTarget,
)

_VOID = frozenset(
    [
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
    ]
)
_ID = re.compile(IDENTIFIER_PATTERN)
_REF_ATTRIBUTES = frozenset(
    {"src", "href", "poster", "data", "action", "formaction", "xlink:href"}
)
_ASCII_SPACE = " \t\n\f\r"
_DENSITY = re.compile(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?x")


def _srcset_urls(value: str) -> list[str]:
    """Strict author syntax for canonical managed URLs and optional w/x descriptors.

    Managed filenames cannot contain commas, so commas unambiguously separate
    candidates. Reject malformed candidates rather than letting a browser drop
    them and silently choose a different resource.
    """
    urls = []
    seen: set[tuple[str, int | float]] = set()
    mode: str | None = None
    for candidate in value.split(","):
        fields = re.split(r"[ \t\n\f\r]+", candidate.strip(_ASCII_SPACE))
        if len(fields) not in {1, 2} or not fields[0]:
            raise ContractError("broken_reference")
        descriptor = fields[1] if len(fields) == 2 else "1x"
        number: int | float
        try:
            if re.fullmatch(r"[0-9]+w", descriptor):
                number = int(descriptor[:-1])
                if number <= 0:
                    raise ContractError("broken_reference")
            elif _DENSITY.fullmatch(descriptor):
                number = float(descriptor[:-1])
                if not math.isfinite(number):
                    raise ContractError("broken_reference")
            else:
                raise ContractError("broken_reference")
        except ValueError as exc:
            raise ContractError("broken_reference") from exc
        unit = descriptor[-1]
        if (mode is not None and unit != mode) or (unit, number) in seen:
            raise ContractError("broken_reference")
        mode = unit
        seen.add((unit, number))
        urls.append(fields[0])
    return urls


@dataclass
class Block:
    start: int
    end: int = 0
    text: list[str] = field(default_factory=list)


class HTMLIndex(StrictHTMLParser):
    def __init__(self, html: str):
        super().__init__(convert_charrefs=True)
        self.html = html
        self.blocks: dict[str, Block] = {}
        self.ids: set[str] = set()
        self.references: list[tuple[str, bool]] = []
        self.reference_blocks: dict[str, set[str]] = {}
        self.unowned_references: set[str] = set()
        self.assets: set[str] = set()
        self.asset_blocks: dict[str, set[str]] = {}
        self.asset_sources: list[tuple[str, str]] = []
        self.figures: set[str] = set()
        self.figure_blocks: dict[str, set[str]] = {}
        self.evidence: set[str] = set()
        self.stack: list[tuple[str, str | None]] = []
        self._ignore_initial_lf = False
        # HTML stream preprocessing precedes character-reference expansion.
        # Retain a sparse offset map so block_html still returns original bytes
        # even when CRLF in text or tag attributes shrinks to one LF for parsing.
        self._crlf_ends = [
            match.start() - removed + 1
            for removed, match in enumerate(re.finditer("\r\n", html))
        ]
        normalized = html.replace("\r\n", "\n").replace("\r", "\n")
        self.line_offsets = [0]
        for match in re.finditer("\n", normalized):
            self.line_offsets.append(match.end())
        try:
            self.feed(normalized)
            self.close()
        except (ValueError, AssertionError) as exc:
            raise ContractError("invalid_document") from exc
        if self.stack or not self.blocks:
            raise ContractError("invalid_document")

    def _source_offset(self, additional: int = 0) -> int:
        line, column = self.getpos()
        offset = self.line_offsets[line - 1] + column + additional
        return offset + bisect_right(self._crlf_ends, offset)

    def _add_reference(self, value: str, external_link: bool, blocks: set[str]) -> None:
        self.references.append((value, external_link))
        self.reference_blocks.setdefault(value, set()).update(blocks)
        if not blocks:
            self.unowned_references.add(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._ignore_initial_lf = False
        names = [key for key, _ in attrs]
        if len(names) != len(set(names)) or tag == "base":
            raise ContractError("invalid_document")
        attributes = dict(attrs)
        for key, destination in (
            ("id", self.ids),
            ("data-asset-id", self.assets),
            ("data-figure-id", self.figures),
            ("data-evidence-id", self.evidence),
        ):
            if key not in attributes:
                continue
            value = attributes[key]
            if not value or (key != "id" and not _ID.fullmatch(value)):
                raise ContractError("invalid_document")
            if key in {"id", "data-figure-id"} and value in destination:
                raise ContractError("duplicate_id")
            destination.add(value)
        content_id = attributes.get("data-content-id")
        if "data-content-id" in attributes:
            if not content_id or not _ID.fullmatch(content_id):
                raise ContractError("invalid_document")
            if content_id in self.blocks:
                raise ContractError("duplicate_id")
            self.blocks[content_id] = Block(self._source_offset())
        blocks = {item for _, item in self.stack if item}
        if content_id:
            blocks.add(content_id)
        asset_id = attributes.get("data-asset-id")
        if asset_id:
            self.asset_blocks.setdefault(asset_id, set()).update(blocks)
            source = attributes.get("src") or attributes.get("href")
            if source:
                self.asset_sources.append((asset_id, source))
        figure_id = attributes.get("data-figure-id")
        if figure_id:
            self.figure_blocks[figure_id] = blocks
        for key, value in attrs:
            if key in _REF_ATTRIBUTES and value is not None:
                self._add_reference(value, tag == "a" and key == "href", blocks)
            if key in {"srcset", "imagesrcset"}:
                if value is None:
                    raise ContractError("broken_reference")
                for source in _srcset_urls(value):
                    self._add_reference(source, False, blocks)
            # Inline CSS URLs remain outside the structural contract.
            if key == "style":
                reject_css_resources(value or "")
        if tag in _VOID:
            if content_id:
                self.blocks[content_id].end = self._source_offset(
                    len(self._start_tag())
                )
        else:
            self.stack.append((tag, content_id))
            self._ignore_initial_lf = tag in {"pre", "textarea", "listing"}

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self._end(tag, self._source_offset(len(self._start_tag())))

    def _start_tag(self) -> str:
        result = self.get_starttag_text()
        if result is None:
            raise ContractError("invalid_document")
        return result

    def _end(self, tag: str, end: int) -> None:
        self._ignore_initial_lf = False
        if not self.stack or self.stack[-1][0] != tag:
            raise ContractError("invalid_document")
        _, content_id = self.stack.pop()
        if content_id:
            self.blocks[content_id].end = end

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            raise ContractError("invalid_document")
        end = self.html.find(">", self._source_offset())
        if end == -1:
            raise ContractError("invalid_document")
        self._end(tag, end + 1)

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1][0] in {"textarea", "title"}:
            data = unescape(data)
        if self._ignore_initial_lf:
            if data.startswith("\n"):
                data = data[1:]
            self._ignore_initial_lf = False
        if self.stack and self.stack[-1][0] == "style":
            reject_css_resources(data)
        for _, content_id in self.stack:
            if content_id:
                self.blocks[content_id].text.append(data)

    def handle_comment(self, data: str) -> None:
        self._ignore_initial_lf = False

    def handle_decl(self, decl: str) -> None:
        self._ignore_initial_lf = False

    def handle_pi(self, data: str) -> None:
        self._ignore_initial_lf = False

    def block_html(self, content_id: str) -> str:
        block = self.blocks[content_id]
        return self.html[block.start : block.end]


def _unique(values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ContractError("duplicate_id")


def inspect_document(document: Document) -> HTMLIndex:
    index = HTMLIndex(document.html)
    _unique([asset.asset_id for asset in document.assets])
    _unique([asset.path for asset in document.assets])
    _unique([figure.figure_id for figure in document.figures])
    _unique([item.evidence_id for item in document.evidence])
    assets = {asset.asset_id: asset for asset in document.assets}
    paths = {
        asset_path(asset.asset_id, asset.path): asset.asset_id
        for asset in document.assets
    }
    figures = {figure.figure_id for figure in document.figures}
    evidence = {item.evidence_id for item in document.evidence}
    if (
        not index.assets <= assets.keys()
        or not index.figures <= figures
        or not index.evidence <= evidence
    ):
        raise ContractError("broken_reference")
    if any(not blocks for blocks in index.asset_blocks.values()):
        raise ContractError("broken_reference")
    for asset_id, source in index.asset_sources:
        if source.split("#", 1)[0] != assets[asset_id].path:
            raise ContractError("broken_reference")
    for figure in document.figures:
        if (
            figure.content_id not in index.blocks
            or not set(figure.asset_ids) <= assets.keys()
            or figure.figure_id not in index.figures
            or figure.content_id not in index.figure_blocks.get(figure.figure_id, set())
        ):
            raise ContractError("broken_reference")
        _unique(figure.asset_ids)
        if figure.renderer and (
            figure.renderer.asset_id not in assets
            or assets[figure.renderer.asset_id].kind != "renderer"
        ):
            raise ContractError("broken_reference")
        if figure.kind == "graph" and not figure.renderer:
            raise ContractError("broken_reference")
    for reference, external_link in index.references:
        try:
            parsed = urlsplit(reference)
        except ValueError as exc:
            raise ContractError("unsafe_path") from exc
        if parsed.scheme or parsed.netloc:
            # External citations are links, never fetched as local artifacts.
            if not external_link or parsed.scheme not in {"https", "http", "mailto"}:
                raise ContractError("unsafe_path")
            continue
        if parsed.query or "%" in reference or "\\" in reference:
            raise ContractError("unsafe_path")
        if parsed.path:
            relative_path(parsed.path)
            if parsed.path not in paths.keys() | {
                "index.html",
                "evidence.json",
                "figures.json",
            }:
                raise ContractError("broken_reference")
            if parsed.path in paths:
                if reference in index.unowned_references:
                    raise ContractError("broken_reference")
                index.asset_blocks.setdefault(paths[parsed.path], set()).update(
                    index.reference_blocks[reference]
                )
        if (
            parsed.fragment
            and (not parsed.path or parsed.path == "index.html")
            and unquote(parsed.fragment) not in index.ids
        ):
            raise ContractError("broken_reference")
    return index


def validate_request(request: EditRequest, current: Revision | None) -> None:
    if current is None:
        if request.base_revision is not None or not isinstance(
            request.target, DocumentTarget
        ):
            raise ContractError("invalid_target")
        return
    if request.document_id != current.document.document_id:
        raise ContractError("invalid_request")
    if request.base_revision != current.revision_id:
        code: ErrorCode = (
            "document_exists" if request.base_revision is None else "stale_revision"
        )
        raise ContractError(code, current_revision=current.revision_id)
    if isinstance(request.target, DocumentTarget):
        return
    index = inspect_document(current.document)
    if request.target.content_id not in index.blocks:
        raise ContractError("invalid_target")
    if isinstance(request.target, SelectionTarget):
        target = request.target
        text = "".join(index.blocks[target.content_id].text)
        encoded = text.encode("utf-16-le")
        try:
            before = encoded[: target.start * 2].decode("utf-16-le")
            selected = encoded[target.start * 2 : target.end * 2].decode("utf-16-le")
            after = encoded[target.end * 2 :].decode("utf-16-le")
        except UnicodeError as exc:
            raise ContractError("invalid_target") from exc
        if (
            target.start >= target.end
            or target.end * 2 > len(encoded)
            or selected != target.exact
            or not before.endswith(target.prefix)
            or not after.startswith(target.suffix)
        ):
            raise ContractError("invalid_target")


def content_changes(before: Document | None, after: Document) -> list[ContentChange]:
    old = inspect_document(before) if before else None
    new = inspect_document(after)
    old_ids = set(old.blocks) if old else set()
    old_assets = {asset.asset_id: asset for asset in before.assets} if before else {}
    new_assets = {asset.asset_id: asset for asset in after.assets}
    changed_assets = {
        item
        for item in old_assets.keys() | new_assets.keys()
        if old_assets.get(item) != new_assets.get(item)
    }
    old_figures = (
        {figure.figure_id: figure for figure in before.figures} if before else {}
    )
    new_figures = {figure.figure_id: figure for figure in after.figures}
    affected = set()
    for item in old_figures.keys() | new_figures.keys():
        previous, current = old_figures.get(item), new_figures.get(item)
        for figure in (previous, current):
            if figure and (
                previous != current
                or changed_assets.intersection(figure.asset_ids)
                or (figure.renderer and figure.renderer.asset_id in changed_assets)
            ):
                affected.add(figure.content_id)
    for item in changed_assets:
        affected.update(new.asset_blocks.get(item, set()))
        if old:
            affected.update(old.asset_blocks.get(item, set()))
    result = [
        ContentChange(content_id=item, kind="removed")
        for item in sorted(old_ids - new.blocks.keys())
    ]
    for item in sorted(new.blocks):
        if item not in old_ids:
            result.append(ContentChange(content_id=item, kind="added"))
        elif old and (item in affected or old.block_html(item) != new.block_html(item)):
            result.append(ContentChange(content_id=item, kind="updated"))
    return result


def evidence_changes(
    before: Document | None, after: Document, *, reason: str
) -> list[EvidenceChange]:
    old = {item.evidence_id: item for item in before.evidence} if before else {}
    new = {item.evidence_id: item for item in after.evidence}
    result = [
        EvidenceChange(evidence_id=item, kind="removed", reason=reason)
        for item in sorted(old.keys() - new.keys())
    ]
    for item in sorted(new):
        if item not in old:
            result.append(EvidenceChange(evidence_id=item, kind="added", reason=reason))
        elif old[item] != new[item]:
            result.append(
                EvidenceChange(evidence_id=item, kind="updated", reason=reason)
            )
    return result


def validate_candidate(candidate: Candidate, current: Revision | None) -> None:
    validate_request(candidate.request, current)
    if candidate.document.document_id != candidate.request.document_id:
        raise ContractError("invalid_candidate")
    previous = current.document if current else None
    expected = {
        (item.content_id, item.kind)
        for item in content_changes(previous, candidate.document)
    }
    reported = {(item.content_id, item.kind) for item in candidate.changes}
    if expected != reported or len(reported) != len(candidate.changes):
        raise ContractError("invalid_candidate")
    expected_evidence = {
        (item.evidence_id, item.kind)
        for item in evidence_changes(previous, candidate.document, reason="check")
    }
    reported_evidence = {
        (item.evidence_id, item.kind) for item in candidate.evidence_changes
    }
    if expected_evidence != reported_evidence or len(reported_evidence) != len(
        candidate.evidence_changes
    ):
        raise ContractError("invalid_candidate")
