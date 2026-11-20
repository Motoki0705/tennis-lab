"""Validate and audit a checkout or an in-memory repository file map."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .layout import DEDICATED_PATH
from .markdown import references
from .render import render_documents
from .schema import (
    DEDICATED_AREA,
    ENTRY_MAX_LINES,
    INDEX_MAX_ENTRIES,
    INDEX_NAME,
    MEMORY_DIR,
    SHARED_LAYERS,
    STALE_DAYS,
    SUMMARY_MAX_BYTES,
    SUMMARY_MAX_LINES,
    SUMMARY_PATH,
    CitationError,
    Entry,
    markdown_body,
    parse_entry,
    private_data_kinds,
)
from .tree import Checkout, FileMap, MappedTree, Tree


@dataclass(frozen=True)
class Problem:
    path: str
    code: str
    message: str


@dataclass(frozen=True)
class LimitUsage:
    path: str
    unit: str
    used: int
    maximum: int
    ratio: float
    exceeded: bool


@dataclass(frozen=True)
class StaleEntry:
    path: str
    last_verified: str
    age_days: int


@dataclass(frozen=True)
class AuditReport:
    today: str
    stale: list[StaleEntry]
    broken_citations: list[Problem]
    limits: list[LimitUsage]
    problems: list[Problem]


@dataclass
class _Inspection:
    tree: Tree
    today: date
    entries: list[Entry] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    limits: list[LimitUsage] = field(default_factory=list)
    changes: dict[str, bytes] = field(default_factory=dict)
    markdown: dict[str, str] = field(default_factory=dict)

    def _directory(self, path: str) -> bool:
        if self.tree.kind(path, follow_links=False) == "directory":
            return True
        self.problems.append(Problem(path, "layout", "expected a directory"))
        return False

    def _read(self, path: str) -> bytes | None:
        kind = self.tree.kind(path, follow_links=False)
        if kind is None:
            return None
        if kind != "file":
            self.problems.append(Problem(path, "layout", "expected a regular file"))
            return None
        return self.tree.read_bytes(path)

    def _load_layout(self) -> dict[str, bytes]:
        """Read inputs and existing generated files, without entering ops-report."""
        if not self._directory(".agents") or not self._directory(MEMORY_DIR):
            return {}
        expected = {"SUMMARY.md", DEDICATED_AREA, *SHARED_LAYERS}
        children = set(self.tree.children(MEMORY_DIR))
        for name in sorted(children - expected):
            self.problems.append(
                Problem(f"{MEMORY_DIR}/{name}", "layout", "unexpected memory path")
            )
        if DEDICATED_AREA in children:
            self._directory(DEDICATED_PATH)

        stored: dict[str, bytes] = {}
        if (content := self._read(SUMMARY_PATH)) is not None:
            stored[SUMMARY_PATH] = content
        for layer in SHARED_LAYERS:
            directory = f"{MEMORY_DIR}/{layer}"
            if not self._directory(directory):
                continue
            for name in self.tree.children(directory):
                path = f"{directory}/{name}"
                if not name.endswith(".md"):
                    self.problems.append(
                        Problem(
                            path, "layout", "shared layers contain Markdown files only"
                        )
                    )
                elif (content := self._read(path)) is not None:
                    if name == INDEX_NAME:
                        stored[path] = content
                    else:
                        self._entry(layer, name, path, content)
        return stored

    def _measure(self, path: str, unit: str, used: int, maximum: int) -> None:
        self.limits.append(
            LimitUsage(path, unit, used, maximum, used / maximum, used > maximum)
        )
        if used > maximum:
            self.problems.append(
                Problem(path, "limit", f"{used} {unit} exceeds the limit of {maximum}")
            )

    def _text(self, path: str, content: bytes) -> str | None:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            self.problems.append(Problem(path, "encoding", "expected UTF-8 text"))
            return None
        for kind in private_data_kinds(text):
            self.problems.append(Problem(path, "private_data", f"contains {kind}"))
        return text

    def _entry(self, layer: str, name: str, path: str, content: bytes) -> None:
        text = self._text(path, content)
        if text is None:
            return
        self._measure(path, "lines", len(text.splitlines()), ENTRY_MAX_LINES)
        self.markdown[path] = text
        try:
            entry = parse_entry(layer, name, text, today=self.today)
        except CitationError as exc:
            self.problems.append(Problem(path, "citation", str(exc)))
        except ValueError as exc:
            self.problems.append(Problem(path, "entry_schema", str(exc)))
        else:
            self._check_summary_scope(path, entry.summary)
            self.entries.append(entry)

    def _check_summary_scope(self, path: str, text: str) -> None:
        if DEDICATED_AREA in text.casefold():
            self.problems.append(
                Problem(
                    path,
                    "dedicated_reference",
                    "summary mentions dedicated ops-report memory",
                )
            )

    def _generated(self, stored: dict[str, bytes], *, sync: bool) -> None:
        for path, document in render_documents(self.entries).items():
            wanted = document.encode("utf-8")
            found = stored.get(path)
            if wanted != found:
                self.changes[path] = wanted
                if not sync:
                    self.problems.append(
                        Problem(path, "out_of_sync", "run shared_memory sync-index")
                    )
            content = wanted if sync else found
            if content is None:
                continue
            if path == SUMMARY_PATH:
                self._measure(path, "bytes", len(content), SUMMARY_MAX_BYTES)
            text = self._text(path, content)
            if text is None:
                continue
            self.markdown[path] = text
            lines = text.splitlines()
            if path == SUMMARY_PATH:
                self._check_summary_scope(path, text)
                self._measure(path, "lines", len(lines), SUMMARY_MAX_LINES)
            else:
                count = sum(line.startswith("- [") for line in lines)
                self._measure(path, "entries", count, INDEX_MAX_ENTRIES)

    def _target(self, path: str, *, file: bool = False) -> str:
        missing = "target file does not exist" if file else "target does not exist"
        dedicated = "target is in dedicated ops-report memory"
        normalized = posixpath.normpath(path)
        if normalized == DEDICATED_PATH or normalized.startswith(f"{DEDICATED_PATH}/"):
            raise ValueError(dedicated)
        try:
            resolved = self.tree.resolve(path)
        except (FileNotFoundError, NotADirectoryError):
            raise ValueError(missing) from None
        if resolved == DEDICATED_PATH or resolved.startswith(f"{DEDICATED_PATH}/"):
            raise ValueError(dedicated)
        # In sync mode, generated documents already represent the proposed tree.
        if resolved in self.markdown:
            return resolved
        kind = self.tree.kind(resolved)
        if kind != "file" and (file or kind != "directory"):
            raise ValueError(missing)
        return resolved

    def _target_text(self, path: str) -> str:
        if path in self.markdown:
            return self.markdown[path]
        try:
            return self.tree.read_bytes(path).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("target is not UTF-8 text") from None

    def _check_references(self) -> None:
        parsed = {
            path: references(markdown_body(text))
            for path, text in self.markdown.items()
        }
        anchors = {path: fragments for path, (_, fragments) in parsed.items()}

        def has_anchor(path: str, anchor: str) -> bool:
            if path not in anchors:
                anchors[path] = references(markdown_body(self._target_text(path)))[1]
            return anchor in anchors[path]

        for entry in self.entries:
            for position, citation in enumerate(entry.citations):
                try:
                    path = self._target(citation.path, file=True)
                    symbol = citation.symbol
                    if symbol is None:
                        continue
                    if symbol.startswith("#"):
                        if not path.lower().endswith(".md"):
                            raise ValueError("anchor symbols require a Markdown file")
                        anchor = unquote(symbol[1:])
                        if not anchor or not has_anchor(path, anchor):
                            raise ValueError("Markdown anchor was not found")
                    elif symbol not in self._target_text(path):
                        raise ValueError("symbol was not found")
                except (OSError, ValueError) as exc:
                    issue = (
                        str(exc)
                        if isinstance(exc, ValueError)
                        else "cannot read target"
                    )
                    self.problems.append(
                        Problem(
                            entry.path, "citation", f"citations[{position}]: {issue}"
                        )
                    )

        for owner, (targets, _) in parsed.items():
            for target in targets:
                try:
                    parts = urlsplit(target)
                    if parts.scheme or parts.netloc:
                        continue
                    relative = unquote(parts.path)
                    path = self._target(
                        posixpath.join(posixpath.dirname(owner), relative)
                        if relative
                        else owner
                    )
                    anchor = unquote(parts.fragment)
                    if (
                        anchor
                        and path.lower().endswith(".md")
                        and not has_anchor(path, anchor)
                    ):
                        raise ValueError("Markdown anchor was not found")
                except (OSError, ValueError) as exc:
                    issue = (
                        str(exc)
                        if isinstance(exc, ValueError)
                        else "cannot read target"
                    )
                    label = (
                        "[redacted]" if private_data_kinds(unquote(target)) else target
                    )
                    self.problems.append(Problem(owner, "link", f"{label}: {issue}"))

    def stale_entries(self) -> list[StaleEntry]:
        stale = []
        for entry in self.entries:
            age = (self.today - entry.last_verified).days
            if age > STALE_DAYS:
                stale.append(
                    StaleEntry(entry.path, entry.last_verified.isoformat(), age)
                )
        return stale


def _inspect(
    source: Path | FileMap | Tree, today: date | None, *, sync: bool
) -> _Inspection:
    tree: Tree
    if isinstance(source, Path):
        tree = Checkout(source)
    elif isinstance(source, Mapping):
        tree = MappedTree(source)
    else:
        tree = source
    inspection = _Inspection(tree, today or date.today())
    stored = inspection._load_layout()
    inspection._generated(stored, sync=sync)
    inspection._check_references()
    return inspection


def validate(
    source: Path | FileMap | Tree,
    *,
    today: date | None = None,
    require_fresh: bool = False,
) -> list[Problem]:
    """Check layout, sources, references, size and generated byte equality."""
    inspection = _inspect(source, today, sync=False)
    if require_fresh:
        inspection.problems.extend(
            Problem(entry.path, "stale", f"last verified {entry.age_days} days ago")
            for entry in inspection.stale_entries()
        )
    return inspection.problems


def plan_sync(
    source: Path | FileMap | Tree, *, today: date | None = None
) -> tuple[dict[str, bytes], list[Problem]]:
    """Validate the proposed tree and return only the generated documents to replace."""
    inspection = _inspect(source, today, sync=True)
    return inspection.changes, inspection.problems


def audit(source: Path | FileMap | Tree, *, today: date | None = None) -> AuditReport:
    """Report freshness, broken citations and use of each document size budget."""
    inspection = _inspect(source, today, sync=False)
    return AuditReport(
        today=inspection.today.isoformat(),
        stale=inspection.stale_entries(),
        broken_citations=[p for p in inspection.problems if p.code == "citation"],
        limits=inspection.limits,
        problems=inspection.problems,
    )
