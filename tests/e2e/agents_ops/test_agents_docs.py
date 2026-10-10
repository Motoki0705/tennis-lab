"""Validate the AI-operations entry points (AGENTS.md -> .agents/README.md -> memory).

The shared memory is meant to be cheap to edit and delete, so these checks keep
the index, the entry files and the cross-document links from silently drifting.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
AGENTS_MD = ROOT / "AGENTS.md"
AGENTS_README = ROOT / ".agents/README.md"
MEMORY_DIR = ROOT / ".agents/memory"
MEMORY_INDEX = MEMORY_DIR / "INDEX.md"
MEMORY_META_FILES = {"README.md", "INDEX.md"}

DOCS = [
    AGENTS_MD,
    ROOT / "CLAUDE.md",
    AGENTS_README,
    MEMORY_DIR / "README.md",
    MEMORY_INDEX,
    ROOT / ".github/copilot-instructions.md",
    *sorted((ROOT / ".github/agents").glob("*.md")),
]

REQUIRED_KEYS = ("title", "type", "applies_to", "source", "created", "last_verified", "evidence")
MEMORY_TYPES = {"gotcha", "environment", "decision", "workflow"}
APPLIES_TO = {"all", "claude", "codex"}

# A link may point at a sibling issue's file that is not merged yet, but only when
# the same line says so explicitly ("#1057で実装中").
PENDING_MARK = re.compile(r"#\d+で実装中")
LINK = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
FENCE = re.compile(r"^\s*(```|~~~)")
SECRET_PATTERNS = (
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),  # e-mail address
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub token
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),  # API key
)


def _memory_entries() -> list[Path]:
    return sorted(p for p in MEMORY_DIR.glob("*.md") if p.name not in MEMORY_META_FILES)


def _lines_outside_fences(text: str) -> list[str]:
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return lines


def _slug(heading: str) -> str:
    """GitHub-style heading anchor: lowercase, drop punctuation, spaces -> '-'."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    return {
        _slug(line.lstrip("#"))
        for line in _lines_outside_fences(path.read_text(encoding="utf-8"))
        if line.startswith("#")
    }


def _links(path: Path) -> list[tuple[str, str, str]]:
    """Return (text, target, line) for every Markdown link outside code fences."""
    found: list[tuple[str, str, str]] = []
    for line in _lines_outside_fences(path.read_text(encoding="utf-8")):
        for match in LINK.finditer(line):
            found.append((match.group(1), match.group(2), line))
    return found


def _frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{path.name}: missing YAML frontmatter"
    data = yaml.safe_load(match.group(1))
    assert isinstance(data, dict), f"{path.name}: frontmatter must be a mapping"
    return data


def _as_date(value: object, field: str, name: str) -> dt.date:
    if isinstance(value, dt.date):
        return value
    raise AssertionError(f"{name}: {field} must be an ISO date (YYYY-MM-DD), got {value!r}")


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc: Path) -> None:
    broken: list[str] = []
    for _text, target, line in _links(doc):
        if re.match(r"^[a-z]+:", target):
            continue  # external URL
        file_part, _, anchor = target.partition("#")
        resolved = (doc.parent / file_part).resolve() if file_part else doc
        if not resolved.exists():
            if not PENDING_MARK.search(line):
                broken.append(f"{target} (missing file)")
            continue
        if anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            broken.append(f"{target} (missing anchor)")
    assert not broken, f"{doc.relative_to(ROOT)}: broken links: {broken}"


def test_entry_chain_from_agents_md_to_memory_index() -> None:
    agents_targets = {target for _, target, _ in _links(AGENTS_MD)}
    assert ".agents/README.md" in agents_targets
    assert ".agents/memory/INDEX.md" in agents_targets
    readme_targets = {target for _, target, _ in _links(AGENTS_README)}
    assert "memory/INDEX.md" in readme_targets
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()[0] == "@AGENTS.md"


def test_memory_index_matches_entry_files() -> None:
    entries = _memory_entries()
    assert entries, "shared memory has no entries"
    indexed = [(text, target) for text, target, line in _links(MEMORY_INDEX) if line.startswith("- ")]
    indexed_files = [target for _, target in indexed]
    assert len(indexed_files) == len(set(indexed_files)), "INDEX.md lists an entry twice"
    assert set(indexed_files) == {p.name for p in entries}, "INDEX.md and entry files differ"
    for text, target in indexed:
        title = _frontmatter(MEMORY_DIR / target)["title"]
        assert text == title, f"INDEX.md link text for {target} must equal its title"


@pytest.mark.parametrize("entry", _memory_entries(), ids=lambda p: p.name)
def test_memory_entry_frontmatter(entry: Path) -> None:
    meta = _frontmatter(entry)
    missing = [key for key in REQUIRED_KEYS if not meta.get(key)]
    assert not missing, f"{entry.name}: missing frontmatter keys {missing}"
    assert meta["type"] in MEMORY_TYPES, f"{entry.name}: unknown type {meta['type']!r}"
    assert meta["applies_to"] in APPLIES_TO, f"{entry.name}: unknown applies_to {meta['applies_to']!r}"
    created = _as_date(meta["created"], "created", entry.name)
    verified = _as_date(meta["last_verified"], "last_verified", entry.name)
    assert created <= verified <= dt.date.today(), f"{entry.name}: need created <= last_verified <= today"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*\.md", entry.name), f"{entry.name}: use kebab-case"


@pytest.mark.parametrize("entry", _memory_entries(), ids=lambda p: p.name)
def test_memory_entry_has_no_personal_data(entry: Path) -> None:
    text = entry.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text), f"{entry.name}: looks like personal data or a secret ({pattern.pattern})"
    assert "/home/" not in text, f"{entry.name}: do not write machine-specific home paths"
