"""Validate the AI-operations entry points (AGENTS.md -> memory SUMMARY -> layer index -> entry).

The shared memory is meant to be cheap to edit and delete, so these checks keep
the summary, the layer indexes, the entry files and the cross-document links from
silently drifting, and keep the always-read files small.
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
MEMORY_SUMMARY = MEMORY_DIR / "SUMMARY.md"
MEMORY_SKILL = ROOT / ".agents/skills/memory/SKILL.md"
DSH_SKILL = ROOT / ".agents/skills/deepseek-harness/SKILL.md"
BOARD_SKILL = ROOT / ".agents/skills/parent-board/SKILL.md"
LAYERS = ("workflow", "decision", "environment", "gotcha", "review")
DEDICATED = "ops-report"
AREAS = (*LAYERS, DEDICATED)
LAYER_INDEX = "index.md"

DOCS = [
    AGENTS_MD,
    ROOT / "CLAUDE.md",
    AGENTS_README,
    MEMORY_SKILL,
    DSH_SKILL,
    BOARD_SKILL,
    DSH_SKILL.with_name("task-template.md"),
    ROOT / ".agents/ops/parent/README.md",
    ROOT / ".agents/ops/board/README.md",
    ROOT / ".agents/ops/herdr_labels/README.md",
    ROOT / ".agents/skills/parent-session/SKILL.md",
    MEMORY_SUMMARY,
    *(MEMORY_DIR / layer / LAYER_INDEX for layer in LAYERS),
    MEMORY_DIR / DEDICATED / "SUMMARY.md",
    MEMORY_DIR / DEDICATED / LAYER_INDEX,
    ROOT / ".github/copilot-instructions.md",
    *sorted((ROOT / ".github/agents").glob("*.md")),
]

REQUIRED_KEYS = (
    "title",
    "applies_to",
    "source",
    "created",
    "last_verified",
    "evidence",
)
APPLIES_TO = {"all", "claude", "codex"}

# Size caps keep the always-read files cheap. The values are hypotheses: revise
# them from operation (the memory skill points here as their single source).
SUMMARY_MAX_LINES = 40
INDEX_MAX_ENTRIES = 30
ENTRY_MAX_LINES = 40

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


def _layer_entries(layer: str) -> list[Path]:
    return sorted(
        p
        for p in (MEMORY_DIR / layer).glob("*.md")
        if p.name != LAYER_INDEX and (layer != DEDICATED or p.name != "SUMMARY.md")
    )


def _memory_entries() -> list[Path]:
    return [entry for layer in AREAS for entry in _layer_entries(layer)]


def _entry_id(path: Path) -> str:
    return str(path.relative_to(MEMORY_DIR))


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
    raise AssertionError(
        f"{name}: {field} must be an ISO date (YYYY-MM-DD), got {value!r}"
    )


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


def test_entry_chain_from_agents_md_to_memory_summary() -> None:
    agents_targets = {target for _, target, _ in _links(AGENTS_MD)}
    assert ".agents/README.md" in agents_targets
    assert ".agents/memory/SUMMARY.md" in agents_targets
    assert ".agents/skills/memory/SKILL.md" in agents_targets
    readme_targets = {target for _, target, _ in _links(AGENTS_README)}
    assert "memory/SUMMARY.md" in readme_targets
    summary_targets = {target for _, target, _ in _links(MEMORY_SUMMARY)}
    missing = [
        layer for layer in LAYERS if f"{layer}/{LAYER_INDEX}" not in summary_targets
    ]
    assert not missing, f"SUMMARY.md does not link the layer indexes {missing}"
    assert (ROOT / "CLAUDE.md").read_text(encoding="utf-8").splitlines()[
        0
    ] == "@AGENTS.md"


def test_deepseek_harness_skill_is_discoverable_from_both_agents() -> None:
    meta = _frontmatter(DSH_SKILL)
    name = meta.get("name")
    assert isinstance(name, str) and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name)
    assert name == DSH_SKILL.parent.name and len(name) <= 64
    description = meta.get("description")
    assert isinstance(description, str) and description.strip()
    assert len(description) <= 1024
    claude_skills = ROOT / ".claude/skills"
    assert claude_skills.is_symlink()
    assert claude_skills.resolve() == DSH_SKILL.parent.parent.resolve()
    assert (claude_skills / name / "SKILL.md").samefile(DSH_SKILL)
    agents_targets = {target for _, target, _ in _links(AGENTS_MD)}
    assert str(DSH_SKILL.relative_to(ROOT)) in agents_targets


def test_parent_board_skill_is_discoverable_from_both_agents_and_parent_workflow() -> (
    None
):
    meta = _frontmatter(BOARD_SKILL)
    assert meta["name"] == BOARD_SKILL.parent.name == "parent-board"
    assert isinstance(meta["description"], str) and meta["description"].strip()
    assert (ROOT / ".claude/skills/parent-board/SKILL.md").samefile(BOARD_SKILL)
    assert str(BOARD_SKILL.relative_to(ROOT)) in {
        target for _, target, _ in _links(AGENTS_MD)
    }
    assert "skills/parent-board/SKILL.md" in {
        target for _, target, _ in _links(AGENTS_README)
    }
    parent_skill = ROOT / ".agents/skills/parent-session/SKILL.md"
    assert "../parent-board/SKILL.md" in {
        target for _, target, _ in _links(parent_skill)
    }


def test_memory_dir_holds_only_summary_layers_and_dedicated_area() -> None:
    stray = sorted(
        p.name
        for p in MEMORY_DIR.iterdir()
        if not (p == MEMORY_SUMMARY or (p.is_dir() and p.name in AREAS))
    )
    assert not stray, (
        f"put memory entries in a layer directory, not at the top: {stray}"
    )
    for layer in AREAS:
        nested = [
            p.name for p in (MEMORY_DIR / layer).iterdir() if not p.name.endswith(".md")
        ]
        assert not nested, f"{layer}/ must hold only Markdown files: {nested}"


@pytest.mark.parametrize("layer", AREAS)
def test_layer_index_matches_entry_files(layer: str) -> None:
    index = MEMORY_DIR / layer / LAYER_INDEX
    indexed = [
        (text, target, line)
        for text, target, line in _links(index)
        if line.startswith("- ")
    ]
    indexed_files = [target for _, target, _ in indexed]
    assert len(indexed_files) == len(set(indexed_files)), (
        f"{layer}/index.md lists an entry twice"
    )
    assert set(indexed_files) == {p.name for p in _layer_entries(layer)}, (
        f"{layer}/index.md and entry files differ"
    )
    assert len(indexed) <= INDEX_MAX_ENTRIES, (
        f"{layer}/ has {len(indexed)} entries; compress it"
    )
    for text, target, line in indexed:
        meta = _frontmatter(MEMORY_DIR / layer / target)
        assert text == meta["title"], (
            f"{layer}/index.md link text for {target} must equal its title"
        )
        scope = meta["applies_to"]
        assert line.endswith(f" ({scope})") == (scope != "all"), (
            f"{layer}/index.md line for {target} must end with ' ({scope})' only when not 'all'"
        )


def test_memory_is_not_empty() -> None:
    assert any(_layer_entries(layer) for layer in LAYERS), (
        "shared memory has no entries"
    )


def test_memory_summary_stays_small() -> None:
    for summary in (MEMORY_SUMMARY, MEMORY_DIR / DEDICATED / "SUMMARY.md"):
        lines = len(summary.read_text(encoding="utf-8").splitlines())
        assert lines <= SUMMARY_MAX_LINES, f"{summary}: {lines} lines; compress it"


def test_dedicated_memory_is_not_in_the_always_loaded_summary() -> None:
    assert "ops-report" not in MEMORY_SUMMARY.read_text(encoding="utf-8")
    summary = MEMORY_DIR / DEDICATED / "SUMMARY.md"
    assert LAYER_INDEX in {target for _, target, _ in _links(summary)}


@pytest.mark.parametrize("entry", _memory_entries(), ids=_entry_id)
def test_memory_entry_frontmatter(entry: Path) -> None:
    meta = _frontmatter(entry)
    missing = [key for key in REQUIRED_KEYS if not meta.get(key)]
    assert not missing, f"{entry.name}: missing frontmatter keys {missing}"
    assert "type" not in meta, (
        f"{entry.name}: the layer directory is the type; drop the 'type' key"
    )
    lines = len(entry.read_text(encoding="utf-8").splitlines())
    assert lines <= ENTRY_MAX_LINES, (
        f"{entry.name}: {lines} lines; split or compress it"
    )
    assert meta["applies_to"] in APPLIES_TO, (
        f"{entry.name}: unknown applies_to {meta['applies_to']!r}"
    )
    created = _as_date(meta["created"], "created", entry.name)
    verified = _as_date(meta["last_verified"], "last_verified", entry.name)
    assert created <= verified <= dt.date.today(), (
        f"{entry.name}: need created <= last_verified <= today"
    )
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*\.md", entry.name), (
        f"{entry.name}: use kebab-case"
    )


@pytest.mark.parametrize(
    "entry",
    [
        MEMORY_SUMMARY,
        MEMORY_DIR / DEDICATED / "SUMMARY.md",
        *(MEMORY_DIR / layer / LAYER_INDEX for layer in AREAS),
        *_memory_entries(),
    ],
    ids=_entry_id,
)
def test_memory_entry_has_no_personal_data(entry: Path) -> None:
    text = entry.read_text(encoding="utf-8")
    for pattern in SECRET_PATTERNS:
        assert not pattern.search(text), (
            f"{entry.name}: looks like personal data or a secret ({pattern.pattern})"
        )
    assert "/home/" not in text, (
        f"{entry.name}: do not write machine-specific home paths"
    )
