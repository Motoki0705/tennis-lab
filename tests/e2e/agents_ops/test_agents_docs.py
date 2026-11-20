"""Validate the AI-operations entry points (AGENTS.md -> memory SUMMARY -> layer index -> entry).

The shared memory is meant to be cheap to edit and delete, so these checks keep
the summary, the layer indexes, the entry files and the cross-document links from
silently drifting, and keep the always-read files small.
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import re
import shlex
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops"))
memory_schema = importlib.import_module("shared_memory.schema")
memory_checks = importlib.import_module("shared_memory.validate")

AGENTS_MD = ROOT / "AGENTS.md"
AGENTS_README = ROOT / ".agents/README.md"
MEMORY_DIR = ROOT / memory_schema.MEMORY_DIR
MEMORY_SUMMARY = ROOT / memory_schema.SUMMARY_PATH
MEMORY_SKILL = ROOT / ".agents/skills/memory/SKILL.md"
DSH_SKILL = ROOT / ".agents/skills/deepseek-harness/SKILL.md"
BOARD_SKILL = ROOT / ".agents/skills/parent-board/SKILL.md"
LAYERS = memory_schema.SHARED_LAYERS
DEDICATED_DIR = MEMORY_DIR / memory_schema.DEDICATED_AREA
LAYER_INDEX = memory_schema.INDEX_NAME

DOCS = [
    AGENTS_MD,
    ROOT / "CLAUDE.md",
    AGENTS_README,
    MEMORY_SKILL,
    DSH_SKILL,
    BOARD_SKILL,
    DSH_SKILL.with_name("task-template.md"),
    ROOT / ".agents/ops/parent/README.md",
    ROOT / ".agents/ops/parent/worker_prompt.md",
    ROOT / ".agents/ops/worker_logs/README.md",
    ROOT / ".agents/ops/shared_memory/README.md",
    ROOT / ".agents/ops/decision_logs/README.md",
    ROOT / ".agents/ops/decision_logs/README.ja.md",
    ROOT / ".agents/ops/decision_logs/README.zh-CN.md",
    ROOT / ".agents/ops/lib/README.md",
    ROOT / ".agents/ops/review/README.md",
    ROOT / ".agents/ops/dispatcher/parent_bootstrap.md",
    *sorted((ROOT / ".agents/skills/parent-session/references").glob("*.md")),
    ROOT / ".agents/ops/board/README.md",
    ROOT / ".agents/ops/herdr_labels/README.md",
    ROOT / ".agents/skills/parent-session/SKILL.md",
    ROOT / ".agents/skills/herdr-session-label/SKILL.md",
    ROOT / ".agents/skills/explain-html/SKILL.md",
    ROOT / ".agents/skills/explain-html/references/authoring.md",
    ROOT / ".agents/skills/explain-html/references/claim-review.md",
    DEDICATED_DIR / "SUMMARY.md",
    DEDICATED_DIR / LAYER_INDEX,
    ROOT / ".github/copilot-instructions.md",
    *sorted((ROOT / ".github/agents").glob("*.md")),
]

DEDICATED_KEYS = (
    "title",
    "applies_to",
    "source",
    "created",
    "last_verified",
    "evidence",
)
DEDICATED_APPLIES_TO = {"all", "claude", "codex"}
DEDICATED_SUMMARY_MAX_LINES = 40
DEDICATED_INDEX_MAX_ENTRIES = 30
DEDICATED_ENTRY_MAX_LINES = 40

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


def _dedicated_entries() -> list[Path]:
    return sorted(
        path
        for path in DEDICATED_DIR.glob("*.md")
        if path.name not in {LAYER_INDEX, "SUMMARY.md"}
    )


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


def test_worker_wait_is_reachable_from_the_operational_entry_points() -> None:
    workers = ROOT / ".agents/skills/parent-session/references/workers.md"
    assert {"開発ルール", "学習の実行"} <= _anchors(AGENTS_MD)
    for doc in (AGENTS_MD, AGENTS_README):
        assert any(
            (doc.parent / target.partition("#")[0]).resolve() == workers
            and target.endswith("#ciローカルqueueの待機")
            for _, target, _ in _links(doc)
        )
    gotcha = "queue-worker-reaped-with-agent-exec-child.md"
    assert not (MEMORY_DIR / "gotcha" / gotcha).exists()
    assert gotcha not in (MEMORY_DIR / "gotcha/index.md").read_text(encoding="utf-8")


def test_local_queue_policy_has_one_operational_source_and_preserves_colab() -> None:
    legacy = "workers-do-not-wait-on-queue-jobs.md"
    assert not (MEMORY_DIR / "decision" / legacy).exists()
    assert legacy not in {
        target for _, target, _ in _links(MEMORY_DIR / "decision/index.md")
    }
    assert "親が完了イベントで再開" not in MEMORY_SUMMARY.read_text(encoding="utf-8")
    workers = ROOT / ".agents/skills/parent-session/references/workers.md"
    assert {"ciローカルqueueの待機", "colabの長時間ジョブ"} <= _anchors(workers)
    assert (
        ".agents/skills/parent-session/references/workers.md#colabの長時間ジョブ"
        in {target for _, target, _ in _links(AGENTS_MD)}
    )
    assert "../../tennis-colab/SKILL.md" in {target for _, target, _ in _links(workers)}


def test_parent_cli_help_examples_match_the_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
    cli = importlib.import_module("parent.cli_impl")
    guidance = importlib.import_module("parent.cli_help")
    service = importlib.import_module("herdr_labels.service")
    root = cli.parser()
    for name, example in guidance.EXAMPLES.items():
        tokens = shlex.split(example.splitlines()[0])
        args = root.parse_args(["--parent", "1216", *tokens])
        assert args.command == name
        if name in {"register", "handoff"}:
            assert service.validate_summary(args.summary) == args.summary
        if name == "review-request":
            assert args.review_model is None and args.model_reason_file is None


def test_parent_cli_completion_examples_cover_the_agreement_and_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
    examples = importlib.import_module("parent.cli_help").EXAMPLES
    notifications = importlib.import_module("parent.notifications")
    completion = importlib.import_module("parent.completion")
    notice = json.loads(examples["notify"].splitlines()[-1])
    assert notifications.validate(notice)["kind"] == "completion"
    finish = json.loads(examples["release"].splitlines()[-1])
    assert finish.pop("evidence") and finish["schema_version"] == 2
    finish["schema_version"] = 1
    requirements = {
        "basis_ref": finish["basis_ref"],
        "criteria": [{"id": "agreement"}, {"id": "ac-1"}],
    }
    completion.validate_finish(finish, requirements, {}, 1216)


def test_skill_routes_each_cli_topic_to_a_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / ".agents/ops"))
    topics = importlib.import_module("parent.cli_help").TOPICS
    skill = ROOT / ".agents/skills/parent-session/SKILL.md"
    references = {
        path for _, path, _ in _links(skill) if path.startswith("references/")
    }
    assert references == {"references/" + name + ".md" for name in topics}
    assert references == {
        str(p.relative_to(skill.parent))
        for p in (skill.parent / "references").glob("*.md")
    }


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


def test_decision_record_entry_is_reachable_from_operational_docs() -> None:
    specification = ROOT / ".agents/ops/decision_logs/README.ja.md"
    for name in (
        ".agents/skills/parent-session/references/finish.md",
        ".agents/ops/parent/README.md",
        ".agents/ops/worker_logs/README.md",
        ".agents/ops/lib/README.md",
    ):
        doc = ROOT / name
        assert any(
            (doc.parent / target.partition("#")[0]).resolve() == specification
            for _, target, _ in _links(doc)
        )


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


def test_parent_session_skill_is_discoverable_from_agents() -> None:
    parent_skill = ROOT / ".agents/skills/parent-session/SKILL.md"
    assert str(parent_skill.relative_to(ROOT)) in {
        target for _, target, _ in _links(AGENTS_MD)
    }
    assert parent_skill.is_file()


def test_shared_memory_matches_the_generated_contract() -> None:
    problems = memory_checks.validate(ROOT)
    assert not problems, "\n".join(
        f"{problem.path}: [{problem.code}] {problem.message}" for problem in problems
    )


def test_dino_polling_observation_describes_the_measured_stop() -> None:
    path = MEMORY_DIR / "environment/wsl2-nvidia-smi-polling-stalls-gpu.md"
    meta = _frontmatter(path)
    title, summary = meta["title"], meta["summary"]
    assert isinstance(title, str) and isinstance(summary, str)
    assert all(
        term in summary for term in ("タイムアウト", "監視", "0.17秒前", "heartbeat")
    )
    assert all(term not in title + summary for term in ("画面", "フリーズ", "固ま"))
    body = path.read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "235/235" in body and "53フレーム目" in body
    assert "0.17秒前" in body and "過去のGPU停止の原因は未特定" in body
    assert f"- {summary}\n" in MEMORY_SUMMARY.read_text(encoding="utf-8")


def test_shared_memory_is_fresh() -> None:
    report = memory_checks.audit(ROOT)
    assert not report.stale, "\n".join(
        f"{entry.path}: {entry.age_days} days since {entry.last_verified}"
        for entry in report.stale
    )


def test_dedicated_memory_dir_holds_only_markdown_files() -> None:
    stray = [
        path.name
        for path in DEDICATED_DIR.iterdir()
        if path.suffix != ".md" or not path.is_file()
    ]
    assert not stray, f"{DEDICATED_DIR.name}/ must hold only Markdown files: {stray}"


def test_dedicated_memory_index_matches_entry_files() -> None:
    index = DEDICATED_DIR / LAYER_INDEX
    indexed = [
        (text, target, line)
        for text, target, line in _links(index)
        if line.startswith("- ")
    ]
    indexed_files = [target for _, target, _ in indexed]
    assert len(indexed_files) == len(set(indexed_files)), (
        "ops-report/index.md lists an entry twice"
    )
    assert set(indexed_files) == {path.name for path in _dedicated_entries()}, (
        "ops-report/index.md and entry files differ"
    )
    assert len(indexed) <= DEDICATED_INDEX_MAX_ENTRIES, (
        f"ops-report/ has {len(indexed)} entries; compress it"
    )
    for text, target, line in indexed:
        meta = _frontmatter(DEDICATED_DIR / target)
        assert text == meta["title"], (
            f"ops-report/index.md link text for {target} must equal its title"
        )
        scope = meta["applies_to"]
        assert line.endswith(f" ({scope})") == (scope != "all"), (
            f"ops-report/index.md line for {target} must end with ' ({scope})' only when not 'all'"
        )


def test_memory_is_not_empty() -> None:
    assert any(
        path.name != LAYER_INDEX
        for layer in LAYERS
        for path in (MEMORY_DIR / layer).glob("*.md")
    ), "shared memory has no entries"


def test_dedicated_memory_summary_stays_small() -> None:
    summary = DEDICATED_DIR / "SUMMARY.md"
    lines = len(summary.read_text(encoding="utf-8").splitlines())
    assert lines <= DEDICATED_SUMMARY_MAX_LINES, (
        f"{summary}: {lines} lines; compress it"
    )


def test_dedicated_memory_is_not_in_the_always_loaded_summary() -> None:
    assert "ops-report" not in MEMORY_SUMMARY.read_text(encoding="utf-8")
    summary = DEDICATED_DIR / "SUMMARY.md"
    assert LAYER_INDEX in {target for _, target, _ in _links(summary)}


def test_dedicated_memory_entry_frontmatter() -> None:
    for entry in _dedicated_entries():
        meta = _frontmatter(entry)
        missing = [key for key in DEDICATED_KEYS if not meta.get(key)]
        assert not missing, f"{entry.name}: missing frontmatter keys {missing}"
        assert "type" not in meta, (
            f"{entry.name}: the layer directory is the type; drop the 'type' key"
        )
        lines = len(entry.read_text(encoding="utf-8").splitlines())
        assert lines <= DEDICATED_ENTRY_MAX_LINES, (
            f"{entry.name}: {lines} lines; split or compress it"
        )
        assert meta["applies_to"] in DEDICATED_APPLIES_TO, (
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


def test_dedicated_memory_has_no_personal_data() -> None:
    entries = [
        DEDICATED_DIR / "SUMMARY.md",
        DEDICATED_DIR / LAYER_INDEX,
        *_dedicated_entries(),
    ]
    for entry in entries:
        text = entry.read_text(encoding="utf-8")
        for pattern in SECRET_PATTERNS:
            assert not pattern.search(text), (
                f"{entry.name}: looks like personal data or a secret ({pattern.pattern})"
            )
        assert "/home/" not in text, (
            f"{entry.name}: do not write machine-specific home paths"
        )
