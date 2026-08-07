from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AGENTS_DIR = ROOT / ".codex" / "agents"
SKILL_DIR = ROOT / ".agents" / "skills" / "issue-subagent-workflow"

EXPECTED_MODELS = {
    "codebase-scout.toml": "gpt-5.6-luna",
    "codebase-explorer.toml": "gpt-5.6-terra",
    "issue-implementer.toml": "gpt-5.6-sol",
    "test-writer.toml": "gpt-5.6-sol",
    "issue-validator.toml": "gpt-5.6-sol",
}

TERMINAL_ONLY_SENTINELS = (
    "Communication mode: terminal-only.",
    "Do not send commentary, milestone, percentage, command-in-progress, "
    "or routine progress messages to the parent.",
    "Return exactly one compact final handoff when the assignment is complete.",
)


def load_agent(filename: str) -> dict[str, object]:
    with (AGENTS_DIR / filename).open("rb") as handle:
        return tomllib.load(handle)


def test_custom_agents_use_canonical_model_slugs() -> None:
    actual: dict[str, str] = {}

    for filename, expected_model in EXPECTED_MODELS.items():
        config = load_agent(filename)
        model = config.get("model")
        assert isinstance(model, str)
        assert model == expected_model
        actual[filename] = model

    assert "gpt-5.6" not in actual.values()


def test_custom_agents_enforce_terminal_only_parent_communication() -> None:
    for filename in EXPECTED_MODELS:
        instructions = load_agent(filename).get("developer_instructions")
        assert isinstance(instructions, str)
        for sentinel in TERMINAL_ONLY_SENTINELS:
            assert sentinel in instructions, filename


def test_workflow_requires_terminal_only_spawn_and_event_driven_waiting() -> None:
    skill = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    spawn_contracts = (SKILL_DIR / "references/spawn-contracts.md").read_text(
        encoding="utf-8"
    )
    workflow = (SKILL_DIR / "references/workflow.md").read_text(encoding="utf-8")

    assert "exact terminal-only footer" in skill
    assert "timeout_ms = 3_600_000" in skill
    assert "Treat only `FINAL_ANSWER` as completion." in spawn_contracts
    assert "Do not use shorter waits as polling intervals" in spawn_contracts
    assert "Do not pair waiting with repeated `list_agents`" in workflow

    for sentinel in TERMINAL_ONLY_SENTINELS:
        assert sentinel in spawn_contracts
