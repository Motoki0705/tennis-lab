from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_user_directed_single_implementer_is_explicitly_compliant() -> None:
    skill = (
        ROOT / ".agents/skills/issue-subagent-workflow/SKILL.md"
    ).read_text(encoding="utf-8")
    workflow = (
        ROOT / ".agents/skills/issue-subagent-workflow/references/workflow.md"
    ).read_text(encoding="utf-8")
    assert "one Implementer or sequential execution is compliant" in skill
    assert "Parallelism is a latency optimization, not an acceptance criterion" in workflow


def test_default_implementer_does_not_own_shared_artifacts() -> None:
    path = ROOT / ".codex/agents/issue-implementer.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    instructions = payload["developer_instructions"]
    assert "By default, do not write shared workflow artifacts" in instructions
    assert "artifact_integrator = true" in instructions
    assert "Only when the spawn message explicitly grants" in instructions


def test_test_writer_and_validator_preserve_independent_gates() -> None:
    tester = tomllib.loads(
        (ROOT / ".codex/agents/test-writer.toml").read_text(encoding="utf-8")
    )["developer_instructions"]
    validator = tomllib.loads(
        (ROOT / ".codex/agents/issue-validator.toml").read_text(encoding="utf-8")
    )["developer_instructions"]
    assert "Never modify production code" in tester
    assert "run-check <task-dir> test <check-id>" in tester
    assert "sole task specification" in validator
    assert "sealed candidate fingerprint" in validator


def test_contracts_are_machine_backed_not_instruction_only() -> None:
    schema = (
        ROOT
        / ".agents/skills/issue-subagent-workflow/scripts/issue_task_schema.py"
    ).read_text(encoding="utf-8")
    state = (
        ROOT
        / ".agents/skills/issue-subagent-workflow/scripts/issue_task_state.py"
    ).read_text(encoding="utf-8")
    finalization = (
        ROOT
        / ".agents/skills/issue-subagent-workflow/scripts/issue_task_finalization.py"
    ).read_text(encoding="utf-8")
    transitions = (
        ROOT
        / ".agents/skills/issue-subagent-workflow/scripts/issue_task_transitions.py"
    ).read_text(encoding="utf-8")
    assert "ARTIFACT_CONTRACTS" in schema
    assert 'CURRENT_SCHEMA_VERSION = 5' in state
    assert 'state["status"] = "validated"' in transitions
    assert "compute_revision_fingerprint" in finalization
