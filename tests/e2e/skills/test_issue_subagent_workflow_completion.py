from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def load_agent(filename: str) -> dict[str, Any]:
    path = ROOT / ".codex/agents" / filename
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_user_directed_single_implementer_is_explicitly_compliant() -> None:
    skill = (ROOT / ".agents/skills/issue-subagent-workflow/SKILL.md").read_text(
        encoding="utf-8"
    )
    workflow = (
        ROOT / ".agents/skills/issue-subagent-workflow/references/workflow.md"
    ).read_text(encoding="utf-8")
    assert "one Implementer or sequential execution is compliant" in skill
    assert (
        "Parallelism is a latency optimization, not an acceptance criterion" in workflow
    )


def test_default_implementer_does_not_own_shared_artifacts() -> None:
    payload = load_agent("issue-implementer.toml")
    instructions = payload["developer_instructions"]
    assert "By default, do not write shared workflow artifacts" in instructions
    assert "artifact_integrator = true" in instructions
    assert "replace only `implementation.md`" in instructions
    assert "Never write `preflight.md` or `seal.md`" in instructions


def test_reviewer_sequence_is_explicit() -> None:
    skill = (ROOT / ".agents/skills/issue-subagent-workflow/SKILL.md").read_text(
        encoding="utf-8"
    )
    stages = (
        "discovery `preflight_reviewer`",
        "independent adversarial Test Writer",
        "`seal_reviewer` with no source/test edits",
        "The Validator receives",
    )
    offsets = [skill.index(stage) for stage in stages]
    assert offsets == sorted(offsets)


def test_dedicated_reviewers_own_only_gate_evidence() -> None:
    cases = (
        (
            "preflight-reviewer.toml",
            "preflight_reviewer",
            "preflight.md",
            "preflight-checks.json",
            "run-check <task-dir> preflight <check-id>",
            "artifact-check <task-dir> preflight",
            "preflight-verdict",
        ),
        (
            "seal-reviewer.toml",
            "seal_reviewer",
            "seal.md",
            "seal-checks.json",
            "run-check <task-dir> seal <check-id>",
            "artifact-check <task-dir> seal",
            "seal-verdict",
        ),
    )
    for filename, name, artifact, results, run_check, artifact_check, verdict in cases:
        payload = load_agent(filename)
        instructions = payload["developer_instructions"]
        assert payload["name"] == name
        assert payload["model"] == "gpt-5.6-luna"
        assert payload["model_reasoning_effort"] == "xhigh"
        assert payload["sandbox_mode"] == "workspace-write"
        assert f"only authored Markdown artifact is `{artifact}`" in instructions
        assert results in instructions
        assert run_check in instructions
        assert artifact_check in instructions
        assert f"Do not call `{verdict}`" in instructions
        assert "Do not modify production code" in instructions
        assert "Communication mode: terminal-only." in instructions


def test_test_writer_and_validator_preserve_independent_gates() -> None:
    tester = load_agent("test-writer.toml")["developer_instructions"]
    validator = load_agent("issue-validator.toml")["developer_instructions"]
    assert "Never modify production code" in tester
    assert "run-check <task-dir> test <check-id>" in tester
    assert "mandatory minimum coverage only" in tester
    assert "run-test-probe <task-dir> <AT-NNN>" in tester
    assert "sole task specification" in validator
    assert "sealed candidate fingerprint" in validator


def test_contracts_are_machine_backed_not_instruction_only() -> None:
    schema = (
        ROOT / ".agents/skills/issue-subagent-workflow/scripts/issue_task_schema.py"
    ).read_text(encoding="utf-8")
    state = (
        ROOT / ".agents/skills/issue-subagent-workflow/scripts/issue_task_state.py"
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
    assert "CURRENT_SCHEMA_VERSION = 6" in state
    assert 'state["status"] = "validated"' in transitions
    assert "compute_revision_fingerprint" in finalization
