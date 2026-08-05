from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/issue-subagent-workflow/scripts"
HELPERS = ROOT / "tests/e2e/skills/test_issue_subagent_workflow.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manage = load_module("manage_issue_task_completion", SCRIPTS / "manage_issue_task.py")
helpers = load_module("issue_workflow_test_helpers", HELPERS)


def prepare_validation(tmp_path: Path) -> Path:
    task_dir = helpers.write_task(tmp_path)
    helpers.pass_preflight(task_dir, 1)
    helpers.write_tests_cycle(task_dir, 1, "PASS")
    manage.apply_test_verdict(task_dir, "PASS")
    manage.transition(task_dir, "validation")
    helpers.write_validation(task_dir, ac2_verdict="PASS", final_verdict="PASS")
    return task_dir


def test_artifact_check_catches_missing_exploration_heading(tmp_path: Path) -> None:
    task_dir = prepare_validation(tmp_path)
    exploration = task_dir / "01-exploration/exploration.md"
    exploration.write_text(
        exploration.read_text(encoding="utf-8").replace(
            "## Relevant files and symbols",
            "## Relevant files, symbols, and entry points",
        ),
        encoding="utf-8",
    )

    errors = manage.check_artifact(task_dir, "exploration")
    assert errors == [
        "01-exploration/exploration.md is missing heading: "
        "## Relevant files and symbols"
    ]


def test_validator_pass_does_not_mutate_state_when_precheck_fails(
    tmp_path: Path,
) -> None:
    task_dir = prepare_validation(tmp_path)
    exploration = task_dir / "01-exploration/exploration.md"
    exploration.write_text(
        exploration.read_text(encoding="utf-8").replace(
            "## Existing tests and fixtures",
            "## Tests and fixtures",
        ),
        encoding="utf-8",
    )
    state_path = task_dir / "state.toml"
    state_before = state_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="pre-completion check failed"):
        manage.apply_validation_verdict(task_dir, "PASS")

    assert state_path.read_text(encoding="utf-8") == state_before
    state = manage.load_state(task_dir)
    assert state["phase"] == "validation"
    assert state["status"] == "in_progress"
    assert state["verdict"] == ""


def test_validator_pass_completes_after_all_artifacts_validate(tmp_path: Path) -> None:
    task_dir = prepare_validation(tmp_path)
    assert manage.check_artifact(task_dir, "validation") == []

    manage.apply_validation_verdict(task_dir, "PASS")

    state = manage.load_state(task_dir)
    assert state["status"] == "complete"
    assert state["verdict"] == "PASS"
    assert manage.check(task_dir) == []


def test_completion_contract_documents_user_topology_and_command_authority() -> None:
    contract = (
        ROOT
        / ".agents/skills/issue-subagent-workflow/references/completion-hardening.md"
    ).read_text(encoding="utf-8")
    assert "An explicit user request to use one Implementer" in contract
    assert "Do not emit Tester RETURN" in contract
    assert "Artifact validation precedes state mutation" in contract
