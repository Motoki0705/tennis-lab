"""Regression tests for bounded Preflight and Seal review cycles."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPERS_PATH = Path(__file__).with_name("test_issue_subagent_workflow.py")


def _load_workflow_helpers() -> ModuleType:
    module_name = "_issue_subagent_workflow_review_bound_helpers"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, HELPERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_preflight_verdict(
    task: Path,
    *,
    cycle: int,
    fingerprint: str,
    verdict: str,
    findings: str,
) -> None:
    flow = _load_workflow_helpers()
    flow.write_preflight(task, cycle, fingerprint)
    path = task / "03-implementation/preflight.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "## Final production preflight verdict\nPASS",
        f"## Final production preflight verdict\n{verdict}",
    )
    text = text.replace(
        "## RETURN implementation findings\nNone",
        f"## RETURN implementation findings\n{findings}",
    )
    path.write_text(text, encoding="utf-8")


def _write_seal_verdict(
    task: Path,
    *,
    cycle: int,
    fingerprint: str,
    verdict: str,
    findings: str,
) -> None:
    flow = _load_workflow_helpers()
    flow.write_seal(task, cycle, fingerprint)
    path = task / "03-implementation/seal.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "## Final candidate seal verdict\nPASS",
        f"## Final candidate seal verdict\n{verdict}",
    )
    text = text.replace(
        "## RETURN implementation findings\nNone",
        f"## RETURN implementation findings\n{findings}",
    )
    path.write_text(text, encoding="utf-8")


def test_second_consecutive_preflight_return_requires_strategy_review(
    tmp_path: Path,
) -> None:
    flow = _load_workflow_helpers()
    root, task = flow.setup_task(tmp_path)
    flow.advance_to_implementation(root, task)
    flow.write_implementation(task, 1)

    first_candidate = flow.candidate.compute_candidate_fingerprint(
        task,
        flow.manage.load_state(task),
    )
    _write_preflight_verdict(
        task,
        cycle=1,
        fingerprint=first_candidate,
        verdict="RETURN",
        findings="AC-001: first bounded finding.",
    )
    flow.manage.apply_preflight_verdict(task, "RETURN")
    first_state = flow.manage.load_state(task)
    assert first_state["preflight_cycle"] == 1
    assert first_state["preflight_verdict"] == "RETURN"
    assert first_state["return_review_required"] is False

    (root / "src.txt").write_text("repaired once\n", encoding="utf-8")
    flow.write_implementation(task, 1)
    closure_candidate = flow.candidate.compute_candidate_fingerprint(
        task,
        flow.manage.load_state(task),
    )
    _write_preflight_verdict(
        task,
        cycle=1,
        fingerprint=closure_candidate,
        verdict="RETURN",
        findings="AC-001: frozen finding remains open.",
    )
    flow.manage.apply_preflight_verdict(task, "RETURN")
    closure_state = flow.manage.load_state(task)
    assert closure_state["return_review_required"] is True
    assert flow.manage.check(task) == []

    with pytest.raises(ValueError, match="return review is required"):
        flow.manage.apply_preflight_verdict(task, "RETURN")

    flow.manage.apply_return_review(
        task,
        "implementation",
        "Preflight closure remained RETURN; redesign the frozen checks.",
    )
    reviewed_state = flow.manage.load_state(task)
    assert reviewed_state["phase"] == "implementation"
    assert reviewed_state["test_cycle"] == 0
    assert reviewed_state["preflight_cycle"] == 0
    assert reviewed_state["preflight_verdict"] == ""
    assert reviewed_state["test_verdict"] == ""
    assert reviewed_state["seal_verdict"] == ""
    assert reviewed_state["return_review_required"] is False
    assert reviewed_state["return_review_action"] == "implementation"


def test_seal_return_requires_strategy_review_before_another_cycle(
    tmp_path: Path,
) -> None:
    flow = _load_workflow_helpers()
    root, task = flow.setup_task(tmp_path)
    flow.advance_to_implementation(root, task)
    flow.write_implementation(task, 1)
    candidate = flow.candidate.compute_candidate_fingerprint(
        task,
        flow.manage.load_state(task),
    )

    flow.write_preflight(task, 1, candidate)
    assert flow.manage.run_check(task, "preflight", "py-ok") == 0
    flow.manage.apply_preflight_verdict(task, "PASS")
    flow.write_tests(task, 1, candidate)
    assert flow.manage.run_check(task, "test", "py-ok") == 0
    flow.manage.apply_test_verdict(task, "PASS")

    _write_seal_verdict(
        task,
        cycle=1,
        fingerprint=candidate,
        verdict="RETURN",
        findings="Approved-scope evidence is incomplete.",
    )
    flow.manage.apply_seal_verdict(task, "RETURN")
    seal_state = flow.manage.load_state(task)
    assert seal_state["seal_verdict"] == "RETURN"
    assert seal_state["return_review_required"] is True
    assert flow.manage.check(task) == []

    with pytest.raises(ValueError, match="return review is required"):
        flow.manage.apply_seal_verdict(task, "RETURN")

    flow.manage.apply_return_review(
        task,
        "implementation",
        "Seal evidence gap requires a fresh production/test cycle.",
    )
    reviewed_state = flow.manage.load_state(task)
    assert reviewed_state["phase"] == "implementation"
    assert reviewed_state["test_cycle"] == 1
    assert reviewed_state["preflight_verdict"] == "PASS"
    assert reviewed_state["test_verdict"] == "PASS"
    assert reviewed_state["seal_verdict"] == "RETURN"
    assert reviewed_state["return_review_required"] is False

    flow.write_seal(task, 1, candidate)
    assert flow.manage.run_check(task, "seal", "py-ok") == 0
    flow.manage.apply_seal_verdict(task, "PASS")
    assert flow.manage.load_state(task)["seal_verdict"] == "PASS"


def test_packaging_repair_reopens_validated_task_and_invalidates_evidence(
    tmp_path: Path,
) -> None:
    flow = _load_workflow_helpers()
    root, task = flow.setup_task(tmp_path)
    candidate = flow.advance_to_validation(root, task)
    flow.write_validation(task, candidate)
    flow.manage.apply_validation_verdict(task, "PASS")
    validated = flow.manage.load_state(task)
    validated.update(
        {
            "packaging_candidate_sha256": candidate,
            "pr_number": 808,
            "pr_head_sha": "a" * 40,
            "remote_checks_verdict": "PASS",
            "pr_evidence_sha256": "sha256:" + "b" * 64,
        }
    )
    flow.manage._state.write_state(task, validated)
    frozen_artifacts = {
        relative: (task / relative).read_bytes()
        for relative in (
            "issue.json",
            "issue.md",
            "00-feasibility/feasibility.md",
            "01-exploration/exploration.md",
            "02-planning/plan.md",
            "02-planning/checks.json",
        )
    }

    (root / "src.txt").write_text("packaging repair\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / ".agents/skills/issue-subagent-workflow/scripts/manage_issue_task.py"
            ),
            "packaging-repair",
            str(task),
            "--reason",
            "  Remote CI exposed a content defect after Validator PASS.  ",
        ],
        cwd=root,
        check=True,
    )

    reopened = flow.manage.load_state(task)
    assert reopened["phase"] == "implementation"
    assert reopened["status"] == "in_progress"
    assert reopened["verdict"] == ""
    assert reopened["attempt"] == validated["attempt"]
    assert reopened["base_revision"] == validated["base_revision"]
    assert reopened["issue_sha256"] == validated["issue_sha256"]
    assert reopened["preflight_cycle"] == 0
    assert reopened["test_cycle"] == 0
    assert reopened["seal_cycle"] == 0
    assert reopened["preflight_verdict"] == ""
    assert reopened["test_verdict"] == ""
    assert reopened["seal_verdict"] == ""
    assert reopened["preflight_candidate_sha256"] == ""
    assert reopened["test_candidate_sha256"] == ""
    assert reopened["sealed_candidate_sha256"] == ""
    assert reopened["validation_candidate_sha256"] == ""
    assert reopened["packaging_candidate_sha256"] == ""
    assert reopened["pr_number"] == 0
    assert reopened["pr_head_sha"] == ""
    assert reopened["remote_checks_verdict"] == ""
    assert reopened["pr_evidence_sha256"] == ""
    assert reopened["return_review_action"] == "implementation"
    assert reopened["return_review_reason"] == (
        "Remote CI exposed a content defect after Validator PASS."
    )
    assert {
        relative: (task / relative).read_bytes() for relative in frozen_artifacts
    } == frozen_artifacts
    assert flow.manage.check(task) == []


def test_packaging_repair_rejects_non_validated_state(tmp_path: Path) -> None:
    flow = _load_workflow_helpers()
    _, task = flow.setup_task(tmp_path)

    with pytest.raises(ValueError, match="requires packaging/validated state"):
        flow.manage.reopen_packaging_repair(task, "Not yet validated.")


def test_reviewer_contracts_freeze_retry_scope_and_keep_seal_narrow() -> None:
    preflight = tomllib.loads(
        (ROOT / ".codex/agents/preflight-reviewer.toml").read_text(encoding="utf-8")
    )["developer_instructions"]
    seal = tomllib.loads(
        (ROOT / ".codex/agents/seal-reviewer.toml").read_text(encoding="utf-8")
    )["developer_instructions"]
    skill = (
        ROOT / ".agents/skills/issue-subagent-workflow/SKILL.md"
    ).read_text(encoding="utf-8")

    assert "Discovery mode" in preflight
    assert "Closure mode" in preflight
    assert "Do not invent a new mutation category" in preflight
    assert "explicitly frozen in `plan.md`" in preflight
    assert "second consecutive Preflight RETURN requires `return-review`" in skill
    assert "Do not conduct new semantic mutation testing" in seal
    assert "any RETURN requires the parent to run `return-review`" in seal
