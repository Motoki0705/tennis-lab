from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/issue-subagent-workflow/scripts"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manage = load_module("manage_issue_task", SCRIPTS / "manage_issue_task.py")


def checklist_digest(items: list[tuple[str, str]]) -> str:
    canonical = json.dumps(
        [{"id": item_id, "text": text} for item_id, text in items],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_task(tmp_path: Path, *, phase: str = "implementation") -> Path:
    task_dir = tmp_path / "issue-1"
    for relative in (
        "01-exploration",
        "02-planning",
        "03-implementation",
        "04-validation",
    ):
        (task_dir / relative).mkdir(parents=True, exist_ok=True)

    items = [("AC-001", "Observable behavior"), ("AC-002", "Regression is covered")]
    digest = checklist_digest(items)
    (task_dir / "issue.md").write_text(
        """# GitHub Issue #1

## Acceptance checklist

- AC-001: Observable behavior (source checkbox: unchecked)
- AC-002: Regression is covered (source checkbox: unchecked)

The source checkbox state is metadata only.

## Title

Example

## Body

Example body
""",
        encoding="utf-8",
    )
    (task_dir / "state.toml").write_text(
        "\n".join(
            (
                "schema_version = 3",
                "issue_number = 1",
                'issue_url = "https://example.test/issues/1"',
                'issue_sha256 = "issue"',
                f'acceptance_checklist_sha256 = "{digest}"',
                "acceptance_checklist_count = 2",
                "attempt = 1",
                "test_cycle = 0",
                'test_verdict = ""',
                f'phase = "{phase}"',
                'status = "in_progress"',
                'verdict = ""',
                'updated_at = "2026-08-04T00:00:00+00:00"',
                "",
            )
        ),
        encoding="utf-8",
    )
    (task_dir / "01-exploration/exploration.md").write_text(
        """# Exploration

- Issue: #1
- Attempt: 1
- Status: COMPLETE

## Relevant files and symbols
Done
## Entry points and execution paths
Done
## Existing tests and fixtures
Done
## Unresolved questions
None
## Evidence table
Done
""",
        encoding="utf-8",
    )
    (task_dir / "02-planning/plan.md").write_text(
        f"""# Plan

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `{digest}`

## Acceptance checklist mapping

| ID | Issue checklist item | Planned implementation | Validation method |
|---|---|---|---|
| AC-001 | Observable behavior | Change | Test |
| AC-002 | Regression is covered | Change | Test |

## Implementation work units and ownership
Done
## Independent test work unit
Done
## Validation strategy
Done
""",
        encoding="utf-8",
    )
    return task_dir


def write_implementation_cycle(task_dir: Path, cycle: int) -> None:
    (task_dir / "03-implementation/implementation.md").write_text(
        f"""# Implementation

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE

## Files and symbols changed
Done
## Behavior implemented
Done
## Commands and results
Done
## Handoff
Done
""",
        encoding="utf-8",
    )


def write_tests_cycle(task_dir: Path, cycle: int, verdict: str) -> None:
    state = manage.load_state(task_dir)
    digest = state["acceptance_checklist_sha256"]
    findings = (
        "Fix the production branch for the failing regression."
        if verdict == "RETURN"
        else ""
    )
    result = "FAIL" if verdict == "RETURN" else "PASS"
    (task_dir / "03-implementation/tests.md").write_text(
        f"""# Tests

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `{digest}`

## Acceptance-checklist-to-test mapping

| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | Observable behavior | test_behavior | PASS |
| AC-002 | Regression is covered | test_regression | {result} |

## Tests added or changed
Done
## Commands and exact outcomes
Done
## Final test verdict

{verdict}

## RETURN implementation findings

{findings}
""",
        encoding="utf-8",
    )


def test_test_return_reenters_implementer_before_validation(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path)
    write_implementation_cycle(task_dir, 1)
    write_tests_cycle(task_dir, 1, "RETURN")

    manage.apply_test_verdict(task_dir, "RETURN")
    state = manage.load_state(task_dir)
    assert state["phase"] == "implementation"
    assert state["test_cycle"] == 1
    assert state["test_verdict"] == "RETURN"

    with pytest.raises(ValueError, match="test-verdict PASS"):
        manage.transition(task_dir, "validation")

    write_implementation_cycle(task_dir, 2)
    write_tests_cycle(task_dir, 2, "PASS")
    manage.apply_test_verdict(task_dir, "PASS")
    manage.transition(task_dir, "validation")

    state = manage.load_state(task_dir)
    assert state["phase"] == "validation"
    assert state["test_cycle"] == 2
    assert state["test_verdict"] == "PASS"


def test_plan_requires_exact_ordered_mapping_rows(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path, phase="planning")
    plan_path = task_dir / "02-planning/plan.md"
    text = plan_path.read_text(encoding="utf-8")
    text = text.replace(
        "| AC-001 | Observable behavior | Change | Test |\n"
        "| AC-002 | Regression is covered | Change | Test |",
        "AC-001 and AC-002 are mentioned in prose, but the table is empty.",
    )
    plan_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="missing checklist IDs"):
        manage.transition(task_dir, "implementation")


def test_return_requires_actionable_findings(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path)
    write_implementation_cycle(task_dir, 1)
    write_tests_cycle(task_dir, 1, "RETURN")
    tests_path = task_dir / "03-implementation/tests.md"
    tests_path.write_text(
        tests_path.read_text(encoding="utf-8").replace(
            "Fix the production branch for the failing regression.",
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="concrete RETURN implementation findings"):
        manage.apply_test_verdict(task_dir, "RETURN")


def write_validation(task_dir: Path, *, ac2_verdict: str, final_verdict: str) -> None:
    state = manage.load_state(task_dir)
    digest = state["acceptance_checklist_sha256"]
    (task_dir / "04-validation/validation.md").write_text(
        f"""# Validation

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `{digest}`

## Acceptance checklist verification

| ID | Issue checklist item | Verdict | Evidence |
|---|---|---|---|
| AC-001 | Observable behavior | PASS | command |
| AC-002 | Regression is covered | {ac2_verdict} | command |

## Code evidence
Done
## Runtime and test evidence
Done
## Final verdict

{final_verdict}

## RETURN exploration questions
Investigate AC-002.
""",
        encoding="utf-8",
    )


def test_validator_return_requires_a_nonpassing_ac_row(tmp_path: Path) -> None:
    task_dir = write_task(tmp_path)
    write_implementation_cycle(task_dir, 1)
    write_tests_cycle(task_dir, 1, "PASS")
    manage.apply_test_verdict(task_dir, "PASS")
    manage.transition(task_dir, "validation")

    write_validation(task_dir, ac2_verdict="PASS", final_verdict="RETURN")
    with pytest.raises(ValueError, match="requires at least one FAIL or NOT VERIFIED"):
        manage.apply_validation_verdict(task_dir, "RETURN")

    write_validation(task_dir, ac2_verdict="FAIL", final_verdict="RETURN")
    manage.apply_validation_verdict(task_dir, "RETURN")
    state = manage.load_state(task_dir)
    assert state["phase"] == "exploration"
    assert state["attempt"] == 2
    assert state["test_cycle"] == 0
    assert state["test_verdict"] == ""
