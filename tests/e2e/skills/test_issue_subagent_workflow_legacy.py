from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/issue-subagent-workflow/scripts"
sys.path.insert(0, str(SCRIPTS))


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


manage = load("manage_issue_task_legacy", "manage_issue_task.py")


def digest(items: list[tuple[str, str]]) -> str:
    canonical = json.dumps(
        [{"id": item_id, "text": text} for item_id, text in items],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_legacy_task(tmp_path: Path, *, version: int, phase: str) -> Path:
    task = tmp_path / f"issue-{version}"
    for directory in ("01-exploration", "02-planning", "03-implementation", "04-validation"):
        (task / directory).mkdir(parents=True, exist_ok=True)
    items = [("AC-001", "Observable behavior"), ("AC-002", "Regression is covered")]
    checklist = digest(items)
    (task / "issue.md").write_text(
        """# GitHub Issue #1

## Acceptance checklist

- AC-001: Observable behavior (source checkbox: unchecked)
- AC-002: Regression is covered (source checkbox: unchecked)

The source checkbox state is metadata only.

## Title

Example

## Body

Example
""",
        encoding="utf-8",
    )
    common = [
        f"schema_version = {version}",
        "issue_number = 1",
        'issue_url = "x"',
        'issue_sha256 = "issue"',
        f'acceptance_checklist_sha256 = "{checklist}"',
        "acceptance_checklist_count = 2",
        "attempt = 1",
    ]
    if version == 4:
        common.extend(
            [
                'feasibility_verdict = "PASS"',
                "preflight_cycle = 1",
                'preflight_verdict = "PASS"',
            ]
        )
    common.extend(
        [
            "test_cycle = 1" if phase == "validation" else "test_cycle = 0",
            'test_verdict = "PASS"' if phase == "validation" else 'test_verdict = ""',
        ]
    )
    if version == 4:
        common.extend(
            [
                "test_return_count = 0",
                "return_review_required = false",
                'return_review_action = ""',
                'return_review_reason = ""',
            ]
        )
    common.extend(
        [
            f'phase = "{phase}"',
            'status = "in_progress"',
            'verdict = ""',
        ]
    )
    if version == 4:
        common.extend(['block_kind = ""', 'block_reason = ""'])
    common.append('updated_at = "x"')
    (task / "state.toml").write_text("\n".join(common) + "\n", encoding="utf-8")

    (task / "01-exploration/exploration.md").write_text("# Exploration\n- Attempt: 1\n", encoding="utf-8")
    (task / "02-planning/plan.md").write_text(
        f"""# Plan
- Attempt: 1
- Frozen acceptance checklist SHA-256: `{checklist}`
## Acceptance checklist mapping
| ID | Issue checklist item | Planned implementation | Validation method |
|---|---|---|---|
| AC-001 | Observable behavior | x | x |
| AC-002 | Regression is covered | x | x |
""",
        encoding="utf-8",
    )
    (task / "03-implementation/implementation.md").write_text(
        "# Implementation\n- Attempt: 1\n- Test cycle: 1\n",
        encoding="utf-8",
    )
    (task / "03-implementation/preflight.md").write_text(
        "# Preflight\n- Attempt: 1\n- Test cycle: 1\n## Final preflight verdict\nPASS\n",
        encoding="utf-8",
    )
    (task / "03-implementation/tests.md").write_text(
        f"""# Tests
- Attempt: 1
- Test cycle: 1
- Frozen acceptance checklist SHA-256: `{checklist}`
## Acceptance-checklist-to-test mapping
| ID | Issue checklist item | Test | Result |
|---|---|---|---|
| AC-001 | Observable behavior | x | PASS |
| AC-002 | Regression is covered | x | PASS |
## Final test verdict
PASS
""",
        encoding="utf-8",
    )
    (task / "04-validation/validation.md").write_text(
        f"""# Validation
- Attempt: 1
- Frozen acceptance checklist SHA-256: `{checklist}`
## Acceptance checklist verification
| ID | Issue checklist item | Verdict | Evidence |
|---|---|---|---|
| AC-001 | Observable behavior | PASS | command |
| AC-002 | Regression is covered | PASS | command |
## Final verdict
PASS
""",
        encoding="utf-8",
    )
    return task


def test_schema_v4_validation_can_complete_without_schema_v5_packaging(tmp_path: Path) -> None:
    task = write_legacy_task(tmp_path, version=4, phase="validation")
    manage.apply_validation_verdict(task, "PASS")
    state = manage.load_state(task)
    assert state["schema_version"] == 5
    assert state["candidate_binding_mode"] == "LEGACY"
    assert state["status"] == "complete"
    assert manage.check(task) == []


def test_schema_v3_in_progress_check_does_not_require_v5_artifacts(tmp_path: Path) -> None:
    task = write_legacy_task(tmp_path, version=3, phase="implementation")
    (task / "03-implementation/preflight.md").unlink()
    assert manage.check(task) == []
