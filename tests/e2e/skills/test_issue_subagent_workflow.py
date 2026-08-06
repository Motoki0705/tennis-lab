from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/issue-subagent-workflow/scripts"


def load(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(SCRIPTS))
init = load("init_issue_task")
manage = load("manage_issue_task")
candidate = load("issue_task_candidate")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def setup_task(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "src.txt").write_text("base\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")

    task = root / ".codex/tasks/issue-1"
    task.mkdir(parents=True)
    payload = {
        "number": 1,
        "title": "Example",
        "body": "## Acceptance checklist\n\n- [ ] Observable behavior\n- [ ] Regression is covered\n",
        "url": "https://github.com/example/repo/issues/1",
        "state": "OPEN",
        "labels": [],
        "updatedAt": "2026-08-06T00:00:00Z",
    }
    issue_hash, issue_md_hash, checklist_hash, items = init.write_issue_snapshot(task, payload)
    (task / "state.toml").write_text(
        init.render_state(
            payload,
            issue_hash,
            checklist_hash,
            len(items),
            issue_snapshot_digest=issue_md_hash,
            base_revision=git(root, "rev-parse", "HEAD"),
        ),
        encoding="utf-8",
    )
    values = {
        "number": 1,
        "attempt": 1,
        "issue_hash": issue_hash,
        "checklist_hash": checklist_hash,
        "feasibility_rows": init.render_feasibility_rows(items),
        "plan_rows": init.render_plan_rows(items),
        "test_rows": init.render_test_rows(items),
        "validation_rows": init.render_validation_rows(items),
    }
    for relative, template in init.TEMPLATES.items():
        path = task / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(template.format(**values), encoding="utf-8")
    return root, task


def write_feasibility(task: Path) -> None:
    state = manage.load_state(task)
    (task / "00-feasibility/feasibility.md").write_text(
        f"""# Feasibility

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Frozen issue SHA-256: `{state['issue_sha256']}`
- Frozen acceptance checklist SHA-256: `{state['acceptance_checklist_sha256']}`

## Allowed and prohibited changes
src and tests are allowed.
## Required checks and baseline
Canonical Python checks are required.
## Breaking-change and compatibility impact
No compatibility is required.
## Acceptance checklist feasibility

| ID | Issue checklist item | Verdict | Required change and evidence |
|---|---|---|---|
| AC-001 | Observable behavior | FEASIBLE | src change |
| AC-002 | Regression is covered | FEASIBLE | test change |

## Constraint conflicts

None

## Final feasibility verdict

PASS

## Blocker resolution required

None
""",
        encoding="utf-8",
    )


def write_exploration(task: Path) -> None:
    (task / "01-exploration/exploration.md").write_text(
        """# Exploration

- Issue: #1
- Attempt: 1
- Status: COMPLETE

## Scope and Issue interpretation
Change src and tests.
## Relevant files and symbols
src.txt
## Entry points and execution paths
Direct file behavior.
## Data, configuration, and interface contracts
Text contract.
## Existing tests and fixtures
tests.txt
## Invariants and compatibility constraints
No compatibility.
## Risks and likely impact radius
Local only.
## Unresolved questions
None
## Evidence table
| Kind | Claim | Evidence |
|---|---|---|
| FACT | src exists | src.txt |
""",
        encoding="utf-8",
    )


def write_plan(task: Path) -> None:
    state = manage.load_state(task)
    (task / "02-planning/plan.md").write_text(
        f"""# Plan

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Frozen issue SHA-256: `{state['issue_sha256']}`
- Frozen acceptance checklist SHA-256: `{state['acceptance_checklist_sha256']}`

## Acceptance checklist mapping
| ID | Issue checklist item | Planned implementation | Validation method |
|---|---|---|---|
| AC-001 | Observable behavior | edit src | canonical check |
| AC-002 | Regression is covered | add tests | canonical check |
## Planned files and symbols
src.txt and tests.txt
## Implementation topology and ownership
One user-directed Implementer, also the explicit integrator.
## Independent test work unit
Test Writer may add tests.txt.
## Canonical verification commands
`py-ok` is authoritative.
## Ordered execution plan
Implement, preflight, tests, seal, validate, package.
## Validation strategy
Independent Validator.
## Non-goals and prohibited changes
No unrelated changes.
## Risks, rollback, and open decisions
None
""",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "checks": [
            {
                "id": "py-ok",
                "argv": [sys.executable, "-c", "raise SystemExit(0)"],
                "cwd": ".",
                "env": {},
                "stages": ["preflight", "test", "seal"],
                "required": True,
                "authority": ["AC-001", "AC-002"],
            }
        ],
    }
    (task / "02-planning/checks.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_implementation(task: Path, cycle: int) -> None:
    (task / "03-implementation/implementation.md").write_text(
        f"""# Implementation

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE

## Assigned ownership
Explicit integrator.
## Files and symbols changed
src.txt
## Behavior implemented
New behavior.
## Plan deviations and rationale
None
## Commands and results
Focused checks pass.
## Known limitations and remaining risks
None
## Handoff
Ready for preflight.
""",
        encoding="utf-8",
    )


def write_preflight(task: Path, cycle: int, fp: str) -> None:
    (task / "03-implementation/preflight.md").write_text(
        f"""# Production preflight

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE
- Candidate SHA-256: `{fp}`

## Candidate identity
{fp}
## Changed scope
src.txt
## Deterministic policy checks
PASS
## Focused checks
PASS
## Canonical command results
py-ok PASS
## Baseline comparison
No regression.
## Commands and exact outcomes
See machine result.
## Final production preflight verdict
PASS
## RETURN implementation findings
None
""",
        encoding="utf-8",
    )


def write_tests(task: Path, cycle: int, fp: str) -> None:
    state = manage.load_state(task)
    (task / "03-implementation/tests.md").write_text(
        f"""# Tests

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `{state['acceptance_checklist_sha256']}`
- Candidate SHA-256: `{fp}`

## Candidate identity
{fp}
## Acceptance-checklist-to-test mapping
| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | Observable behavior | py-ok | PASS |
| AC-002 | Regression is covered | tests.txt | PASS |
## Tests added or changed
tests.txt
## Normal, boundary, invalid, and regression cases
Covered.
## Canonical command results
py-ok PASS
## Commands and exact outcomes
See machine result.
## Failures encountered
None
## Untested risks and reasons
None
## Final test verdict
PASS
## RETURN implementation findings
None
""",
        encoding="utf-8",
    )


def write_seal(task: Path, cycle: int, fp: str) -> None:
    (task / "03-implementation/seal.md").write_text(
        f"""# Final candidate seal

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE
- Candidate SHA-256: `{fp}`

## Candidate identity
{fp}
## Changed-since-test inspection
No changes.
## Canonical command results
py-ok PASS
## Complete scope inspection
Allowed scope only.
## Commands and exact outcomes
See machine result.
## Final candidate seal verdict
PASS
## RETURN implementation findings
None
""",
        encoding="utf-8",
    )


def write_validation(task: Path, fp: str) -> None:
    state = manage.load_state(task)
    (task / "04-validation/validation.md").write_text(
        f"""# Validation

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Frozen issue SHA-256: `{state['issue_sha256']}`
- Frozen acceptance checklist SHA-256: `{state['acceptance_checklist_sha256']}`
- Candidate SHA-256: `{fp}`

## Inspection scope and revision
Inspected {fp}.
## Acceptance checklist verification
| ID | Issue checklist item | Verdict | Evidence |
|---|---|---|---|
| AC-001 | Observable behavior | PASS | src.txt direct inspection |
| AC-002 | Regression is covered | PASS | tests.txt and py-ok |
## Code evidence
src.txt
## Runtime and test evidence
py-ok PASS
## Regression and repository-rule checks
Scope PASS
## Final verdict
PASS
## RETURN exploration questions
None
""",
        encoding="utf-8",
    )


def advance_to_implementation(root: Path, task: Path) -> None:
    write_feasibility(task)
    manage.apply_feasibility_verdict(task, "PASS", kind=None, reason=None)
    write_exploration(task)
    manage.transition(task, "planning")
    write_plan(task)
    manage.transition(task, "implementation")
    (root / "src.txt").write_text("changed\n", encoding="utf-8")


def advance_to_validation(root: Path, task: Path) -> str:
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    write_preflight(task, 1, fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")

    (root / "tests.txt").write_text("new test\n", encoding="utf-8")
    test_fp = cast(
        str,
        candidate.compute_candidate_fingerprint(task, manage.load_state(task)),
    )
    assert test_fp != fp
    write_tests(task, 1, test_fp)
    assert manage.run_check(task, "test", "py-ok") == 0
    manage.apply_test_verdict(task, "PASS")

    write_seal(task, 1, test_fp)
    assert manage.run_check(task, "seal", "py-ok") == 0
    manage.apply_seal_verdict(task, "PASS")
    manage.transition(task, "validation")
    return test_fp



def install_fake_gh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    head: str,
    checks_pass: bool,
) -> None:
    directory = tmp_path / "fake-bin"
    directory.mkdir(exist_ok=True)
    script = directory / "gh"
    script.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys

head = os.environ["FAKE_PR_HEAD"]
checks_pass = os.environ["FAKE_CHECKS_PASS"] == "1"
if sys.argv[1:3] == ["pr", "view"]:
    conclusion = "SUCCESS" if checks_pass else "FAILURE"
    print(json.dumps({
        "number": 706,
        "url": "https://github.com/example/repo/pull/706",
        "headRefOid": head,
        "isDraft": False,
        "state": "OPEN",
        "statusCheckRollup": [{
            "__typename": "CheckRun",
            "name": "CI",
            "status": "COMPLETED",
            "conclusion": conclusion,
        }],
    }))
elif sys.argv[1] == "api":
    print(json.dumps([[{"filename": "src.txt"}, {"filename": "tests.txt"}]]))
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_PR_HEAD", head)
    monkeypatch.setenv("FAKE_CHECKS_PASS", "1" if checks_pass else "0")
    monkeypatch.setenv("PATH", f"{directory}:{os.environ.get('PATH', '')}")


def test_full_v5_flow_uses_validated_then_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, task = setup_task(tmp_path)
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")
    state = manage.load_state(task)
    assert state["phase"] == "packaging"
    assert state["status"] == "validated"
    assert state["verdict"] == "VALIDATED"

    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    assert candidate.compute_revision_fingerprint(task, state, head) == fp
    install_fake_gh(tmp_path, monkeypatch, head=head, checks_pass=True)
    manage.capture_pr_evidence(task, pr_number=706)
    state = manage.load_state(task)
    evidence_digest = state["pr_evidence_sha256"]
    (task / "05-packaging/packaging.md").write_text(
        f"""# Packaging

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Candidate SHA-256: `{fp}`
- PR number: 706
- PR head SHA: `{head}`
- Remote checks: PASS
- PR evidence SHA-256: `{evidence_digest}`

## Final candidate binding
Matches Validator candidate.
## Pull request identity
PR #706 at {head}.
## Complete paginated diff scope
Allowed scope PASS.
## Remote required checks
All required checks PASS.
## Packaging evidence
Remote inspection recorded.
## Final packaging verdict
PASS
""",
        encoding="utf-8",
    )
    manage.finalize_pr(task, pr_number=706, head_sha=head)
    state = manage.load_state(task)
    assert state["status"] == "complete"
    assert state["verdict"] == "PASS"
    assert manage.check(task) == []


def test_candidate_change_after_tester_pass_requires_retest(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    write_preflight(task, 1, fp)
    manage.run_check(task, "preflight", "py-ok")
    manage.apply_preflight_verdict(task, "PASS")
    write_tests(task, 1, fp)
    manage.run_check(task, "test", "py-ok")
    manage.apply_test_verdict(task, "PASS")
    (root / "src.txt").write_text("changed again\n", encoding="utf-8")
    new_fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    write_seal(task, 1, new_fp)
    with pytest.raises(ValueError, match="rerun the Test Writer"):
        manage.apply_seal_verdict(task, "PASS")


def test_issue_body_tampering_blocks_transition(tmp_path: Path) -> None:
    _, task = setup_task(tmp_path)
    write_feasibility(task)
    (task / "issue.md").write_text(
        (task / "issue.md").read_text(encoding="utf-8").replace("Example", "Tampered"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="issue.md does not exactly match"):
        manage.apply_feasibility_verdict(task, "PASS", kind=None, reason=None)


def test_transition_runs_artifact_check_automatically(tmp_path: Path) -> None:
    _, task = setup_task(tmp_path)
    write_feasibility(task)
    manage.apply_feasibility_verdict(task, "PASS", kind=None, reason=None)
    write_exploration(task)
    path = task / "01-exploration/exploration.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "## Data, configuration, and interface contracts",
            "## Combined contracts",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Data, configuration"):
        manage.transition(task, "planning")


def test_changed_canonical_invocation_rejects_old_result(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    write_preflight(task, 1, fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0

    manifest_path = task / "02-planning/checks.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checks"][0]["argv"] = [sys.executable, "-c", "raise SystemExit(0)", "changed"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical invocation mismatch"):
        manage.apply_preflight_verdict(task, "PASS")


def test_validation_requires_substantive_evidence(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    path = task / "04-validation/validation.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "| AC-002 | Regression is covered | PASS | tests.txt and py-ok |",
            "| AC-002 | Regression is covered | PASS | None |",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="evidence is not substantive"):
        manage.apply_validation_verdict(task, "PASS")


def test_finalize_pr_failure_keeps_validated_state_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, task = setup_task(tmp_path)
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")

    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(tmp_path, monkeypatch, head=head, checks_pass=False)
    manage.capture_pr_evidence(task, pr_number=706)
    state = manage.load_state(task)
    evidence_digest = state["pr_evidence_sha256"]
    state_before = (task / "state.toml").read_bytes()
    (task / "05-packaging/packaging.md").write_text(
        f"""# Packaging

- Issue: #1
- Attempt: 1
- Status: COMPLETE
- Candidate SHA-256: `{fp}`
- PR number: 706
- PR head SHA: `{head}`
- Remote checks: FAIL
- PR evidence SHA-256: `{evidence_digest}`

## Final candidate binding
Matches Validator candidate.
## Pull request identity
PR #706 at {head}.
## Complete paginated diff scope
Allowed scope PASS.
## Remote required checks
A required check failed.
## Packaging evidence
Remote inspection recorded.
## Final packaging verdict
PASS
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Remote checks PASS|remote required checks"):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert (task / "state.toml").read_bytes() == state_before
