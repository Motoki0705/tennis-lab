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
remote = sys.modules["issue_task_remote"]

PACKAGING_PENDING_EVIDENCE = (
    "post-Validator packaging must run capture-pr, write packaging.md from captured evidence, "
    "run finalize-pr, then run final workflow check"
)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def setup_task(
    tmp_path: Path,
    *,
    extra_base_files: tuple[str, ...] = (),
    acceptance_count: int = 2,
) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / "src.txt").write_text("base\n", encoding="utf-8")
    for relative in extra_base_files:
        (root / relative).write_text("base\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "base")

    task = root / ".codex/tasks/issue-1"
    task.mkdir(parents=True)
    acceptance_descriptions = [
        "Observable behavior",
        "Regression is covered",
        *[
            "Final PR packaging succeeds"
            if index == 79
            else f"Requirement {index:03d} is satisfied"
            for index in range(3, acceptance_count + 1)
        ],
    ]
    checklist = "\n".join(
        f"- [ ] {description}" for description in acceptance_descriptions
    )
    payload = {
        "number": 1,
        "title": "Example",
        "body": f"## Acceptance checklist\n\n{checklist}\n",
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


def task_acceptance_items(task: Path) -> list[tuple[str, str]]:
    return cast(list[tuple[str, str]], manage._state.acceptance_items(task))


def write_feasibility(task: Path) -> None:
    state = manage.load_state(task)
    rows = "\n".join(
        f"| {item_id} | {text} | FEASIBLE | implementation evidence |"
        for item_id, text in task_acceptance_items(task)
    )
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
{rows}

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
    items = task_acceptance_items(task)
    rows = "\n".join(
        f"| {item_id} | {text} | implement requirement | canonical check |"
        for item_id, text in items
    )
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
{rows}
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
                "authority": [item_id for item_id, _ in items],
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
    rows = "\n".join(
        f"| {item_id} | {text} | canonical evidence | PASS |"
        for item_id, text in task_acceptance_items(task)
    )
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
{rows}
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


def write_validation(
    task: Path,
    fp: str,
    *,
    packaging_pending: bool = False,
    final_verdict: str = "PASS",
) -> None:
    state = manage.load_state(task)
    rows: list[str] = []
    for item_id, text in task_acceptance_items(task):
        if packaging_pending and item_id == "AC-079":
            verdict = "NOT VERIFIED"
            evidence = PACKAGING_PENDING_EVIDENCE
        elif final_verdict == "RETURN" and item_id == "AC-001":
            verdict = "FAIL"
            evidence = "Concrete failing code and runtime evidence"
        else:
            verdict = "PASS"
            evidence = "Code inspection and canonical checks PASS"
        rows.append(f"| {item_id} | {text} | {verdict} | {evidence} |")
    rendered_rows = "\n".join(rows)
    questions = "Concrete re-exploration question" if final_verdict == "RETURN" else "None"
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
{rendered_rows}
## Code evidence
src.txt
## Runtime and test evidence
py-ok PASS
## Regression and repository-rule checks
Scope PASS
## Final verdict
{final_verdict}
## RETURN exploration questions
{questions}
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


def advance_to_validation(
    root: Path,
    task: Path,
    *,
    renames: tuple[tuple[str, str], ...] = (),
) -> str:
    advance_to_implementation(root, task)
    for source, destination in renames:
        git(root, "mv", source, destination)
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
    files_payload: object | None = None,
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
files_payload = json.loads(os.environ["FAKE_PR_FILES"])
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
    print(json.dumps(files_payload))
else:
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_PR_HEAD", head)
    monkeypatch.setenv("FAKE_CHECKS_PASS", "1" if checks_pass else "0")
    monkeypatch.setenv(
        "FAKE_PR_FILES",
        json.dumps(
            files_payload
            if files_payload is not None
            else [[{"filename": "src.txt"}, {"filename": "tests.txt"}]]
        ),
    )
    monkeypatch.setenv("PATH", f"{directory}:{os.environ.get('PATH', '')}")


def write_packaging(
    task: Path,
    *,
    fp: str,
    head: str,
    remote_checks: str = "PASS",
) -> None:
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
- Remote checks: {remote_checks}
- PR evidence SHA-256: `{evidence_digest}`

## Final candidate binding
Matches Validator candidate.
## Pull request identity
PR #706 at {head}.
## Complete paginated diff scope
The complete paginated file inventory matches the final revision.
## Remote required checks
Required checks recorded as {remote_checks}.
## Packaging evidence
AC-079: capture-pr bound the exact PR head, complete paginated files, candidate equality, and
required remote checks to captured evidence in
packaging.md; finalize-pr and the final workflow check remain mandatory.
## Final packaging verdict
PASS
""",
        encoding="utf-8",
    )


def test_full_v5_flow_uses_validated_then_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
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
    write_packaging(task, fp=fp, head=head)
    manage.finalize_pr(task, pr_number=706, head_sha=head)
    state = manage.load_state(task)
    assert state["status"] == "complete"
    assert state["verdict"] == "PASS"
    assert manage.check(task) == []


def test_ac079_packaging_pending_is_required_for_enforced_validator_pass(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(
        ValueError,
        match="AC-079 to be exactly NOT VERIFIED until post-Validator packaging",
    ):
        manage.apply_validation_verdict(task, "PASS")
    assert (task / "state.toml").read_bytes() == state_before


def test_other_not_verified_row_blocks_validator_pass(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    path = task / "04-validation/validation.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "| AC-078 | Requirement 078 is satisfied | PASS | "
            "Code inspection and canonical checks PASS |",
            "| AC-078 | Requirement 078 is satisfied | NOT VERIFIED | "
            f"{PACKAGING_PENDING_EVIDENCE} |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="every pre-packaging AC row PASS; not passed: AC-078",
    ):
        manage.apply_validation_verdict(task, "PASS")


def test_all_prepackaging_ac_rows_must_pass(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    path = task / "04-validation/validation.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "| AC-078 | Requirement 078 is satisfied | PASS | "
            "Code inspection and canonical checks PASS |",
            "| AC-078 | Requirement 078 is satisfied | FAIL | Concrete defect |",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="every pre-packaging AC row PASS; not passed: AC-078",
    ):
        manage.apply_validation_verdict(task, "PASS")


def test_packaging_pending_not_verified_applies_as_validator_return(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(
        task,
        fp,
        packaging_pending=True,
        final_verdict="RETURN",
    )

    manage.apply_validation_verdict(task, "RETURN")
    state = manage.load_state(task)
    assert state["phase"] == "exploration"
    assert state["status"] == "in_progress"
    assert state["verdict"] == "RETURN"
    assert state["attempt"] == 2


def test_packaging_pending_pass_artifact_cannot_be_applied_as_validator_return(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)

    with pytest.raises(
        ValueError,
        match="validation.md records PASS under ## Final verdict; expected RETURN",
    ):
        manage.apply_validation_verdict(task, "RETURN")


def test_packaging_pending_requires_exact_substantive_evidence(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    path = task / "04-validation/validation.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            PACKAGING_PENDING_EVIDENCE,
            "post-Validator packaging happens later",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="NOT VERIFIED evidence must name mandatory post-validation packaging",
    ):
        manage.apply_validation_verdict(task, "PASS")


def test_packaging_pending_exception_is_not_available_to_legacy_binding(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    state = manage.load_state(task)
    state["candidate_binding_mode"] = "LEGACY"
    manage._state.write_state(task, state)

    with pytest.raises(
        ValueError,
        match="every issue checklist item must have verdict PASS; not passed: AC-079",
    ):
        manage.apply_validation_verdict(task, "PASS")


def test_capture_and_finalize_cannot_precede_validated_packaging(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    advance_to_validation(root, task)
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(ValueError, match="capture-pr requires packaging/validated state"):
        manage.capture_pr_evidence(task, pr_number=706)
    with pytest.raises(ValueError, match="finalize-pr requires packaging/validated state"):
        manage.finalize_pr(task, pr_number=706, head_sha="a" * 40)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


def test_capture_requires_exact_checked_out_remote_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    install_fake_gh(tmp_path, monkeypatch, head="a" * 40, checks_pass=True)
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(ValueError, match="local HEAD does not match the remote PR head"):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


def test_capture_requires_current_candidate_equality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    (root / "src.txt").write_text("changed after validation\n", encoding="utf-8")
    install_fake_gh(tmp_path, monkeypatch, head=head, checks_pass=True)
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(ValueError, match="current content differs from the validated candidate"):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


def test_finalize_requires_captured_pr_evidence(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    manage.apply_validation_verdict(task, "PASS")
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(
        ValueError,
        match="missing required file: 05-packaging/pr-evidence.json",
    ):
        manage.finalize_pr(task, pr_number=706, head_sha="a" * 40)
    assert (task / "state.toml").read_bytes() == state_before


def test_finalize_requires_packaging_artifact_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(tmp_path, monkeypatch, head=head, checks_pass=True)
    manage.capture_pr_evidence(task, pr_number=706)
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(ValueError, match="packaging.md does not record a valid Candidate"):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert (task / "state.toml").read_bytes() == state_before


def test_capture_and_finalize_reconcile_renamed_files_with_no_renames_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, task = setup_task(
        tmp_path,
        extra_base_files=("src-second.txt", "src-third.txt"),
    )
    fp = advance_to_validation(
        root,
        task,
        renames=(
            ("src.txt", "renamed.txt"),
            ("src-second.txt", "renamed-second.txt"),
            ("src-third.txt", "renamed-third.txt"),
        ),
    )
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")

    git(root, "add", "-A")
    git(root, "commit", "-m", "renamed candidate")
    head = git(root, "rev-parse", "HEAD")
    state = manage.load_state(task)
    revision_files = candidate.revision_changed_paths(task, state, head)
    assert revision_files == [
        "renamed-second.txt",
        "renamed-third.txt",
        "renamed.txt",
        "src-second.txt",
        "src-third.txt",
        "src.txt",
        "tests.txt",
    ]
    install_fake_gh(
        tmp_path,
        monkeypatch,
        head=head,
        checks_pass=True,
        files_payload=[
            [
                {
                    "filename": "renamed.txt",
                    "previous_filename": "src.txt",
                    "status": "renamed",
                },
                {
                    "filename": "renamed-second.txt",
                    "previous_filename": "src-second.txt",
                    "status": "renamed",
                },
            ],
            [
                {
                    "filename": "renamed-third.txt",
                    "previous_filename": "src-third.txt",
                    "status": "renamed",
                },
                {"filename": "tests.txt", "status": "added"},
            ],
        ],
    )

    manage.capture_pr_evidence(task, pr_number=706)
    state = manage.load_state(task)
    evidence = json.loads(
        (task / "05-packaging/pr-evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["files"] == revision_files
    write_packaging(task, fp=fp, head=head)
    manage.finalize_pr(task, pr_number=706, head_sha=head)
    state = manage.load_state(task)
    assert state["status"] == "complete"
    assert state["verdict"] == "PASS"
    assert manage.check(task) == []


@pytest.mark.parametrize(
    "renamed_entry",
    [
        {"filename": "src.txt", "status": "renamed"},
        {"filename": "src.txt", "status": "renamed", "previous_filename": None},
        {"filename": "src.txt", "status": "renamed", "previous_filename": 7},
        {"filename": "src.txt", "status": "renamed", "previous_filename": ""},
    ],
)
def test_capture_rejects_renamed_file_without_valid_previous_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    renamed_entry: dict[str, object],
) -> None:
    root, task = setup_task(tmp_path)
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")

    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(
        tmp_path,
        monkeypatch,
        head=head,
        checks_pass=True,
        files_payload=[
            [
                renamed_entry,
                {"filename": "tests.txt", "status": "added"},
            ]
        ],
    )
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(ValueError, match="valid previous_filename"):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


def test_capture_rejects_valid_renamed_inventory_mismatch_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path)
    fp = advance_to_validation(
        root,
        task,
        renames=(("src.txt", "renamed.txt"),),
    )
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")

    git(root, "add", "-A")
    git(root, "commit", "-m", "renamed candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(
        tmp_path,
        monkeypatch,
        head=head,
        checks_pass=True,
        files_payload=[
            [
                {
                    "filename": "renamed.txt",
                    "previous_filename": "different-old-name.txt",
                    "status": "renamed",
                },
                {"filename": "tests.txt", "status": "added"},
            ]
        ],
    )
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(
        ValueError,
        match="complete paginated PR file list differs from the validated revision",
    ):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


def test_finalize_rechecks_complete_paginated_files_against_final_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(tmp_path, monkeypatch, head=head, checks_pass=True)
    manage.capture_pr_evidence(task, pr_number=706)

    evidence_path = task / "05-packaging/pr-evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["files"] = ["src.txt"]
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    state = manage.load_state(task)
    state["pr_evidence_sha256"] = remote.evidence_digest(evidence)
    manage._state.write_state(task, state)
    write_packaging(task, fp=fp, head=head)
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(
        ValueError,
        match="captured complete paginated PR files differ from the final revision",
    ):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert (task / "state.toml").read_bytes() == state_before


def test_finalize_requires_packaging_artifact_to_establish_ac079(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path, acceptance_count=79)
    fp = advance_to_validation(root, task)
    write_validation(task, fp, packaging_pending=True)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(tmp_path, monkeypatch, head=head, checks_pass=True)
    manage.capture_pr_evidence(task, pr_number=706)
    write_packaging(task, fp=fp, head=head)
    packaging_path = task / "05-packaging/packaging.md"
    packaging_path.write_text(
        packaging_path.read_text(encoding="utf-8").replace(
                "AC-079: capture-pr bound the exact PR head, complete paginated files, "
                "candidate equality, and\nrequired remote checks to captured evidence in\n"
                "packaging.md; finalize-pr and the final workflow check remain mandatory.",
            "Remote inspection recorded.",
        ),
        encoding="utf-8",
    )
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(
        ValueError,
        match="packaging.md does not establish AC-079",
    ):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert (task / "state.toml").read_bytes() == state_before


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
            "| AC-002 | Regression is covered | PASS | "
            "Code inspection and canonical checks PASS |",
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
    state_before = (task / "state.toml").read_bytes()
    write_packaging(task, fp=fp, head=head, remote_checks="FAIL")
    with pytest.raises(ValueError, match="Remote checks PASS|remote required checks"):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert (task / "state.toml").read_bytes() == state_before
