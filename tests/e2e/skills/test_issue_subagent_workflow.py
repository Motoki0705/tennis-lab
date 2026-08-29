from __future__ import annotations

import hashlib
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
issue = sys.modules["issue_task_issue"]
remote = sys.modules["issue_task_remote"]


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def setup_task(
    tmp_path: Path,
    *,
    extra_base_files: tuple[str, ...] = (),
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
    payload = {
        "number": 1,
        "title": "Example",
        "body": "## Acceptance checklist\n\n- [ ] Observable behavior\n- [ ] Regression is covered\n",
        "url": "https://github.com/example/repo/issues/1",
        "state": "OPEN",
        "labels": [],
        "updatedAt": "2026-08-06T00:00:00Z",
    }
    issue_hash, issue_md_hash, checklist_hash, items = init.write_issue_snapshot(
        task, payload
    )
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
- Frozen issue SHA-256: `{state["issue_sha256"]}`
- Frozen acceptance checklist SHA-256: `{state["acceptance_checklist_sha256"]}`

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
- Frozen issue SHA-256: `{state["issue_sha256"]}`
- Frozen acceptance checklist SHA-256: `{state["acceptance_checklist_sha256"]}`

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
    (task / "02-planning/checks.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


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


def write_tests(
    task: Path,
    cycle: int,
    fp: str,
    *,
    adversarial_rows: str = "None — independent impact review found no additional executable case.",
    adversarial_probe_results: str = "None",
) -> None:
    state = manage.load_state(task)
    (task / "03-implementation/tests.md").write_text(
        f"""# Tests

- Issue: #1
- Attempt: 1
- Test cycle: {cycle}
- Status: COMPLETE
- Frozen acceptance checklist SHA-256: `{state["acceptance_checklist_sha256"]}`
- Candidate SHA-256: `{fp}`

## Candidate identity
{fp}
## Acceptance-checklist-to-test mapping
| ID | Issue checklist item | Test or authoritative evidence | Result |
|---|---|---|---|
| AC-001 | Observable behavior | py-ok | PASS |
| AC-002 | Regression is covered | tests.txt | PASS |
## Independent adversarial test design
Inspected public behavior and changed callers independently of the planned minimum.
## Independently derived adversarial tests
{adversarial_rows}
## Adversarial probe results
{adversarial_probe_results}
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
- Frozen issue SHA-256: `{state["issue_sha256"]}`
- Frozen acceptance checklist SHA-256: `{state["acceptance_checklist_sha256"]}`
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
    base: object,
    head: str,
    checks_pass: bool,
    base_ref_name: object = "main",
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

base_ref_name = json.loads(os.environ["FAKE_PR_BASE_REF_NAME"])
base = json.loads(os.environ["FAKE_PR_BASE"])
head = os.environ["FAKE_PR_HEAD"]
checks_pass = os.environ["FAKE_CHECKS_PASS"] == "1"
files_payload = json.loads(os.environ["FAKE_PR_FILES"])
if sys.argv[1:3] == ["pr", "view"]:
    conclusion = "SUCCESS" if checks_pass else "FAILURE"
    print(json.dumps({
        "number": 706,
        "url": "https://github.com/example/repo/pull/706",
        "baseRefName": base_ref_name,
        "baseRefOid": base,
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
    monkeypatch.setenv("FAKE_PR_BASE_REF_NAME", json.dumps(base_ref_name))
    monkeypatch.setenv("FAKE_PR_BASE", json.dumps(base))
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


def test_multiline_acceptance_context_is_preserved_in_canonical_artifacts(
    tmp_path: Path,
) -> None:
    body = """## Acceptance checklist

<!--
- [ ] This template example is not an acceptance item
  Context from the template is ignored too.
-->

- [ ] 出力契約を満たす
  文脈: 生成物を後続処理が入力として利用する。
  下流への影響: consumer | renderer が追加変換なしで読み込める。
  検証方法・証拠: 契約テストの結果を提示する。
- [x] 従来の1行ACも維持する
- [ ]
"""
    items = issue.extract_acceptance_items(body)
    expected = (
        "出力契約を満たす 文脈: 生成物を後続処理が入力として利用する。 "
        "下流への影響: consumer | renderer が追加変換なしで読み込める。 "
        "検証方法・証拠: 契約テストの結果を提示する。"
    )
    assert [(item.item_id, item.text, item.source_checked) for item in items] == [
        ("AC-001", expected, False),
        ("AC-002", "従来の1行ACも維持する", True),
    ]
    escaped_expected = issue.escape_table_cell(expected)
    assert (
        f"| AC-001 | {escaped_expected} | PENDING | PENDING |"
        in init.render_plan_rows(items)
    )

    payload = {
        "number": 7,
        "title": "Detailed acceptance context",
        "body": body,
        "url": "https://github.com/example/repo/issues/7",
        "state": "OPEN",
        "labels": [],
        "updatedAt": "2026-08-28T00:00:00Z",
    }
    issue_hash, issue_md_hash, checklist_hash, snapshot_items = (
        issue.write_issue_snapshot(tmp_path, payload)
    )
    state = {
        "candidate_binding_mode": "CONTENT",
        "issue_sha256": issue_hash,
        "issue_snapshot_sha256": issue_md_hash,
        "acceptance_checklist_sha256": checklist_hash,
        "acceptance_checklist_count": len(snapshot_items),
        "issue_number": payload["number"],
        "issue_url": payload["url"],
    }
    assert issue.validate_issue_snapshot(tmp_path, state) == []
    issue_markdown = (tmp_path / "issue.md").read_text(encoding="utf-8")
    assert f"- AC-001: {expected} (source checkbox: unchecked)" in issue_markdown
    assert body in issue_markdown


def test_multiline_acceptance_hash_uses_normalized_full_context() -> None:
    compact = """## Acceptance checklist
- [ ] Summary
  Context: downstream contract
  Evidence: focused test
"""
    reformatted = """## Acceptance checklist
- [ ] Summary
  Context:   downstream   contract
  Evidence:\tfocused test
"""
    compact_items = issue.extract_acceptance_items(compact)
    reformatted_items = issue.extract_acceptance_items(reformatted)
    assert compact_items == reformatted_items
    assert issue.acceptance_hash(compact_items) == issue.acceptance_hash(
        reformatted_items
    )


@pytest.mark.parametrize(
    ("detail", "message"),
    [
        ("detail without indentation", "at least two ASCII spaces"),
        (" detail with one space", "at least two ASCII spaces"),
        ("\tdetail with a tab", "at least two ASCII spaces"),
        ("\n  detail after a blank line", "must immediately follow"),
    ],
)
def test_acceptance_context_rejects_detail_that_cannot_be_safely_attached(
    detail: str, message: str
) -> None:
    body = f"## Acceptance checklist\n- [ ] Summary\n{detail}\n"
    with pytest.raises(ValueError, match=message):
        issue.extract_acceptance_items(body)


def test_indented_legacy_task_item_remains_a_separate_acceptance_item() -> None:
    body = """## Acceptance checklist
- [ ] Parent item
  - [x] Indented item
    Its own detailed context.
"""
    items = issue.extract_acceptance_items(body)
    assert [(item.text, item.source_checked) for item in items] == [
        ("Parent item", False),
        ("Indented item Its own detailed context.", True),
    ]


def test_full_v6_flow_uses_validated_then_complete(
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
    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=state["base_revision"],
        head=head,
        checks_pass=True,
    )
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


def test_capture_and_finalize_use_advanced_pr_base_with_tracked_task_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path)
    frozen_base = manage.load_state(task)["base_revision"]
    (root / "base-shared.txt").write_text("advanced base\n", encoding="utf-8")
    git(root, "add", "base-shared.txt")
    git(root, "commit", "-m", "advance PR base")
    pr_base = git(root, "rev-parse", "HEAD")

    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "-A")
    git(root, "commit", "-m", "candidate with workflow artifacts")
    head = git(root, "rev-parse", "HEAD")
    state = manage.load_state(task)

    candidate_files = candidate.revision_changed_paths(task, state, head)
    assert "base-shared.txt" in candidate_files
    assert not any(path.startswith(".codex/tasks/") for path in candidate_files)
    pr_files = candidate.revision_path_inventory(task, pr_base, head)
    assert "base-shared.txt" not in pr_files
    assert "src.txt" in pr_files
    assert "tests.txt" in pr_files
    assert any(path.startswith(".codex/tasks/issue-1/") for path in pr_files)
    files_payload = [[{"filename": path, "status": "modified"} for path in pr_files]]

    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=frozen_base,
        head=head,
        checks_pass=True,
        files_payload=files_payload,
    )
    state_before = (task / "state.toml").read_bytes()
    with pytest.raises(
        ValueError,
        match="complete paginated PR file list differs from the remote PR base",
    ):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()

    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=pr_base,
        head=head,
        checks_pass=True,
        files_payload=files_payload,
    )
    manage.capture_pr_evidence(task, pr_number=706)
    state = manage.load_state(task)
    evidence = json.loads(
        (task / "05-packaging/pr-evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["schema_version"] == 2
    assert evidence["base_ref_name"] == "main"
    assert evidence["base_sha"] == pr_base
    assert evidence["head_sha"] == head
    assert evidence["files"] == pr_files

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
PR #706 at {head}, based on main at {pr_base}.
## Complete paginated diff scope
PR-base inventory includes tracked workflow artifacts.
## Remote required checks
All required checks PASS.
## Packaging evidence
Remote base, head, files, and checks recorded.
## Final packaging verdict
PASS
""",
        encoding="utf-8",
    )

    evidence_path = task / "05-packaging/pr-evidence.json"
    state_path = task / "state.toml"
    packaging_path = task / "05-packaging/packaging.md"
    correct_evidence_text = evidence_path.read_text(encoding="utf-8")
    tampered_evidence = dict(evidence)
    tampered_evidence["base_sha"] = frozen_base
    tampered_evidence_text = (
        json.dumps(tampered_evidence, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )
    tampered_digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                tampered_evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )
    evidence_path.write_text(tampered_evidence_text, encoding="utf-8")
    state_path.write_text(
        state_path.read_text(encoding="utf-8").replace(
            evidence_digest,
            tampered_digest,
        ),
        encoding="utf-8",
    )
    packaging_path.write_text(
        packaging_path.read_text(encoding="utf-8").replace(
            evidence_digest,
            tampered_digest,
        ),
        encoding="utf-8",
    )
    state_before_finalize = state_path.read_bytes()
    with pytest.raises(
        ValueError,
        match="PR evidence files differ from the recorded PR base revision",
    ):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert state_path.read_bytes() == state_before_finalize

    evidence_path.write_text(correct_evidence_text, encoding="utf-8")
    state_path.write_text(
        state_path.read_text(encoding="utf-8").replace(
            tampered_digest,
            evidence_digest,
        ),
        encoding="utf-8",
    )
    packaging_path.write_text(
        packaging_path.read_text(encoding="utf-8").replace(
            tampered_digest,
            evidence_digest,
        ),
        encoding="utf-8",
    )
    manage.finalize_pr(task, pr_number=706, head_sha=head)
    state = manage.load_state(task)
    assert state["status"] == "complete"
    assert state["verdict"] == "PASS"
    assert manage.check(task) == []


def test_capture_finalize_and_check_use_diverged_pr_merge_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path)
    frozen_base = manage.load_state(task)["base_revision"]
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "-A")
    git(root, "commit", "-m", "candidate with workflow artifacts")
    head = git(root, "rev-parse", "HEAD")

    git(root, "checkout", "--detach", frozen_base)
    (root / "base-only.txt").write_text("advanced base\n", encoding="utf-8")
    git(root, "add", "base-only.txt")
    git(root, "commit", "-m", "diverged PR base")
    pr_base = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "--detach", head)
    assert git(root, "merge-base", pr_base, head) == frozen_base

    state = manage.load_state(task)
    candidate_files = candidate.revision_changed_paths(task, state, head)
    assert candidate_files == ["src.txt", "tests.txt"]
    assert candidate.compute_candidate_fingerprint(task, state) == fp
    assert candidate.compute_revision_fingerprint(task, state, head) == fp

    endpoint_files = candidate.revision_path_inventory(task, pr_base, head)
    pr_files = candidate.pr_evidence_path_inventory(task, pr_base, head)
    assert endpoint_files == sorted([*pr_files, "base-only.txt"])
    assert "base-only.txt" not in pr_files
    assert set(candidate_files) < set(pr_files)
    assert any(path.startswith(".codex/tasks/issue-1/") for path in pr_files)

    files_payload = [[{"filename": path, "status": "modified"} for path in pr_files]]
    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=pr_base,
        head=head,
        checks_pass=True,
        files_payload=files_payload,
    )
    manage.capture_pr_evidence(task, pr_number=706)
    state = manage.load_state(task)
    evidence_path = task / "05-packaging/pr-evidence.json"
    state_path = task / "state.toml"
    packaging_path = task / "05-packaging/packaging.md"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["base_sha"] == pr_base
    assert evidence["head_sha"] == head
    assert evidence["candidate_sha256"] == fp
    assert evidence["files"] == pr_files

    evidence_digest = state["pr_evidence_sha256"]
    packaging_path.write_text(
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
Matches the frozen-base Validator candidate.
## Pull request identity
PR #706 at {head}, based on diverged main at {pr_base}.
## Complete paginated diff scope
Merge-base-to-head inventory includes tracked workflow artifacts.
## Remote required checks
All required checks PASS.
## Packaging evidence
Actual remote base, head, files, and checks recorded.
## Final packaging verdict
PASS
""",
        encoding="utf-8",
    )

    correct_evidence_text = evidence_path.read_text(encoding="utf-8")
    correct_state_text = state_path.read_text(encoding="utf-8")
    correct_packaging_text = packaging_path.read_text(encoding="utf-8")
    mismatched_evidence = dict(evidence)
    mismatched_evidence["files"] = sorted([*pr_files, "base-only.txt"])
    mismatched_digest = remote.evidence_digest(mismatched_evidence)
    evidence_path.write_text(
        json.dumps(
            mismatched_evidence,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        correct_state_text.replace(evidence_digest, mismatched_digest),
        encoding="utf-8",
    )
    packaging_path.write_text(
        correct_packaging_text.replace(evidence_digest, mismatched_digest),
        encoding="utf-8",
    )
    state_before_failed_finalize = state_path.read_bytes()
    with pytest.raises(
        ValueError,
        match="PR evidence files differ from the recorded PR base revision",
    ):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert state_path.read_bytes() == state_before_failed_finalize

    evidence_path.write_text(correct_evidence_text, encoding="utf-8")
    state_path.write_text(correct_state_text, encoding="utf-8")
    packaging_path.write_text(correct_packaging_text, encoding="utf-8")
    manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert manage.check(task) == []

    complete_state_text = state_path.read_text(encoding="utf-8")
    evidence_path.write_text(
        json.dumps(
            mismatched_evidence,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        complete_state_text.replace(evidence_digest, mismatched_digest),
        encoding="utf-8",
    )
    state_before_failed_check = state_path.read_bytes()
    assert (
        "PR evidence files differ from the recorded PR base revision"
        in manage.check(task)
    )
    assert state_path.read_bytes() == state_before_failed_check


def test_capture_finalize_and_check_reject_pr_without_merge_base_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, task = setup_task(tmp_path)
    fp = advance_to_validation(root, task)
    write_validation(task, fp)
    manage.apply_validation_verdict(task, "PASS")
    git(root, "add", "src.txt", "tests.txt")
    git(root, "commit", "-m", "candidate")
    head = git(root, "rev-parse", "HEAD")
    frozen_base = manage.load_state(task)["base_revision"]
    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=frozen_base,
        head=head,
        checks_pass=True,
    )
    manage.capture_pr_evidence(task, pr_number=706)
    state = manage.load_state(task)
    evidence_path = task / "05-packaging/pr-evidence.json"
    state_path = task / "state.toml"
    packaging_path = task / "05-packaging/packaging.md"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_digest = state["pr_evidence_sha256"]
    packaging_path.write_text(
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
Merge-base-to-head inventory recorded.
## Remote required checks
All required checks PASS.
## Packaging evidence
Remote base, head, files, and checks recorded.
## Final packaging verdict
PASS
""",
        encoding="utf-8",
    )

    empty_tree = git(root, "hash-object", "-w", "-t", "tree", "/dev/null")
    unrelated_base = git(root, "commit-tree", empty_tree, "-m", "unrelated base")
    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=unrelated_base,
        head=head,
        checks_pass=True,
    )
    correct_evidence_text = evidence_path.read_text(encoding="utf-8")
    correct_state_text = state_path.read_text(encoding="utf-8")
    correct_packaging_text = packaging_path.read_text(encoding="utf-8")
    state_before_failed_recapture = state_path.read_bytes()
    evidence_before_failed_recapture = evidence_path.read_bytes()

    with pytest.raises(ValueError, match=r"git merge-base .* failed"):
        manage.capture_pr_evidence(task, pr_number=706)
    assert state_path.read_bytes() == state_before_failed_recapture
    assert evidence_path.read_bytes() == evidence_before_failed_recapture

    unrelated_evidence = dict(evidence)
    unrelated_evidence["base_sha"] = unrelated_base
    unrelated_digest = remote.evidence_digest(unrelated_evidence)
    evidence_path.write_text(
        json.dumps(
            unrelated_evidence,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        correct_state_text.replace(evidence_digest, unrelated_digest),
        encoding="utf-8",
    )
    packaging_path.write_text(
        correct_packaging_text.replace(evidence_digest, unrelated_digest),
        encoding="utf-8",
    )
    state_before_failed_finalize = state_path.read_bytes()
    evidence_before_failed_finalize = evidence_path.read_bytes()
    with pytest.raises(ValueError, match=r"git merge-base .* failed"):
        manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert state_path.read_bytes() == state_before_failed_finalize
    assert evidence_path.read_bytes() == evidence_before_failed_finalize

    evidence_path.write_text(correct_evidence_text, encoding="utf-8")
    state_path.write_text(correct_state_text, encoding="utf-8")
    packaging_path.write_text(correct_packaging_text, encoding="utf-8")
    manage.finalize_pr(task, pr_number=706, head_sha=head)
    assert manage.check(task) == []

    complete_state_text = state_path.read_text(encoding="utf-8")
    evidence_path.write_text(
        json.dumps(
            unrelated_evidence,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        complete_state_text.replace(evidence_digest, unrelated_digest),
        encoding="utf-8",
    )
    packaging_path.write_text(
        correct_packaging_text.replace(evidence_digest, unrelated_digest),
        encoding="utf-8",
    )
    state_before_failed_check = state_path.read_bytes()
    evidence_before_failed_check = evidence_path.read_bytes()
    assert any(
        "git merge-base" in error and "failed" in error
        for error in manage.check(task)
    )
    assert state_path.read_bytes() == state_before_failed_check
    assert evidence_path.read_bytes() == evidence_before_failed_check


@pytest.mark.parametrize(
    "base",
    [None, "", "a" * 39, "A" * 40, "g" * 40],
)
def test_capture_rejects_malformed_remote_pr_base_sha_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base: object,
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
        base=base,
        head=head,
        checks_pass=True,
    )
    state_before = (task / "state.toml").read_bytes()

    with pytest.raises(ValueError, match="remote PR base SHA is invalid"):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


def test_legacy_pr_evidence_schema_requires_base_bound_recapture(
    tmp_path: Path,
) -> None:
    task = tmp_path / ".codex/tasks/issue-1"
    evidence_path = task / "05-packaging/pr-evidence.json"
    evidence_path.parent.mkdir(parents=True)
    evidence_path.write_text('{"schema_version": 1}\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="schema_version 1 does not bind the remote PR base; rerun capture-pr",
    ):
        remote.load_pr_evidence(task)


def test_schema_v5_task_loads_with_legacy_adversarial_contract(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    state_path = task / "state.toml"
    state_path.write_text(
        state_path.read_text(encoding="utf-8")
        .replace("schema_version = 6", "schema_version = 5")
        .replace('adversarial_testing_mode = "ENFORCED"\n', ""),
        encoding="utf-8",
    )

    state = manage.load_state(task)
    assert state["schema_version"] == 6
    assert state["adversarial_testing_mode"] == "LEGACY"

    advance_to_implementation(root, task)
    write_implementation(task, 1)
    preflight_fp = candidate.compute_candidate_fingerprint(
        task, manage.load_state(task)
    )
    write_preflight(task, 1, preflight_fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")
    write_tests(task, 1, preflight_fp)
    tests_path = task / "03-implementation/tests.md"
    tests_text = tests_path.read_text(encoding="utf-8")
    adversarial_start = tests_text.index("## Independent adversarial test design")
    tests_start = tests_text.index("## Tests added or changed")
    tests_path.write_text(
        tests_text[:adversarial_start] + tests_text[tests_start:],
        encoding="utf-8",
    )
    assert manage.run_check(task, "test", "py-ok") == 0
    manage.apply_test_verdict(task, "PASS")
    assert manage.load_state(task)["test_verdict"] == "PASS"


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

    git(root, "add", "-u")
    git(root, "add", "tests.txt")
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
        base=state["base_revision"],
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
Both old and new rename paths match the no-renames revision inventory.
## Remote required checks
All required checks PASS.
## Packaging evidence
Two paginated file pages recorded.
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
        base=manage.load_state(task)["base_revision"],
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

    git(root, "add", "-u")
    git(root, "add", "tests.txt")
    git(root, "commit", "-m", "renamed candidate")
    head = git(root, "rev-parse", "HEAD")
    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=manage.load_state(task)["base_revision"],
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
        match="complete paginated PR file list differs from the remote PR base",
    ):
        manage.capture_pr_evidence(task, pr_number=706)
    assert (task / "state.toml").read_bytes() == state_before
    assert not (task / "05-packaging/pr-evidence.json").exists()


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


def test_adversarial_probe_can_return_with_every_ac_passing(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    preflight_fp = candidate.compute_candidate_fingerprint(
        task, manage.load_state(task)
    )
    write_preflight(task, 1, preflight_fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")

    (root / "tests.txt").write_text("adversarial regression\n", encoding="utf-8")
    test_fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "manage_issue_task.py"),
            "run-test-probe",
            str(task),
            "AT-001",
            "--authority",
            "PUBLIC_CONTRACT",
            "--authority-ref",
            "documented caller accepts the boundary input",
            "--argv",
            sys.executable,
            "-c",
            "raise SystemExit(7)",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 7
    write_tests(
        task,
        1,
        test_fp,
        adversarial_rows=(
            "| AT-001 | boundary input rejected | PUBLIC_CONTRACT | "
            "documented caller accepts it | test-probes.json and log | FAIL |"
        ),
        adversarial_probe_results="AT-001 FAIL with exit code 7.",
    )
    tests_path = task / "03-implementation/tests.md"
    tests_path.write_text(
        tests_path.read_text(encoding="utf-8")
        .replace("## Final test verdict\nPASS", "## Final test verdict\nRETURN")
        .replace(
            "## RETURN implementation findings\nNone",
            "## RETURN implementation findings\nAT-001: production rejects a documented boundary input.",
        ),
        encoding="utf-8",
    )

    manage.apply_test_verdict(task, "RETURN")
    state = manage.load_state(task)
    assert state["test_verdict"] == "RETURN"
    assert state["test_candidate_sha256"] == test_fp


def test_adversarial_row_requires_candidate_bound_probe_evidence(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    preflight_fp = candidate.compute_candidate_fingerprint(
        task, manage.load_state(task)
    )
    write_preflight(task, 1, preflight_fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")

    test_fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    write_tests(
        task,
        1,
        test_fp,
        adversarial_rows=(
            "| AT-001 | downstream caller | REPO_INVARIANT | caller remains compatible | "
            "claimed command | PASS |"
        ),
        adversarial_probe_results="AT-001 claimed PASS.",
    )
    assert "missing adversarial test probe results" in "; ".join(
        manage.check_artifact(task, "tests")
    )


def test_passing_adversarial_probe_can_complete_tester_gate(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    preflight_fp = candidate.compute_candidate_fingerprint(
        task, manage.load_state(task)
    )
    write_preflight(task, 1, preflight_fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")

    (root / "tests.txt").write_text("adversarial pass\n", encoding="utf-8")
    test_fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    assert (
        manage.run_test_probe(
            task,
            "AT-001",
            authority="BASELINE_REGRESSION",
            authority_ref="base revision accepts the downstream call",
            argv=[sys.executable, "-c", "raise SystemExit(0)"],
        )
        == 0
    )
    write_tests(
        task,
        1,
        test_fp,
        adversarial_rows=(
            "| AT-001 | downstream compatibility | BASELINE_REGRESSION | "
            "base revision accepts the call | test-probes.json and log | PASS |"
        ),
        adversarial_probe_results="AT-001 PASS with candidate-bound evidence.",
    )
    assert manage.run_check(task, "test", "py-ok") == 0
    manage.apply_test_verdict(task, "PASS")
    assert manage.load_state(task)["test_verdict"] == "PASS"


def test_adversarial_probe_result_is_stale_after_candidate_change(
    tmp_path: Path,
) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    preflight_fp = candidate.compute_candidate_fingerprint(
        task, manage.load_state(task)
    )
    write_preflight(task, 1, preflight_fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")

    assert (
        manage.run_test_probe(
            task,
            "AT-001",
            authority="REPO_INVARIANT",
            authority_ref="changed caller remains executable",
            argv=[sys.executable, "-c", "raise SystemExit(0)"],
        )
        == 0
    )
    (root / "tests.txt").write_text("changed after probe\n", encoding="utf-8")
    stale_fp = candidate.compute_candidate_fingerprint(task, manage.load_state(task))
    write_tests(
        task,
        1,
        stale_fp,
        adversarial_rows=(
            "| AT-001 | changed caller | REPO_INVARIANT | caller remains executable | "
            "test-probes.json and log | PASS |"
        ),
        adversarial_probe_results="AT-001 PASS before the later candidate change.",
    )
    assert "candidate fingerprint is stale" in "; ".join(
        manage.check_artifact(task, "tests")
    )


def test_adversarial_probe_rejects_candidate_mutation(tmp_path: Path) -> None:
    root, task = setup_task(tmp_path)
    advance_to_implementation(root, task)
    write_implementation(task, 1)
    preflight_fp = candidate.compute_candidate_fingerprint(
        task, manage.load_state(task)
    )
    write_preflight(task, 1, preflight_fp)
    assert manage.run_check(task, "preflight", "py-ok") == 0
    manage.apply_preflight_verdict(task, "PASS")

    with pytest.raises(ValueError, match="changed candidate content"):
        manage.run_test_probe(
            task,
            "AT-001",
            authority="REPO_INVARIANT",
            authority_ref="tests must not edit production",
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('src.txt').write_text('mutated')",
            ],
        )
    assert not (task / "03-implementation/test-probes.json").exists()


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
    manifest["checks"][0]["argv"] = [
        sys.executable,
        "-c",
        "raise SystemExit(0)",
        "changed",
    ]
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
    install_fake_gh(
        tmp_path,
        monkeypatch,
        base=manage.load_state(task)["base_revision"],
        head=head,
        checks_pass=False,
    )
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
