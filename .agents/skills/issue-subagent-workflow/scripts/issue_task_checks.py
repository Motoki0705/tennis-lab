"""Consistency checks for issue-subagent-workflow artifacts and state."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from issue_task_artifacts import artifact_candidate, check_artifact, packaging_metadata
from issue_task_candidate import (
    compute_candidate_fingerprint,
    compute_revision_fingerprint,
)
from issue_task_schema import ARTIFACT_PATHS
from issue_task_state import (
    CORE_REQUIRED_FILES,
    ENFORCED_REQUIRED_FILES,
    VERSIONED_ARTIFACT_RE,
    acceptance_items,
    assert_artifact_test_cycle,
    assert_checklist_hash_present,
    assert_mapping_table,
    assert_standalone_value,
    load_state,
    validate_state,
    validation_matrix_errors,
)
from issue_task_remote import PR_EVIDENCE_PATH, pr_evidence_errors
from issue_task_verification import stage_result_errors


def _required_artifacts(state: dict[str, Any]) -> list[str]:
    phase = state.get("phase")
    status = state.get("status")
    required = ["feasibility"]
    if phase in {"planning", "implementation", "validation", "packaging"} or status in {
        "validated",
        "complete",
    }:
        required.append("exploration")
    if phase in {"implementation", "validation", "packaging"} or status in {
        "validated",
        "complete",
    }:
        required.append("plan")
    if (
        int(state.get("preflight_cycle", 0)) > 0
        or phase in {"validation", "packaging"}
        or status in {"validated", "complete"}
    ):
        required.extend(("implementation", "preflight"))
    if (
        int(state.get("test_cycle", 0)) > 0
        or phase in {"validation", "packaging"}
        or status in {"validated", "complete"}
    ):
        required.append("tests")
    if (
        int(state.get("seal_cycle", 0)) > 0
        or phase in {"validation", "packaging"}
        or status in {"validated", "complete"}
    ):
        required.append("seal")
    if phase == "packaging" or status in {"validated", "complete"}:
        required.append("validation")
    if status == "complete":
        required.append("packaging")
    return list(dict.fromkeys(required))


def _legacy_checks(task_dir: Path, state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    phase = state.get("phase")
    status = state.get("status")
    try:
        items = acceptance_items(task_dir)
    except ValueError as exc:
        return [str(exc)]
    checklist_hash = str(state.get("acceptance_checklist_sha256", ""))
    if phase in {"implementation", "validation"} or status == "complete":
        plan_path = task_dir / "02-planning/plan.md"
        try:
            assert_checklist_hash_present(plan_path, checklist_hash)
            assert_mapping_table(plan_path, "## Acceptance checklist mapping", items)
        except ValueError as exc:
            errors.append(str(exc))
    if phase == "validation" or status == "complete":
        tests_path = task_dir / "03-implementation/tests.md"
        validation_path = task_dir / "04-validation/validation.md"
        cycle = int(state.get("test_cycle", 0))
        try:
            assert_artifact_test_cycle(tests_path, cycle)
            assert_checklist_hash_present(tests_path, checklist_hash)
            assert_mapping_table(
                tests_path,
                "## Acceptance-checklist-to-test mapping",
                items,
            )
            assert_standalone_value(
                tests_path,
                "## Final test verdict",
                "PASS",
                {"PASS", "RETURN"},
            )
        except ValueError as exc:
            errors.append(str(exc))
        if status == "complete":
            try:
                errors.extend(validation_matrix_errors(task_dir, require_all_pass=True))
                assert_standalone_value(
                    validation_path,
                    "## Final verdict",
                    "PASS",
                    {"PASS", "RETURN"},
                )
            except ValueError as exc:
                errors.append(str(exc))
    return errors


def check(
    task_dir: Path,
    *,
    state_override: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    for relative in ("issue.md", "state.toml"):
        path = task_dir / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"empty required file: {relative}")
    if errors:
        return errors

    try:
        state = state_override or load_state(task_dir)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [str(exc)]

    mode = state.get("candidate_binding_mode")
    required_files = list(CORE_REQUIRED_FILES)
    if mode == "ENFORCED":
        required_files.extend(ENFORCED_REQUIRED_FILES)
        if state.get("status") == "complete":
            required_files.append(ARTIFACT_PATHS["packaging"])
            required_files.append(PR_EVIDENCE_PATH)
    for relative in dict.fromkeys(required_files):
        path = task_dir / relative
        if not path.is_file():
            errors.append(f"missing required file: {relative}")
        elif not path.read_text(encoding="utf-8").strip():
            errors.append(f"empty required file: {relative}")

    for path in task_dir.rglob("*.md"):
        if VERSIONED_ARTIFACT_RE.search(path.name):
            errors.append(
                f"versioned workflow artifact is forbidden: {path.relative_to(task_dir)}"
            )
    if errors:
        return errors

    errors.extend(validate_state(task_dir, state))
    if mode != "ENFORCED":
        errors.extend(_legacy_checks(task_dir, state))
        return list(dict.fromkeys(errors))

    for artifact in _required_artifacts(state):
        errors.extend(check_artifact(task_dir, artifact))

    if state.get("preflight_verdict") == "PASS":
        errors.extend(
            stage_result_errors(
                task_dir,
                "preflight",
                str(state.get("preflight_candidate_sha256", "")),
            )
        )
    if state.get("test_verdict") == "PASS":
        errors.extend(
            stage_result_errors(
                task_dir,
                "test",
                str(state.get("test_candidate_sha256", "")),
            )
        )
    if state.get("seal_verdict") == "PASS":
        errors.extend(
            stage_result_errors(
                task_dir,
                "seal",
                str(state.get("sealed_candidate_sha256", "")),
            )
        )

    try:
        current = compute_candidate_fingerprint(task_dir, state)
        phase = state.get("phase")
        status = state.get("status")
        if phase == "implementation":
            if state.get("seal_verdict") == "PASS":
                if current != state.get("sealed_candidate_sha256"):
                    errors.append("current candidate differs from sealed candidate")
            elif state.get("test_verdict") == "PASS":
                if current != state.get("test_candidate_sha256"):
                    errors.append("current candidate differs from Tester candidate")
        if phase in {"validation", "packaging"} or status in {"validated", "complete"}:
            if current != state.get("sealed_candidate_sha256"):
                errors.append("current candidate differs from sealed candidate")
        if phase == "packaging" or status in {"validated", "complete"}:
            if artifact_candidate(task_dir, "validation") != state.get(
                "validation_candidate_sha256"
            ):
                errors.append("validation artifact candidate differs from state")
        if status == "complete":
            errors.extend(pr_evidence_errors(task_dir, state))
            pr_number, head_sha, remote = packaging_metadata(task_dir)
            if pr_number != state.get("pr_number"):
                errors.append("packaging PR number differs from state")
            if head_sha != state.get("pr_head_sha"):
                errors.append("packaging PR head SHA differs from state")
            if remote != "PASS":
                errors.append("packaging remote checks are not PASS")
            revision_candidate = compute_revision_fingerprint(task_dir, state, head_sha)
            if revision_candidate != state.get("packaging_candidate_sha256"):
                errors.append("PR head content differs from packaged candidate")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return list(dict.fromkeys(errors))
