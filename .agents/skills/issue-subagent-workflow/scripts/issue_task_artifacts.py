"""Per-artifact readiness checks for issue-subagent-workflow."""

from __future__ import annotations

import re
from pathlib import Path

from issue_task_candidate import candidate_metadata
from issue_task_remote import evidence_digest, load_pr_evidence
from issue_task_schema import ARTIFACT_CONTRACTS, ARTIFACT_PATHS
from issue_task_state import (
    acceptance_items,
    assert_artifact_test_cycle,
    assert_artifacts_ready,
    assert_checklist_hash_present,
    assert_issue_hash_present,
    assert_mapping_table,
    extract_section,
    feasibility_matrix_errors,
    heading_count,
    load_state,
    standalone_value,
    test_matrix_errors,
    validate_state,
    validation_matrix_errors,
)
from issue_task_verification import manifest_errors, test_probe_result_errors

NONE_VALUES = {"None", "N/A", "なし"}


def _expected_cycle(state: dict[str, object], artifact: str) -> int:
    test_cycle = int(state.get("test_cycle", 0))
    preflight_cycle = int(state.get("preflight_cycle", 0))
    if artifact == "seal":
        return test_cycle if test_cycle > 0 else 1
    if artifact == "tests":
        return preflight_cycle if preflight_cycle > test_cycle else max(test_cycle, 1)
    if artifact in {"implementation", "preflight"}:
        if state.get("test_verdict") == "RETURN":
            return test_cycle + 1
        if preflight_cycle > test_cycle:
            return preflight_cycle
        return max(test_cycle, 1)
    return max(test_cycle, 1)


def _metadata_int(path: Path, label: str) -> int:
    pattern = re.compile(rf"(?m)^- {re.escape(label)}: (\d+)\s*$")
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path.name} does not record {label}")
    return int(match.group(1))


def _metadata_text(path: Path, label: str) -> str:
    pattern = re.compile(rf"(?m)^- {re.escape(label)}: `?([^`\n]+)`?\s*$")
    match = pattern.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"{path.name} does not record {label}")
    return match.group(1).strip()


ADVERSARIAL_TEST_HEADINGS = {
    "## Independent adversarial test design",
    "## Independently derived adversarial tests",
    "## Adversarial probe results",
}


def _section_errors(
    path: Path,
    artifact: str,
    state: dict[str, object],
) -> list[str]:
    contract = ARTIFACT_CONTRACTS[artifact]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    headings = contract.headings
    nonempty_headings = contract.nonempty_headings
    if artifact == "tests" and state.get("adversarial_testing_mode") == "LEGACY":
        headings = tuple(
            heading for heading in headings if heading not in ADVERSARIAL_TEST_HEADINGS
        )
        nonempty_headings = tuple(
            heading
            for heading in nonempty_headings
            if heading not in ADVERSARIAL_TEST_HEADINGS
        )
    for heading in headings:
        count = heading_count(text, heading)
        if count != 1:
            errors.append(
                f"{contract.path} must contain exactly one heading: {heading}; found {count}"
            )
    if errors:
        return errors
    for heading in nonempty_headings:
        section = extract_section(text, heading)
        allow_none = heading in contract.allow_none_headings
        if not section:
            errors.append(f"{contract.path} has an empty section: {heading}")
        elif not allow_none and section.strip() in NONE_VALUES:
            errors.append(f"{contract.path} requires substantive content: {heading}")
    return errors


def check_artifact(task_dir: Path, artifact: str) -> list[str]:
    """Return all contract errors for one completed workflow artifact."""
    if artifact not in ARTIFACT_PATHS:
        return [f"unknown artifact: {artifact}"]

    state = load_state(task_dir)
    contract = ARTIFACT_CONTRACTS[artifact]
    path = task_dir / contract.path
    errors: list[str] = []
    if state.get("candidate_binding_mode") == "ENFORCED":
        errors.extend(validate_state(task_dir, state))
    try:
        assert_artifacts_ready(task_dir, (contract.path,), int(state["attempt"]))
    except ValueError as exc:
        return [str(exc)]

    errors.extend(_section_errors(path, artifact, state))
    if errors:
        return errors

    items = acceptance_items(task_dir)
    try:
        if contract.requires_issue_hash:
            assert_issue_hash_present(path, str(state["issue_sha256"]))
        if contract.requires_checklist_hash:
            assert_checklist_hash_present(
                path,
                str(state["acceptance_checklist_sha256"]),
            )
        if contract.requires_cycle:
            assert_artifact_test_cycle(path, _expected_cycle(state, artifact))
        if contract.requires_candidate:
            candidate_metadata(path)

        if artifact == "feasibility":
            verdict = standalone_value(
                path,
                "## Final feasibility verdict",
                {"PASS", "BLOCKED"},
            )
            errors.extend(
                feasibility_matrix_errors(
                    task_dir,
                    require_all_feasible=verdict == "PASS",
                )
            )
            conflicts = extract_section(
                path.read_text(encoding="utf-8"),
                "## Constraint conflicts",
            ).strip()
            resolution = extract_section(
                path.read_text(encoding="utf-8"),
                "## Blocker resolution required",
            ).strip()
            if verdict == "PASS" and (
                conflicts not in NONE_VALUES or resolution not in NONE_VALUES
            ):
                errors.append("feasibility PASS must not retain an unresolved conflict")
            if verdict == "BLOCKED" and (
                conflicts in NONE_VALUES or resolution in NONE_VALUES
            ):
                errors.append(
                    "feasibility BLOCKED requires conflict and resolution evidence"
                )
        elif artifact == "plan":
            assert_mapping_table(path, "## Acceptance checklist mapping", items)
            errors.extend(manifest_errors(task_dir))
        elif artifact == "preflight":
            verdict = standalone_value(
                path,
                "## Final production preflight verdict",
                {"PASS", "RETURN"},
            )
            findings = extract_section(
                path.read_text(encoding="utf-8"),
                "## RETURN implementation findings",
            ).strip()
            if verdict == "RETURN" and findings in NONE_VALUES:
                errors.append("preflight RETURN requires actionable findings")
            if verdict == "PASS" and findings not in NONE_VALUES:
                errors.append("preflight PASS must not retain RETURN findings")
        elif artifact == "tests":
            assert_mapping_table(
                path,
                "## Acceptance-checklist-to-test mapping",
                items,
            )
            verdict = standalone_value(
                path,
                "## Final test verdict",
                {"PASS", "RETURN"},
            )
            errors.extend(
                test_matrix_errors(
                    task_dir,
                    require_all_pass=verdict == "PASS",
                )
            )
            if state.get("adversarial_testing_mode") == "ENFORCED":
                errors.extend(
                    test_probe_result_errors(
                        task_dir,
                        candidate_metadata(path),
                    )
                )
            findings = extract_section(
                path.read_text(encoding="utf-8"),
                "## RETURN implementation findings",
            ).strip()
            if verdict == "RETURN" and findings in NONE_VALUES:
                errors.append("Tester RETURN requires actionable findings")
            if verdict == "PASS" and findings not in NONE_VALUES:
                errors.append("Tester PASS must not retain RETURN findings")
        elif artifact == "seal":
            verdict = standalone_value(
                path,
                "## Final candidate seal verdict",
                {"PASS", "RETURN"},
            )
            findings = extract_section(
                path.read_text(encoding="utf-8"),
                "## RETURN implementation findings",
            ).strip()
            if verdict == "RETURN" and findings in NONE_VALUES:
                errors.append("candidate seal RETURN requires actionable findings")
            if verdict == "PASS" and findings not in NONE_VALUES:
                errors.append("candidate seal PASS must not retain RETURN findings")
        elif artifact == "validation":
            verdict = standalone_value(
                path,
                "## Final verdict",
                {"PASS", "RETURN"},
            )
            errors.extend(
                validation_matrix_errors(
                    task_dir,
                    require_all_pass=verdict == "PASS",
                )
            )
            questions = extract_section(
                path.read_text(encoding="utf-8"),
                "## RETURN exploration questions",
            ).strip()
            if verdict == "RETURN" and questions in NONE_VALUES:
                errors.append("Validator RETURN requires exploration questions")
            if verdict == "PASS" and questions not in NONE_VALUES:
                errors.append("Validator PASS must not retain RETURN questions")
        elif artifact == "packaging":
            standalone_value(
                path,
                "## Final packaging verdict",
                {"PASS"},
            )
            if _metadata_int(path, "PR number") < 1:
                errors.append("packaging.md requires a positive PR number")
            head = _metadata_text(path, "PR head SHA")
            if not re.fullmatch(r"[0-9a-f]{40}|WORKTREE", head):
                errors.append("packaging.md PR head SHA is invalid")
            if _metadata_text(path, "Remote checks") != "PASS":
                errors.append("packaging.md requires Remote checks PASS")
            evidence = load_pr_evidence(task_dir)
            recorded = _metadata_text(path, "PR evidence SHA-256")
            if recorded != evidence_digest(evidence):
                errors.append("packaging.md PR evidence digest is stale")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def artifact_candidate(task_dir: Path, artifact: str) -> str:
    if artifact not in ARTIFACT_PATHS:
        raise ValueError(f"unknown artifact: {artifact}")
    return candidate_metadata(task_dir / ARTIFACT_PATHS[artifact])


def packaging_metadata(task_dir: Path) -> tuple[int, str, str]:
    path = task_dir / ARTIFACT_PATHS["packaging"]
    return (
        _metadata_int(path, "PR number"),
        _metadata_text(path, "PR head SHA"),
        _metadata_text(path, "Remote checks"),
    )
