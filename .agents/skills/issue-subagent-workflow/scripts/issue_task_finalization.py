"""Fail-closed validation and PR finalization."""

from __future__ import annotations

from pathlib import Path

from issue_task_artifacts import (
    artifact_candidate,
    check_artifact,
    packaging_metadata,
)
from issue_task_candidate import (
    compute_candidate_fingerprint,
    compute_revision_fingerprint,
    current_revision,
    revision_changed_paths,
)
from issue_task_checks import check
from issue_task_remote import load_pr_evidence, pr_evidence_errors
from issue_task_state import (
    PACKAGING_PENDING_AC_ID,
    acceptance_items,
    assert_standalone_value,
    extract_section,
    load_state,
    validate_state,
    validation_matrix_errors,
    write_state,
)
from issue_task_transitions import apply_validation_verdict as _apply_validation_verdict

PACKAGING_AC079_EVIDENCE_TOKENS = (
    "AC-079",
    "capture-pr",
    "exact PR head",
    "complete paginated files",
    "candidate equality",
    "required remote checks",
    "packaging.md",
    "captured evidence",
    "finalize-pr",
    "final workflow check",
)


def _raise_errors(errors: list[str]) -> None:
    if errors:
        raise ValueError("pre-completion check failed: " + "; ".join(dict.fromkeys(errors)))


def _packaging_ac079_evidence_errors(task_dir: Path) -> list[str]:
    if PACKAGING_PENDING_AC_ID not in {
        item_id for item_id, _ in acceptance_items(task_dir)
    }:
        return []
    packaging_path = task_dir / "05-packaging/packaging.md"
    try:
        evidence = extract_section(
            packaging_path.read_text(encoding="utf-8"),
            "## Packaging evidence",
        )
    except (OSError, ValueError) as exc:
        return [str(exc)]
    missing = [token for token in PACKAGING_AC079_EVIDENCE_TOKENS if token not in evidence]
    if not missing:
        return []
    return [
        "packaging.md does not establish AC-079; missing evidence tokens: "
        + ", ".join(missing)
    ]


def apply_validation_verdict(task_dir: Path, verdict: str) -> None:
    """Validate the artifact set before applying a Validator verdict."""
    state = load_state(task_dir)
    if state.get("candidate_binding_mode") == "ENFORCED":
        artifacts = (
            "feasibility",
            "exploration",
            "plan",
            "implementation",
            "preflight",
            "tests",
            "seal",
            "validation",
        )
        errors: list[str] = []
        for artifact in artifacts:
            errors.extend(check_artifact(task_dir, artifact))
        errors.extend(validate_state(task_dir, state))
        errors.extend(check(task_dir))
        if verdict == "PASS":
            current = compute_candidate_fingerprint(task_dir, state)
            if current != state.get("sealed_candidate_sha256"):
                errors.append("candidate changed after the final seal")
            if artifact_candidate(task_dir, "validation") != current:
                errors.append("validation.md does not identify the sealed candidate")
        validation_path = task_dir / "04-validation/validation.md"
        errors.extend(
            validation_matrix_errors(
                task_dir,
                require_all_pass=verdict == "PASS",
            )
        )
        try:
            assert_standalone_value(
                validation_path,
                "## Final verdict",
                verdict,
                {"PASS", "RETURN"},
            )
        except ValueError as exc:
            errors.append(str(exc))
        _raise_errors(errors)
    else:
        validation_path = task_dir / "04-validation/validation.md"
        errors = validation_matrix_errors(
            task_dir,
            require_all_pass=verdict == "PASS",
        )
        try:
            assert_standalone_value(
                validation_path,
                "## Final verdict",
                verdict,
                {"PASS", "RETURN"},
            )
        except ValueError as exc:
            errors.append(str(exc))
        if verdict == "PASS":
            errors.extend(check(task_dir))
        _raise_errors(errors)
    _apply_validation_verdict(task_dir, verdict)


def finalize_pr(
    task_dir: Path,
    *,
    pr_number: int,
    head_sha: str,
) -> None:
    """Bind the validated candidate to a checked remote PR head and complete."""
    state = load_state(task_dir)
    if state.get("candidate_binding_mode") != "ENFORCED":
        raise ValueError("finalize-pr is available only for schema v5 tasks")
    if state.get("phase") != "packaging" or state.get("status") != "validated":
        raise ValueError("finalize-pr requires packaging/validated state")

    errors = check_artifact(task_dir, "packaging")
    errors.extend(validate_state(task_dir, state))
    errors.extend(pr_evidence_errors(task_dir, state))
    errors.extend(_packaging_ac079_evidence_errors(task_dir))
    evidence = load_pr_evidence(task_dir)
    packaged_pr, packaged_head, remote = packaging_metadata(task_dir)
    candidate = compute_candidate_fingerprint(task_dir, state)
    packaged_candidate = artifact_candidate(task_dir, "packaging")
    if evidence.get("pr_number") != pr_number:
        errors.append("captured PR evidence number does not match finalize-pr")
    if evidence.get("head_sha") != head_sha:
        errors.append("captured PR evidence head does not match finalize-pr")
    if packaged_pr != pr_number:
        errors.append("packaging.md PR number does not match finalize-pr")
    if packaged_head != head_sha:
        errors.append("packaging.md PR head SHA does not match finalize-pr")
    if remote != "PASS":
        errors.append("remote required checks are not PASS")
    if candidate != state.get("validation_candidate_sha256"):
        errors.append("working candidate differs from the validated candidate")
    if packaged_candidate != candidate:
        errors.append("packaging.md candidate differs from the validated candidate")
    if current_revision(task_dir) != head_sha:
        errors.append("local HEAD does not match the supplied PR head SHA")
    try:
        revision_candidate = compute_revision_fingerprint(task_dir, state, head_sha)
    except ValueError as exc:
        errors.append(str(exc))
        revision_candidate = ""
    if revision_candidate != candidate:
        errors.append("PR head content differs from the validated candidate")
    try:
        expected_files = revision_changed_paths(task_dir, state, head_sha)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        if evidence.get("files") != expected_files:
            errors.append(
                "captured complete paginated PR files differ from the final revision"
            )
    _raise_errors(errors)

    next_state = dict(state)
    next_state["packaging_candidate_sha256"] = candidate
    next_state["pr_number"] = pr_number
    next_state["pr_head_sha"] = head_sha
    next_state["remote_checks_verdict"] = "PASS"
    next_state["status"] = "complete"
    next_state["verdict"] = "PASS"
    _raise_errors(validate_state(task_dir, next_state))
    _raise_errors(check(task_dir, state_override=next_state))
    write_state(task_dir, next_state)
