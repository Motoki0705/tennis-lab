"""Per-artifact readiness checks for issue-subagent-workflow."""

from __future__ import annotations

from pathlib import Path

from issue_task_state import (
    REQUIRED_HEADINGS,
    acceptance_items,
    assert_artifact_test_cycle,
    assert_artifacts_ready,
    assert_checklist_hash_present,
    assert_mapping_table,
    feasibility_matrix_errors,
    load_state,
    standalone_value,
    validation_matrix_errors,
)

ARTIFACT_PATHS = {
    "feasibility": "00-feasibility/feasibility.md",
    "exploration": "01-exploration/exploration.md",
    "plan": "02-planning/plan.md",
    "implementation": "03-implementation/implementation.md",
    "preflight": "03-implementation/preflight.md",
    "tests": "03-implementation/tests.md",
    "validation": "04-validation/validation.md",
}


def _expected_cycle(state: dict[str, object]) -> int:
    cycle = int(state.get("test_cycle", 0))
    return cycle + 1 if state.get("phase") == "implementation" else cycle


def check_artifact(task_dir: Path, artifact: str) -> list[str]:
    """Return contract errors for one completed workflow artifact."""
    if artifact not in ARTIFACT_PATHS:
        return [f"unknown artifact: {artifact}"]

    state = load_state(task_dir)
    relative = ARTIFACT_PATHS[artifact]
    path = task_dir / relative
    errors: list[str] = []

    try:
        assert_artifacts_ready(task_dir, (relative,), int(state["attempt"]))
    except ValueError as exc:
        errors.append(str(exc))
        return errors

    text = path.read_text(encoding="utf-8")
    for heading in REQUIRED_HEADINGS.get(relative, ()):
        if heading not in text:
            errors.append(f"{relative} is missing heading: {heading}")

    items = acceptance_items(task_dir)
    checklist_hash = str(state["acceptance_checklist_sha256"])

    try:
        if artifact == "feasibility":
            assert_checklist_hash_present(path, checklist_hash)
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
        elif artifact == "plan":
            assert_checklist_hash_present(path, checklist_hash)
            assert_mapping_table(path, "## Acceptance checklist mapping", items)
        elif artifact == "implementation":
            assert_artifact_test_cycle(path, _expected_cycle(state))
        elif artifact == "preflight":
            assert_artifact_test_cycle(path, _expected_cycle(state))
            standalone_value(
                path,
                "## Final preflight verdict",
                {"PASS", "RETURN"},
            )
        elif artifact == "tests":
            assert_artifact_test_cycle(path, _expected_cycle(state))
            assert_checklist_hash_present(path, checklist_hash)
            assert_mapping_table(
                path,
                "## Acceptance-checklist-to-test mapping",
                items,
            )
            standalone_value(
                path,
                "## Final test verdict",
                {"PASS", "RETURN"},
            )
        elif artifact == "validation":
            assert_checklist_hash_present(path, checklist_hash)
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
    except ValueError as exc:
        errors.append(str(exc))

    return errors
