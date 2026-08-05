"""Consistency checks for issue-subagent-workflow artifacts."""

from __future__ import annotations

import tomllib
from pathlib import Path

from issue_task_state import (
    CORE_REQUIRED_FILES,
    REQUIRED_HEADINGS,
    VERSIONED_ARTIFACT_RE,
    acceptance_items,
    assert_artifact_test_cycle,
    assert_checklist_hash_present,
    assert_mapping_table,
    assert_standalone_value,
    feasibility_matrix_errors,
    load_state,
    validate_state,
    validation_matrix_errors,
)


def check(task_dir: Path) -> list[str]:
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
        state = load_state(task_dir)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [str(exc)]

    required_files = list(CORE_REQUIRED_FILES)
    if state.get("feasibility_verdict") != "LEGACY":
        required_files.append("00-feasibility/feasibility.md")
    if (
        state.get("feasibility_verdict") != "LEGACY"
        or int(state.get("preflight_cycle", 0)) > 0
    ):
        required_files.append("03-implementation/preflight.md")
    for relative in required_files:
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

    headings_to_check: dict[str, tuple[str, ...]] = {}
    for relative, headings in REQUIRED_HEADINGS.items():
        if relative == "00-feasibility/feasibility.md":
            if state.get("feasibility_verdict") != "LEGACY":
                headings_to_check[relative] = headings
            continue
        if relative == "03-implementation/preflight.md":
            if (
                state.get("feasibility_verdict") != "LEGACY"
                or int(state.get("preflight_cycle", 0)) > 0
            ):
                headings_to_check[relative] = headings
            continue
        headings_to_check[relative] = headings
    for relative, headings in headings_to_check.items():
        text = (task_dir / relative).read_text(encoding="utf-8")
        for heading in headings:
            if heading not in text:
                errors.append(f"{relative} is missing heading: {heading}")

    try:
        items = acceptance_items(task_dir)
    except ValueError:
        items = []

    phase = state.get("phase")
    status = state.get("status")
    checklist_hash = str(state.get("acceptance_checklist_sha256", ""))
    feasibility_verdict = state.get("feasibility_verdict")

    if items and feasibility_verdict in {"PASS", "BLOCKED"}:
        path = task_dir / "00-feasibility/feasibility.md"
        try:
            assert_checklist_hash_present(path, checklist_hash)
            errors.extend(
                feasibility_matrix_errors(
                    task_dir,
                    require_all_feasible=feasibility_verdict == "PASS",
                )
            )
            assert_standalone_value(
                path,
                "## Final feasibility verdict",
                feasibility_verdict,
                {"PASS", "BLOCKED"},
            )
        except ValueError as exc:
            errors.append(str(exc))

    if items and (phase in {"implementation", "validation"} or status == "complete"):
        plan_path = task_dir / "02-planning/plan.md"
        try:
            assert_checklist_hash_present(plan_path, checklist_hash)
            assert_mapping_table(plan_path, "## Acceptance checklist mapping", items)
        except ValueError as exc:
            errors.append(str(exc))

    preflight_cycle = int(state.get("preflight_cycle", 0))
    if preflight_cycle > 0 and (task_dir / "03-implementation/preflight.md").is_file():
        preflight_path = task_dir / "03-implementation/preflight.md"
        try:
            assert_artifact_test_cycle(preflight_path, preflight_cycle)
            assert_standalone_value(
                preflight_path,
                "## Final preflight verdict",
                str(state.get("preflight_verdict")),
                {"PASS", "RETURN"},
            )
        except ValueError as exc:
            errors.append(str(exc))

    if items and (phase == "validation" or status == "complete"):
        implementation_path = task_dir / "03-implementation/implementation.md"
        tests_path = task_dir / "03-implementation/tests.md"
        cycle = int(state.get("test_cycle", 0))
        try:
            assert_artifact_test_cycle(implementation_path, cycle)
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
        attempt = int(state.get("attempt", 0))
        required_headings = headings_to_check
        for relative in required_headings:
            text = (task_dir / relative).read_text(encoding="utf-8")
            if "PENDING" in text or "Replace this" in text:
                errors.append(f"complete task contains placeholders: {relative}")
            if f"- Attempt: {attempt}" not in text:
                errors.append(
                    f"complete task artifact does not record attempt {attempt}: {relative}"
                )

        validation_path = task_dir / "04-validation/validation.md"
        try:
            assert_checklist_hash_present(validation_path, checklist_hash)
            errors.extend(
                validation_matrix_errors(task_dir, require_all_pass=True)
            )
            assert_standalone_value(
                validation_path,
                "## Final verdict",
                "PASS",
                {"PASS", "RETURN"},
            )
        except ValueError as exc:
            errors.append(str(exc))
        if state.get("verdict") != "PASS":
            errors.append("complete task state verdict must be PASS")

    return errors
