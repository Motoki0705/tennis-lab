"""Workflow transitions for issue-subagent-workflow."""

from __future__ import annotations

from pathlib import Path

from issue_task_state import (
    EFFICIENCY_REQUIRED_FILES,
    NEXT_PHASE,
    REQUIRED_HEADINGS,
    acceptance_items,
    assert_artifact_test_cycle,
    assert_artifacts_ready,
    assert_checklist_hash_present,
    assert_mapping_table,
    assert_nonempty_section,
    assert_standalone_value,
    feasibility_matrix_errors,
    load_state,
    validate_state,
    validation_matrix_errors,
    write_state,
)


def apply_feasibility_verdict(
    task_dir: Path,
    verdict: str,
    *,
    kind: str | None,
    reason: str | None,
) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "feasibility" or state.get("status") != "in_progress":
        raise ValueError(
            "a feasibility verdict is valid only during in-progress feasibility"
        )
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    attempt = int(state["attempt"])
    path = task_dir / "00-feasibility/feasibility.md"
    assert_artifacts_ready(task_dir, ("00-feasibility/feasibility.md",), attempt)
    assert_checklist_hash_present(path, str(state["acceptance_checklist_sha256"]))

    if verdict == "PASS":
        if kind is not None or reason is not None:
            raise ValueError("feasibility PASS must not include block kind or reason")
        errors = feasibility_matrix_errors(task_dir, require_all_feasible=True)
        if errors:
            raise ValueError("; ".join(errors))
        assert_standalone_value(
            path,
            "## Final feasibility verdict",
            "PASS",
            {"PASS", "BLOCKED"},
        )
        state["feasibility_verdict"] = "PASS"
        state["phase"] = "exploration"
        state["verdict"] = ""
    else:
        if kind is None or reason is None:
            raise ValueError("feasibility BLOCKED requires --kind and --reason")
        errors = feasibility_matrix_errors(task_dir, require_all_feasible=False)
        if errors:
            raise ValueError("; ".join(errors))
        assert_standalone_value(
            path,
            "## Final feasibility verdict",
            "BLOCKED",
            {"PASS", "BLOCKED"},
        )
        assert_nonempty_section(
            path,
            "## Constraint conflicts",
            "constraint conflict evidence",
        )
        assert_nonempty_section(
            path,
            "## Blocker resolution required",
            "blocker resolution",
        )
        state["feasibility_verdict"] = "BLOCKED"
        state["status"] = "blocked"
        state["verdict"] = "BLOCKED"
        state["block_kind"] = kind
        state["block_reason"] = reason
    write_state(task_dir, state)


def transition(task_dir: Path, requested: str) -> None:
    state = load_state(task_dir)
    if state.get("status") != "in_progress":
        raise ValueError("cannot transition a task that is not in_progress")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    current = str(state.get("phase"))
    expected = NEXT_PHASE.get(current)
    if requested != expected:
        raise ValueError(
            f"invalid transition: {current!r} -> {requested!r}; expected {expected!r}"
        )

    attempt = int(state["attempt"])
    items = acceptance_items(task_dir)
    checklist_hash = str(state["acceptance_checklist_sha256"])

    if requested == "planning":
        assert_artifacts_ready(
            task_dir,
            ("01-exploration/exploration.md",),
            attempt,
        )
    elif requested == "implementation":
        plan_path = task_dir / "02-planning/plan.md"
        assert_artifacts_ready(task_dir, ("02-planning/plan.md",), attempt)
        assert_checklist_hash_present(plan_path, checklist_hash)
        assert_mapping_table(plan_path, "## Acceptance checklist mapping", items)
        state["preflight_cycle"] = 0
        state["preflight_verdict"] = ""
        state["test_cycle"] = 0
        state["test_verdict"] = ""
        state["test_return_count"] = 0
        state["return_review_required"] = False
        state["return_review_action"] = ""
        state["return_review_reason"] = ""
    else:
        if state.get("test_verdict") != "PASS":
            raise ValueError("cannot enter validation before test-verdict PASS")
        cycle = int(state["test_cycle"])
        implementation_path = task_dir / "03-implementation/implementation.md"
        tests_path = task_dir / "03-implementation/tests.md"
        assert_artifacts_ready(
            task_dir,
            (
                "03-implementation/implementation.md",
                "03-implementation/tests.md",
            ),
            attempt,
        )
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
        if (
            state.get("feasibility_verdict") != "LEGACY"
            or int(state.get("preflight_cycle", 0)) > 0
        ):
            preflight_path = task_dir / "03-implementation/preflight.md"
            assert_artifacts_ready(
                task_dir,
                ("03-implementation/preflight.md",),
                attempt,
            )
            assert_artifact_test_cycle(preflight_path, cycle)
            assert_standalone_value(
                preflight_path,
                "## Final preflight verdict",
                "PASS",
                {"PASS", "RETURN"},
            )

    state["phase"] = requested
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_preflight_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError(
            "a preflight verdict is valid only during in-progress implementation"
        )
    if state.get("return_review_required"):
        raise ValueError("return review is required before another preflight cycle")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    attempt = int(state["attempt"])
    cycle = int(state["test_cycle"]) + 1
    implementation_path = task_dir / "03-implementation/implementation.md"
    preflight_path = task_dir / "03-implementation/preflight.md"
    assert_artifacts_ready(
        task_dir,
        (
            "03-implementation/implementation.md",
            "03-implementation/preflight.md",
        ),
        attempt,
    )
    assert_artifact_test_cycle(implementation_path, cycle)
    assert_artifact_test_cycle(preflight_path, cycle)
    assert_standalone_value(
        preflight_path,
        "## Final preflight verdict",
        verdict,
        {"PASS", "RETURN"},
    )
    if verdict == "RETURN":
        assert_nonempty_section(
            preflight_path,
            "## RETURN implementation findings",
            "RETURN implementation findings",
        )

    state["preflight_cycle"] = cycle
    state["preflight_verdict"] = verdict
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_test_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError(
            "a test verdict is valid only during in-progress implementation"
        )
    if state.get("return_review_required"):
        raise ValueError("return review is required before another test cycle")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    attempt = int(state["attempt"])
    cycle = int(state["test_cycle"]) + 1
    if (
        state.get("preflight_verdict") != "PASS"
        or state.get("preflight_cycle") != cycle
    ):
        raise ValueError(
            "test verdict requires a matching preflight-verdict PASS for this cycle"
        )

    items = acceptance_items(task_dir)
    implementation_path = task_dir / "03-implementation/implementation.md"
    tests_path = task_dir / "03-implementation/tests.md"

    assert_artifacts_ready(
        task_dir,
        (
            "03-implementation/implementation.md",
            "03-implementation/tests.md",
        ),
        attempt,
    )
    assert_artifact_test_cycle(implementation_path, cycle)
    assert_artifact_test_cycle(tests_path, cycle)
    assert_checklist_hash_present(
        tests_path,
        str(state["acceptance_checklist_sha256"]),
    )
    assert_mapping_table(
        tests_path,
        "## Acceptance-checklist-to-test mapping",
        items,
    )
    assert_standalone_value(
        tests_path,
        "## Final test verdict",
        verdict,
        {"PASS", "RETURN"},
    )
    if verdict == "RETURN":
        assert_nonempty_section(
            tests_path,
            "## RETURN implementation findings",
            "RETURN implementation findings",
        )

    state["test_cycle"] = cycle
    state["test_verdict"] = verdict
    state["verdict"] = ""
    if verdict == "RETURN":
        state["test_return_count"] = int(state["test_return_count"]) + 1
        if state["test_return_count"] >= 2:
            state["return_review_required"] = True
    else:
        state["test_return_count"] = 0
        state["return_review_required"] = False
        state["return_review_action"] = ""
        state["return_review_reason"] = ""
    write_state(task_dir, state)


def apply_return_review(task_dir: Path, action: str, reason: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError("return review is valid only during in-progress implementation")
    if not state.get("return_review_required"):
        raise ValueError("return review is not currently required")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    state["return_review_required"] = False
    state["return_review_action"] = action
    state["return_review_reason"] = reason
    state["test_return_count"] = 0
    if action == "exploration":
        state["attempt"] = int(state["attempt"]) + 1
        state["preflight_cycle"] = 0
        state["preflight_verdict"] = ""
        state["test_cycle"] = 0
        state["test_verdict"] = ""
        state["phase"] = "exploration"
        state["verdict"] = "RETURN_REVIEW"
    write_state(task_dir, state)


def block_task(task_dir: Path, kind: str, reason: str) -> None:
    state = load_state(task_dir)
    if state.get("status") != "in_progress":
        raise ValueError("only an in-progress task can be blocked")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))
    state["status"] = "blocked"
    state["verdict"] = "BLOCKED"
    state["block_kind"] = kind
    state["block_reason"] = reason
    state["return_review_required"] = False
    write_state(task_dir, state)


def apply_validation_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "validation" or state.get("status") != "in_progress":
        raise ValueError("a validator verdict is valid only during in-progress validation")
    state_errors = validate_state(task_dir, state)
    if state_errors:
        raise ValueError("; ".join(state_errors))

    attempt = int(state["attempt"])
    validation_path = task_dir / "04-validation/validation.md"
    if verdict == "PASS":
        required = tuple(
            path
            for path in REQUIRED_HEADINGS
            if state.get("feasibility_verdict") != "LEGACY"
            or path not in EFFICIENCY_REQUIRED_FILES
        )
        assert_artifacts_ready(task_dir, required, attempt)
        assert_checklist_hash_present(
            validation_path,
            str(state["acceptance_checklist_sha256"]),
        )
        errors = validation_matrix_errors(task_dir, require_all_pass=True)
        if errors:
            raise ValueError("; ".join(errors))
        assert_standalone_value(
            validation_path,
            "## Final verdict",
            "PASS",
            {"PASS", "RETURN"},
        )
        state["status"] = "complete"
        state["verdict"] = "PASS"
    else:
        assert_artifacts_ready(
            task_dir,
            ("04-validation/validation.md",),
            attempt,
        )
        assert_checklist_hash_present(
            validation_path,
            str(state["acceptance_checklist_sha256"]),
        )
        errors = validation_matrix_errors(task_dir, require_all_pass=False)
        if errors:
            raise ValueError("; ".join(errors))
        assert_standalone_value(
            validation_path,
            "## Final verdict",
            "RETURN",
            {"PASS", "RETURN"},
        )
        state["attempt"] = attempt + 1
        state["preflight_cycle"] = 0
        state["preflight_verdict"] = ""
        state["test_cycle"] = 0
        state["test_verdict"] = ""
        state["test_return_count"] = 0
        state["return_review_required"] = False
        state["return_review_action"] = ""
        state["return_review_reason"] = ""
        state["phase"] = "exploration"
        state["status"] = "in_progress"
        state["verdict"] = "RETURN"
    write_state(task_dir, state)
