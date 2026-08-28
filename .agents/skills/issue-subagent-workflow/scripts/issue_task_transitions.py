"""Workflow transitions for issue-subagent-workflow."""

from __future__ import annotations

from pathlib import Path

from issue_task_artifacts import artifact_candidate, check_artifact
from issue_task_candidate import compute_candidate_fingerprint
from issue_task_state import (
    CORE_REQUIRED_FILES,
    NEXT_PHASE,
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
from issue_task_verification import load_check_manifest, stage_result_errors


def _enforced(state: dict[str, object]) -> bool:
    return state.get("candidate_binding_mode") == "ENFORCED"


def _raise_errors(errors: list[str]) -> None:
    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))


def _check_artifacts(task_dir: Path, *artifacts: str) -> None:
    errors: list[str] = []
    for artifact in artifacts:
        errors.extend(check_artifact(task_dir, artifact))
    _raise_errors(errors)


def _reset_candidate_evidence(state: dict[str, object]) -> None:
    state["preflight_cycle"] = 0
    state["preflight_verdict"] = ""
    state["preflight_candidate_sha256"] = ""
    state["test_cycle"] = 0
    state["test_verdict"] = ""
    state["test_candidate_sha256"] = ""
    state["seal_cycle"] = 0
    state["seal_verdict"] = ""
    state["sealed_candidate_sha256"] = ""
    state["validation_candidate_sha256"] = ""
    state["packaging_candidate_sha256"] = ""
    state["pr_number"] = 0
    state["pr_head_sha"] = ""
    state["remote_checks_verdict"] = ""
    state["pr_evidence_sha256"] = ""
    state["test_return_count"] = 0
    state["return_review_required"] = False
    state["return_review_action"] = ""
    state["return_review_reason"] = ""


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
    _raise_errors(validate_state(task_dir, state))

    attempt = int(state["attempt"])
    path = task_dir / "00-feasibility/feasibility.md"
    if _enforced(state):
        _check_artifacts(task_dir, "feasibility")
    else:
        assert_artifacts_ready(task_dir, ("00-feasibility/feasibility.md",), attempt)
        assert_checklist_hash_present(path, str(state["acceptance_checklist_sha256"]))

    if verdict == "PASS":
        if kind is not None or reason is not None:
            raise ValueError("feasibility PASS must not include block kind or reason")
        errors = feasibility_matrix_errors(task_dir, require_all_feasible=True)
        _raise_errors(errors)
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
        _raise_errors(errors)
        assert_standalone_value(
            path,
            "## Final feasibility verdict",
            "BLOCKED",
            {"PASS", "BLOCKED"},
        )
        assert_nonempty_section(
            path, "## Constraint conflicts", "constraint conflict evidence"
        )
        assert_nonempty_section(
            path, "## Blocker resolution required", "blocker resolution"
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
    _raise_errors(validate_state(task_dir, state))

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
        if _enforced(state):
            _check_artifacts(task_dir, "exploration")
        else:
            assert_artifacts_ready(
                task_dir, ("01-exploration/exploration.md",), attempt
            )
    elif requested == "implementation":
        plan_path = task_dir / "02-planning/plan.md"
        if _enforced(state):
            _check_artifacts(task_dir, "plan")
            load_check_manifest(task_dir)
        else:
            assert_artifacts_ready(task_dir, ("02-planning/plan.md",), attempt)
            assert_checklist_hash_present(plan_path, checklist_hash)
            assert_mapping_table(plan_path, "## Acceptance checklist mapping", items)
        _reset_candidate_evidence(state)
    else:
        if state.get("test_verdict") != "PASS":
            raise ValueError("cannot enter validation before test-verdict PASS")
        cycle = int(state["test_cycle"])
        if _enforced(state):
            if state.get("seal_verdict") != "PASS" or state.get("seal_cycle") != cycle:
                raise ValueError("cannot enter validation before candidate seal PASS")
            _check_artifacts(task_dir, "implementation", "tests", "seal")
            current_candidate = compute_candidate_fingerprint(task_dir, state)
            expected_candidate = str(state.get("sealed_candidate_sha256", ""))
            if current_candidate != expected_candidate:
                raise ValueError("candidate changed after the final seal")
            if state.get("test_candidate_sha256") != expected_candidate:
                raise ValueError(
                    "Tester and sealed candidate fingerprints do not match"
                )
            if artifact_candidate(task_dir, "tests") != expected_candidate:
                raise ValueError(
                    "tests.md candidate fingerprint does not match the seal"
                )
            if artifact_candidate(task_dir, "seal") != expected_candidate:
                raise ValueError("seal.md candidate fingerprint does not match state")
            _raise_errors(stage_result_errors(task_dir, "test", expected_candidate))
            _raise_errors(stage_result_errors(task_dir, "seal", expected_candidate))
        else:
            implementation_path = task_dir / "03-implementation/implementation.md"
            tests_path = task_dir / "03-implementation/tests.md"
            assert_artifacts_ready(
                task_dir,
                ("03-implementation/implementation.md", "03-implementation/tests.md"),
                attempt,
            )
            assert_artifact_test_cycle(implementation_path, cycle)
            assert_artifact_test_cycle(tests_path, cycle)
            assert_checklist_hash_present(tests_path, checklist_hash)
            assert_mapping_table(
                tests_path, "## Acceptance-checklist-to-test mapping", items
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
                    task_dir, ("03-implementation/preflight.md",), attempt
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
    _raise_errors(validate_state(task_dir, state))

    attempt = int(state["attempt"])
    cycle = int(state["test_cycle"]) + 1
    implementation_path = task_dir / "03-implementation/implementation.md"
    preflight_path = task_dir / "03-implementation/preflight.md"
    if _enforced(state):
        _check_artifacts(task_dir, "implementation", "preflight")
        candidate = compute_candidate_fingerprint(task_dir, state)
        if artifact_candidate(task_dir, "preflight") != candidate:
            raise ValueError("preflight.md does not identify the current candidate")
        if verdict == "PASS":
            _raise_errors(stage_result_errors(task_dir, "preflight", candidate))
        state["preflight_candidate_sha256"] = candidate
        state["test_candidate_sha256"] = ""
        state["seal_cycle"] = 0
        state["seal_verdict"] = ""
        state["sealed_candidate_sha256"] = ""
        state["validation_candidate_sha256"] = ""
    else:
        assert_artifacts_ready(
            task_dir,
            ("03-implementation/implementation.md", "03-implementation/preflight.md"),
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
    _raise_errors(validate_state(task_dir, state))

    attempt = int(state["attempt"])
    cycle = int(state["test_cycle"]) + 1
    if (
        state.get("preflight_verdict") != "PASS"
        or state.get("preflight_cycle") != cycle
    ):
        raise ValueError("test verdict requires a matching production preflight PASS")

    items = acceptance_items(task_dir)
    implementation_path = task_dir / "03-implementation/implementation.md"
    tests_path = task_dir / "03-implementation/tests.md"
    if _enforced(state):
        _check_artifacts(task_dir, "tests")
        candidate = compute_candidate_fingerprint(task_dir, state)
        if artifact_candidate(task_dir, "tests") != candidate:
            raise ValueError("tests.md does not identify the current candidate")
        if verdict == "PASS":
            _raise_errors(stage_result_errors(task_dir, "test", candidate))
        state["test_candidate_sha256"] = candidate
        state["seal_cycle"] = 0
        state["seal_verdict"] = ""
        state["sealed_candidate_sha256"] = ""
        state["validation_candidate_sha256"] = ""
    else:
        assert_artifacts_ready(
            task_dir,
            ("03-implementation/implementation.md", "03-implementation/tests.md"),
            attempt,
        )
        assert_artifact_test_cycle(implementation_path, cycle)
        assert_artifact_test_cycle(tests_path, cycle)
        assert_checklist_hash_present(
            tests_path, str(state["acceptance_checklist_sha256"])
        )
        assert_mapping_table(
            tests_path, "## Acceptance-checklist-to-test mapping", items
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


def apply_seal_verdict(task_dir: Path, verdict: str) -> None:
    state = load_state(task_dir)
    if not _enforced(state):
        raise ValueError(
            "candidate seal is available only for schema-v5-or-newer tasks"
        )
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError(
            "a seal verdict is valid only during in-progress implementation"
        )
    if state.get("test_verdict") != "PASS" or int(state.get("test_cycle", 0)) < 1:
        raise ValueError("candidate seal requires Tester PASS")
    _raise_errors(validate_state(task_dir, state))
    _check_artifacts(task_dir, "seal")

    candidate = compute_candidate_fingerprint(task_dir, state)
    if candidate != state.get("test_candidate_sha256"):
        raise ValueError("candidate changed after Tester PASS; rerun the Test Writer")
    if artifact_candidate(task_dir, "seal") != candidate:
        raise ValueError("seal.md does not identify the current candidate")
    if verdict == "PASS":
        _raise_errors(stage_result_errors(task_dir, "seal", candidate))
    state["seal_cycle"] = int(state["test_cycle"])
    state["seal_verdict"] = verdict
    state["sealed_candidate_sha256"] = candidate
    state["validation_candidate_sha256"] = ""
    state["verdict"] = ""
    write_state(task_dir, state)


def apply_return_review(task_dir: Path, action: str, reason: str) -> None:
    state = load_state(task_dir)
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError(
            "return review is valid only during in-progress implementation"
        )
    if not state.get("return_review_required"):
        raise ValueError("return review is not currently required")
    _raise_errors(validate_state(task_dir, state))

    state["return_review_required"] = False
    state["return_review_action"] = action
    state["return_review_reason"] = reason
    state["test_return_count"] = 0
    if action == "exploration":
        state["attempt"] = int(state["attempt"]) + 1
        _reset_candidate_evidence(state)
        state["phase"] = "exploration"
        state["verdict"] = "RETURN_REVIEW"
    write_state(task_dir, state)


def block_task(task_dir: Path, kind: str, reason: str) -> None:
    state = load_state(task_dir)
    if state.get("status") != "in_progress":
        raise ValueError("only an in-progress task can be blocked")
    _raise_errors(validate_state(task_dir, state))
    state["status"] = "blocked"
    state["verdict"] = "BLOCKED"
    state["block_kind"] = kind
    state["block_reason"] = reason
    state["return_review_required"] = False
    write_state(task_dir, state)


def apply_validation_verdict(task_dir: Path, verdict: str) -> None:
    """Apply a validated artifact verdict; caller performs complete prechecks."""
    state = load_state(task_dir)
    if state.get("phase") != "validation" or state.get("status") != "in_progress":
        raise ValueError(
            "a Validator verdict is valid only during in-progress validation"
        )
    _raise_errors(validate_state(task_dir, state))

    attempt = int(state["attempt"])
    validation_path = task_dir / "04-validation/validation.md"
    if verdict == "PASS":
        if _enforced(state):
            candidate = compute_candidate_fingerprint(task_dir, state)
            if candidate != state.get("sealed_candidate_sha256"):
                raise ValueError(
                    "candidate changed after seal and before Validator PASS"
                )
            if artifact_candidate(task_dir, "validation") != candidate:
                raise ValueError("validation.md does not identify the sealed candidate")
            state["validation_candidate_sha256"] = candidate
            state["phase"] = "packaging"
            state["status"] = "validated"
            state["verdict"] = "VALIDATED"
        else:
            required = tuple(
                path
                for path in CORE_REQUIRED_FILES
                if path not in {"issue.md", "state.toml"}
            )
            if (task_dir / "03-implementation/preflight.md").is_file():
                required += ("03-implementation/preflight.md",)
            assert_artifacts_ready(task_dir, required, attempt)
            assert_checklist_hash_present(
                validation_path, str(state["acceptance_checklist_sha256"])
            )
            _raise_errors(validation_matrix_errors(task_dir, require_all_pass=True))
            assert_standalone_value(
                validation_path,
                "## Final verdict",
                "PASS",
                {"PASS", "RETURN"},
            )
            state["status"] = "complete"
            state["verdict"] = "PASS"
    else:
        if _enforced(state):
            candidate = compute_candidate_fingerprint(task_dir, state)
            if candidate != state.get("sealed_candidate_sha256"):
                raise ValueError(
                    "candidate changed before Validator RETURN was applied"
                )
            if artifact_candidate(task_dir, "validation") != candidate:
                raise ValueError("validation.md does not identify the sealed candidate")
        else:
            assert_artifacts_ready(task_dir, ("04-validation/validation.md",), attempt)
            assert_checklist_hash_present(
                validation_path, str(state["acceptance_checklist_sha256"])
            )
            _raise_errors(validation_matrix_errors(task_dir, require_all_pass=False))
            assert_standalone_value(
                validation_path,
                "## Final verdict",
                "RETURN",
                {"PASS", "RETURN"},
            )
        state["attempt"] = attempt + 1
        _reset_candidate_evidence(state)
        state["phase"] = "exploration"
        state["status"] = "in_progress"
        state["verdict"] = "RETURN"
    write_state(task_dir, state)
