"""Install bounded Preflight and Seal retry policy on public workflow commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import issue_task_artifacts as _artifacts
import issue_task_checks as _checks
import issue_task_finalization as _finalization
import issue_task_remote as _remote
import issue_task_state as _state
import issue_task_transitions as _transitions
from issue_task_verification import run_check as _run_check
from issue_task_verification import run_test_probe as _run_test_probe

_BASE_VALIDATE_STATE = _state.validate_state


def _pending_preflight_cycle(state: dict[str, Any]) -> int:
    return int(state.get("test_cycle", 0)) + 1


def _review_source(state: dict[str, Any]) -> str:
    if state.get("return_review_required") is not True:
        return ""
    pending_preflight_cycle = _pending_preflight_cycle(state)
    if (
        state.get("phase") == "implementation"
        and state.get("preflight_verdict") == "RETURN"
        and int(state.get("preflight_cycle", 0)) == pending_preflight_cycle
    ):
        return "preflight"
    if (
        state.get("phase") == "implementation"
        and state.get("test_verdict") == "RETURN"
        and int(state.get("test_return_count", 0)) >= 2
    ):
        return "test"
    if (
        state.get("phase") == "implementation"
        and state.get("test_verdict") == "PASS"
        and state.get("seal_verdict") == "RETURN"
        and int(state.get("seal_cycle", 0)) == int(state.get("test_cycle", 0))
    ):
        return "seal"
    return "invalid"


def _bounded_validate_state(
    task_dir: Path,
    state: dict[str, Any],
) -> list[str]:
    """Extend enforced candidate validation to bounded Preflight and Seal gates."""
    if state.get("return_review_required") is not True:
        return _BASE_VALIDATE_STATE(task_dir, state)

    base_state = dict(state)
    base_state["return_review_required"] = False
    errors = _BASE_VALIDATE_STATE(task_dir, base_state)
    if _review_source(state) == "invalid":
        errors.append(
            "return_review_required must follow repeated Preflight RETURN, "
            "two Tester RETURNs, or Seal RETURN"
        )
    return list(dict.fromkeys(errors))


def _install_validator() -> None:
    """Install one validator across modules imported by the public CLI."""
    _state.validate_state = _bounded_validate_state
    _artifacts.validate_state = _bounded_validate_state
    _checks.validate_state = _bounded_validate_state
    _finalization.validate_state = _bounded_validate_state
    _remote.validate_state = _bounded_validate_state
    _transitions.validate_state = _bounded_validate_state


_install_validator()


def _require_open_gate(task_dir: Path, operation: str) -> dict[str, Any]:
    state = _state.load_state(task_dir)
    if state.get("return_review_required"):
        raise ValueError(f"return review is required before {operation}")
    return state


def check(task_dir: Path) -> list[str]:
    """Run the whole-task check with bounded-review state semantics."""
    return _checks.check(task_dir)


def transition(task_dir: Path, requested: str) -> None:
    _require_open_gate(task_dir, "a phase transition")
    _transitions.transition(task_dir, requested)


def reopen_packaging_repair(task_dir: Path, reason: str) -> None:
    _transitions.reopen_packaging_repair(task_dir, reason)


def apply_feasibility_verdict(
    task_dir: Path,
    verdict: str,
    *,
    kind: str | None,
    reason: str | None,
) -> None:
    _require_open_gate(task_dir, "a feasibility verdict")
    _transitions.apply_feasibility_verdict(
        task_dir,
        verdict,
        kind=kind,
        reason=reason,
    )


def apply_preflight_verdict(task_dir: Path, verdict: str) -> None:
    state = _require_open_gate(task_dir, "another preflight cycle")
    cycle = _pending_preflight_cycle(state)
    previous_return = (
        state.get("preflight_verdict") == "RETURN"
        and int(state.get("preflight_cycle", 0)) == cycle
    )
    _transitions.apply_preflight_verdict(task_dir, verdict)
    if verdict == "RETURN" and previous_return:
        updated = _state.load_state(task_dir)
        updated["return_review_required"] = True
        updated["return_review_action"] = ""
        updated["return_review_reason"] = ""
        _state.write_state(task_dir, updated)


def apply_test_verdict(task_dir: Path, verdict: str) -> None:
    _require_open_gate(task_dir, "another test cycle")
    _transitions.apply_test_verdict(task_dir, verdict)


def apply_seal_verdict(task_dir: Path, verdict: str) -> None:
    _require_open_gate(task_dir, "another seal cycle")
    _transitions.apply_seal_verdict(task_dir, verdict)
    if verdict == "RETURN":
        updated = _state.load_state(task_dir)
        updated["return_review_required"] = True
        updated["return_review_action"] = ""
        updated["return_review_reason"] = ""
        _state.write_state(task_dir, updated)


def run_check(task_dir: Path, stage: str, check_id: str) -> int:
    _require_open_gate(task_dir, "another canonical check")
    return _run_check(task_dir, stage, check_id)


def run_test_probe(
    task_dir: Path,
    probe_id: str,
    *,
    authority: str,
    authority_ref: str,
    argv: list[str],
    cwd: str = ".",
    env: dict[str, str] | None = None,
) -> int:
    _require_open_gate(task_dir, "another adversarial test probe")
    return _run_test_probe(
        task_dir,
        probe_id,
        authority=authority,
        authority_ref=authority_ref,
        argv=argv,
        cwd=cwd,
        env=env,
    )


def _clear_preflight_and_downstream(state: dict[str, Any]) -> None:
    state["preflight_cycle"] = 0
    state["preflight_verdict"] = ""
    state["preflight_candidate_sha256"] = ""
    state["test_verdict"] = ""
    state["test_candidate_sha256"] = ""
    state["seal_cycle"] = 0
    state["seal_verdict"] = ""
    state["sealed_candidate_sha256"] = ""
    state["validation_candidate_sha256"] = ""
    state["packaging_candidate_sha256"] = ""


def _reset_candidate_evidence(state: dict[str, Any]) -> None:
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


def apply_return_review(task_dir: Path, action: str, reason: str) -> None:
    state = _state.load_state(task_dir)
    if state.get("phase") != "implementation" or state.get("status") != "in_progress":
        raise ValueError(
            "return review is valid only during in-progress implementation"
        )
    source = _review_source(state)
    if source not in {"preflight", "test", "seal"}:
        raise ValueError("return review is not currently required")
    if action not in {"implementation", "exploration"}:
        raise ValueError("return review action must be implementation or exploration")
    if not reason.strip():
        raise ValueError("return review reason must not be blank")
    errors = _bounded_validate_state(task_dir, state)
    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))

    if action == "exploration":
        state["attempt"] = int(state["attempt"]) + 1
        _reset_candidate_evidence(state)
        state["phase"] = "exploration"
        state["verdict"] = "RETURN_REVIEW"
    else:
        if source == "preflight":
            _clear_preflight_and_downstream(state)
        state["test_return_count"] = 0
        state["return_review_required"] = False
        state["verdict"] = ""

    state["return_review_action"] = action
    state["return_review_reason"] = reason.strip()
    _state.write_state(task_dir, state)


def block_task(task_dir: Path, kind: str, reason: str) -> None:
    _require_open_gate(task_dir, "blocking the task")
    _transitions.block_task(task_dir, kind, reason)
