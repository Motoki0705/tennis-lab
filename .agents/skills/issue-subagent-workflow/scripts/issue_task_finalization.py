"""Fail-closed finalization wrapper for issue-subagent-workflow."""

from __future__ import annotations

from pathlib import Path

from issue_task_artifacts import check_artifact
from issue_task_checks import check
from issue_task_transitions import apply_validation_verdict as _apply_validation_verdict


def apply_validation_verdict(task_dir: Path, verdict: str) -> None:
    """Validate all relevant artifacts before mutating final workflow state."""
    errors = check_artifact(task_dir, "validation")
    if verdict == "PASS":
        errors.extend(check(task_dir))
    if errors:
        joined = "; ".join(dict.fromkeys(errors))
        raise ValueError(f"pre-completion check failed: {joined}")
    _apply_validation_verdict(task_dir, verdict)
