"""An independent approval observation supersedes earlier planning facts."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_improvement_approval_does_not_resurrect_a_revoked_planning_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    execution = importlib.import_module("parent.execution")
    improvements = importlib.import_module("parent.improvements")
    binding = {"decision_id": "approved-improvement"}
    old = [{"event": "agreed", "improvement_approval": binding}]
    observed = []

    def history() -> list[dict[str, str]]:
        observed.append("revocation")
        return [{"event": "basis_revoked"}]

    def approval(*args: object, **kwargs: object) -> dict[str, str]:
        observed.append("fresh-approval")
        return binding

    monkeypatch.setattr(improvements, "require", approval)
    monkeypatch.setattr(
        execution, "human_basis", lambda event: {"kind": "human_agreement"}
    )
    session = SimpleNamespace(history=history)
    with pytest.raises(execution.ExecutionBasisStale, match="有効な実行根拠"):
        execution._current_basis(session, history=old)
    assert observed == ["fresh-approval", "revocation"]
