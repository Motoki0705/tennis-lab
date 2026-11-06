"""Planning approval belongs to one report and one current execution context."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture
def planning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Any, Any, dict[str, Any]]:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    module = importlib.import_module("parent.worker_preflight")
    current = {
        "repo": "o/r",
        "parent": 1,
        "child": 10,
        "authorization": {
            "generation": 2,
            "execution_basis_kind": "human_agreement",
            "execution_basis_sha256": "a" * 64,
        },
        "parent_requirements": "b" * 64,
        "requirements": "c" * 64,
        "dependencies": [{"issue": 11, "head": "d" * 40}],
        "worktree": str(tmp_path),
        "branch": "codex/worker",
    }
    phase = module.new("delegation-one")
    phase.update(
        context=copy.deepcopy(current),
        runtime={"contract_sha256": "contract", "session_id": "private-session"},
    )
    snapshot = {
        "schema_version": 1,
        "lineage_id": phase["lineage_id"],
        "turn_id": "turn-one",
        "previous_turn_id": None,
        "phase": "planning",
        "decision_id": None,
        "context_sha256": module.digest(current),
    }
    phase["turns"].append(
        {
            "snapshot": snapshot,
            "delegation_id": "delegation-one",
            "source": str(tmp_path),
            "final_sha256": "final",
        }
    )
    task = {
        "delegated": True,
        "delegation_id": "delegation-one",
        "preflight": phase,
        "worktree": str(tmp_path),
    }
    session = SimpleNamespace(
        row={"tasks": {"10": task}},
        store=SimpleNamespace(save=lambda: None),
        owner=lambda: None,
    )
    monkeypatch.setattr(module, "context", lambda *_: copy.deepcopy(current))
    return module, session, current


def decision(module: Any, phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": "decision-one",
        "lineage_id": phase["lineage_id"],
        "delegation_id": "delegation-one",
        "turn_id": "turn-one",
        "final_sha256": "final",
        "context_sha256": module.digest(phase["context"]),
        "runtime": copy.deepcopy(phase["runtime"]),
        "invalidated": False,
        "delivered_turn_id": None,
        "action": "approve",
        "answer": "提案どおり実装する。",
    }


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "child",
        "delegation",
        "turn",
        "generation",
        "authorization",
        "parent_requirements",
        "requirements",
        "dependency",
        "head",
        "session",
        "model",
    ],
)
def test_decision_matches_current_plan(
    planning: tuple[Any, Any, dict[str, Any]], change: str
) -> None:
    module, session, _ = planning
    phase = session.row["tasks"]["10"]["preflight"]
    answer = decision(module, phase)
    if change == "child":
        phase["context"]["child"] = 12
    elif change == "delegation":
        answer["delegation_id"] = "another-delegation"
    elif change == "turn":
        answer["turn_id"] = "another-turn"
    elif change in {"generation", "authorization"}:
        phase["context"]["authorization"][
            "generation" if change == "generation" else "execution_basis_sha256"
        ] = 3 if change == "generation" else "e" * 64
    elif change in {"parent_requirements", "requirements"}:
        phase["context"][change] = "e" * 64
    elif change == "dependency":
        phase["context"]["dependencies"].append({"issue": 13, "head": "f" * 40})
    elif change == "head":
        phase["context"]["dependencies"][0]["head"] = "f" * 40
    elif change in {"session", "model"}:
        phase["runtime"]["session_id" if change == "session" else "contract_sha256"] = (
            "changed"
        )
    assert module.decision_matches(phase, answer) is (change == "none")


@pytest.mark.parametrize(
    "report", ["complete", "missing", "empty", "fenced", "duplicate"]
)
def test_planning_report_requires_complete_evidence(
    planning: tuple[Any, Any, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report: str,
) -> None:
    module, session, _ = planning
    text = "\n\n".join(f"## {h}\n\nなし" for h in module.HEADINGS)
    if report == "missing":
        text = text.replace("## テスト設計", "## 別の節")
    elif report == "empty":
        text = text.replace("## 実装案\n\nなし", "## 実装案\n")
    elif report == "fenced":
        text = "```markdown\n" + text + "\n```"
    elif report == "duplicate":
        text += "\n\n## 実装案\n\n別案"
    (tmp_path / "final.md").write_text(text)
    monkeypatch.setattr(
        module, "latest_evidence", lambda *_: (tmp_path, {"result": "completed"})
    )
    monkeypatch.setattr(module, "bind_runtime", lambda *_: None)
    if report == "complete":
        assert (
            module.decide(session, 10, "approve", "確認済み、実装へ。")["action"]
            == "approve"
        )
    else:
        with pytest.raises(ValueError, match="headings"):
            module.decide(session, 10, "approve", "見出しなしのまま承認しない。")
        assert not session.row["tasks"]["10"]["preflight"]["decisions"]
        assert (
            module.decide(session, 10, "investigate", "必須6節で再提出する。")["action"]
            == "investigate"
        )


@pytest.mark.parametrize(
    "case",
    [
        "skip",
        "empty_skip",
        "late_skip",
        "no_decision",
        "replace_answer",
        "unbound_additional",
    ],
)
def test_phase_actions_and_skip_rules(
    planning: tuple[Any, Any, dict[str, Any]], case: str
) -> None:
    module, session, _ = planning
    phase = session.row["tasks"]["10"]["preflight"]
    if case in {"skip", "empty_skip"}:
        phase.update(turns=[], context=None)
    kwargs: dict[str, Any] = {"resume": case not in {"skip", "empty_skip"}}
    if case in {"skip", "empty_skip", "late_skip"}:
        kwargs["skip_reason"] = "自明な文言1行の訂正。" if case != "empty_skip" else " "
    if case == "replace_answer":
        phase["decisions"].append(decision(module, phase))
    if case == "unbound_additional":
        phase.update(phase="implementing", skip_reason="自明な訂正。")
        kwargs["additional"] = "回答がない追加指示。"
    if case == "skip":
        prepared, _ = module.prepare_turn(session, 10, "訂正する。", **kwargs)
        assert prepared["phase"] == "implementing" and prepared["skip_reason"]
        assert prepared["turns"][-1]["snapshot"]["phase"] == "implementing"
    else:
        before = copy.deepcopy(phase)
        with pytest.raises(ValueError):
            module.prepare_turn(session, 10, "回答の置換は禁止。", **kwargs)
        assert phase == before


@pytest.mark.parametrize("continuation", ["verified", "unrelated", "different_lineage"])
def test_continuation_preserves_approval_binding(
    planning: tuple[Any, Any, dict[str, Any]], continuation: str
) -> None:
    module, session, _ = planning
    phase = session.row["tasks"]["10"]["preflight"]
    answer = decision(module, phase)
    phase["delegation_id"] = "delegation-two"
    if continuation != "unrelated":
        phase["continuations"].append(
            {"from": "delegation-one", "to": "delegation-two"}
        )
    if continuation == "different_lineage":
        phase["lineage_id"] = "new-lineage"
    assert module.decision_matches(phase, answer) is (continuation == "verified")
