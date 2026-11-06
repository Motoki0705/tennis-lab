"""Planning approval belongs to one report and one current execution context."""

from __future__ import annotations

import copy
import importlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_context_tracks_external_dependency_prs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    phase = importlib.import_module("parent.worker_preflight")
    github = importlib.import_module("parent.github").GitHub("o/r")
    github.parent = 1
    monkeypatch.setattr(github, "family", lambda _parent: {1: None, 10: None})
    issue = {
        "number": 20,
        "repository_url": "https://api.github.com/repos/o/r",
        "state": "closed",
        "state_reason": "completed",
        "body": "## 要件\n外部依存の入力仕様。\n",
    }
    pulls: list[dict[str, Any]] = [
        {
            "number": 200,
            "body": "Closes #20",
            "state": "closed",
            "merged_at": "2026-11-06T00:00:00Z",
            "head": {
                "sha": "a" * 40,
                "ref": "codex/external",
                "repo": {"full_name": "o/r"},
            },
        }
    ]
    reads: list[str] = []

    def get(endpoint: str, **_kwargs: object) -> object:
        reads.append(endpoint)
        if endpoint == "repos/o/r/issues/20":
            return copy.deepcopy(issue)
        if match := re.fullmatch(
            r"repos/o/r/issues/(\d+)/timeline\?per_page=100&page=1", endpoint
        ):
            return (
                [
                    {
                        "id": pr["number"],
                        "source": {
                            "issue": {
                                **pr,
                                "pull_request": {
                                    "url": "https://api.github.com/repos/o/r/pulls/200"
                                },
                                "repository_url": "https://api.github.com/repos/o/r",
                            }
                        },
                    }
                    for pr in pulls
                ]
                if match[1] == "20"
                else []
            )
        if match := re.fullmatch(r"repos/o/r/pulls/(\d+)", endpoint):
            return copy.deepcopy(
                next(pr for pr in pulls if pr["number"] == int(match[1]))
            )
        raise AssertionError(endpoint)

    monkeypatch.setattr(github, "get", get)
    child = SimpleNamespace(body="## 依存関係\n#20 の完了後。\n")
    session = SimpleNamespace(
        gh=github,
        parent=1,
        row={"tasks": {"10": {"branch": "codex/worker", "worktree": str(tmp_path)}}},
        work_items=lambda: (SimpleNamespace(body="親の要件。"), {10: child}),
    )
    first = phase.context(session, 10, authorized={"generation": 1})
    assert [p["head"] for p in first["dependencies"][0]["prs"]] == ["a" * 40]
    pulls[0]["head"]["sha"] = "b" * 40
    advanced = phase.context(session, 10, authorized={"generation": 1})
    assert phase.digest(first) != phase.digest(advanced)
    pulls.append({**copy.deepcopy(pulls[0]), "number": 201})
    added = phase.context(session, 10, authorized={"generation": 1})
    assert phase.digest(advanced) != phase.digest(added)
    assert any("/issues/20/timeline?" in endpoint for endpoint in reads)


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


@pytest.mark.parametrize("failure", ["http", "malformed"])
def test_context_failure_does_not_invalidate_or_save(
    planning: tuple[Any, Any, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module, session, _ = planning
    phase = session.row["tasks"]["10"]["preflight"]
    before = copy.deepcopy(phase)
    error = (
        importlib.import_module("issues.cli").GitHubHTTPError(
            "repos/o/r/issues/20", 503, "unavailable"
        )
        if failure == "http"
        else ValueError("malformed dependency response")
    )
    saved: list[bool] = []
    monkeypatch.setattr(session.store, "save", lambda: saved.append(True))

    def unavailable(*_args: object) -> None:
        raise error

    monkeypatch.setattr(module, "context", unavailable)
    with pytest.raises(type(error), match="unavailable|malformed"):
        module.observe_context(session, 10, phase, persist=True)
    assert phase == before
    assert saved == []


def test_new_result_does_not_revive_invalidated_review(
    planning: tuple[Any, Any, dict[str, Any]],
) -> None:
    module, session, current = planning
    task = session.row["tasks"]["10"]
    task["branch"] = "codex/worker"
    phase = task["preflight"]
    phase.update(phase="implementing", skip_reason="レビュー鮮度のunit fixture。")
    task["verification"] = {"evidence": "元の実装を確認した。"}
    request = {"review_run_id": "review-one", "prs": [{"branch": "codex/worker"}]}
    module.bind_review(session, request)
    assert module.review_errors(task) == ()
    module.invalidate(phase, "dependency changed", task=task)
    current["requirements"] = "new requirements"
    phase.update(
        context=copy.deepcopy(current), phase="implementing", refresh_required=False
    )
    phase["turns"][-1]["final_sha256"] = "new final"
    phase["turns"][-1]["snapshot"]["context_sha256"] = module.digest(current)
    answer = decision(module, phase)
    answer.update(final_sha256="new final", delivered_turn_id="new implementation turn")
    phase["decisions"].append(answer)
    task["verification"] = {"evidence": "新計画の実装を再検証した。"}
    assert module.implementation_authorized(phase)
    assert "worker_review_inputs_changed" in module.review_errors(task)
    module.bind_review(session, {**request, "review_run_id": "review-two"})
    assert module.review_errors(task) == ()


def test_observed_basis_change_expires_approval(
    planning: tuple[Any, Any, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, session, _ = planning

    def revoked(*_args: object) -> None:
        raise module.ExecutionBasisStale("execution basis revoked")

    monkeypatch.setattr(module, "context", revoked)
    value = session.row["tasks"]["10"]["preflight"]
    fresh, stale = module.observe_context(session, 10, value, persist=True)
    assert fresh is None and stale
    assert value["refresh_required"] and value["phase"] == "planning"


def test_legacy_review_revalidates_worker_inputs(
    planning: tuple[Any, Any, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    phase, session, current = planning
    execution = importlib.import_module("parent.execution")
    basis = {
        "kind": "human_agreement",
        **{
            key: key
            for key in (
                "purpose",
                "scope",
                "behavior",
                "acceptance",
                "human_confirmation",
            )
        },
    }
    session.owner = lambda: {"generation": 2}
    session.parent = 1
    session.work_items = lambda: (None, {})
    session.row["tasks"]["10"]["branch"] = "codex/worker"
    request = {
        "schema_version": 1,
        "review_run_id": "old-review",
        "prs": [{"branch": "codex/worker"}],
    }
    session.row["reviews"] = [{"request": request}]
    monkeypatch.setattr(execution, "current_basis", lambda _: basis)
    monkeypatch.setattr(
        execution, "observed_authorization", lambda _: current["authorization"]
    )
    monkeypatch.setattr(
        phase, "context", lambda *_args, **_kwargs: copy.deepcopy(current)
    )
    with pytest.raises(ValueError, match="planning approval|review inputs"):
        execution.authorization(session, request)


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
