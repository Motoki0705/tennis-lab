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


def test_content_change_is_an_observation_without_mutating_provenance(
    planning: tuple[Any, Any, dict[str, Any]],
) -> None:
    module, session, current = planning
    value = session.row["tasks"]["10"]["preflight"]
    before = copy.deepcopy(value)
    current["requirements"] = "updated explanation"
    current["authorization"]["generation"] += 1
    fresh, changed = module.observe_context(session, 10, value, persist=True)
    assert changed and fresh == current
    assert value == before


@pytest.mark.parametrize("planning_only", [False, True])
def test_parent_selects_the_turn_without_a_required_plan_approval(
    planning: tuple[Any, Any, dict[str, Any]], planning_only: bool
) -> None:
    module, session, current = planning
    before = copy.deepcopy(session.row)
    value, text = module.prepare_turn(
        session, 10, "次の作業を進める。", resume=True, planning=planning_only
    )
    assert value["phase"] == ("planning" if planning_only else "implementing")
    assert value["turns"][-1]["snapshot"]["context_sha256"] == module.digest(current)
    assert text == "次の作業を進める。"
    assert session.row == before


def test_revocation_is_reported_without_destroying_worker_evidence(
    planning: tuple[Any, Any, dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, session, _ = planning

    def revoked(*_args: object) -> None:
        raise module.ExecutionBasisStale("explicit revocation")

    monkeypatch.setattr(module, "context", revoked)
    before = copy.deepcopy(session.row)
    assert module.observe_context(
        session, 10, session.row["tasks"]["10"]["preflight"], persist=True
    ) == (None, True)
    with pytest.raises(module.ExecutionBasisStale):
        module.prepare_turn(session, 10, "continue", resume=True)
    assert session.row == before


def test_dependency_declaration_distinguishes_typed_pr_references(
    planning: tuple[Any, Any, dict[str, Any]],
) -> None:
    module, _, _ = planning
    assert module.dependencies(
        "## 依存関係\n#10 と #11。参考: PR #20、[#21](https://github.com/o/r/pull/21)\n"
    ) == [10, 11]


def test_runtime_binding_compares_execution_inputs_and_session(
    planning: tuple[Any, Any, dict[str, Any]],
) -> None:
    module, _, _ = planning
    value = module.new("delegation")
    record: dict[str, Any] = {
        "runtime": {"session_id": "same-session"},
        "contract": {
            "task_id": "delegation",
            "settings": {
                "selected": {"model": "model-a"},
                "source": "/old/path",
                "sha256": "old",
            },
        },
    }
    module.bind_runtime(value, record)
    relocated = copy.deepcopy(record)
    relocated["contract"]["settings"].update(source="/new/path", sha256="new")
    module.bind_runtime(value, relocated)
    relocated["runtime"]["session_id"] = "other-session"
    with pytest.raises(ValueError, match="session/contract"):
        module.bind_runtime(value, relocated)
    relocated["runtime"]["session_id"] = "same-session"
    relocated["contract"]["settings"]["selected"]["model"] = "other-model"
    with pytest.raises(ValueError, match="session/contract"):
        module.bind_runtime(value, relocated)


@pytest.mark.parametrize("mutation", ["none", "old_session", "other_run", "contract"])
def test_replacement_binding_requires_reserved_run_contract_and_new_session(
    planning: tuple[Any, Any, dict[str, Any]], mutation: str, tmp_path: Path
) -> None:
    module, _, _ = planning
    old_record: dict[str, Any] = {
        "runtime": {"session_id": "old-session"},
        "contract": {
            "task_id": "same-delegation",
            "settings": {"selected": {"model": "model-a"}},
        },
    }
    value = module.new("same-delegation")
    value["session_replacements"] = [
        {
            "from_runtime": module.runtime_binding(old_record),
            "to_run": str(tmp_path / "replacement"),
            "to_runtime": None,
        }
    ]
    record = copy.deepcopy(old_record)
    record.update(
        run_dir=str(tmp_path / "replacement"), runtime={"session_id": "new-session"}
    )
    if mutation == "old_session":
        record["runtime"]["session_id"] = "old-session"
    elif mutation == "other_run":
        record["run_dir"] = str(tmp_path / "unreserved-run")
    elif mutation == "contract":
        record["contract"]["settings"]["selected"]["model"] = "model-b"
    before = copy.deepcopy(value)
    if mutation == "none":
        module.bind_runtime(value, record)
        assert value["runtime"]["session_id"] == "new-session"
        assert value["session_replacements"][0]["to_runtime"] == value["runtime"]
    else:
        with pytest.raises(ValueError, match="explicit worker session replacement"):
            module.bind_runtime(value, record)
        assert value == before
