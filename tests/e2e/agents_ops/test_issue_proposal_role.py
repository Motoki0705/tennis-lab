"""Proposal vetoes at independent automation boundaries; no real network/agents."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def ops(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[3] / ".agents/ops"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.syspath_prepend(str(root / "ops-report"))


@pytest.mark.parametrize(
    "labels,body",
    [
        (["issue:proposal"], "自由な本文"),
        (["issue:proposal", "issue:parent"], "自由な本文"),
        ([], "<!-- parent-improvement-proposal:v1 parent=1 generation=1 -->"),
    ],
)
def test_work_gate_uses_both_role_and_provenance(
    ops: None, labels: list[str], body: str
) -> None:
    api = importlib.import_module("issues.evidence")
    with pytest.raises(ValueError, match="proposal"):
        api.require_work_issue({"labels": labels, "body": body})
    for role in ("issue:parent", "issue:child", "issue:standalone"):
        api.require_work_issue({"labels": [role], "body": "元の作業Issue"})


def test_dispatcher_excludes_proposals_and_vetoes_explicit_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.e2e.agents_ops.test_dispatcher_judgement import Harness, issue

    h = Harness(tmp_path, monkeypatch)
    h.gh.issues[2] = issue(2)
    h.gh.issues[2]["labels"] = [{"name": "issue:proposal"}]
    with h.store.locked():
        snapshot = h.collector.collect(h.gh, h.store, h.work)
        assert snapshot["subjects"] == [1]
        with pytest.raises(ValueError, match="proposal"):
            h.collector.collect(h.gh, h.store, h.work, [2])
        assert h.collector.is_container(h.gh.issues[2])
    assert not h.gh.writes and not h.calls.exists()


def test_supervisor_rechecks_proposal_role_immediately_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.e2e.agents_ops.test_dispatcher_supervisor import DispatchHarness

    d = DispatchHarness(tmp_path, monkeypatch)
    d.prepare()
    d.control.assign()
    job = d.job()
    assert d.control.claim("o/r#1", job["id"]) is not None
    d.h.gh.issues[1]["labels"].append({"name": "issue:proposal"})
    runtime = importlib.import_module("dispatcher.runtime").Runtime(Path(job["runner"]))
    runtime.launch_context = lambda: d.control.launch_admission("o/r#1", job["id"])
    calls: list[Any] = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="proposal"):
        runtime.execute(job, 0)
    assert calls == [] and d.job()["cancelled_before_start"]


def test_project_tree_rejects_proposal_before_projection(ops: None) -> None:
    from tests.e2e.agents_ops.test_project_progress import base_state

    state = base_state()
    state["issues"]["1204"]["labels"].append("issue:proposal")

    class Api:
        def issue(self, repo: str, n: int) -> dict[str, Any]:
            return dict(state["issues"][str(n)])

    with pytest.raises(ValueError, match="proposal"):
        importlib.import_module("project_progress.progress").read_tree(
            Api(), "o/r", 1204
        )


def test_report_and_triage_enumeration_exclude_proposals(
    ops: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    api = importlib.import_module("opsreport.shell")
    monkeypatch.setattr(
        api,
        "gh_json",
        lambda *a, **kw: [
            {"number": 1, "labels": ["issue:standalone"], "body": "作業"},
            {"number": 2, "labels": ["issue:proposal", "issue:parent"], "body": "提案"},
            {
                "number": 3,
                "labels": [],
                "body": "<!-- parent-improvement-proposal:v1 parent=1 generation=1 -->",
            },
        ],
    )
    assert [
        r["number"]
        for r in api.gh_list(
            ["issue", "list", "--json", "number"], limit=100, cwd=tmp_path
        )
    ] == [1]


def test_organization_collection_excludes_proposals_before_loading_work_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.e2e.agents_ops.test_dispatcher_judgement import Harness, issue

    h = Harness(tmp_path, monkeypatch)
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report")
    )
    h.gh.issues[2] = issue(2)
    h.gh.issues[2]["labels"] = [{"name": "issue:proposal"}]
    monkeypatch.setattr(
        h.gh, "issue_rows", lambda: list(h.gh.issues.values()), raising=False
    )
    with h.store.locked():
        snapshot = importlib.import_module("opsreport.organization").collect(
            h.gh, h.store, h.work
        )
    assert set(snapshot["issues"]) == {"1"}
    assert not h.gh.writes
