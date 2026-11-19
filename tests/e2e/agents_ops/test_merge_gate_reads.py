"""Read budgets and pre-merge freshness through the public gate and fake gh."""

from __future__ import annotations

import importlib
import subprocess
from datetime import datetime

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def test_explicit_observation_cache_is_private_and_ends_with_its_scope(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("parent.github")
    store = importlib.import_module("parent.state").Store(
        parent_env.root / "observations"
    )
    gh = module.GitHub("o/r", store=store)
    gh.parent = 1
    remote = {"revision": 1}
    calls = []

    def fetch(self: object, endpoint: str, **kwargs: object) -> object:
        calls.append(remote["revision"])
        return dict(remote)

    monkeypatch.setattr(module.IssueGitHub, "get", fetch)
    with store.locked():
        assert gh.get("user") == {"revision": 1}
        remote["revision"] = 2
        with gh.io_scope():
            external = gh.detached()
            with external.read_observation():
                assert external.get("user") == {"revision": 2}
                remote["revision"] = 3
                assert external.get("user") == {"revision": 2}
                detached = external.detached()
                assert detached.get("user") == {"revision": 3}
                remote["revision"] = 4
                assert detached.get("user") == {"revision": 4}
                # An explicit freshness check bypasses the observation's cache.
                with external.fresh_reads():
                    assert external.get("user") == {"revision": 4}
            remote["revision"] = 5
            assert external.get("user") == {"revision": 5}
            with external.read_observation():
                assert external.get("user") == {"revision": 5}
        assert gh.get("user") == {"revision": 1}
    with store.locked():
        assert gh.get("user") == {"revision": 5}
    assert calls == [1, 2, 3, 4, 4, 5, 5, 5]


def test_ci_rollups_are_shared_only_within_one_observation(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    for key in ("PATH", "PARENT_GH_STATE", "PARENT_GH_CALLS"):
        monkeypatch.setenv(key, p.env[key])
    module = importlib.import_module("parent.github")
    gh = module.GitHub("o/r")
    gh.parent = 1
    state = p.state()
    state["prs"] = [{"number": 100, "head": {"sha": "a" * 40}}]
    p.save(state)
    fields = ("headRefOid", "statusCheckRollup")
    with gh.read_observation():
        assert gh.pr_view(100, fields)["headRefOid"] == "a" * 40
        state["prs"][0]["head"]["sha"] = "b" * 40
        p.save(state)
        assert gh.pr_view(100, fields)["headRefOid"] == "a" * 40
        with gh.fresh_reads():
            assert gh.pr_view(100, fields)["headRefOid"] == "b" * 40
    state["prs"][0]["head"]["sha"] = "c" * 40
    p.save(state)
    assert gh.pr_view(100, fields)["headRefOid"] == "c" * 40
    with gh.read_observation():
        assert gh.pr_view(100, fields)["headRefOid"] == "c" * 40
    assert sum(c[:2] == ["pr", "view"] for c in p.calls()) == 4
    # Moving the transport/cache must retain the gate's strict JSON parsing.
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(
            a, 0, '{"headRefOid":"old","headRefOid":"new","statusCheckRollup":[]}'
        ),
    )
    with gh.read_observation(), pytest.raises(ValueError, match="duplicate keys"):
        gh.pr_view(100, fields)


def bind_task_adoption(gate_env: GateEnv, issue: int) -> None:
    p = gate_env.review.p
    proposals = importlib.import_module("parent.proposals")
    model = importlib.import_module("opsreport.proposals")
    adoption = importlib.import_module("issues.adoption")
    text, _ = model.render_proposals(
        [
            model.Proposal(
                "コード負債",
                "責務を分割する",
                evidence="change.txt と回帰テスト。",
                purpose="保守性を改善する。",
                scope="change.txt。",
                expected_effect="変更を検証できる。",
                acceptance=("回帰テストが成功する。",),
            )
        ],
        "20261019",
    )
    marker = model.report_marker(
        slot="20261019",
        since=datetime.fromisoformat("2026-10-16T08:00:00+09:00"),
        until=datetime.fromisoformat("2026-10-19T08:00:00+09:00"),
        proposals=1,
    )
    body = (
        f"{marker}\n\n{model.PROPOSALS_BEGIN}\n"
        + text.replace("- [ ] **", "- [x] **")
        + f"\n{model.PROPOSALS_END}\n"
    )
    _, (proposal,) = proposals.parse_report(body)
    state = p.state()
    state["issues"]["77"] = {
        **state["issues"]["1"],
        "id": 10077,
        "number": 77,
        "body": body,
        "labels": [{"name": "ops-report"}],
    }
    state["issues"][str(issue)]["body"] = adoption.bind_task(
        state["issues"][str(issue)]["body"],
        plan="a" * 64,
        node="implementation",
        bindings=[
            {
                "source": {
                    "report_number": 77,
                    "proposal_id": proposal.pid,
                    "option": None,
                },
                "proposal_sha256": importlib.import_module("parent.policy").digest(
                    proposal.snapshot()
                ),
            }
        ],
    )
    p.save(state)
