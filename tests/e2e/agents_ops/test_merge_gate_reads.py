"""Read budgets and pre-merge freshness through the public gate and fake gh."""

from __future__ import annotations

import importlib
import json
import subprocess
from collections import Counter
from datetime import datetime

import pytest

from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.merge_gate_reads import (
    add_unrelated_parents,
    api_endpoints,
)


@pytest.mark.parametrize("peers", [0, 20])
@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_dry_run_reads_each_endpoint_once_and_never_reads_other_parents(
    gate_env: GateEnv, peers: int
) -> None:
    g, p = gate_env, gate_env.review.p
    unrelated = add_unrelated_parents(p, peers)
    offset = len(p.calls())
    metrics = p.root / "gate-metrics.jsonl"
    response = g.cli("merge", AGENT_AUTO_MERGE="0", PARENT_METRICS_FILE=str(metrics))
    assert response.returncode == 0, response.stderr
    assert all(row["decision"] == "merge" for row in g.results(response).values())
    calls = p.calls()[offset:]
    endpoints = api_endpoints(calls)
    # This fixture needs 28 distinct REST facts: auth, formal family, timeline
    # PR discovery, snapshots/CI and publisher-validated review evidence.
    assert len(endpoints) <= 28
    assert set(Counter(endpoints).values()) == {1}
    assert not any(
        e.split("?")[0].startswith(f"repos/o/r/issues/{n}/")
        or e == f"repos/o/r/issues/{n}"
        for n in unrelated
        for e in endpoints
    )
    assert not any(
        e.startswith("repos/o/r/issues?state=all&per_page") for e in endpoints
    )
    (row,) = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert row["github_requests"] == len(endpoints)
    assert row["cache_hits"] > 0
    assert row["exit_code"] == 0
    assert sum(c[:2] == ["pr", "view"] for c in calls) == 3
    assert not g.merge_calls()


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


@pytest.mark.parametrize("change", ["review", "ci", "owner", "base"])
def test_changes_after_collection_are_rechecked_before_merge(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    fake = p.root / "bin/gh"
    source = fake.read_text()
    # Mutate at the next observation's first parent GET. Review changes are
    # injected later, during that observation's CI I/O, to protect read ordering.
    hook = """
    state["gate_boundary_reads"] = state.get("gate_boundary_reads", 0) + 1
    if state["gate_boundary_reads"] == 2:
        change = os.environ["GATE_CHANGE"]
        if change == "ci":
            state["prs"][0]["statusCheckRollup"][0].update(status="QUEUED", conclusion="")
        elif change == "review":
            for rows in state["comments"].values():
                rows[:] = [c for c in rows if '"kind":"result"' not in c["body"]]
        elif change == "base":
            state["prs"][0]["base"]["ref"] = "other-default"
        elif change == "owner":
            from pathlib import Path
            registry = Path(os.environ["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions/registry.json"
            data = json.loads(registry.read_text())
            assert data["version"] == 1
            data["parents"]["o/r#1"]["session"] = "claude:replacement-owner"
            registry.write_text(json.dumps(data))
    save()
"""
    if change == "review":
        source = source.replace(
            'if args[:2] == ["pr", "view"]:\n',
            'if args[:2] == ["pr", "view"]:\n' + hook,
        )
    else:
        line = 'payload = json.load(sys.stdin) if method != "GET" else None\n'
        source = source.replace(
            line,
            line + 'if method == "GET" and endpoint == "repos/o/r/issues/1":\n' + hook,
        )
    fake.write_text(source)
    response = g.cli("merge", GATE_CHANGE=change)
    assert response.returncode in {1, 2}, response.stdout
    assert not g.merge_calls()
    if change == "ci":
        assert "ci_python_pending" in g.results(response)[100]["reasons"]
    elif change == "base":
        assert "base_not_default_branch" in g.results(response)[100]["reasons"]
    elif change == "owner":
        assert "registry changed" in response.stderr
    else:
        assert "published record missing" in response.stderr


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_merging_three_prs_keeps_explicit_rechecks_within_request_budget(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    add_unrelated_parents(p, 20)
    offset = len(p.calls())
    response = g.cli("merge", PARENT_MERGE_ADVANCE_BASE="1")
    assert response.returncode == 0, response.stderr
    assert all(row["action"] == "merged" for row in g.results(response).values())
    assert len(g.merge_calls()) == 3
    # The same fixture used 1,283 requests before scoped observations. The 250
    # request budget gains only two fresh bodies (parent + closing child) per
    # merge: 3 * 2 = 6, without rediscovering the formal family.
    assert len(api_endpoints(p.calls()[offset:])) <= 256


def bind_task_adoption(gate_env: GateEnv, issue: int) -> None:
    g, p = gate_env, gate_env.review.p
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
                "proposal_sha256": g.policy.digest(proposal.snapshot()),
            }
        ],
    )
    p.save(state)


@pytest.mark.parametrize("issue", [1, 10], ids=["parent", "child"])
@pytest.mark.parametrize("revoke", [False, True])
def test_adoption_source_is_needed_and_rechecked_after_review_evidence(
    gate_env: GateEnv, revoke: bool, issue: int
) -> None:
    g, p = gate_env, gate_env.review.p
    bind_task_adoption(g, issue)
    fake = p.root / "bin/gh"
    line = 'payload = json.load(sys.stdin) if method != "GET" else None\n'
    fake.write_text(
        fake.read_text().replace(
            line,
            line
            + """if method == "GET" and endpoint == "repos/o/r/issues/77":
    state["source_reads"] = state.get("source_reads", 0) + 1
    if state["source_reads"] == 3 and os.environ.get("REVOKE_SOURCE") == "1":
        state["issues"]["77"]["state"] = "closed"
    save()
""",
        )
    )
    response = g.cli("merge", REVOKE_SOURCE="1" if revoke else "0")
    assert p.state()["source_reads"] >= 3
    if revoke:
        assert response.returncode == 2
        assert "source report must remain open and complete" in response.stderr
        assert not g.merge_calls()
    else:
        assert response.returncode == 0, response.stderr
        assert g.results(response)[100]["action"] == "merged"


@pytest.mark.parametrize("issue", [1, 10], ids=["parent", "child"])
@pytest.mark.parametrize("change", ["report", "option", "proposal_sha256"])
def test_adoption_binding_changes_during_review_evidence_block_merge(
    gate_env: GateEnv, issue: int, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    bind_task_adoption(g, issue)
    adoption = importlib.import_module("issues.adoption")
    state = p.state()
    original = state["issues"][str(issue)]["body"]
    metadata = adoption.parse_task_metadata(original)
    binding = metadata["bindings"][0]
    if change == "report":
        state["issues"]["78"] = {
            **state["issues"]["77"],
            "id": 10078,
            "number": 78,
            "state": "closed",
        }
        binding["source"]["report_number"] = 78
        expected = "source report must remain open and complete"
    elif change == "option":
        binding["source"]["option"] = "other option"
        expected = "task adoption/option changed"
    else:
        binding["proposal_sha256"] = "b" * 64
        expected = "task proposal digest changed"
    _, _, rest = original.partition("\n")
    state["changed_binding_body"] = (
        adoption.task_marker(
            metadata["plan_sha256"], metadata["node"], metadata["bindings"]
        )
        + "\n"
        + rest
    )
    p.save(state)
    fake = p.root / "bin/gh"
    line = 'payload = json.load(sys.stdin) if method != "GET" else None\n'
    fake.write_text(
        fake.read_text().replace(
            line,
            line
            + """if method == "GET" and endpoint.split("?")[0] == "repos/o/r/issues/100/comments":
    state["binding_evidence_reads"] = state.get("binding_evidence_reads", 0) + 1
    if state["binding_evidence_reads"] == 2:
        # The final observation already collected parent/child bodies. Change
        # the binding during review evidence I/O, leaving the old report valid.
        state["issues"][os.environ["BINDING_ISSUE"]]["body"] = state["changed_binding_body"]
    save()
""",
        )
    )
    response = g.cli("merge", BINDING_ISSUE=str(issue))
    assert p.state()["binding_evidence_reads"] >= 2
    assert (
        adoption.parse_task_metadata(p.state()["issues"][str(issue)]["body"])
        == metadata
    )
    assert not g.merge_calls()
    assert response.returncode == 2, response.stdout
    assert expected in response.stderr
