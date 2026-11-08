"""Human requirement changes and renewed consent through disposable parent state."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_execution_basis import registry
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv, review_setup
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def agreements(p: ParentEnv) -> list[dict[str, Any]]:
    policy = importlib.import_module("parent.policy")
    return [
        e
        for e in policy.events(p.state()["comments"]["1"], "operator", 1)
        if e["event"] == "agreed"
    ]


def delegation_args(worktree: Path, child: int = 10) -> tuple[str, ...]:
    return (
        "delegate",
        "--child",
        str(child),
        "--agent",
        "codex",
        "--branch",
        "codex/" + worktree.name,
        "--worktree",
        str(worktree),
    )


@pytest.mark.parametrize(
    "heading", ["全体のスコープ", "全体の完了条件", "作業状況・停止事項"]
)
def test_human_body_changes_are_parent_observations(
    parent_env: ParentEnv, heading: str
) -> None:
    p = parent_env
    selected = review_setup(p)
    policy = importlib.import_module("parent.policy")
    state = p.state()
    state["issues"]["1"]["body"] = policy.replace_section(
        state["issues"]["1"]["body"], heading, "説明を整理し、確認の順序を更新した。"
    )
    p.save(state)
    before = registry(p).read_bytes()
    assert json.loads(p.cli("status").stdout)["input_changes"]["body_changed"]
    assert registry(p).read_bytes() == before
    p.child("changed-body")
    p.cli(*delegation_args(p.worktree("changed-body"), child=13))
    p.cli("review-prepare", "--prs", *map(str, selected))
    assert len(agreements(p)) == 1


def test_human_progress_preserves_body_digest_authorization_and_review(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    policy = importlib.import_module("parent.policy")
    execution = importlib.import_module("parent.execution")
    state = p.state()
    for heading in ("全体の完了条件", "Acceptance checklist"):
        state["issues"]["1"]["body"] = policy.replace_section(
            state["issues"]["1"]["body"], heading, "- [ ] 動作を確認する。"
        )
    p.save(state)
    selected = review_setup(p)
    args = ("review-prepare", "--prs", *map(str, selected))
    prepared = p.cli(*args).stdout
    [agreement] = agreements(p)
    assert agreement["requirements_sha256"] == policy.digest(
        execution.requirement_body(p.state()["issues"]["1"]["body"])
    )
    state = p.state()
    body = state["issues"]["1"]["body"]
    for heading in execution.PROGRESS_SECTIONS:
        if heading != "子タスクのチェックリスト":
            body = policy.replace_section(body, heading, "進捗を更新した。")
    state["issues"]["1"]["body"] = body.replace(
        "[ ] 動作を確認する。", "[x] 動作を確認する。"
    )
    p.save(state)
    p.child("progress")  # The child checklist and progress summary change as well.
    result = json.loads(
        p.cli(*delegation_args(p.worktree("progress"), child=13)).stdout
    )
    assert result["action"] == "launch"
    assert (
        result["assignment"]["authorization"] == json.loads(prepared)["authorization"]
    )
    assert p.cli(*args).stdout == prepared
    p.cli("agree", "--spec", p.spec(agreement["agreement"]))
    assert agreements(p) == [agreement]


@pytest.mark.parametrize("comment_snapshot", [False, True])
def test_legacy_human_agreement_is_usable_without_fabricating_history(
    parent_env: ParentEnv, comment_snapshot: bool
) -> None:
    p = parent_env
    selected = review_setup(p)
    policy = importlib.import_module("parent.policy")
    state = p.state()
    event = agreements(p)[-1]
    original = event.pop("body")
    event.pop("requirements_sha256", None)
    if not comment_snapshot:
        event.pop("comment_inputs")
    next(c for c in state["comments"]["1"] if c["body"] == original)["body"] = (
        policy.marker(event)
    )
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    status = json.loads(p.cli("status").stdout)
    assert status["execution_basis"]["kind"] == "human_agreement"
    assert (p.state(), registry(p).read_bytes()) == before
    p.child("legacy")
    p.cli(*delegation_args(p.worktree("legacy"), child=13))
    p.cli("review-prepare", "--prs", *map(str, selected))
    assert len(agreements(p)) == 1
    assert "requirements_sha256" not in agreements(p)[0]


@pytest.mark.parametrize("value", [None, True, "", "A" * 64, "g" * 64])
def test_malformed_human_body_digest_is_rejected_on_read(
    parent_env: ParentEnv, value: Any
) -> None:
    p = parent_env
    p.agreed()
    policy = importlib.import_module("parent.policy")
    event = agreements(p)[-1]
    original = event.pop("body")
    event["requirements_sha256"] = value
    state = p.state()
    next(c for c in state["comments"]["1"] if c["body"] == original)["body"] = (
        policy.marker(event)
    )
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    response = p.cli("status", ok=False)
    assert response.returncode == 2
    assert "SHA-256" in response.stderr
    assert (p.state(), registry(p).read_bytes()) == before


@pytest.mark.parametrize("change", ["body", "legacy"])
def test_human_body_observation_preserves_review_evidence(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    policy = importlib.import_module("parent.policy")
    state = p.state()
    if change == "body":
        state["issues"]["1"]["body"] += "\n\n## 補足\n確認順序を説明する。\n"
    else:
        event = agreements(p)[-1]
        original = event.pop("body")
        event.pop("requirements_sha256", None)
        next(c for c in state["comments"]["1"] if c["body"] == original)["body"] = (
            policy.marker(event)
        )
    p.save(state)
    saved = {path.name: path.read_bytes() for path in g.review.run_dir.glob("*.json")}
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert all(row["decision"] == "merge" for row in rows.values())
    assert {
        path.name: path.read_bytes() for path in g.review.run_dir.glob("*.json")
    } == saved
    assert not g.merge_calls() and len(g.review.calls()) == 1
    p.cli("basis-revoke", "--reason-file", p.note("人間の停止指示を親が確認。"))
    assert g.cli("merge").returncode == 2


@pytest.mark.parametrize("recovery", ["checking", "agree"])
def test_same_human_reagreement_after_revocation_resumes_delegation_idempotently(
    parent_env: ParentEnv, recovery: str
) -> None:
    p = parent_env
    p.agreed()
    p.child("worker")
    args = delegation_args(p.worktree("worker"))
    initial = json.loads(p.cli(*args).stdout)["assignment"]["authorization"]
    agreement = agreements(p)[-1]["agreement"]
    authorizations = [initial]
    for _ in range(2):
        revoke = ("basis-revoke", "--reason-file", p.note("人間の同じ停止指示。"))
        p.cli(*revoke)
        before = p.state(), registry(p).read_bytes()
        p.cli(*revoke)
        p.cli(*args, ok=False)
        assert (p.state(), registry(p).read_bytes()) == before
        if recovery == "checking":
            p.cli("checking")
            before = p.state(), registry(p).read_bytes()
            p.cli("checking")
            assert (p.state(), registry(p).read_bytes()) == before
        # A new human confirmation may repeat the exact same spec and comment inputs.
        p.cli("agree", "--spec", p.spec(agreement))
        before = p.state(), registry(p).read_bytes()
        p.cli("agree", "--spec", p.spec(agreement))
        assert (p.state(), registry(p).read_bytes()) == before
        resumed = json.loads(
            p.cli(
                *args,
                "--restart-evidence-file",
                p.note("既存workerの終了とworktreeを確認した。"),
            ).stdout
        )
        assert resumed["action"] == "launch"
        authorizations.append(resumed["assignment"]["authorization"])
    assert len({json.dumps(auth, sort_keys=True) for auth in authorizations}) == 3
    assert len({e["revision"] for e in agreements(p)}) == 3
    policy = importlib.import_module("parent.policy")
    history = policy.events(p.state()["comments"]["1"], "operator", 1)
    assert [e["revision"] for e in history if e["event"] == "basis_revoked"] == [1, 2]
    assert all(e["agreement"] == agreement for e in agreements(p))
    assert len({json.dumps(e["comment_inputs"]) for e in agreements(p)}) == 1
    p.cli("checking")  # Also advance after a renewal without an intervening checking.
    p.cli("agree", "--spec", p.spec(agreement))
    assert len(agreements(p)) == 4
