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
def test_human_body_changes_block_split_delegation_and_review_prepare(
    parent_env: ParentEnv, heading: str
) -> None:
    p = parent_env
    selected = review_setup(p)
    policy = importlib.import_module("parent.policy")
    worktree = p.worktree("changed-body")
    state = p.state()
    state["issues"]["1"]["body"] = policy.replace_section(
        state["issues"]["1"]["body"],
        heading,
        "合意後の変更。この範囲の作業は停止する。",
    )
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    responses = [
        p.child("changed-body", ok=False),
        p.cli(*delegation_args(worktree), ok=False),
        p.cli("review-prepare", "--prs", *map(str, selected), ok=False),
        p.cli("agree", "--spec", p.spec(agreements(p)[-1]["agreement"]), ok=False),
    ]
    for response in responses:
        assert response.returncode == 2
        assert "human agreement requirements changed" in response.stderr
        assert "checking → agree" in response.stderr
    p.cli("status")
    assert (p.state(), registry(p).read_bytes()) == before
    assert not (registry(p).parent / "reviews").exists()


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
def test_legacy_human_body_digest_missing_preserves_reads_and_requires_reagreement(
    parent_env: ParentEnv, comment_snapshot: bool
) -> None:
    p = parent_env
    selected = review_setup(p)
    policy = importlib.import_module("parent.policy")
    state = p.state()
    event = agreements(p)[-1]
    original_body = event.pop("body")
    event.pop("requirements_sha256", None)
    if not comment_snapshot:
        event.pop("comment_inputs")
    next(c for c in state["comments"]["1"] if c["body"] == original_body)["body"] = (
        policy.marker(event)
    )
    p.save(state)
    worktree = p.worktree("legacy")
    before = p.state(), registry(p).read_bytes()
    status = json.loads(p.cli("status").stdout)
    assert status["phase"] == "要件合意済み"
    assert status["execution_basis"]["kind"] == "human_agreement"
    responses = [
        p.child("legacy", ok=False),
        p.cli(*delegation_args(worktree), ok=False),
        p.cli("review-prepare", "--prs", *map(str, selected), ok=False),
        p.cli("agree", "--spec", p.spec(event["agreement"]), ok=False),
    ]
    for response in responses:
        assert response.returncode == 2
        assert "legacy human agreement lacks requirements_sha256" in response.stderr
        assert "checking → agree" in response.stderr
    assert (p.state(), registry(p).read_bytes()) == before
    p.cli("checking")
    p.cli("agree", "--spec", p.spec(event["agreement"]))
    p.child("reagreed")
    result = json.loads(p.cli(*delegation_args(worktree, child=13)).stdout)
    assert result["action"] == "launch"
    assert len(agreements(p)) == 2
    assert "requirements_sha256" in agreements(p)[-1]


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
def test_human_body_freshness_blocks_existing_review_and_merge(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    policy = importlib.import_module("parent.policy")
    state = p.state()
    if change == "body":
        state["issues"]["1"]["body"] += "\n\n## 追加指示\n作業を停止する。\n"
        expected = "human agreement requirements changed"
    else:
        event = agreements(p)[-1]
        original = event.pop("body")
        event.pop("requirements_sha256", None)
        next(c for c in state["comments"]["1"] if c["body"] == original)["body"] = (
            policy.marker(event)
        )
        expected = "legacy human agreement lacks requirements_sha256"
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    saved = {path.name: path.read_bytes() for path in g.review.run_dir.glob("*.json")}
    request = p.cli(
        "review-request",
        "--run-id",
        g.review.request["review_run_id"],
        "--instruction-file",
        p.note(),
        ok=False,
    )
    merge = g.cli("merge")
    for response in (request, merge):
        assert response.returncode == 2
        assert expected in response.stderr
    p.cli("status")
    assert (p.state(), registry(p).read_bytes()) == before
    assert {
        path.name: path.read_bytes() for path in g.review.run_dir.glob("*.json")
    } == saved
    assert g.merge_calls() == [] and len(g.review.calls()) == 1


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
    assert [e["revision"] for e in agreements(p)] == [1, 2, 3]
    policy = importlib.import_module("parent.policy")
    history = policy.events(p.state()["comments"]["1"], "operator", 1)
    assert [e["revision"] for e in history if e["event"] == "basis_revoked"] == [1, 2]
    assert all(e["agreement"] == agreement for e in agreements(p))
    assert len({json.dumps(e["comment_inputs"]) for e in agreements(p)}) == 1
    p.cli("checking")  # Also advance after a renewal without an intervening checking.
    p.cli("agree", "--spec", p.spec(agreement))
    assert [e["revision"] for e in agreements(p)] == [1, 2, 3, 4]
