"""Freeform progress notes through fake GitHub and disposable parent state."""

from __future__ import annotations

import importlib
import json
from typing import cast

import pytest

from tests.e2e.agents_ops.test_execution_basis import (
    auto_gate,
    autonomous,
    ordinary_comment,
    registry,
)
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv, review_setup
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


@pytest.mark.parametrize("phase", ["assigned", "checking", "agreed", "autonomous"])
def test_owner_note_is_idempotent_and_preserves_state(
    parent_env: ParentEnv, phase: str
) -> None:
    p = parent_env
    if phase == "agreed":
        p.agreed()
    elif phase == "autonomous":
        autonomous(p)
    else:
        p.cli("register")
        if phase == "checking":
            p.cli("checking")
    policy = importlib.import_module("parent.policy")
    before = registry(p).read_bytes(), p.cli("status").stdout
    inputs = p.cli("basis-input").stdout
    text = "## 進捗\n\n検証を完了した。次は依存先の結果を確認する。"
    result = p.cli("note", "--body-file", p.note("\n" + text + "\n"))
    assert json.loads(result.stdout) == {
        "parent": 1,
        "body_sha256": policy.digest(text),
    }
    comment = p.state()["comments"]["1"][-1]
    event = json.loads(comment["body"].splitlines()[0][len(policy.PREFIX) : -4])
    assert event == {
        "parent": 1,
        "event": "note",
        "generation": 1,
        "body_sha256": policy.digest(text),
    }
    assert comment["body"] == policy.marker(event) + "\n\n" + text
    assert p.env["CLAUDE_CODE_SESSION_ID"] not in comment["body"]
    assert p.env["XDG_STATE_HOME"] not in comment["body"]
    state = p.state()
    assert p.cli("note", "--body-file", p.note(text)).stdout == result.stdout
    assert p.state() == state
    assert (registry(p).read_bytes(), p.cli("status").stdout) == before
    assert p.cli("basis-input").stdout == inputs


@pytest.mark.parametrize(
    "assignment", ["unassigned", "other", "released", "ended", "handoff_pending"]
)
def test_note_requires_active_owner(parent_env: ParentEnv, assignment: str) -> None:
    p = parent_env
    if assignment != "unassigned":
        p.cli("register")
    if assignment in {"released", "ended"}:
        p.cli(
            "release",
            *(["--ended"] if assignment == "ended" else []),
            "--note-file",
            p.note(),
        )
    elif assignment == "handoff_pending":
        p.cli("handoff", "--note-file", p.note())
    extra = (
        {"CLAUDE_CODE_SESSION_ID": "private-other-session"}
        if assignment == "other"
        else {}
    )
    before = p.state(), registry(p).read_bytes() if registry(p).exists() else None
    result = p.cli("note", "--body-file", p.note(), extra=extra, ok=False)
    assert result.returncode == 2
    assert "担当未指定・別セッション・終了済み・引継ぎ待ち" in result.stderr
    assert (
        p.state(),
        registry(p).read_bytes() if registry(p).exists() else None,
    ) == before


@pytest.mark.parametrize(
    "text",
    [
        "",
        " \n\t",
        "進捗\n<!-- parent-session:v1 {} -->",
        "本文中の <!-- HTML comment",
        "終端だけ -->",
        "parent-session:v1",
        "PARENT-REVIEW:v1",
        "x" * 60_001,
        "非公開のprivate-session-a-uniqueを含む本文",
    ],
    ids=[
        "empty",
        "whitespace",
        "marker",
        "html-open",
        "html-close",
        "session",
        "review",
        "large",
        "private",
    ],
)
def test_invalid_note_has_no_write(parent_env: ParentEnv, text: str) -> None:
    p = parent_env
    p.cli("register")
    before = p.state(), registry(p).read_bytes()
    assert p.cli("note", "--body-file", p.note(text), ok=False).returncode == 2
    assert (p.state(), registry(p).read_bytes()) == before


def test_note_character_limit_accepts_multibyte_text(parent_env: ParentEnv) -> None:
    p = parent_env
    p.cli("register")
    text = "進" * 60_000
    p.cli("note", "--body-file", p.note(text))
    assert p.state()["comments"]["1"][-1]["body"].endswith("\n\n" + text)
    p.cli("status")


def test_note_keeps_frozen_review_requirements_and_request(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    selected = review_setup(p)
    args = ("review-prepare", "--prs", *map(str, selected))
    prepared = p.cli(*args).stdout
    p.cli("note", "--body-file", p.note("対象PR群の確認を完了した。"))
    assert p.cli(*args).stdout == prepared
    run = json.loads(prepared)["review_run_id"]
    instructed = p.cli(
        "review-request",
        "--run-id",
        run,
        "--instruction-file",
        p.note("対象をレビューする。"),
    ).stdout
    p.cli("note", "--body-file", p.note("レビューを指示した。"))
    assert (
        p.cli(
            "review-request",
            "--run-id",
            run,
            "--instruction-file",
            p.note("対象をレビューする。"),
        ).stdout
        == instructed
    )


def test_note_does_not_classify_existing_unclassified_comments(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    comment = ordinary_comment(p, "範囲を変更してください。", "operator")
    p.cli("note", "--body-file", p.note("変更の確認待ち。"))
    response = p.child("unclassified", ok=False)
    assert f"ID={comment['id']}" in response.stderr
    assert "agreement-ignore-comments" in response.stderr


@pytest.mark.parametrize("kind", ["human", "autonomous"])
@pytest.mark.parametrize("comment", ["same-body", "human-instruction"])
def test_note_preserves_review_and_merge_but_ordinary_comments_block(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    kind: str,
    comment: str,
) -> None:
    p = parent_env
    g = (
        auto_gate(p, monkeypatch)
        if kind == "autonomous"
        else cast(GateEnv, request.getfixturevalue("gate_env"))
    )
    inputs = p.cli("basis-input").stdout
    frozen = registry(p).read_bytes(), (g.review.run_dir / "result.json").read_bytes()
    text = "## 進捗\n\n親が結果・差分・CI・回帰テストを確認した。"
    p.cli("note", "--body-file", p.note(text))
    assert p.cli("basis-input").stdout == inputs
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert all(row["decision"] == "merge" for row in rows.values())
    assert (
        registry(p).read_bytes(),
        (g.review.run_dir / "result.json").read_bytes(),
    ) == frozen
    child = str(p.state()["next"])
    p.child("note-preserves-basis")
    worktree = p.worktree("note-worker")
    p.cli(
        "delegate",
        "--child",
        child,
        "--agent",
        "codex",
        "--branch",
        "codex/note-worker",
        "--worktree",
        str(worktree),
    )

    added = ordinary_comment(
        p,
        text if comment == "same-body" else "作業を停止して要件を確認してください。",
        "operator" if comment == "same-body" else "human",
    )
    current = json.loads(p.cli("basis-input").stdout)
    assert current["comments"][-1]["id"] == added["id"]
    before = p.state(), registry(p).read_bytes()
    responses = [
        p.child("unclassified", ok=False),
        p.cli(
            "delegate",
            "--child",
            child,
            "--agent",
            "codex",
            "--branch",
            "codex/note-worker",
            "--worktree",
            str(worktree),
            ok=False,
        ),
        p.cli(
            "review-request",
            "--run-id",
            g.review.request["review_run_id"],
            "--instruction-file",
            p.note(),
            ok=False,
        ),
        g.cli("merge"),
    ]
    assert all(response.returncode == 2 for response in responses)
    assert (p.state(), registry(p).read_bytes()) == before
    assert not g.merge_calls() and len(g.review.calls()) == 1


@pytest.mark.parametrize("change", ["body", "digest", "generation", "field"])
def test_corrupt_note_marker_is_rejected(parent_env: ParentEnv, change: str) -> None:
    p = parent_env
    p.agreed()
    p.cli("note", "--body-file", p.note("進捗だけの連絡。"))
    policy = importlib.import_module("parent.policy")
    state = p.state()
    comment = state["comments"]["1"][-1]
    event = json.loads(comment["body"].splitlines()[0][len(policy.PREFIX) : -4])
    text = "進捗だけの連絡。"
    if change == "body":
        text = "作業を停止してください。"
    elif change == "digest":
        event["body_sha256"] = "f" * 64
    elif change == "generation":
        event["generation"] = 0
    else:
        event["unexpected"] = True
    comment["body"] = policy.marker(event) + "\n\n" + text
    p.save(state)
    before = p.state(), registry(p).read_bytes()
    assert p.cli("status", ok=False).returncode == 2
    p.child("corrupt-note", ok=False)
    assert (p.state(), registry(p).read_bytes()) == before
