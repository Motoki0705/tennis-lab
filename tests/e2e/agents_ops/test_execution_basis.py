"""Execution evidence through fake GitHub/runtime and private disposable state."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    evidence_comments,
    review_policy_module,
    untracked_legacy_fixture,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.worker_preflight import completed_implementation


def registry(p: ParentEnv) -> Path:
    return (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )


def spec(p: ParentEnv, ignored: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    policy = importlib.import_module("parent.policy")
    args = ("--ignored-comments", p.spec(ignored)) if ignored else ()
    inputs = json.loads(p.cli("basis-input", *args).stdout)
    return {
        "basis": {
            "schema_version": 1,
            "kind": "autonomous_decision",
            "purpose": "未担当の具体的な作業を進める",
            "scope": "issueに記載した変更範囲と対象外を守る",
            "behavior": "共通APIから委任とレビューを要求する",
            "acceptance": "fake CLIと一時stateで境界を確認する",
            "rationale": "issueと関連コードから実装範囲と完了条件を確定できる",
            "sources": ["https://github.com/o/r/issues/1", "code:fixed-commit:path"],
            "input_sha256": policy.digest(inputs),
            "policy_version": "autonomous-start/v1",
        },
        "inputs": inputs,
        "ignored_comments": ignored or [],
    }


def record(p: ParentEnv, value: dict[str, Any] | None = None) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(p.cli("basis-record", "--spec", p.spec(value or spec(p))).stdout),
    )


def autonomous(p: ParentEnv) -> None:
    p.cli("register")
    record(p)


def test_valid_autonomous_basis_allows_split_delegation_and_result(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    p.child("unrecorded", ok=False)
    value = spec(p)
    recorded = record(p, value)
    record(p, value)
    assert sum('"event": "execution_basis"' in c["body"] for c in p.comments()) == 1
    assert not any('"event": "agreed"' in c["body"] for c in p.comments())
    p.child("worker")
    worktree = p.worktree("worker")
    launch = json.loads(
        p.cli(
            "delegate",
            "--child",
            "10",
            "--agent",
            "codex",
            "--branch",
            "codex/worker",
            "--worktree",
            str(worktree),
        ).stdout
    )
    auth = launch["assignment"]["authorization"]
    assert auth == {
        "generation": 1,
        "execution_basis_kind": "autonomous_decision",
        "execution_basis_sha256": recorded["basis_sha256"],
    }
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/worker")]
    p.save(state)
    completed_implementation(p)
    p.cli("record-result", "--child", "10", "--evidence-file", p.note())
    row = json.loads(registry(p).read_text())["parents"]["o/r#1"]
    assert row["tasks"]["10"]["verification"]["authorization"] == auth


@pytest.mark.parametrize(
    "case",
    [
        "version",
        "policy",
        "human",
        "digest",
        "empty",
        "input",
        "input-schema",
        "unknown",
    ],
)
def test_invalid_autonomous_basis_has_no_write(
    parent_env: ParentEnv, case: str
) -> None:
    p = parent_env
    p.cli("register")
    value = spec(p)
    if case == "version":
        value["basis"]["schema_version"] = 99
    elif case == "policy":
        value["basis"]["policy_version"] = "unreviewed/v2"
    elif case == "human":
        value["basis"]["human_confirmation"] = "generated confirmation"
    elif case == "digest":
        value["basis"]["input_sha256"] = "f" * 64
    elif case == "empty":
        value["basis"]["rationale"] = " "
    elif case == "input":
        value["inputs"]["issue"]["body_sha256"] = "a" * 64
        value["basis"]["input_sha256"] = importlib.import_module(
            "parent.policy"
        ).digest(value["inputs"])
    elif case == "input-schema":
        value["inputs"]["schema_version"] = 99
        value["basis"]["input_sha256"] = importlib.import_module(
            "parent.policy"
        ).digest(value["inputs"])
    else:
        value["unexpected"] = True
    before = p.state(), registry(p).read_bytes()
    p.cli("basis-record", "--spec", p.spec(value), ok=False)
    assert (p.state(), registry(p).read_bytes()) == before
    p.child("invalid", ok=False)


def test_unassigned_record_and_owner_generation_handoff(parent_env: ParentEnv) -> None:
    p = parent_env
    candidate = record(p)
    assert candidate["generation"] == 1
    args = (
        "basis-check",
        "--basis-sha256",
        candidate["basis_sha256"],
        "--generation",
        "1",
    )
    assert (
        json.loads(p.cli(*args).stdout)["execution_basis_sha256"]
        == candidate["basis_sha256"]
    )
    assert not registry(p).exists()  # A judgment does not claim a parent.
    p.cli("register")
    p.cli(*args, ok=False)  # A manual registration wins over a stale candidate.
    p.child("existing")
    value = spec(p)
    second = {"CLAUDE_CODE_SESSION_ID": "private-other-session"}
    p.cli("basis-record", "--spec", p.spec(value), extra=second, ok=False)
    p.cli("handoff", "--note-file", p.note())
    p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    p.child("stale-owner", ok=False)
    p.env.update(second)
    p.child("inherited-basis")
    assert record(p)["generation"] == 2
    p.child("next")
    assert "10" in json.loads(registry(p).read_text())["parents"]["o/r#1"]["tasks"]


def test_candidate_rechecks_inputs_without_creating_assignment(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    candidate = record(p)
    state = p.state()
    state["issues"]["1"]["body"] += "\n\n## 指示\n変更された範囲。\n"
    p.save(state)
    p.cli(
        "basis-check",
        "--basis-sha256",
        candidate["basis_sha256"],
        "--generation",
        "1",
        ok=False,
    )
    assert not registry(p).exists()


def test_judgment_job_needs_no_runtime_identity_but_rejects_ambiguous_identity(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    value = spec(p)
    p.cli(
        "basis-record",
        "--spec",
        p.spec(value),
        extra={"CLAUDE_CODE_SESSION_ID": "", "CODEX_SESSION_ID": ""},
    )
    before = p.state()
    p.cli(
        "basis-record",
        "--spec",
        p.spec(value),
        extra={"CLAUDE_CODE_SESSION_ID": "one", "CODEX_SESSION_ID": "two"},
        ok=False,
    )
    assert p.state() == before


def test_long_issue_and_instruction_are_bound_without_copying_them_into_public_record(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["body"] += "\n\n## 詳細\n" + "具体的な要件。" * 10000
    state["comments"]["1"] = [
        {"id": 900, "user": {"login": "human"}, "body": "重要な追加指示。" * 10000}
    ]
    p.save(state)
    value = spec(p)
    assert "body_sha256" in value["inputs"]["issue"]
    assert "sha256" in value["inputs"]["comments"][0]
    record(p, value)
    assert len(p.comments()[-1]["body"]) < 60000
    state = p.state()
    state["comments"]["1"][0]["body"] += "停止の指示。"
    p.save(state)
    p.cli(
        "basis-check",
        "--basis-sha256",
        importlib.import_module("parent.policy").digest(value["basis"]),
        "--generation",
        "1",
        ok=False,
    )


@pytest.mark.parametrize(
    "heading,stable",
    [
        ("Acceptance checklist", True),
        ("全体の完了条件", True),
        ("対象外", False),
        ("方針の選択", False),
    ],
)
def test_only_acceptance_completion_checks_are_nonmaterial(
    parent_env: ParentEnv, heading: str, stable: bool
) -> None:
    execution = importlib.import_module("parent.execution")
    before = f"## {heading}\n\n- [ ] この項目\n"
    after = before.replace("[ ]", "[x]")
    assert (
        execution.requirement_body(before) == execution.requirement_body(after)
    ) is stable


def test_returning_to_a_previous_basis_and_repeated_revocation_are_not_lost(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    autonomous(p)
    original = spec(p)
    changed = copy.deepcopy(original)
    changed["basis"]["scope"] = "別の範囲"
    record(p, changed)
    restored = record(p, original)
    history = importlib.import_module("parent.policy").events(
        p.comments(), "operator", 1
    )
    roots = [e for e in history if e["event"] == "execution_basis"]
    assert [e["revision"] for e in roots] == [1, 2, 3]
    assert (
        roots[-1]["basis_sha256"]
        == restored["basis_sha256"]
        == roots[0]["basis_sha256"]
    )
    for _ in range(2):
        p.cli("basis-revoke", "--reason-file", p.note("人間の同じ停止指示"))
        p.cli("basis-revoke", "--reason-file", p.note("人間の同じ停止指示"))
        p.child("revoked", ok=False)
        p.agreed()
    history = importlib.import_module("parent.policy").events(
        p.comments(), "operator", 1
    )
    assert [e["revision"] for e in history if e["event"] == "basis_revoked"] == [1, 2]


@pytest.mark.parametrize(
    "change", ["body", "instruction", "checking", "revocation", "needs-human"]
)
def test_material_changes_and_explicit_holds_block_autonomous_work(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    autonomous(p)
    value = spec(p)
    state = p.state()
    if change == "body":
        state["issues"]["1"]["body"] += "\n\n## 追加指示\n変更範囲を修正する。\n"
        p.save(state)
    elif change == "instruction":
        state["comments"]["1"].append(
            {
                "id": 900,
                "user": {"login": "human"},
                "body": "この作業は停止し、追加の変更は見送ってください。",
            }
        )
        p.save(state)
    elif change == "checking":
        p.cli("checking")
    elif change == "revocation":
        p.cli(
            "basis-revoke",
            "--reason-file",
            p.note("人間の明示的な見送り。再合意まで停止。"),
        )
    else:
        state["issues"]["1"]["labels"].append({"name": "needs-human"})
        p.save(state)
    if change == "checking":
        p.child("independent")
        return
    if change in {"body", "instruction"}:
        notices = json.loads(p.cli("status").stdout)["input_changes"]
        assert notices["body_changed"] or notices["comments"]
        p.cli(
            "basis-revoke",
            "--reason-file",
            p.note("親が実質変更・停止指示を確認した。"),
        )
    p.child("blocked", ok=False)
    before = p.state()
    p.cli("basis-record", "--spec", p.spec(value), ok=False)
    assert p.state() == before
    if change == "revocation":
        p.agreed()
        p.child("human-resumed")


def test_progress_and_classified_unrelated_comments_keep_basis_identity(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    autonomous(p)
    original = record(p)
    p.child("first")  # Changes the child list and progress, not requirements.
    policy = importlib.import_module("parent.policy")
    state = p.state()
    state["issues"]["1"]["body"] = policy.replace_section(
        state["issues"]["1"]["body"], "進捗概要", "検証中。"
    )
    unrelated = {"id": 900, "user": {"login": "human"}, "body": "表示を確認しました。"}
    state["comments"]["1"].append(unrelated)
    p.save(state)
    p.child("assessed-comment")
    execution = importlib.import_module("parent.execution")
    ignored = [
        {
            "id": 900,
            "sha256": policy.digest(execution.comment_snapshot(unrelated)),
            "reason": "要件の変更を含まない表示確認",
        }
    ]
    assert record(p, spec(p, ignored))["basis_sha256"] == original["basis_sha256"]
    p.child("after-observation")
    state = p.state()
    # Find the observed comment rather than depending on publication count.
    next(c for c in state["comments"]["1"] if c["id"] == 900)["body"] = (
        "停止してください。"
    )
    p.save(state)
    notices = json.loads(p.cli("status").stdout)["input_changes"]["comments"]
    assert any(c["id"] == 900 and c["change"] == "edited" for c in notices)
    p.cli("basis-revoke", "--reason-file", p.note("親が停止指示を確認した。"))
    p.child("edited-observation", ok=False)


@pytest.mark.parametrize("change", ["requirements", "basis", "generation"])
def test_stale_delegation_needs_explicit_parent_result_verification(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    autonomous(p)
    p.child("worker")
    worktree = p.worktree("worker")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/worker",
        "--worktree",
        str(worktree),
    )
    state = p.state()
    state["prs"] = [p.pr(100, 10, branch="codex/worker")]
    if change == "requirements":
        state["issues"]["10"]["body"] = state["issues"]["10"]["body"].replace(
            "動作する", "追加の境界を確認できる"
        )
    p.save(state)
    if change == "basis":
        value = spec(p)
        value["basis"]["scope"] = "変更後の範囲"
        record(p, value)
    elif change == "generation":
        data = json.loads(registry(p).read_text())
        data["parents"]["o/r#1"]["generation"] += 1
        registry(p).write_text(json.dumps(data))
        record(p)
    assert (
        "no recorded run"
        in p.cli(
            "record-result", "--child", "10", "--evidence-file", p.note(), ok=False
        ).stderr
    )
    completed_implementation(p)
    p.cli(
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("現在の要件・差分・回帰テストを確認して既存成果を採用する。"),
    )
    assert json.loads(registry(p).read_text())["parents"]["o/r#1"]["tasks"]["10"][
        "verification"
    ]["prs"] == [100]
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/worker",
        "--worktree",
        str(worktree),
        ok=False,
    )


def test_recording_basis_does_not_resume_a_stopped_child(parent_env: ParentEnv) -> None:
    p = parent_env
    autonomous(p)
    p.child("stopped")
    p.cli("stop", "--child", "10", "--spec", p.detail())
    record(p)
    worktree = p.worktree("stopped")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/stopped",
        "--worktree",
        str(worktree),
        ok=False,
    )


def test_registry_migration_preserves_claims_reservations_and_unknown_fields(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("reserved")
    worktree = p.worktree("reserved")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/reserved",
        "--worktree",
        str(worktree),
    )
    path = registry(p)
    old = json.loads(path.read_text())
    old["version"] = 1
    task = old["parents"]["o/r#1"]["tasks"]["10"]
    task.pop("authorization")
    task.pop("requirements_sha256")
    old["parents"]["o/r#1"]["extension"] = {
        "receipts": ["existing"],
        "notes": "preserve",
    }
    path.write_text(json.dumps(old))
    state_mod = importlib.import_module("parent.state")
    with state_mod.Store(path.parent).locked() as store:
        assert store.data["version"] == 1
        assert store.parents == old["parents"]
        assert json.loads(path.read_text())["version"] == 1
        store.save()
    with state_mod.Store(path.parent).locked() as store:
        assert store.parents == old["parents"]
    assert (
        json.loads(p.cli("reconcile").stdout)["tasks"][0]["action"] == "do_not_launch"
    )


@pytest.mark.parametrize(
    "bad", ["truncated", "duplicate", "version", "generation", "task", "review"]
)
def test_malformed_registry_never_becomes_an_empty_assignment(
    parent_env: ParentEnv, bad: str
) -> None:
    p = parent_env
    p.agreed()
    path = registry(p)
    data = json.loads(path.read_text())
    if bad == "truncated":
        path.write_text('{"version":1,"parents":')
    elif bad == "duplicate":
        path.write_text('{"version":1,"parents":{},"parents":{}}')
    else:
        if bad == "version":
            data["version"] = 100
        elif bad == "generation":
            data["parents"]["o/r#1"]["generation"] = True
        elif bad == "task":
            data["parents"]["o/r#1"]["tasks"] = {"10": None}
        else:
            data["parents"]["o/r#1"]["reviews"] = [{"request": {"schema_version": 99}}]
        path.write_text(json.dumps(data))
    before = path.read_bytes()
    calls = len(p.calls())
    p.cli("register", ok=False)
    assert path.read_bytes() == before
    assert len(p.calls()) == calls


def test_parent_cli_saves_legacy_registry_with_interrupted_write_residue(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.cli("register")
    path = registry(p)
    data = json.loads(path.read_text())
    data["version"] = 1
    path.write_text(json.dumps(data))
    temporary = path.with_name("registry.json.tmp")
    temporary.write_text("interrupted write")
    p.cli("release", "--note-file", p.note())
    saved = json.loads(path.read_text())
    assert saved["version"] == 1
    assert saved["parents"]["o/r#1"]["status"] == "released"
    assert temporary.read_text() == "interrupted write"


def test_missing_registry_symlink_target_is_not_an_empty_assignment(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    path = registry(p)
    path.unlink()
    missing = path.with_name("missing-registry.json")
    path.symlink_to(missing)
    p.cli("register", ok=False)
    assert path.is_symlink() and not missing.exists()


@pytest.mark.parametrize("command", ["register", "release"])
def test_parent_cli_refuses_symlinked_state_directory(
    parent_env: ParentEnv, command: str
) -> None:
    p = parent_env
    link = registry(p).parent
    target = p.root / "private-target"
    args: tuple[str, ...]
    if command == "register":
        target.mkdir(mode=0o700)
        args = ("register",)
    else:
        p.cli("register")
        link.rename(target)
        args = ("--state-dir", str(link), "release", "--note-file", p.note())
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    before = {file.name: file.read_bytes() for file in target.iterdir() if file.is_file()}
    entries = set(target.iterdir())
    calls = len(p.calls())

    result = p.cli(*args, ok=None)

    assert result.returncode != 0 and "symlink" in result.stderr
    assert set(target.iterdir()) == entries and link.is_symlink()
    assert {name: (target / name).read_bytes() for name in before} == before
    assert len(p.calls()) == calls


@pytest.mark.parametrize("kind", ["parent-session", "parent-review"])
def test_unknown_public_marker_version_is_rejected(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    comment = {
        "user": {"login": "operator"},
        "body": f'<!-- {kind}:v99 {{"parent":1}} -->',
    }
    if kind == "parent-session":
        state = parent_env.state()
        state["comments"]["1"] = [{"id": 9, **comment}]
        parent_env.save(state)
        parent_env.cli("register", ok=False)
    else:
        policy = review_policy_module(monkeypatch)
        with pytest.raises(ValueError, match="unknown parent-review marker version"):
            policy.records([comment], "operator")


def ordinary_comment(p: ParentEnv, body: str, login: str = "human") -> dict[str, Any]:
    state = p.state()
    comment = {
        "id": 1 + max(c["id"] for rows in state["comments"].values() for c in rows),
        "user": {"login": login},
        "body": body,
    }
    state["comments"]["1"].append(comment)
    p.save(state)
    return comment


def ignored_comment(comment: dict[str, Any]) -> dict[str, Any]:
    execution = importlib.import_module("parent.execution")
    return {
        "id": comment["id"],
        "sha256": execution.digest(execution.comment_snapshot(comment)),
        "reason": "要件と関係しない雑談であることを本文から確認した。",
    }


@pytest.mark.parametrize(
    "change", ["digest", "duplicate", "missing", "shape", "partial"]
)
def test_human_comment_classification_rejects_invalid_or_incomplete_input(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.agreed()
    comment = ordinary_comment(p, "雑談。")
    entries = [ignored_comment(comment)]
    if change == "digest":
        entries[0]["sha256"] = "a" * 64
    elif change == "duplicate":
        entries *= 2
    elif change == "missing":
        entries[0]["id"] += 100
    elif change == "shape":
        entries[0]["unexpected"] = True
    else:
        ordinary_comment(p, "別の未分類指示。")
    before = p.state(), registry(p).read_bytes()
    p.cli("agreement-ignore-comments", "--ignored-comments", p.spec(entries), ok=False)
    assert (p.state(), registry(p).read_bytes()) == before


def test_human_comment_reclassification_records_return_to_previous_text(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    comment = ordinary_comment(p, "要件と無関係な雑談A。")
    for body in (
        "要件と無関係な雑談A。",
        "要件と無関係な雑談B。",
        "要件と無関係な雑談A。",
    ):
        state = p.state()
        comment = next(c for c in state["comments"]["1"] if c["id"] == comment["id"])
        comment["body"] = body
        p.save(state)
        for _ in range(2):
            p.cli(
                "agreement-ignore-comments",
                "--ignored-comments",
                p.spec([ignored_comment(comment)]),
            )
    policy = importlib.import_module("parent.policy")
    records = [
        e
        for e in policy.events(p.comments(), "operator", 1)
        if e["event"] == "agreement_comments"
    ]
    assert [e["classification_revision"] for e in records] == [1, 2, 3]
    assert records[0]["comment_inputs"] == records[2]["comment_inputs"]
    p.child("reclassified")


@pytest.mark.parametrize("change", ["added", "edited"])
def test_legacy_human_agreement_remains_readable_and_requires_reagreement_for_changes(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.cli("register")
    comment = ordinary_comment(p, "合意前の指示。")
    p.agreed()
    policy = importlib.import_module("parent.policy")
    state = p.state()
    agreed = next(
        c for c in evidence_comments(state) if '"event": "agreed"' in c["body"]
    )
    [event] = policy.events([agreed], "operator", 1)
    agreement = event["agreement"]
    event.pop("body")
    event.pop("comment_inputs")
    agreed["body"] = policy.marker(event)
    agreed["created_at"] = "2026-10-22T03:00:00Z"
    p.save(state)
    untracked_legacy_fixture(p)
    before = p.state(), registry(p).read_bytes()
    p.cli("status")
    assert (p.state(), registry(p).read_bytes()) == before
    p.child("legacy-compatible")
    if change == "added":
        comment = ordinary_comment(p, "作業を止めてください。")
    else:
        state = p.state()
        original = next(c for c in state["comments"]["1"] if c["id"] == comment["id"])
        original.update(
            body="対象範囲を変更します。", updated_at="2026-10-22T03:01:00Z"
        )
        p.save(state)
    notices = json.loads(p.cli("status").stdout)["input_changes"]["comments"]
    assert any(c["id"] == comment["id"] and c["change"] == change for c in notices)
    p.cli("basis-revoke", "--reason-file", p.note("親が変更・停止指示を確認した。"))
    p.child("blocked", ok=False)
    p.cli("agree", "--spec", p.spec(agreement))
    p.child("reagreed")


@pytest.mark.parametrize("change", ["edited", "deleted"])
def test_human_agreement_detects_changes_to_existing_instruction_comments(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    p.cli("register")
    comment = ordinary_comment(p, "合意に含まれた追加要件。")
    p.agreed()
    state = p.state()
    if change == "edited":
        next(c for c in state["comments"]["1"] if c["id"] == comment["id"])["body"] = (
            "要件を変更する。"
        )
    else:
        state["comments"]["1"] = [
            c for c in state["comments"]["1"] if c["id"] != comment["id"]
        ]
    p.save(state)
    notices = json.loads(p.cli("status").stdout)["input_changes"]["comments"]
    assert any(c["id"] == comment["id"] and c["change"] == change for c in notices)
