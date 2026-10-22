"""Adoption → explicit receipt → separate agreement → child, with fake gh only."""

from __future__ import annotations

import fcntl
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
sys.path.insert(0, str(OPS))
formats: Any = importlib.import_module("issues")


def _issue(ops: Any, number: int = 100) -> dict[str, Any]:
    return next(i for i in ops.state["issues"] if i["number"] == number)


def _edit(ops: Any, edit: Any, number: int = 100) -> None:
    state = ops.state
    row = next(i for i in state["issues"] if i["number"] == number)
    row["body"] = edit(row["body"])
    ops.write_state(state)


def _parent(
    ops: Any,
    *args: str,
    parent: int = 1,
    ok: bool = True,
    extra: dict[str, str] | None = None,
) -> Any:
    clean = {
        k: v
        for k, v in ops.env.items()
        if k not in {"CODEX_THREAD_ID", "CODEX_SESSION_ID", "CLAUDE_CODE_SESSION_ID"}
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "parent",
            "--repo",
            "o/r",
            "--parent",
            str(parent),
            *args,
        ],
        cwd=ops.repo,
        env={
            **clean,
            "PYTHONPATH": str(OPS),
            "CLAUDE_CODE_SESSION_ID": "private-parent-session",
            **(extra or {}),
        },
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert (result.returncode == 0) == ok, result.stderr + result.stdout
    return result


def _json(ops: Any, value: Any) -> str:
    path = ops.tmp / "input.json"
    path.write_text(json.dumps(value, ensure_ascii=False))
    return str(path)


def _agree(ops: Any, *, pid: str | None = None, report: int = 100) -> None:
    if pid is not None:
        status = json.loads(_parent(ops, "status").stdout)
        if status["phase"] != "要件合意済み":
            _agree(ops)
    else:
        _parent(ops, "checking")
    _parent(
        ops,
        "agree",
        *(["--report", str(report), "--proposal", pid] if pid is not None else []),
        "--spec",
        _json(
            ops,
            {
                "purpose": "採用した改善を具体化する",
                "scope": "big.pyと関連テスト",
                "behavior": "合意後に分割する",
                "acceptance": "fakeと回帰テストで確認する",
                "human_confirmation": "この目的・範囲・動作・受入条件で合意します。",
            },
        ),
    )


def _setup(ops: Any, *, route: bool = True) -> dict[str, Any]:
    state = ops.state
    sections = {
        s: "親の目的と範囲。"
        for s in formats.PARENT_REQUIRED
        if s != formats.TASKS_HEADING
    }
    template = state["issues"][0]
    for n in (1, 2):
        state["issues"].append(
            {
                **template,
                "number": n,
                "title": f"[改善] 親{n}",
                "body": formats.render_parent_body(sections, []),
                "state": "OPEN",
                "labels": ["issue:parent"],
                "url": f"https://github.com/o/r/issues/{n}",
            }
        )
    ops.write_state(state)
    if route:
        data = json.loads((ops.tmp / "agent_output.json").read_text())
        for prop in data["proposals"]:
            prop["related_parent"] = 1
        ops.set_agent_output(data)
    generated = ops.run("report")
    assert generated.returncode == 0, generated.stderr
    return _issue(ops)


def _select(ops: Any, pid: str = "R-20261012-02") -> None:
    _edit(ops, lambda b: b.replace(f"- [ ] **{pid}**", f"- [x] **{pid}**"))


def _receive(ops: Any, pid: str = "R-20261012-02", ok: bool = True, **env: str) -> Any:
    return _parent(
        ops, "receive-proposal", "--report", "100", "--proposal", pid, ok=ok, extra=env
    )


def _manifest(
    pid: str = "R-20261012-02", *, key: str = "split", option: str | None = None
) -> dict[str, Any]:
    sections = {s: s + "を実装する。" for s in formats.CHILD_REQUIRED}
    sections["依存関係"] = "なし。"
    sections["Acceptance checklist"] = (
        "- [ ] 責務を分割できる\n  文脈: 保守性。\n  下流への影響: 変更しやすくなる。\n  検証方法・証拠: 回帰テスト。"
    )
    return {
        "key": key,
        "dependencies": [],
        "spec": {
            "role": "child",
            "summary": "責務を分割する",
            "parent": {"identifier": "改善", "number": 1},
            "sections": sections,
            "source": {"proposal_id": pid, "report_number": 100, "option": option},
        },
    }


def _create(ops: Any, *, ok: bool = True, manifest: Any = None, **env: str) -> Any:
    return _parent(
        ops,
        "create-child",
        "--spec",
        _json(ops, manifest or _manifest()),
        ok=ok,
        extra=env,
    )


def _children(ops: Any) -> list[dict[str, Any]]:
    return [i for i in ops.state["issues"] if "issue:child" in i["labels"]]


def _events(ops: Any, kind: str) -> list[dict[str, Any]]:
    found = []
    for comment in ops.state.get("comments", {}).get("1", []):
        first = comment["body"].splitlines()[0]
        event = json.loads(first[len("<!-- parent-session:v1 ") : -4])
        if event["event"] == kind:
            found.append(event)
    return found


def test_adoption_receipt_and_agreement_are_three_distinct_states(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    assert ops.run("triage").returncode == 0
    assert _events(ops, "proposal_pending") == []
    _receive(ops, ok=False)
    _create(ops, ok=False)
    _select(ops)
    assert ops.run("triage").returncode == 0
    assert len(_events(ops, "proposal_pending")) == 1
    assert "受領待ち" in _issue(ops)["body"]
    assert _children(ops) == []
    _create(ops, ok=False)
    _receive(ops)
    _create(ops, ok=False)
    _parent(
        ops,
        "delegate",
        "--child",
        "101",
        "--agent",
        "codex",
        "--branch",
        "codex/split",
        "--worktree",
        str(ops.repo),
        ok=False,
    )
    assert _children(ops) == []
    _agree(ops, pid="R-20261012-02")
    created = json.loads(_create(ops).stdout)
    assert created["child"] == 101 and not created["reused"]
    child = _children(ops)[0]
    assert child["title"] == "[改善 #1] 責務を分割する"
    assert child["body"].startswith(
        "<!-- ops-report-proposal: R-20261012-02 report=#100 option= -->\n"
    )
    assert ops.state["parents"] == {"101": 1}
    assert "- [ ] #101 責務を分割する" in _issue(ops, 1)["body"]
    assert ops.run("triage").returncode == 0
    assert "→ #101" in _issue(ops)["body"]
    _receive(ops)  # Does not revoke the later agreement.
    assert json.loads(_create(ops).stdout)["reused"]
    changed = _manifest()
    changed["spec"]["sections"]["具体的な変更内容・対象範囲"] += "追加の変更。"
    _create(ops, manifest=changed, ok=False)
    assert len(_events(ops, "proposal_received")) == 1
    assert "private-parent-session" not in json.dumps(ops.state)


def test_old_parent_agreement_does_not_authorize_newly_received_proposal(
    ops_env: Any,
) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _agree(ops)
    _select(ops)
    agreement = _events(ops, "agreed")[0]["agreement"]
    before = ops.state
    rejected = _parent(
        ops,
        "agree",
        "--report",
        "100",
        "--proposal",
        "R-20261012-02",
        "--spec",
        _json(ops, agreement),
        ok=False,
    )
    assert "explicit receipt" in rejected.stderr
    assert ops.state == before
    _create(ops, ok=False)  # No explicit receipt, even with a parent agreement.
    _receive(ops)
    _create(ops, ok=False)
    assert _children(ops) == []
    _agree(ops, pid="R-20261012-02")
    _create(ops)


def test_proposal_agreement_checks_plain_comment_after_parent_note(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    _receive(ops)
    _agree(ops, pid="R-20261012-02")
    body = "調査を終えたので、合意済みの範囲で子issueを作成します。"
    note = ops.tmp / "note.md"
    note.write_text(body, encoding="utf-8")
    _parent(ops, "note", "--body-file", str(note))
    _create(ops)

    # The same login/body without a progress marker still needs classification.
    posted = subprocess.run(
        [
            str(ops.tmp / "bin/gh"),
            "api",
            "--method",
            "POST",
            "repos/o/r/issues/1/comments",
            "--input",
            "-",
        ],
        input=json.dumps({"body": body}, ensure_ascii=False),
        env=ops.env,
        text=True,
        capture_output=True,
        timeout=45,
        check=True,
    )
    comment = json.loads(posted.stdout)
    before = ops.state
    rejected = _create(ops, ok=False)
    assert "unclassified agreement comments" in rejected.stderr
    assert f"ID={comment['id']}" in rejected.stderr
    assert comment["html_url"] in rejected.stderr and "(added)" in rejected.stderr
    assert ops.state == before
    assert len(_children(ops)) == 1


def test_pending_proposal_preserves_independent_creation_and_delegation(
    ops_env: Any,
) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _agree(ops)
    independent = _manifest(key="independent")
    independent["spec"].pop("source")
    child = json.loads(_create(ops, manifest=independent).stdout)["child"]
    _select(ops)
    _receive(ops)
    status = json.loads(_parent(ops, "status").stdout)
    assert status["phase"] == "要件合意済み"
    assert len(status["proposal_receipts"]) == 1
    assert status["proposal_agreements"] == []
    before = ops.state
    assert "separate human requirements agreement" in _create(ops, ok=False).stderr
    assert ops.state == before
    assert json.loads(_create(ops, manifest=independent).stdout)["reused"]
    worktree = ops.repo / ".claude/worktrees/independent"
    subprocess.run(
        ["git", "worktree", "add", "-b", "codex/independent", str(worktree)],
        cwd=ops.repo,
        check=True,
        capture_output=True,
    )
    delegated = _parent(
        ops,
        "delegate",
        "--child",
        str(child),
        "--agent",
        "codex",
        "--branch",
        "codex/independent",
        "--worktree",
        str(worktree),
    )
    assert json.loads(delegated.stdout)["action"] == "launch"
    # An existing child from the pending proposal is guarded before any reservation/write.
    _edit(
        ops,
        lambda body: (
            "<!-- ops-report-proposal: R-20261012-02 report=#100 option= -->\n" + body
        ),
        child,
    )
    before = ops.state
    rejected = _parent(
        ops,
        "delegate",
        "--child",
        str(child),
        "--agent",
        "codex",
        "--branch",
        "codex/independent",
        "--worktree",
        str(worktree),
        ok=False,
    )
    assert "separate human requirements agreement" in rejected.stderr
    assert ops.state == before


def test_proposal_agreements_are_scoped_and_parent_checking_revokes_them(
    ops_env: Any,
) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _agree(ops)
    _select(ops)
    _receive(ops)
    _edit(ops, lambda b: b.replace("  - [ ] 5件に絞る", "  - [x] 5件に絞る"))
    _receive(ops, "R-20261012-03")
    _agree(ops, pid="R-20261012-02")
    _create(ops)
    other = _manifest("R-20261012-03", key="other", option="5件に絞る")
    _create(ops, manifest=other, ok=False)
    assert len(_children(ops)) == 1
    assert len(_events(ops, "proposal_agreed")) == 1
    _agree(ops, pid="R-20261012-02")  # Idempotent without revoking any work.
    assert len(_events(ops, "proposal_agreed")) == 1
    changed = dict(_events(ops, "proposal_agreed")[0]["agreement"])
    changed["scope"] = "既存作業の要件も変更する。"
    before = ops.state
    rejected = _parent(
        ops,
        "agree",
        "--report",
        "100",
        "--proposal",
        "R-20261012-02",
        "--spec",
        _json(ops, changed),
        ok=False,
    )
    assert "agreement changed" in rejected.stderr
    assert ops.state == before
    _agree(ops, pid="R-20261012-03")
    _create(ops, manifest=other)
    assert (
        len(_events(ops, "agreed")) == 1
    )  # Proposal consent leaves the parent digest intact.
    _parent(ops, "checking")
    independent = _manifest(key="independent")
    independent["spec"].pop("source")
    _create(ops, manifest=independent, ok=False)
    _create(ops, ok=False)
    _agree(ops)
    _create(ops, manifest=independent)
    _create(ops, ok=False)  # Re-agreeing the parent never authorizes its proposals.
    _agree(ops, pid="R-20261012-02")
    assert json.loads(_create(ops).stdout)["reused"]
    assert len(_events(ops, "proposal_agreed")) == 3


@pytest.mark.parametrize("stopped", [False, True])
def test_unassigned_or_stopped_parent_then_manual_assignment_receives_once(
    ops_env: Any, stopped: bool
) -> None:
    ops = ops_env
    _setup(ops, route=False)
    if stopped:
        _parent(ops, "register")
        note = ops.tmp / "release.md"
        note.write_text("手動で再開する。")
        _parent(ops, "release", "--note-file", str(note))
        _edit(ops, lambda b: b.replace("関連親issue: 未指定", "関連親issue: #1"))
    _select(ops)
    result = ops.run("triage")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["unassigned"] == ["R-20261012-02"]
    assert ("未割当" if stopped else "保留: 評価対象の選択が不足") in _issue(ops)[
        "body"
    ]
    assert _events(ops, "proposal_pending") == []
    _receive(ops, ok=False)
    _edit(ops, lambda b: b.replace("関連親issue: 未指定", "関連親issue: #1"))
    _parent(ops, "register")
    for _ in range(2):
        assert ops.run("triage").returncode == 0
        _receive(ops)
    assert (
        len(_events(ops, "proposal_pending"))
        == len(_events(ops, "proposal_received"))
        == 1
    )
    assert _children(ops) == []


@pytest.mark.parametrize("failure", ["create", "relation", "progress", "annotation"])
def test_interruptions_never_duplicate_children(ops_env: Any, failure: str) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    _receive(ops)
    _agree(ops, pid="R-20261012-02")
    if failure == "annotation":
        _create(ops)
        assert (
            ops.run("triage", FAKE_GH_API_FAIL="repos/o/r/issues/100").returncode == 1
        )
    else:
        env = (
            {"FAKE_GH_API_FAIL_AFTER_SAVE": "repos/o/r/issues"}
            if failure == "create"
            else {
                "FAKE_GH_API_FAIL": "repos/o/r/issues/1/sub_issues"
                if failure == "relation"
                else "repos/o/r/issues/1"
            }
        )
        _create(ops, ok=False, **env)
    assert len(_children(ops)) == 1
    assert json.loads(_create(ops).stdout)["reused"]
    assert ops.run("triage").returncode == 0
    assert len(_children(ops)) == 1 and "→ #101" in _issue(ops)["body"]


def test_pending_and_receipt_response_loss_are_idempotent(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    assert (
        ops.run(
            "triage", FAKE_GH_API_FAIL_AFTER_SAVE="repos/o/r/issues/1/comments"
        ).returncode
        == 1
    )
    assert ops.run("triage").returncode == 0
    _receive(ops, ok=False, FAKE_GH_API_FAIL_AFTER_SAVE="repos/o/r/issues/1/comments")
    _create(ops, ok=False)
    _receive(ops)
    assert (
        len(_events(ops, "proposal_pending"))
        == len(_events(ops, "proposal_received"))
        == 1
    )
    assert _children(ops) == []


@pytest.mark.parametrize("after_child", [False, True])
def test_option_changes_are_rejected_after_delivery_or_creation(
    ops_env: Any, after_child: bool
) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _edit(ops, lambda b: b.replace("  - [ ] 5件に絞る", "  - [x] 5件に絞る"))
    _receive(ops, "R-20261012-03")
    _agree(ops, pid="R-20261012-03")
    manifest = _manifest("R-20261012-03", option="5件に絞る")
    if after_child:
        _create(ops, manifest=manifest)
    _edit(
        ops,
        lambda b: b.replace("  - [x] 5件に絞る", "  - [ ] 5件に絞る").replace(
            "  - [ ] 上限なし", "  - [x] 上限なし"
        ),
    )
    result = ops.run("triage")
    assert result.returncode == 1 and "changed" in result.stderr
    _create(ops, manifest=manifest, ok=False)
    assert len(_children(ops)) == int(after_child)


def test_selection_changes_during_delivery_preserve_human_edit(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    result = ops.run(
        "triage",
        FAKE_GH_API_EDIT=json.dumps(
            [["- [x] **R-20261012-02**", "- [ ] **R-20261012-02**"]]
        ),
    )
    assert result.returncode == 1
    assert "not annotated" in result.stderr or "changed" in result.stdout
    assert "引渡し:" not in _issue(ops)["body"]
    _receive(ops, ok=False)
    assert _children(ops) == []


def test_renaming_reparenting_and_new_task_key_reuse_metadata_identity(
    ops_env: Any,
) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    _receive(ops)
    _agree(ops, pid="R-20261012-02")
    _create(ops)
    state = ops.state
    child = next(i for i in state["issues"] if i["number"] == 101)
    child["title"] = "[改善 #2] 名前を変更した提案"
    child["body"] = child["body"].replace("親issue: #1", "親issue: #2")
    state["parents"]["101"] = 2
    ops.write_state(state)
    assert ops.run("triage").returncode == 0
    reused = json.loads(_create(ops, manifest=_manifest(key="renamed")).stdout)
    assert reused["child"] == 101 and reused["actual_parent"] == 2
    assert len(_children(ops)) == 1
    assert ops.state["parents"]["101"] == 2


def test_duplicate_proposal_metadata_fails_before_writes(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    _receive(ops)
    _agree(ops, pid="R-20261012-02")
    _create(ops)
    state = ops.state
    child = _children(ops)[0]
    state["issues"].append({**child, "number": 102, "title": "改名した別issue"})
    ops.write_state(state)
    before = ops.state
    assert ops.run("triage").returncode == 1
    _create(ops, ok=False)
    assert ops.state == before


def test_human_issue_uses_same_registration_agreement_and_create_entry(
    ops_env: Any,
) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    manifest = _manifest()
    manifest["spec"].pop("source")
    _create(ops, ok=False, manifest=manifest)
    _agree(ops)
    _create(ops, manifest=manifest)
    assert ops.state["parents"]["101"] == 1


def test_dry_run_relative_logs_and_concurrent_triage(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    before = ops.state
    caller = ops.tmp / "caller"
    caller.mkdir()
    result = ops.run("triage", "--dry-run", "--log-dir", "relative logs", cwd=caller)
    assert result.returncode == 0 and "would deliver" in result.stderr
    assert ops.state == before
    runs = [ops.start("triage", FAKE_GH_CREATE_DELAY="1") for _ in range(2)]
    results = [p.communicate(timeout=90) + (p.returncode,) for p in runs]
    assert all(r[2] == 0 for r in results), results
    assert any("waiting for lock" in r[1] for r in results)
    assert len(_events(ops, "proposal_pending")) == 1
    assert _children(ops) == []


def test_multiple_options_and_forged_links_fail_closed(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _edit(
        ops,
        lambda b: b.replace("  - [ ] 5件に絞る", "  - [x] 5件に絞る").replace(
            "  - [ ] 上限なし", "  - [x] 上限なし"
        ),
    )
    assert ops.run("triage").returncode == 1
    _edit(
        ops,
        lambda b: b.replace("  - [x] 上限なし", "  - [ ] 上限なし").replace(
            "（優先度: 高）", "（優先度: 高） → #50"
        ),
    )
    assert ops.run("triage").returncode == 1
    assert _children(ops) == []


def test_old_report_version_prevents_all_mutations(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _select(ops)
    state = ops.state
    state["issues"].append(
        {
            **_issue(ops),
            "number": 60,
            "body": _issue(ops)["body"].replace(
                "ops-report:v2 slot=20261012", "ops-report:v1 week=2026W42"
            ),
        }
    )
    ops.write_state(state)
    before = ops.state
    result = ops.run("triage")
    assert result.returncode == 1 and "unsupported ops-report:v1" in result.stderr
    assert ops.state == before


def test_triage_lock_failure_is_not_success(ops_env: Any) -> None:
    path = Path(ops_env.env["OPS_REPORT_LOCK_FILE"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = ops_env.run("triage", OPS_REPORT_LOCK_TIMEOUT="1")
    assert result.returncode == 1 and "still held" in result.stderr


def test_github_decisions_reach_next_report_and_explicit_deferral(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    _receive(ops)
    _agree(ops, pid="R-20261012-02")
    _create(ops)
    _edit(
        ops,
        lambda b: b.replace(
            "見送り理由: 未記入", "見送り理由: 次の構成変更まで保留する。", 1
        ),
    )
    # Metadata supplies the child reference even before the next daily annotation.
    assert "→ #101" not in _issue(ops)["body"]
    assert ops.run("report", "--now", "2026-10-15T08:00:00+09:00").returncode == 0
    logs = ops.tmp / "logs"
    collected = [json.loads(p.read_text()) for p in logs.glob("*/collected.json")]
    current = next(
        c for c in collected if c["meta"]["period_end"].startswith("2026-10-15")
    )
    decisions = current["ops_memory"]["decisions"]
    assert {d["proposal_id"]: d["decision"] for d in decisions} == {
        "R-20261012-01": "deferred",
        "R-20261012-02": "selected",
    }
    assert "次の構成変更" in decisions[0]["reason"]
    assert "https://github.com/o/r/issues/100" in decisions[1]["source"]
    assert "子issue #101" in decisions[1]["source"]


def test_parent_closed_or_handoff_pending_defers_without_receipt(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    state = ops.state
    next(i for i in state["issues"] if i["number"] == 1)["state"] = "CLOSED"
    ops.write_state(state)
    result = ops.run("triage")
    assert result.returncode == 0 and "未割当" in _issue(ops)["body"]
    _receive(ops, ok=False)
    state = ops.state
    next(i for i in state["issues"] if i["number"] == 1)["state"] = "OPEN"
    ops.write_state(state)
    note = ops.tmp / "handoff.md"
    note.write_text("次の指定を待つ。")
    _parent(ops, "handoff", "--note-file", str(note))
    assert ops.run("triage").returncode == 0
    _receive(ops, ok=False)
    assert _events(ops, "proposal_pending") == _events(ops, "proposal_received") == []
    _parent(ops, "handoff", "--accept", "--note-file", str(note))
    _receive(ops)
    assert len(_events(ops, "proposal_received")) == 1
    assert _children(ops) == []


def test_adoption_changed_during_creation_is_not_success(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    _select(ops)
    _receive(ops)
    _agree(ops, pid="R-20261012-02")
    _create(
        ops,
        ok=False,
        FAKE_GH_API_EDIT=json.dumps(
            [["- [x] **R-20261012-02**", "- [ ] **R-20261012-02**"]]
        ),
    )
    assert len(_children(ops)) == 1
    assert ops.run("triage").returncode == 1
    assert "→ #101" not in _issue(ops)["body"]
    _select(ops)
    assert json.loads(_create(ops).stdout)["reused"]
    assert len(_children(ops)) == 1


def test_stale_local_decision_snapshot_cannot_override_github(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    decisions = _json(
        ops,
        [
            {
                "proposal_id": "R-20261012-02",
                "decision": "deferred",
                "reason": "古い見送り",
                "source": "https://github.com/o/r/issues/100 / R-20261012-02",
            }
        ],
    )
    before = ops.state
    (ops.tmp / "agent_used.txt").unlink()
    result = ops.run(
        "report", "--now", "2026-10-15T08:00:00+09:00", "--memory-decisions", decisions
    )
    assert result.returncode == 1 and "no longer matches GitHub" in result.stderr
    assert ops.state == before
    assert not (ops.tmp / "agent_used.txt").exists()


def test_three_slots_in_one_week_keep_distinct_proposal_identity(ops_env: Any) -> None:
    ops = ops_env
    _setup(ops)
    _parent(ops, "register")
    created = []
    for day in ("12", "15", "18"):
        if day != "12":
            result = ops.run("report", "--now", f"2026-10-{day}T08:00:00+09:00")
            assert result.returncode == 0, result.stderr
        report = next(
            i
            for i in ops.state["issues"]
            if i["title"] == f"運用レポート 2026-10-{day}"
        )
        pid = f"R-202610{day}-02"
        _edit(
            ops,
            lambda b, pid=pid: b.replace(f"- [ ] **{pid}**", f"- [x] **{pid}**"),
            report["number"],
        )
        assert ops.run("triage").returncode == 0
        _parent(
            ops,
            "receive-proposal",
            "--report",
            str(report["number"]),
            "--proposal",
            pid,
        )
        _agree(ops, pid=pid, report=report["number"])
        manifest = _manifest(pid, key=f"split-{day}")
        manifest["spec"]["source"]["report_number"] = report["number"]
        child = json.loads(_create(ops, manifest=manifest).stdout)["child"]
        created.append(child)
        assert _issue(ops, child)["body"].startswith(
            f"<!-- ops-report-proposal: {pid} report=#{report['number']} option= -->"
        )
    assert ops.run("triage").returncode == 0
    assert len(set(created)) == len(_children(ops)) == 3
    assert len(_events(ops, "proposal_received")) == 3
