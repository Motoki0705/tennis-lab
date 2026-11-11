"""One formal parent issue through execution, review, merge and completion.

GitHub and agent CLIs are offline fixtures. Target resolution, registry writes,
git snapshots, review publication, merge gate and completion use real entrypoints.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import manual, store
from tests.e2e.agents_ops.test_merge_gate import GateEnv
from tests.e2e.agents_ops.test_merge_gate import gate_env as gate_env
from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def completion_spec(p: ParentEnv) -> str:
    return str(
        p.spec(
            {
                "schema_version": 1,
                "state": "completed",
                "reason": "受入条件・実merge・close・実行終了を確認した。",
                "resume_condition": "完了。追加要件は別途判断する。",
                "evidence": "自身の要件を差分と回帰テストで確認し、PR #100 の対象HEAD・CI・正式レビュー・mergeを照合した。",
            }
        )
    )


def assert_no_hierarchy_writes(p: ParentEnv) -> None:
    writes = [
        c for c in p.calls() if "--method" in c and c[c.index("--method") + 1] != "GET"
    ]
    assert not any(
        "repos/o/r/issues" in c or any("/sub_issues" in arg for arg in c)
        for c in writes
    )
    assert p.state()["parents"] == {}


def introduce_split(p: ParentEnv) -> None:
    """Model a concurrent remote split without using the parent's guarded CLI."""
    state = p.state()
    sections = {name: "独立した成果を検証する。" for name in p.formats.CHILD_REQUIRED}
    sections["依存関係"] = "なし。"
    sections["Acceptance checklist"] = (
        "- [ ] 独立して完成する\n  文脈: 分割後。\n  下流への影響: 二重起動を防ぐ。\n  検証方法・証拠: fake gh。"
    )
    state["issues"]["10"] = {
        **state["issues"]["2"],
        "number": 10,
        "id": 10010,
        "title": "[開発 #1] 独立成果",
        "labels": [{"name": "issue:child"}],
        "body": p.formats.render_child_body(1, sections),
    }
    state["parents"]["10"] = 1
    p.save(state)


def test_labelled_parent_resolves_once_and_sync_preserves_requirements(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    before = p.state()["issues"]["1"]["body"]
    execution = importlib.import_module("parent.execution")
    status = json.loads(p.cli("status").stdout)
    assert status["launch_plan"]["launchable_children"] == [1]
    assert [t["child"] for t in status["launch_plan"]["tasks"]] == [1]
    synced = json.loads(p.cli("sync").stdout)
    assert synced["progress"] == {"1": "continue"}
    after = p.state()["issues"]["1"]["body"]
    assert execution.requirement_body(before) == execution.requirement_body(after)
    before_sections = p.formats.parse_sections(before)
    after_sections = p.formats.parse_sections(after)
    for heading in p.formats.PARENT_REQUIRED:
        if heading != "進捗概要":
            assert after_sections[heading] == before_sections[heading]
    assert (
        p.formats.parse_sections(before)[p.formats.TASKS_HEADING]
        == p.formats.parse_sections(after)[p.formats.TASKS_HEADING]
    )
    rejected = p.cli("workflow-report", "--spec", completion_spec(p), ok=False)
    assert "acceptance evidence" in rejected.stderr
    assert_no_hierarchy_writes(p)


def test_subissue_fetch_failure_never_becomes_self_work(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    before = copy.deepcopy(store(p).read_data())
    start = len(p.calls())
    rejected = p.cli(
        "status",
        ok=False,
        extra={
            "PARENT_FAIL_GET_PATH": "repos/o/r/issues/1/sub_issues?per_page=100&page=1"
        },
    )
    assert "GET" in rejected.stderr
    assert store(p).read_data() == before
    assert all(
        c[c.index("--method") + 1] == "GET"
        for c in p.calls()[start:]
        if "--method" in c
    )
    assert_no_hierarchy_writes(p)


def test_completed_children_do_not_reopen_parent_as_work(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    p.child("independent")
    state = p.state()
    state["issues"]["10"].update(state="closed", state_reason="completed")
    p.save(state)
    planned = json.loads(p.cli("sync").stdout)
    assert planned["progress"] == {"10": "completed"}
    assert [t["child"] for t in planned["launch_plan"]["tasks"]] == [10]
    assert planned["launch_plan"]["launchable_children"] == []
    tree = p.worktree("self")
    denied = p.cli(
        "delegate",
        "--child",
        "1",
        "--agent",
        "codex",
        "--branch",
        "codex/self",
        "--worktree",
        str(tree),
        ok=False,
    )
    assert "direct child" in denied.stderr
    tasks = store(p).read_data()["parents"]["o/r#1"]["tasks"]
    assert "1" not in tasks and not any(t.get("delegated") for t in tasks.values())


def test_split_preserves_self_reservation_and_allows_only_stopping_it(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    tree = p.worktree("self")
    args = (
        "delegate",
        "--agent",
        "codex",
        "--branch",
        "codex/self",
        "--worktree",
        str(tree),
    )
    p.cli(*args, "--child", "1")
    saved = copy.deepcopy(store(p).read_data()["parents"]["o/r#1"]["tasks"]["1"])
    refused = p.child("independent", ok=False)
    assert "cannot split" in refused.stderr
    assert_no_hierarchy_writes(p)
    introduce_split(p)
    for child in (1, 10):
        denied = p.cli(*args, "--child", str(child), ok=False)
        assert "work targets changed" in denied.stderr
    denied = manual(
        p, "run-worker", "--child", "1", "--prompt-file", p.note(), ok=False
    )
    assert "work targets changed" in denied.stderr
    assert store(p).read_data()["parents"]["o/r#1"]["tasks"] == {"1": saved}
    stopped = json.loads(p.cli("stop", "--child", "1", "--spec", p.detail()).stdout)
    assert "target_conflict" in stopped
    current = store(p).read_data()["parents"]["o/r#1"]["tasks"]["1"]
    assert current["stopped"] and current["delegation_id"] == saved["delegation_id"]
    denied = p.cli("resume", "--child", "1", "--spec", p.detail(), ok=False)
    assert "work targets changed" in denied.stderr
    # An explicit stop plus termination, with no PR or accepted result, makes
    # the split safe. The old worker's reservation/history stays in the registry.
    p.cli(
        "record-worker-ended",
        "--child",
        "1",
        "--delegation-id",
        saved["delegation_id"],
        "--evidence-file",
        p.note("未起動予約の取消と実行不在を照合した。"),
    )
    status = json.loads(p.cli("status").stdout)
    assert status["launch_plan"]["launchable_children"] == [10]
    assert status["tasks"]["1"]["delegation_id"] == saved["delegation_id"]


def test_relation_change_during_admission_never_saves_self_claim(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    p.agreed()
    tree = p.worktree("self")
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(p.repo)
    snapshot = importlib.import_module("parent.snapshot")
    original = snapshot.validate_worktree

    def split_after_validation(path: Path, branch: str) -> None:
        original(path, branch)
        introduce_split(p)

    monkeypatch.setattr(snapshot, "validate_worktree", split_after_validation)
    registry = store(p)
    session_mod = importlib.import_module("parent.session")
    gh_mod = importlib.import_module("parent.github")
    with registry.locked():
        session = session_mod.ParentSession(
            gh_mod.GitHub("o/r", store=registry),
            registry,
            1,
            "claude:" + p.env["CLAUDE_CODE_SESSION_ID"],
        )
        with pytest.raises(ValueError, match="launch inputs changed"):
            session.delegate(1, "codex", "codex/self", tree, None, None)
    assert registry.read_data()["parents"]["o/r#1"]["tasks"] == {}


def test_removed_formal_children_do_not_silently_switch_to_self(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("worker")
    tree = p.worktree("child")
    p.cli(
        "delegate",
        "--child",
        "10",
        "--agent",
        "codex",
        "--branch",
        "codex/child",
        "--worktree",
        str(tree),
    )
    state = p.state()
    del state["parents"]["10"]
    p.save(state)
    denied = p.cli("status", ok=False)
    assert "work targets changed" in denied.stderr and "#10" in denied.stderr


def test_review_only_history_prevents_silent_target_switch(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    introduce_split(p)
    tree = p.worktree("reviewed")
    (tree / "change.txt").write_text("reviewed implementation\n")
    run_git(tree, "add", "change.txt")
    run_git(tree, "commit", "-m", "review target")
    state = p.state()
    state["prs"] = [p.pr(100, 10, run_git(tree, "rev-parse", "HEAD"), "codex/reviewed")]
    p.save(state)
    p.cli("review-prepare", "--prs", "100")
    row = store(p).read_data()["parents"]["o/r#1"]
    assert row["tasks"] == {} and row["reviews"]
    state = p.state()
    del state["parents"]["10"]
    p.save(state)
    denied = p.cli("status", ok=False)
    assert "work targets changed" in denied.stderr and "#10" in denied.stderr


@pytest.mark.parametrize(
    "gate_env", ["parent_self", "parent_self_worker"], indirect=True
)
def test_self_result_review_merge_completion_without_creating_children(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    body = p.state()["issues"]["1"]["body"]
    requirement_body = importlib.import_module("parent.execution").requirement_body
    saved = store(p).read_data()["parents"]["o/r#1"]
    task = saved["tasks"]["1"]
    assert task["verification"]["prs"] == [100] and task["verification"]["evidence"]
    assert set(g.review.request["requirements"]) == {"1"}
    result = json.loads((g.review.run_dir / "result.json").read_text())
    assert result["prs"][0]["child_issues"] == [1]
    assert g.review.calls()[0]["context"]["children"] == []
    assert result["request"]["prs"][0]["head_sha"] == p.state()["prs"][0]["head"]["sha"]
    if task.get("delegated"):
        assert task["worker_ended"]["delegation_id"] == task["delegation_id"]
        assert len(task["worker_runs"]) == 1
        turn = task["preflight"]["turns"][0]
        journal = json.loads(Path(turn["log"]["journal"]).read_text())
        assert journal["source"]["parent"] == journal["source"]["issue"] == 1
    else:
        assert "worker_ended" not in task and "worker_runs" not in task
    merged = g.cli("merge")
    assert merged.returncode == 0, merged.stderr + merged.stdout
    assert p.state()["issues"]["1"]["state_reason"] == "completed"
    completed = p.cli("workflow-report", "--spec", completion_spec(p))
    assert json.loads(completed.stdout)["state"] == "completed"
    p.cli("release", "--ended", "--note-file", p.note("全体受入と実行終了を照合した。"))
    assert store(p).read_data()["parents"]["o/r#1"]["status"] == "ended"
    assert requirement_body(body) == requirement_body(p.state()["issues"]["1"]["body"])
    assert_no_hierarchy_writes(p)
    assert len(g.merge_calls()) == len(g.review.calls()) == 1


@pytest.mark.parametrize("gate_env", ["parent_self"], indirect=True)
@pytest.mark.parametrize("pending", ["ci", "finding", "acceptance", "worker"])
def test_self_completion_waits_for_real_evidence(
    gate_env: GateEnv, pending: str
) -> None:
    g, p = gate_env, gate_env.review.p
    if pending == "ci":
        state = p.state()
        state["prs"][0]["statusCheckRollup"][0].update(status="QUEUED", conclusion="")
        p.save(state)
    elif pending == "finding":

        def unresolved(result: dict[str, Any]) -> None:
            result["verdict"] = "request_changes"
            result["prs"][0]["findings"] = [
                {
                    "id": "pr-100-1",
                    "severity": "major",
                    "path": "change.txt",
                    "line": 1,
                    "rule": "受入条件",
                    "detail": "受入条件が未達。回帰を確認する。",
                }
            ]

        g.edit_result(unresolved)
    response = g.cli("merge")
    if pending in {"ci", "finding"}:
        assert response.returncode == 1, response.stderr + response.stdout
        assert not g.merge_calls()
    else:
        assert response.returncode == 0, response.stderr + response.stdout
        with store(p).locked() as registry:
            task = registry.parents["o/r#1"]["tasks"]["1"]
            if pending == "acceptance":
                del task["verification"]
            else:
                task.update(delegated=True, delegation_id="pending-worker")
            registry.save()
    denied = p.cli("workflow-report", "--spec", completion_spec(p), ok=False)
    assert "pending" in denied.stderr or "acceptance evidence" in denied.stderr
    assert "development_result" not in store(p).read_data()["parents"]["o/r#1"]
    assert_no_hierarchy_writes(p)


@pytest.mark.parametrize("gate_env", ["parent_self_worker"], indirect=True)
def test_postmerge_worker_check_requires_parent_acceptance(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    response = g.cli("merge")
    assert response.returncode == 0, response.stderr + response.stdout
    # Merge closes the Issue before all requested worker checks have finished.
    # Stop/resume remains a control of that same worker, never a new assignment.
    p.cli("stop", "--child", "1", "--spec", p.detail())
    assert store(p).read_data()["parents"]["o/r#1"]["tasks"]["1"]["stopped"]
    p.cli("resume", "--child", "1", "--spec", p.detail())
    assert p.state()["issues"]["1"]["state"] == "closed"
    previous = store(p).read_data()["parents"]["o/r#1"]["tasks"]["1"]
    restart = p.root / "restart.md"
    restart.write_text("前のrunの終了・回収と同じworkerを確認した。")
    manual(
        p,
        "run-worker",
        "--child",
        "1",
        "--resume",
        "--prompt-file",
        p.note("merge後の動作を確認する。"),
        "--restart-evidence-file",
        str(restart),
    )
    current = store(p).read_data()["parents"]["o/r#1"]["tasks"]["1"]
    assert current["worker_ended"]["delegation_id"] == current["delegation_id"]
    assert current["verification"] == previous["verification"]
    denied = p.cli("workflow-report", "--spec", completion_spec(p), ok=False)
    assert "current result verification missing" in denied.stderr
    manual(
        p,
        "record-result",
        "--child",
        "1",
        "--evidence-file",
        p.note("merge後確認の実結果を親が受け入れた。"),
    )
    accepted = p.cli("workflow-report", "--spec", completion_spec(p))
    assert json.loads(accepted.stdout)["state"] == "completed"
    assert_no_hierarchy_writes(p)
