"""Parallel reservations, dependency PRs and stacked branches in disposable git."""

from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def plan(p: ParentEnv, command: str = "status") -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(p.cli(command).stdout)["launch_plan"])


def reserve(
    p: ParentEnv, child: int, name: str, path: Path, *extra: str
) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            p.cli(
                "delegate",
                "--child",
                str(child),
                "--agent",
                "codex",
                "--branch",
                "codex/" + name,
                "--worktree",
                str(path),
                *extra,
            ).stdout
        ),
    )


def test_default_launches_all_independent_children_and_explains_exceptions(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("upstream")
    p.child("dependent", deps=[10])
    for name in ("decision", "independent-a", "independent-b", "independent-c"):
        p.child(name)
    p.cli("stop", "--child", "12", "--spec", p.detail())
    for command in ("status", "reconcile", "sync"):
        selected = plan(p, command)
        assert selected["launchable_children"] == [10, 13, 14, 15]
        assert selected["policy"]["max_parallel_children"] is None
        reasons = {
            t["child"]: {r["code"] for r in t["reasons"]} for t in selected["tasks"]
        }
        assert reasons[11] == {"dependency_pr_unavailable"}
        assert reasons[12] == {"stopped"}
    for n in (10, 13, 14, 15):
        name = f"worker-{n}"
        outcome = reserve(p, n, name, p.worktree(name))
        assert outcome["action"] == "launch"
        assert n not in outcome["launch_plan"]["launchable_children"]
    selected = plan(p)
    assert selected["active_reservations"] == [10, 13, 14, 15]
    assert selected["launchable_children"] == []
    body = p.state()["issues"]["1"]["body"]
    assert "子の起動予約: #10, #13, #14, #15" in body
    assert "#11: 継続 — 依存先 #10" in body
    assert "#12: 停止" in body and "仕様の確認待ち（詳細・再開条件: #12）" in body
    assert "再現ログを確認済み" not in body
    assert "判断の合意を記録" not in body
    assert str(p.root) not in body
    assert p.env["CLAUDE_CODE_SESSION_ID"] not in body


def test_configured_limit_serializes_reservations_and_requires_worker_end_evidence(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    for name in ("a", "b", "c", "d"):
        p.child(name)
    p.cli(
        "configure-delegation",
        "--max-parallel-children",
        "2",
        "--reason-file",
        p.note("RAM使用量の実測に基づき2件に制限する。"),
    )
    paths = {n: p.worktree(f"worker-{n}") for n in range(10, 14)}

    def invoke(n: int) -> Any:
        return p.cli(
            "delegate",
            "--child",
            str(n),
            "--agent",
            "codex",
            "--branch",
            f"codex/worker-{n}",
            "--worktree",
            str(paths[n]),
            ok=None,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(10, 14)))
    assert [r.returncode for r in results] == [0, 0, 2, 2]
    selected = plan(p)
    assert selected["active_reservations"] == [10, 11]
    assert selected["launchable_children"] == []
    assert selected["tasks"][2]["reasons"][0]["code"] == "parallel_limit"
    p.cli("stop", "--child", "10", "--spec", p.detail())
    state = p.state()
    state["prs"] = [p.pr(100, 11, branch="codex/worker-11")]
    p.save(state)
    assert plan(p)["active_reservations"] == [10, 11]  # Stop/PR is not worker exit.
    token = json.loads(results[0].stdout)["assignment"]["delegation_id"]
    p.cli(
        "record-worker-ended",
        "--child",
        "10",
        "--delegation-id",
        token,
        "--evidence-file",
        p.note("fake runnerの終了と子孫なしを確認。"),
    )
    assert plan(p)["launchable_children"] == [12]
    assert plan(p)["tasks"][0]["action"] == "do_not_launch"
    p.cli("resume", "--child", "10", "--spec", p.detail())
    restarted = reserve(
        p,
        10,
        "worker-10",
        paths[10],
        "--restart-evidence-file",
        p.note("旧workerの終了を確認済み。"),
    )
    assert restarted["assignment"]["delegation_id"] != token
    p.cli(
        "record-worker-ended",
        "--child",
        "10",
        "--delegation-id",
        token,
        "--evidence-file",
        p.note(),
        ok=False,
    )
    assert plan(p)["active_reservations"] == [10, 11]
    p.cli(
        "configure-delegation",
        "--max-parallel-children",
        "all",
        "--reason-file",
        p.note("資源競合をテストlockとqueueで調停する。"),
    )
    assert plan(p)["launchable_children"] == [12, 13]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "auto"])
def test_invalid_limit_is_not_silently_replaced(
    parent_env: ParentEnv, value: str
) -> None:
    p = parent_env
    p.agreed()
    p.cli(
        "configure-delegation",
        "--max-parallel-children",
        value,
        "--reason-file",
        p.note(),
        ok=False,
    )
    assert plan(p)["policy"]["max_parallel_children"] is None


def test_stacked_launch_checks_head_and_rebases_after_squash_merge(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("upstream")
    p.child("downstream", deps=[10])
    upstream = p.worktree("upstream")
    (upstream / "upstream.txt").write_text("upstream implementation\n")
    run_git(upstream, "add", "upstream.txt")
    run_git(upstream, "commit", "-m", "upstream implementation")
    pinned = run_git(upstream, "rev-parse", "HEAD")
    downstream = p.worktree("downstream")
    state = p.state()
    state["prs"] = [p.pr(100, 10, pinned, "codex/upstream")]
    p.save(state)
    selected = plan(p)
    assert selected["launchable_children"] == [11]
    assert selected["tasks"][1]["stacked_on"] == [
        {"child": 10, "pr": 100, "branch": "codex/upstream", "head_sha": pinned}
    ]
    failed = p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/downstream",
        "--worktree",
        str(downstream),
        ok=False,
    )
    assert "merge-base" in failed.stderr
    assert plan(p)["active_reservations"] == []
    run_git(downstream, "reset", "--hard", pinned)
    delegated = reserve(p, 11, "downstream", downstream)
    assert delegated["assignment"]["stacked_on"][0]["head_sha"] == pinned
    (downstream / "downstream.txt").write_text("child implementation\n")
    run_git(downstream, "add", "downstream.txt")
    run_git(downstream, "commit", "-m", "child implementation")
    run_git(p.repo, "merge", "--squash", "codex/upstream")
    run_git(p.repo, "commit", "-m", "squash upstream")
    state = p.state()
    state["prs"][0].update(state="closed", merged_at="2026-10-22T00:00:00Z")
    state["issues"]["10"].update(state="closed", state_reason="completed")
    p.save(state)
    run_git(downstream, "rebase", "--onto", "main", pinned, "codex/downstream")
    assert run_git(downstream, "rev-list", "--count", "main..HEAD") == "1"
    assert (
        run_git(downstream, "diff", "--name-only", "main", "HEAD") == "downstream.txt"
    )
    assert (downstream / "upstream.txt").read_text() == "upstream implementation\n"
    run_git(downstream, "merge-base", "--is-ancestor", "main", "HEAD")
    assert plan(p)["tasks"][1]["stacked_on"] == []


def test_transitive_dependency_and_stopped_pr_do_not_hold_independent_children(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    p.child("c", deps=[11])
    p.child("independent")
    state = p.state()
    state["prs"] = [p.pr(101, 11)]
    p.save(state)
    assert plan(p)["launchable_children"] == [10, 13]
    state["prs"].append(p.pr(100, 10))
    p.save(state)
    assert plan(p)["launchable_children"] == [12, 13]
    p.cli("stop", "--child", "10", "--spec", p.detail())
    selected = plan(p, "sync")
    assert selected["launchable_children"] == [13]
    assert selected["tasks"][2]["state"] == "held"
    p.cli("resume", "--child", "10", "--spec", p.detail())
    assert plan(p)["launchable_children"] == [12, 13]


@pytest.mark.parametrize("variant", ["closed", "ambiguous", "fork"])
def test_unusable_dependency_pr_leaves_other_children_ready(
    parent_env: ParentEnv, variant: str
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    p.child("independent")
    state = p.state()
    pr = p.pr(100, 10)
    if variant == "closed":
        pr["state"] = "closed"
    elif variant == "fork":
        pr["head"]["repo"]["full_name"] = "other/repository"
    state["prs"] = [pr, p.pr(101, 10)] if variant == "ambiguous" else [pr]
    p.save(state)
    selected = plan(p)
    assert selected["launchable_children"] == [12]
    assert selected["tasks"][1]["reasons"]


def test_phase_change_and_unknown_policy_never_allow_launch(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.cli("checking")
    assert plan(p)["launchable_children"] == []
    assert plan(p)["tasks"][0]["reasons"][0]["code"] == "execution_basis_required"
    path = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    registry = json.loads(path.read_text())
    for invalid in (
        None,
        {"schema_version": 99},
        {"schema_version": 1, "max_parallel_children": True, "reason": "invalid"},
    ):
        registry["parents"]["o/r#1"]["launch_policy"] = invalid
        path.write_text(json.dumps(registry))
        assert p.cli("status", ok=False).returncode == 2


def test_dependency_override_cannot_bypass_missing_pr(parent_env: ParentEnv) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b")
    path = p.worktree("b")
    failed = p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/b",
        "--worktree",
        str(path),
        "--dependencies",
        "10",
        ok=False,
    )
    assert "依存先 #10" in failed.stderr
    assert plan(p)["active_reservations"] == []


def test_dependency_head_change_before_reservation_is_rejected(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    p.agreed()
    p.child("a")
    p.child("b", deps=[10])
    original = run_git(p.repo, "rev-parse", "HEAD")
    worktree = p.worktree("b")
    run_git(p.repo, "commit", "--allow-empty", "-m", "updated dependency")
    updated = run_git(p.repo, "rev-parse", "HEAD")
    state = p.state()
    state["prs"] = [p.pr(100, 10, original)]
    p.save(state)
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper = p.root / "bin/git"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        + r"""
import json, os, sys
if sys.argv[1:3] == ["merge-base", "--is-ancestor"]:
    path = os.environ["PARENT_GH_STATE"]
    with open(path) as stream:
        state = json.load(stream)
    state["prs"][0]["head"]["sha"] = os.environ["FAKE_UPDATED_HEAD"]
    with open(path, "w") as stream:
        json.dump(state, stream)
os.execv(os.environ["FAKE_REAL_GIT"], ["git", *sys.argv[1:]])
"""
    )
    wrapper.chmod(0o755)
    result = p.cli(
        "delegate",
        "--child",
        "11",
        "--agent",
        "codex",
        "--branch",
        "codex/b",
        "--worktree",
        str(worktree),
        extra={"FAKE_REAL_GIT": real_git, "FAKE_UPDATED_HEAD": updated},
        ok=False,
    )
    assert "launch inputs changed" in result.stderr
    assert plan(p)["active_reservations"] == []
