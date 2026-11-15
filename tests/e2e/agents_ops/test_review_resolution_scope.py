"""PR fix scope and compact receipts through real git, CLIs and fake GitHub."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import (
    ParentEnv,
    ReviewEnv,
    make_review_env,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import (
    single_review_env as single_review_env,
)
from tests.e2e.agents_ops.test_review_inspection import inspect
from tests.e2e.agents_ops.test_review_recovery import gate_for


def commit(repo: Path, message: str, *paths: str) -> str:
    run_git(repo, "add", *(paths or (".",)))
    run_git(repo, "commit", "-m", message)
    return run_git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("strategy", ["merge", "rebase"])
def test_scope_keeps_conflicts_shared_files_renames_and_absorbed_pr_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    review_policy_module(monkeypatch)
    workspace = importlib.import_module("review_workspace")
    snapshot = importlib.import_module("parent.snapshot").snapshot
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init", "-b", "main")
    run_git(repo, "config", "user.name", "Fixture")
    run_git(repo, "config", "user.email", "fixture@example.test")
    (repo / "shared.txt").write_text("first\n" + "middle\n" * 20 + "last\n")
    (repo / "conflict.txt").write_text("original\n")
    (repo / "rename-old.txt").write_text("source\n")
    base = commit(repo, "base")
    run_git(repo, "checkout", "-b", "codex/fix")
    (repo / "shared.txt").write_text("PR first\n" + "middle\n" * 20 + "last\n")
    (repo / "conflict.txt").write_text("PR original\n")
    (repo / "absorbed.txt").write_text("PR adopted by base\n")
    reviewed_head = commit(repo, "reviewed PR")
    monkeypatch.chdir(repo)

    def observe(head: str, base_sha: str) -> dict[str, Any]:
        return dict(
            snapshot(
                {
                    "number": 100,
                    "head": {"sha": head, "ref": "codex/fix"},
                    "base": {"sha": base_sha, "ref": "main"},
                }
            )[0]
        )

    original = observe(reviewed_head, base)
    run_git(repo, "checkout", "main")
    (repo / "base-only.txt").write_text("base only\n")
    (repo / "shared.txt").write_text("first\n" + "middle\n" * 20 + "BASE last\n")
    (repo / "conflict.txt").write_text("BASE original\n")
    (repo / "absorbed.txt").write_text("PR adopted by base\n")
    tip = commit(repo, "base advances")
    run_git(repo, "checkout", "codex/fix")
    command = (
        ["merge", "--no-ff", "-m", "follow base", tip]
        if strategy == "merge"
        else ["rebase", tip]
    )
    conflict = subprocess.run(
        ["git", *command], cwd=repo, capture_output=True, text=True, check=False
    )
    assert conflict.returncode == 1 and "conflict.txt" in conflict.stdout
    (repo / "conflict.txt").write_text("PR and BASE resolved\n")
    run_git(repo, "add", "conflict.txt")
    if strategy == "merge":
        run_git(repo, "commit", "--no-edit")
    else:
        run_git(repo, "-c", "core.editor=true", "rebase", "--continue")
    run_git(repo, "mv", "rename-old.txt", "rename-new.txt")
    head = commit(repo, "rename after review")
    current = observe(head, tip)
    scope = workspace.resolution_scope(original, current)
    assert scope.changed_paths == [
        "absorbed.txt",
        "conflict.txt",
        "rename-new.txt",
        "rename-old.txt",
        "shared.txt",
    ]
    assert scope.base_only_paths == ["base-only.txt"]
    assert "absorbed.txt" not in scope.head_changed_paths
    assert "shared.txt" in scope.head_changed_paths
    assert (
        repo / "shared.txt"
    ).read_text() == "PR first\n" + "middle\n" * 20 + "BASE last\n"
    assert (
        workspace.verify_rebased_snapshot(original, current) is False
    )  # rename requires resolve
    for invalid in (
        {**current, "merge_base_sha": base},
        {**current, "branch": "codex/other"},
    ):
        with pytest.raises(ValueError):
            workspace.resolution_scope(original, invalid)


def add_paths(repo: Path, directory: str, count: int = 2000) -> list[str]:
    paths = [f"{directory}/file-{i:04d}-with-a-long-name.txt" for i in range(count)]
    (repo / directory).mkdir()
    for path in paths:
        (repo / path).write_text(f"{path}\n")
    return paths


def proof(paths: list[str]) -> dict[str, Any]:
    return {
        "ci": "Python succeeded for the fixed head",
        "regression_tests": "Boundary regression passed",
        "resolution": "Parent checked the fix and regression",
        "reviewed_paths": paths,
    }


def resolve(r: ReviewEnv, head: str, paths: list[str]) -> dict[str, Any]:
    return dict(
        json.loads(
            r.cli(
                "resolve",
                "--run",
                r.request["review_run_id"],
                "--pr",
                "100",
                "--finding",
                "pr-100-1",
                "--fixed-head",
                head,
                "--evidence",
                r.p.spec(proof(paths)),
            ).stdout
        )
    )


@pytest.mark.parametrize("strategy", ["merge", "rebase"])
def test_resolve_large_base_import_is_compact_and_accepted_by_inspect_and_gate(
    single_review_env: ReviewEnv, monkeypatch: pytest.MonkeyPatch, strategy: str
) -> None:
    r, p = single_review_env, single_review_env.p
    output = json.loads(r.output.read_text())
    output["verdict"] = "request_changes"
    output["prs"][0]["findings"] = [
        {
            "severity": "major",
            "path": "change.txt",
            "line": 1,
            "rule": "boundary",
            "detail": "Fix with a regression test",
        }
    ]
    r.output.write_text(json.dumps(output))
    r.run()
    base_paths = add_paths(p.repo, "base-only")
    tip = commit(p.repo, "many base-only changes", "base-only")
    tree = p.repo / ".claude/worktrees/pr-0"
    (tree / "change.txt").write_text("fixed\n")
    (tree / "regression.txt").write_text("checked regression\n")
    commit(tree, "fix finding")
    if strategy == "merge":
        run_git(tree, "merge", "--no-ff", "-m", "follow base", tip)
    else:
        run_git(tree, "rebase", tip)
    head = run_git(tree, "rev-parse", "HEAD")
    state = p.state()
    state["prs"][0]["head"]["sha"] = head
    state["prs"][0]["base"]["sha"] = tip
    p.save(state)
    paths = ["change.txt", "regression.txt"]
    _, value = inspect(r, 100)
    comparison = value["prs"][0]["history"][0]["comparison"]
    assert comparison["changed_paths"] == paths
    assert comparison["base_only_paths"] == base_paths
    fixed = resolve(r, head, paths)
    assert fixed["coverage"] == {"covered": True, "reasons": [], "unresolved": []}
    saved = json.loads(next(r.run_dir.glob("resolution-*.json")).read_text())
    assert saved["paths"] == {
        "changed_paths": paths,
        "head_changed_paths": sorted(paths + base_paths),
        "base_only_paths": base_paths,
    }
    assert saved["record"] == fixed["resolution"] and saved["published"] is True
    assert saved["record"]["schema_version"] == 2
    assert saved["record"]["base_only_paths"]["count"] == 2000
    assert saved["record"]["changed_paths"]["count"] == 2
    assert (
        hashlib.sha256(
            next(r.run_dir.glob("resolution-*.diff")).read_bytes()
        ).hexdigest()
        == saved["record"]["fix_diff_sha256"]
    )
    for n in (1, 100):
        body = next(
            c["body"] for c in r.comments(n) if '"kind":"resolution"' in c["body"]
        )
        assert len(body) < 6000
        assert base_paths[0] not in body and paths[0] in body
    before = copy.deepcopy(p.state()["comments"])
    resolve(r, head, paths)
    assert p.state()["comments"] == before
    gate = gate_for(r, monkeypatch)
    gate.confirm()
    assert (
        gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"]
        == "merge"
    )
    _, value = inspect(r, 100)
    assert value["prs"][0]["history"][0]["followups"][0]["errors"] == []
    assert value["prs"][0]["history"][0]["coverage_check"]["covered"] is True
    assert len(r.calls()) == 1 and gate.merge_calls() == []
    if strategy == "merge":
        # Previously published receipts keep the full-head-diff definition.
        def replace_receipt(old: dict[str, Any], new: dict[str, Any]) -> None:
            key = gate.policy.digest(new)
            (r.run_dir / f"resolution-{key}.json").write_text(
                json.dumps({"record": new, "published": True})
            )
            (r.run_dir / f"resolution-{key}.diff").write_bytes(
                next(r.run_dir.glob("resolution-*.diff")).read_bytes()
            )
            state = p.state()
            for comments in state["comments"].values():
                for comment in comments:
                    if gate.policy.records([comment], "operator") == [old]:
                        comment["body"] = gate.policy.marker(new)
            p.save(state)
            gate.attest_fixture_comments()

        legacy = {
            key: value
            for key, value in saved["record"].items()
            if key not in {"head_changed_paths", "base_only_paths"}
        }
        legacy.update(schema_version=1, changed_paths=sorted(paths + base_paths))
        legacy["evidence"] = {
            **legacy["evidence"],
            "reviewed_paths": legacy["changed_paths"],
        }
        replace_receipt(saved["record"], legacy)
        assert (
            gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"]
            == "merge"
        )
        narrowed = {
            **legacy,
            "changed_paths": paths,
            "evidence": {**legacy["evidence"], "reviewed_paths": paths},
        }
        replace_receipt(legacy, narrowed)
        assert (
            gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"]
            == "blocked"
        )
        assert gate.merge_calls() == []


@pytest.mark.parametrize("kind", ["resolution", "evidence"])
def test_thousands_of_reviewed_paths_stay_local_and_tampering_blocks_gate(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    p = parent_env
    paths: list[str] = []

    def prepare_large_pr(env: ParentEnv) -> None:
        tree = env.repo / ".claude/worktrees/pr-0"
        paths.extend(add_paths(tree, "pr-paths") + ["change.txt"])
        head = commit(tree, "large original PR")
        state = env.state()
        state["prs"][0]["head"]["sha"] = head
        env.save(state)

    r = make_review_env(
        p, pr_count=1, before_review=prepare_large_pr if kind == "evidence" else None
    )
    tree = p.repo / ".claude/worktrees/pr-0"
    head = run_git(tree, "rev-parse", "HEAD")
    output = json.loads(r.output.read_text())
    if kind == "evidence":
        output["prs"][0]["evidence_waits"] = [
            {
                "question": "regression result",
                "required_evidence": "Tests passed for this head",
            }
        ]
    r.output.write_text(json.dumps(output))
    r.run()
    if kind == "resolution":
        paths = add_paths(tree, "pr-paths")
        head = commit(tree, "many fixes")
        state = p.state()
        state["prs"][0]["head"]["sha"] = head
        p.save(state)
    paths.sort()
    args = (
        ("complete-evidence", "--wait", "1")
        if kind == "evidence"
        else ("resolve", "--finding", "changes", "--fixed-head", head)
    )
    args += ("--run", r.request["review_run_id"], "--pr", "100")
    oversized = proof(paths)
    oversized["resolution"] = "x" * 60000
    before = copy.deepcopy(p.state()["comments"])
    refused = r.cli(*args, "--evidence", p.spec(oversized), ok=False)
    assert "GitHub comment limit" in refused.stderr
    assert p.state()["comments"] == before
    r.cli(*args, "--evidence", p.spec(proof(paths)))
    saved_path = next(
        path
        for path in r.run_dir.glob(f"{kind}-*.json")
        if json.loads(path.read_text())["published"]
    )
    saved = json.loads(saved_path.read_text())
    manifest = saved["record"]["evidence"]["reviewed_paths"]
    assert manifest == {
        "count": len(paths),
        "sha256": hashlib.sha256(
            json.dumps(paths, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    key = "reviewed_paths" if kind == "evidence" else "changed_paths"
    assert saved["paths"][key] == paths
    for n in (1, 100):
        body = next(c["body"] for c in r.comments(n) if f'"kind":"{kind}"' in c["body"])
        assert len(body) < 8000 and paths[0] in body and paths[-1] not in body
        assert f"表示 20/{len(paths)}件" in body
    gate = gate_for(r, monkeypatch)
    gate.confirm()
    assert (
        gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"]
        == "merge"
    )
    broken = copy.deepcopy(saved)
    broken["paths"][key].pop()
    saved_path.write_text(json.dumps(broken))
    assert (
        gate.results(gate.cli("merge", AGENT_AUTO_MERGE="0"))[100]["decision"]
        == "blocked"
    )
    saved_path.write_text(json.dumps(saved))
    assert len(r.calls()) == 1 and gate.merge_calls() == []
