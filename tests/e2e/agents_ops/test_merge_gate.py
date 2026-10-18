"""Explicit merge gate against actual parent-run records, fake gh and real git."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import (
    OPS,
    ReviewEnv,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import (
    parent_env as parent_env,
)
from tests.e2e.agents_ops.test_parent_session import (
    review_env as review_env,
)

SCRIPT = OPS / "review/merge_gate.py"


@dataclass
class GateEnv:
    review: ReviewEnv
    policy: Any

    def cli(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        p = self.review.p
        before = len(self.review.calls())
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--repo", "o/r", "--parent", "1", *args],
            cwd=p.repo,
            env={**p.env, "AGENT_AUTO_MERGE": "1", **env},
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert len(self.review.calls()) == before
        assert not (p.root / "unexpected-agent.txt").exists()
        return result

    def merge_calls(self) -> list[list[str]]:
        return [c for c in self.review.p.calls() if c[:2] == ["pr", "merge"]]

    def results(self, result: subprocess.CompletedProcess[str]) -> dict[int, Any]:
        assert result.stdout, result.stderr
        data = json.loads(result.stdout)
        assert data["agent_invocations"] == 0
        assert data["review_run_id"] == self.review.request["review_run_id"]
        return {r["pr"]: r for r in data["results"]}

    def confirm(self) -> None:
        p = self.review.p
        for state in ("result_received", "fixes_verified"):
            p.cli(
                "review-state",
                "--run-id",
                self.review.request["review_run_id"],
                "--state",
                state,
                "--evidence-file",
                p.note("結果・差分・CI・回帰テストを親が確認した。"),
            )

    def edit_result(self, update: Any) -> None:
        """Edit both canonical records and all projections to test genuine bad results."""
        r, p = self.review, self.review.p
        result = json.loads((r.run_dir / "result.json").read_text())
        update(result)
        self.policy.validate_result(result)
        (r.run_dir / "result.json").write_text(json.dumps(result))
        state = p.state()
        for rows in state["comments"].values():
            for comment in rows:
                records = self.policy.records([comment], "operator")
                if not records:
                    continue
                record = records[0]
                if record["kind"] == "result":
                    record = result
                elif record["kind"] == "pr":
                    n = record["pr"]
                    record["result_sha256"] = self.policy.digest(result)
                    record["coverage"] = next(
                        p for p in result["prs"] if p["number"] == n
                    )["coverage"]
                    record["findings"] = self.policy.pr_findings(result, n)
                comment["body"] = self.policy.marker(record)
        p.save(state)


@pytest.fixture
def gate_env(
    review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> GateEnv:
    p = review_env.p
    policy = review_policy_module(monkeypatch)
    if getattr(request, "param", None) == "shared":
        worktree = p.repo / ".claude/worktrees/pr-0"
        (worktree / ".agents").mkdir()
        (worktree / ".agents/shared.txt").write_text("shared foundation\n")
        run_git(worktree, "add", ".agents/shared.txt")
        run_git(worktree, "commit", "-m", "shared changes")
        state = p.state()
        state["prs"][0]["head"]["sha"] = run_git(worktree, "rev-parse", "HEAD")
        p.save(state)
        manifest = json.loads(
            p.cli(
                "review-prepare",
                "--prs",
                "100",
                "101",
                "102",
                "--reason-file",
                p.note("共有基盤を含む対象を親が確定し直す。"),
            ).stdout
        )
        review_env.request = json.loads(
            p.cli(
                "review-request",
                "--run-id",
                manifest["review_run_id"],
                "--instruction-file",
                p.note("この親の3PRを1回で確認する。"),
            ).stdout
        )
        review_env.request_path.write_text(json.dumps(review_env.request))
    state = p.state()
    for pr in state["prs"]:
        pr.update(
            labels=[],
            mergeable=True,
            statusCheckRollup=[
                {
                    "__typename": "CheckRun",
                    "name": "Python",
                    "workflowName": "CI",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
        )
    p.save(state)
    for name in ("codex", "claude"):
        (p.root / "bin" / name).write_text(
            f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(p.root / 'unexpected-agent.txt')!r}).write_text('called')\nraise SystemExit(97)\n"
        )
    review_env.run()
    gate = GateEnv(review_env, policy)
    gate.confirm()
    return gate


def test_three_prs_use_one_run_and_make_individual_decisions(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    g.edit_result(
        lambda result: result.update(
            verdict="request_changes",
            prs=[
                {
                    **pr,
                    "findings": [
                        {
                            "id": "pr-101-1",
                            "severity": "major",
                            "path": "change.txt",
                            "line": 1,
                            "rule": "境界",
                            "detail": "修正が必要",
                        }
                    ],
                }
                if pr["number"] == 101
                else pr
                for pr in result["prs"]
            ],
        )
    )
    state = p.state()
    state["prs"][2]["statusCheckRollup"][0]["status"] = "IN_PROGRESS"
    p.save(state)
    rows = g.results(g.cli("merge", AGENT_AUTO_MERGE="0"))
    assert rows[100]["decision"] == "merge"
    assert "unresolved_findings" in rows[101]["reasons"]
    assert "ci_pending" in rows[102]["reasons"]
    assert len(g.review.calls()) == 1
    assert g.merge_calls() == []


@pytest.mark.parametrize(
    "case,reason",
    [
        ("no_review", "parent_review_missing_or_interrupted"),
        ("not_received", "parent_review_not_confirmed"),
        ("fixing", "parent_review_not_confirmed"),
        ("interrupted", "review_not_published_or_authorized"),
        ("outside", "pr_outside_review_targets"),
        ("not_reviewed", "run_failed_or_incomplete"),
        ("major", "unresolved_findings"),
        ("blocker", "unresolved_findings"),
        ("head", "pr_100_unverified_head"),
        ("base", "pr_100_base_sha_changed"),
        ("base_ref", "pr_100_base_ref_changed"),
        ("draft", "draft"),
        ("fork", "fork"),
        ("closed", "pr_not_open"),
        ("author", "author_unknown"),
        ("conflict", "conflict"),
        ("unknown", "mergeable_unknown"),
        ("ci_before", "ci_missing_required"),
        ("ci_failed", "ci_failed"),
        ("ci_pending", "ci_pending"),
        ("ci_missing", "ci_missing_required"),
        ("human", "human_judgment_pending"),
        ("parent_human", "human_judgment_pending"),
        ("stopped", "child_10_stopped"),
        ("parent_missing", "parent_result_missing_or_inconsistent"),
        ("projection_missing", "pr_100_projection_missing_or_inconsistent"),
        ("projection_digest", "pr_100_projection_missing_or_inconsistent"),
        ("failed_record", "parent_result_missing_or_inconsistent"),
        ("foreign_result", "parent_result_missing_or_inconsistent"),
        ("quoted_result", "parent_result_missing_or_inconsistent"),
    ],
)
def test_rejections_never_merge_or_launch_agent(
    gate_env: GateEnv, case: str, reason: str
) -> None:
    g, r, p = gate_env, gate_env.review, gate_env.review.p
    state = p.state()
    first = state["prs"][0]
    if case == "no_review":
        (r.run_dir / "result.json").unlink()
    elif case in {"not_received", "fixing"}:
        remove = {"fixes_verified"} | (
            {"result_received"} if case == "not_received" else set()
        )
        state["comments"]["1"] = [
            c
            for c in state["comments"]["1"]
            if not any(f'"state": "{s}"' in c["body"] for s in remove)
        ]
    elif case == "interrupted":
        path = r.run_dir / "execution.json"
        execution = json.loads(path.read_text())
        execution["status"] = "running"
        path.write_text(json.dumps(execution))
    elif case == "outside":
        extra = {
            **p.pr(103, 10),
            "labels": [],
            "mergeable": True,
            "statusCheckRollup": first["statusCheckRollup"],
        }
        state["prs"].append(extra)
    elif case in {"not_reviewed", "major", "blocker"}:

        def mutate(result: dict[str, Any]) -> None:
            if case == "not_reviewed":
                result.update(status="failed", verdict="incomplete", error="未確認")
                result["prs"][0]["coverage"] = "not_reviewed"
            else:
                result["verdict"] = "request_changes"
                result["prs"][0]["findings"] = [
                    {
                        "id": "pr-100-1",
                        "severity": case,
                        "path": "change.txt",
                        "line": 1,
                        "rule": "正しさ",
                        "detail": "修正が必要",
                    }
                ]

        g.edit_result(mutate)
        state = p.state()
    elif case in {"head", "base"}:
        worktree = p.repo / ".claude/worktrees/pr-0"
        (worktree / "change.txt").write_text("unexpected\n")
        run_git(worktree, "commit", "-am", "unexpected update")
        first["head" if case == "head" else "base"]["sha"] = run_git(
            worktree, "rev-parse", "HEAD"
        )
    elif case == "base_ref":
        first["base"]["ref"] = "other"
    elif case == "draft":
        first["draft"] = True
    elif case == "fork":
        first["head"]["repo"]["full_name"] = "fork/r"
    elif case == "closed":
        first["state"] = "closed"
    elif case == "author":
        first["head"]["ref"] = "unknown/source"
    elif case in {"conflict", "unknown"}:
        first["mergeable"] = False if case == "conflict" else None
    elif case == "ci_before":
        first["statusCheckRollup"] = []
    elif case == "ci_missing":
        first["statusCheckRollup"][0]["name"] = "Labeler"
    elif case == "ci_failed":
        first["statusCheckRollup"][0]["conclusion"] = "FAILURE"
    elif case == "ci_pending":
        first["statusCheckRollup"][0]["status"] = "QUEUED"
    elif case in {"human", "parent_human"}:
        (first if case == "human" else state["issues"]["1"])["labels"] = [
            {"name": "needs-human"}
        ] + (state["issues"]["1"]["labels"] if case == "parent_human" else [])
    elif case == "stopped":
        p.cli("stop", "--child", "10", "--spec", p.detail())
        state = p.state()
    elif case in {"parent_missing", "foreign_result", "quoted_result", "failed_record"}:
        comment = next(
            c
            for c in state["comments"]["1"]
            if g.policy.records([c], "operator")
            and g.policy.records([c], "operator")[0]["kind"] == "result"
        )
        if case == "foreign_result":
            comment["user"]["login"] = "attacker"
        elif case == "quoted_result":
            comment["body"] = "引用\n" + comment["body"]
        elif case == "parent_missing":
            comment["body"] = "<!-- agent-cross-review:v4 approve -->"
        else:
            failed = json.loads((r.run_dir / "result.json").read_text())
            failed.update(status="failed", verdict="incomplete", error="公開失敗")
            failed["prs"][0]["coverage"] = "not_reviewed"
            state["comments"]["1"].append(
                {
                    "id": 9999,
                    "user": {"login": "operator"},
                    "body": g.policy.marker(failed),
                }
            )
    elif case == "projection_missing":
        state["comments"]["100"] = []
    elif case == "projection_digest":
        record = g.policy.records(state["comments"]["100"], "operator")[0]
        record["result_sha256"] = "f" * 64
        state["comments"]["100"][0]["body"] = g.policy.marker(record)
    p.save(state)
    # Isolate the rejection: prevent the other independent PRs from merging.
    state = p.state()
    for pr in state["prs"]:
        if pr["number"] not in {100, 103}:
            pr["statusCheckRollup"] = []
    p.save(state)
    result = g.cli("merge")
    assert result.returncode == 1, result.stderr
    rows = g.results(result)
    assert reason in rows[103 if case == "outside" else 100]["reasons"]
    assert g.merge_calls() == []


@pytest.mark.parametrize("value", ["", "true", "2", " 1"])
def test_auto_merge_flag_rejects_invalid_values_before_io(
    gate_env: GateEnv, value: str
) -> None:
    g = gate_env
    before = len(g.review.p.calls())
    result = g.cli("merge", AGENT_AUTO_MERGE=value)
    assert result.returncode == 2
    assert "AGENT_AUTO_MERGE" in result.stderr
    assert len(g.review.p.calls()) == before


def test_unset_auto_merge_is_dry_run_without_any_writes(gate_env: GateEnv) -> None:
    g = gate_env
    policy = importlib.import_module("merge_gate_policy")
    assert not policy.auto_merge(None)
    before = len(g.review.p.calls())
    result = g.cli("merge", AGENT_AUTO_MERGE="0")
    assert result.returncode == 0, result.stderr
    assert all(r["action"] == "dry_run" for r in g.results(result).values())
    assert not any(
        "POST" in c or "PATCH" in c or c[:2] == ["pr", "merge"]
        for c in g.review.p.calls()[before:]
    )


def test_dependency_order_merges_a_before_b_while_c_waits(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    policy = importlib.import_module("parent.policy")
    state = p.state()
    # A has the larger PR number: numeric sorting alone would merge B first.
    state["issues"]["10"]["body"] = policy.replace_section(
        state["issues"]["10"]["body"], "依存関係", "#11 のmerge後。"
    )
    state["prs"][2]["statusCheckRollup"] = []
    p.save(state)
    rows = g.results(g.cli("merge"))
    assert [c[2] for c in g.merge_calls()] == ["101", "100"]
    for call in g.merge_calls():
        n = int(call[2])
        assert call[call.index("--match-head-commit") + 1] == next(
            pr["head_sha"] for pr in g.review.request["prs"] if pr["number"] == n
        )
    assert rows[100]["action"] == rows[101]["action"] == "merged"
    assert "ci_missing_required" in rows[102]["reasons"]
    assert p.state()["issues"]["1"]["state"] == "open"
    assert "#10: 完了" in p.state()["issues"]["1"]["body"]
    assert "#11: 完了" in p.state()["issues"]["1"]["body"]
    # A repeated explicit command confirms existing merges without issuing them again.
    rows = g.results(g.cli("merge"))
    assert rows[100]["action"] == rows[101]["action"] == "already_merged"
    assert len(g.merge_calls()) == 2


@pytest.mark.parametrize("change", ["base", "ci"])
def test_remaining_prs_are_recollected_after_each_merge(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    state["prs"][2]["statusCheckRollup"] = []
    p.save(state)
    env = (
        {"PARENT_MERGE_ADVANCE_BASE": "1"}
        if change == "base"
        else {"PARENT_MERGE_CI_PENDING": "101"}
    )
    rows = g.results(g.cli("merge", **env))
    assert [c[2] for c in g.merge_calls()] == ["100"]
    assert (
        "pr_101_base_sha_changed" if change == "base" else "ci_missing_required"
    ) in rows[101]["reasons"]


@pytest.mark.parametrize("failure", ["fail", "head", "response", "close"])
def test_failed_or_lost_merge_is_never_repeated(
    gate_env: GateEnv, failure: str
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    env = {
        "fail": {"PARENT_MERGE_FAIL": "1"},
        "head": {"PARENT_MERGE_HEAD_RACE": state["prs"][1]["head"]["sha"]},
        "response": {"PARENT_MERGE_RESPONSE_LOST": "1"},
        "close": {"PARENT_MERGE_SKIP_CLOSE": "1"},
    }[failure]
    result = g.cli("merge", **env)
    rows = g.results(result)
    assert rows[100]["action"] == (
        "merged" if failure == "response" else "merge_unconfirmed"
    )
    retry = g.cli("merge")
    assert retry.returncode in {1, 2}
    assert len(g.merge_calls()) == 1
    if failure == "fail":
        assert "merge_already_attempted" in g.results(retry)[100]["reasons"]
        dry = g.cli("merge", AGENT_AUTO_MERGE="0")
        assert "merge_already_attempted" in g.results(dry)[100]["reasons"]
    if failure == "close":
        assert "child_10_close_not_confirmed" in g.results(retry)[100]["reasons"]
        assert "#10: 完了" not in p.state()["issues"]["1"]["body"]


@pytest.mark.parametrize("gate_env", ["shared"], indirect=True)
@pytest.mark.parametrize(
    "permission",
    [
        "absent",
        "allow",
        "reallow",
        "deny",
        "foreign",
        "quoted",
        "new_agreement",
        "old_generation",
        "old_run",
    ],
)
def test_shared_foundation_needs_explicit_current_parent_permission(
    gate_env: GateEnv, permission: str
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    if permission != "absent":
        # Human consent already exists: migrate it to the machine-readable marker.
        result = g.cli(
            "shared-permission",
            "--allow-shared-foundation",
            "--evidence-file",
            p.note(
                "人間との合意: この親の共有基盤変更を含めて親がmergeしてよい（合意コメントへのリンク）。"
            ),
        )
        assert result.returncode == 0, result.stderr
        if permission == "deny":
            assert (
                g.cli(
                    "shared-permission",
                    "--deny-shared-foundation",
                    "--evidence-file",
                    p.note("人間が共有基盤の許可を取り消した。"),
                ).returncode
                == 0
            )
        elif permission == "reallow":
            evidence = "人間との合意: この親の共有基盤変更を含めて親がmergeしてよい（合意コメントへのリンク）。"
            before = len(p.state()["comments"]["1"])
            g.cli(
                "shared-permission",
                "--allow-shared-foundation",
                "--evidence-file",
                p.note(evidence),
            )
            assert len(p.state()["comments"]["1"]) == before
            g.cli(
                "shared-permission",
                "--deny-shared-foundation",
                "--evidence-file",
                p.note("許可取り消し"),
            )
            result = g.cli(
                "shared-permission",
                "--allow-shared-foundation",
                "--evidence-file",
                p.note(evidence),
            )
            assert json.loads(result.stdout)["revision"] == 3
        elif permission in {"foreign", "quoted", "old_generation", "old_run"}:
            state = p.state()
            comment = state["comments"]["1"][-1]
            if permission == "foreign":
                comment["user"]["login"] = "attacker"
            elif permission == "quoted":
                comment["body"] = "引用\n" + comment["body"]
            else:
                policy = importlib.import_module("parent.policy")
                record = policy.events([comment], "operator", 1)[0]
                record.pop("body")
                record[
                    "generation" if permission == "old_generation" else "review_run_id"
                ] = 999 if permission == "old_generation" else "parent-1-" + "f" * 32
                comment["body"] = policy.marker(record)
            p.save(state)
        elif permission == "new_agreement":
            # Same purpose/scope/confirmation yields the same agreement digest.
            p.agreed()
    rows = g.results(g.cli("merge"))
    if permission in {"allow", "reallow"}:
        assert [c[2] for c in g.merge_calls()] == ["100"]
        exemption = rows[100]["shared_exemption"]
        assert exemption["allow_shared_foundation"] is True
        assert "合意" in exemption["evidence"]
        assert "#issuecomment-" in exemption["comment_url"]
    else:
        assert g.merge_calls() == []
        assert rows[100]["shared_hits"]
        assert rows[100]["shared_exemption"] is None
        assert (
            "parent_review_not_confirmed"
            if permission == "new_agreement"
            else "touches_shared"
        ) in rows[100]["reasons"]


@pytest.mark.parametrize("extra_path", [False, True])
@pytest.mark.parametrize("resolution", ["valid", "stale", "unpublished", "wrong_diff"])
def test_fix_push_requires_current_published_resolution_and_actual_diff(
    gate_env: GateEnv, extra_path: bool, resolution: str
) -> None:
    g, r, p = gate_env, gate_env.review, gate_env.review.p
    g.edit_result(
        lambda result: result.update(
            verdict="request_changes",
            prs=[
                {
                    **pr,
                    "findings": [
                        {
                            "id": "pr-100-1",
                            "severity": "major",
                            "path": "change.txt",
                            "line": 1,
                            "rule": "境界",
                            "detail": "修正が必要",
                        }
                    ],
                }
                if pr["number"] == 100
                else pr
                for pr in result["prs"]
            ],
        )
    )
    worktree = p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("fix checked\n")
    if extra_path:
        (worktree / "outside.txt").write_text("unreviewed expansion\n")
    run_git(worktree, "add", ".")
    run_git(worktree, "commit", "-m", "fix finding")
    state = p.state()
    fixed = run_git(worktree, "rev-parse", "HEAD")
    state["prs"][0]["head"]["sha"] = fixed
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    evidence = p.spec(
        {
            "ci": f"CI succeeded for {fixed}",
            "regression_tests": "boundary regression passed",
            "resolution": "finding fixed",
        }
    )
    r.cli(
        "resolve",
        "--run",
        r.request["review_run_id"],
        "--pr",
        "100",
        "--finding",
        "pr-100-1",
        "--fixed-head",
        fixed,
        "--evidence",
        evidence,
    )
    if resolution == "stale":
        (worktree / "change.txt").write_text("push after verification\n")
        run_git(worktree, "commit", "-am", "later push")
        state = p.state()
        state["prs"][0]["head"]["sha"] = run_git(worktree, "rev-parse", "HEAD")
        p.save(state)
    elif resolution == "unpublished":
        state = p.state()
        state["comments"]["1"] = [
            c
            for c in state["comments"]["1"]
            if not any(
                rec["kind"] == "resolution" for rec in g.policy.records([c], "operator")
            )
        ]
        p.save(state)
    elif resolution == "wrong_diff":
        path = next(r.run_dir.glob("resolution-*.json"))
        saved = json.loads(path.read_text())
        old = saved["record"]
        saved["record"] = {**old, "fix_diff_sha256": "f" * 64}
        replacement = saved["record"]
        new_key = g.policy.digest(replacement)
        (r.run_dir / f"resolution-{new_key}.json").write_text(json.dumps(saved))
        (r.run_dir / f"resolution-{new_key}.diff").write_bytes(
            next(r.run_dir.glob("resolution-*.diff")).read_bytes()
        )
        state = p.state()
        for comments in state["comments"].values():
            for comment in comments:
                if g.policy.records([comment], "operator") == [old]:
                    comment["body"] = g.policy.marker(replacement)
        p.save(state)
    rows = g.results(g.cli("merge"))
    if resolution == "valid" and not extra_path:
        assert [c[2] for c in g.merge_calls()] == ["100"]
        assert rows[100]["action"] == "merged"
        assert g.merge_calls()[0][-1] == fixed
    else:
        assert g.merge_calls() == []
        assert rows[100]["decision"] == "blocked"
        assert (
            "pr_100_outside_finding_paths" if extra_path else "pr_100_unverified_head"
        ) in rows[100]["reasons"]


def test_missing_parent_run_cannot_use_old_pr_approval(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    state = json.loads(registry.read_text())
    state["parents"]["o/r#1"]["reviews"] = []
    registry.write_text(json.dumps(state))
    state = p.state()
    state["comments"]["100"] = [
        {
            "id": 9999,
            "user": {"login": "operator"},
            "body": "<!-- agent-cross-review:v4 approve -->",
        }
    ]
    p.save(state)
    result = g.cli("merge")
    assert result.returncode == 1, result.stderr
    assert all(
        "parent_review_missing" in row["reasons"]
        for row in json.loads(result.stdout)["results"]
    )
    assert not g.merge_calls()


@pytest.mark.parametrize("when", ["ci", "before_merge"])
def test_head_race_stops_without_merge(gate_env: GateEnv, when: str) -> None:
    g, p = gate_env, gate_env.review.p
    worktree = p.repo / ".claude/worktrees/pr-0"
    (worktree / "change.txt").write_text("newer head\n")
    run_git(worktree, "commit", "-am", "racing push")
    new_head = run_git(worktree, "rev-parse", "HEAD")
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    result = g.cli(
        "merge",
        **{
            (
                "PARENT_VIEW_HEAD_RACE" if when == "ci" else "PARENT_READ_HEAD_RACE"
            ): new_head
        },
    )
    rows = g.results(result)
    assert (
        "head_changed_during_ci" if when == "ci" else "head_changed_before_merge"
    ) in rows[100]["reasons"]
    assert not g.merge_calls()


def test_new_child_pr_invalidates_the_run_set_without_holding_by_ci(
    gate_env: GateEnv,
) -> None:
    g, p = gate_env, gate_env.review.p
    p.child("added-after-review")
    state = p.state()
    extra = {
        **p.pr(103, 13),
        "labels": [],
        "mergeable": True,
        "statusCheckRollup": state["prs"][0]["statusCheckRollup"],
    }
    state["prs"].append(extra)
    p.save(state)
    rows = g.results(g.cli("merge"))
    assert all("pr_set_changed" in rows[n]["reasons"] for n in (100, 101, 102))
    assert "pr_outside_review_targets" in rows[103]["reasons"]
    assert not g.merge_calls()


def test_no_targets_is_incomplete_and_never_invokes_agent(parent_env: Any) -> None:
    p = parent_env
    p.agreed()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", "o/r", "--parent", "1", "merge"],
        cwd=p.repo,
        env={**p.env, "AGENT_AUTO_MERGE": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    assert json.loads(result.stdout)["reasons"] == ["no_target_prs"]
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())


def test_invalid_run_timestamps_still_fail_closed(gate_env: GateEnv) -> None:
    g, r = gate_env, gate_env.review
    path = r.run_dir / "result.json"
    result = json.loads(path.read_text())
    result["finished_at"] = "2026-10-17T23:59:59+00:00"
    path.write_text(json.dumps(result))
    response = g.cli("merge")
    assert response.returncode == 2
    assert "invalid run timestamps" in response.stderr
    assert not g.merge_calls()
