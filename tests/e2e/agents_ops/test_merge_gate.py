"""Explicit merge gate against actual parent-run records, fake gh and real git."""

from __future__ import annotations

import hashlib
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
    ParentEnv,
    ReviewEnv,
    make_review_env,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import (
    parent_env as parent_env,
)

SCRIPT = OPS / "review/merge_gate.py"


def assert_unclassified_agreement_comment(
    response: subprocess.CompletedProcess[str], comment_id: int
) -> None:
    assert response.returncode == 2, response.stderr
    result = json.loads(response.stderr)
    assert result["decision"] == "blocked"
    assert result["reasons"] == ["input_or_collection_error"]
    assert result["agent_invocations"] == 0
    error = result["error"]
    assert "unclassified agreement comments" in error
    assert f"ID={comment_id}" in error
    assert f"https://github.com/o/r/issues/1#issuecomment-{comment_id}" in error
    assert "agreement-ignore-comments" in error and "checking → agree" in error


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
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> GateEnv:
    variant = getattr(request, "param", None)
    review_env = make_review_env(
        parent_env,
        independent=variant in {"independent", "dependency", "research_dependency"},
        standalone=variant == "standalone",
    )
    p = review_env.p
    policy = review_policy_module(monkeypatch)
    if variant in {
        "shared",
        "dependency",
        "research_dependency",
        "stacked",
        "non_main_default",
    }:
        if variant == "research_dependency":
            p.child("research")
        state = p.state()
        if variant == "shared":
            worktree = p.repo / ".claude/worktrees/pr-0"
            (worktree / ".agents").mkdir()
            (worktree / ".agents/shared.txt").write_text("shared foundation\n")
            run_git(worktree, "add", ".agents/shared.txt")
            run_git(worktree, "commit", "-m", "shared changes")
            state["prs"][0]["head"]["sha"] = run_git(worktree, "rev-parse", "HEAD")
        elif variant in {"dependency", "research_dependency"}:
            # Freeze dependencies before the one actual review, like other requirements.
            parent_policy = importlib.import_module("parent.policy")
            dependency = 13 if variant == "research_dependency" else 11
            if variant == "research_dependency":
                state["issues"]["13"].update(state="closed", state_reason="completed")
            state["issues"]["10"]["body"] = parent_policy.replace_section(
                state["issues"]["10"]["body"], "依存関係", f"#{dependency} の完了後。"
            )
        elif variant == "stacked":
            state["prs"][0]["base"]["ref"] = "codex/lower"
            run_git(p.repo, "branch", "codex/lower")
        else:
            state["default_branch"] = "trunk"
            for pr in state["prs"]:
                pr["base"]["ref"] = "trunk"
            run_git(p.repo, "branch", "-m", "main", "trunk")
        p.save(state)
        manifest = json.loads(
            p.cli(
                "review-prepare",
                "--prs",
                "100",
                "101",
                "102",
                "--reason-file",
                p.note("変更パス・依存関係を含む対象をレビュー前に確定し直す。"),
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


@pytest.mark.parametrize("gate_env", ["standalone"], indirect=True)
@pytest.mark.parametrize("close_confirmed", [False, True])
def test_standalone_issue_review_merge_and_completion(
    gate_env: GateEnv, close_confirmed: bool
) -> None:
    g, p = gate_env, gate_env.review.p
    body = p.state()["issues"]["1"]["body"]
    result = json.loads((g.review.run_dir / "result.json").read_text())
    assert result["prs"][0]["child_issues"] == [1]
    response = g.cli(
        "merge", **({} if close_confirmed else {"PARENT_MERGE_SKIP_CLOSE": "1"})
    )
    assert response.returncode == (0 if close_confirmed else 1), response.stderr
    row = g.results(response)[100]
    assert row["action"] == ("merged" if close_confirmed else "merge_unconfirmed")
    assert row["reasons"] == (
        [] if close_confirmed else ["child_1_close_not_confirmed"]
    )
    assert p.state()["parents"] == {}
    assert p.state()["issues"]["1"]["body"] == body
    assert p.state()["issues"]["1"]["state"] == (
        "closed" if close_confirmed else "open"
    )
    if close_confirmed:
        assert p.state()["issues"]["1"]["state_reason"] == "completed"
    g.cli("merge")  # Closed parent or existing merge: never issue a second merge.
    assert len(g.merge_calls()) == len(g.review.calls()) == 1


@pytest.mark.parametrize("gate_env", ["standalone"], indirect=True)
def test_unreviewed_standalone_pr_is_collected_and_blocked(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    registry = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )
    data = json.loads(registry.read_text())
    data["parents"]["o/r#1"]["reviews"] = []
    registry.write_text(json.dumps(data))
    response = g.cli("merge")
    assert response.returncode == 1, response.stderr
    rows = json.loads(response.stdout)["results"]
    assert [r["pr"] for r in rows] == [100]
    assert "parent_review_missing" in rows[0]["reasons"]
    assert g.merge_calls() == []


@pytest.mark.parametrize(
    "change", ["same_agreement", "changed_agreement", "checking", "generation"]
)
def test_review_authorization_survives_only_unrelated_agreement_records(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    policy = importlib.import_module("parent.policy")
    agreed = next(
        e
        for e in policy.events(state["comments"]["1"], "operator", 1)
        if e["event"] == "agreed"
    )
    agreed.pop("body")
    if change == "changed_agreement":
        agreed["agreement"]["scope"] = "既存作業にも関係する新しい範囲"
        agreed["agreement_sha256"] = policy.digest(agreed["agreement"])
    elif change == "checking":
        p.agreed()  # Explicit checking revokes old evidence even with the same digest.
        state = p.state()
    elif change == "generation":
        registry = (
            Path(p.env["XDG_STATE_HOME"])
            / "tennis-lab-agents/parent-sessions/registry.json"
        )
        data = json.loads(registry.read_text())
        data["parents"]["o/r#1"]["generation"] += 1
        registry.write_text(json.dumps(data))
    state["comments"]["1"].append(
        {"id": 9999, "user": {"login": "operator"}, "body": policy.marker(agreed)}
    )
    p.save(state)
    response = g.cli("merge", AGENT_AUTO_MERGE="0")
    rows = g.results(response)
    if change == "same_agreement":
        assert all(r["decision"] == "merge" for r in rows.values())
    else:
        assert all(r["decision"] != "merge" for r in rows.values())
        assert "parent_review_not_confirmed" in rows[100]["reasons"]
    assert len(g.review.calls()) == 1
    assert g.merge_calls() == []


@pytest.mark.skipif(
    not (OPS / "parent/proposals.py").exists(),
    reason="Proposal delivery is supplied by #1223; exercised when the two branches are integrated.",
)
@pytest.mark.parametrize("gate_env", ["shared"], indirect=True)
@pytest.mark.parametrize("proposal_state", ["pending", "agreed", "parent_checking"])
def test_independent_reviewed_pr_merges_while_new_proposal_is_checked(
    gate_env: GateEnv, monkeypatch: pytest.MonkeyPatch, proposal_state: str
) -> None:
    g, p = gate_env, gate_env.review.p
    assert (
        g.cli(
            "shared-permission",
            "--allow-shared-foundation",
            "--evidence-file",
            p.note("人間との合意: 既存の共有基盤変更を親がmergeしてよい。"),
        ).returncode
        == 0
    )
    monkeypatch.syspath_prepend(str(OPS / "ops-report"))
    proposals = importlib.import_module("opsreport.proposals")
    prop = proposals.Proposal(
        section="その他の助言",
        title="独立した追加提案B",
        evidence="使用感",
        purpose="次の改善",
        scope="既存Aとは独立",
        expected_effect="保守性",
        acceptance=("回帰テストで確認",),
        related_parent=1,
    )
    markdown, _ = proposals.render_proposals([prop], "20261012")
    state = p.state()
    state["issues"]["90"] = {
        **state["issues"]["1"],
        "number": 90,
        "id": 10090,
        "title": "運用レポート",
        "labels": [{"name": "ops-report"}],
        "body": "<!-- ops-report:v2 slot=20261012 since=2026-10-09T00:00:00+00:00 until=2026-10-12T00:00:00+00:00 proposals=1 status=complete -->\n"
        + proposals.PROPOSALS_BEGIN
        + "\n"
        + markdown.replace("- [ ] **", "- [x] **")
        + proposals.PROPOSALS_END
        + "\n",
    }
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    p.cli("receive-proposal", "--report", "90", "--proposal", "R-20261012-01")
    status = json.loads(p.cli("status").stdout)
    assert status["phase"] == "要件合意済み"
    assert status["proposal_agreements"] == []
    if proposal_state == "agreed":
        p.cli(
            "agree",
            "--report",
            "90",
            "--proposal",
            "R-20261012-01",
            "--spec",
            p.spec(
                {
                    "purpose": "Bの目的",
                    "scope": "Aには影響しないBの範囲",
                    "behavior": "Bの期待動作",
                    "acceptance": "Bの回帰テスト",
                    "human_confirmation": "Bの要件で合意します。",
                }
            ),
        )
        assert len(json.loads(p.cli("status").stdout)["proposal_agreements"]) == 1
    elif proposal_state == "parent_checking":
        p.cli("checking")
    response = g.cli("merge")
    if proposal_state == "parent_checking":
        assert response.returncode == 2
        assert g.merge_calls() == []
    else:
        rows = g.results(response)
        assert rows[100]["action"] == "merged"
        assert rows[100]["shared_exemption"]["allow_shared_foundation"] is True
        assert [c[2] for c in g.merge_calls()] == ["100"]
    assert (
        len(g.review.calls()) == 1
    )  # Receive, agreement and merge never launch a review.


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
    if case in {"parent_missing", "foreign_result", "quoted_result"}:
        # These records have become ordinary, unclassified discussion. Freshness
        # now refuses them before the public review/coverage checks are reached.
        assert_unclassified_agreement_comment(result, comment["id"])
        assert g.merge_calls() == [] and len(g.review.calls()) == 1
        return
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


@pytest.mark.parametrize("gate_env", ["research_dependency"], indirect=True)
@pytest.mark.parametrize(
    "case,status,satisfied",
    [
        ("no_pr", "completed_without_pr", True),
        ("merged", "merged_into_default_branch", True),
        ("merged_and_closed_unmerged", "merged_into_default_branch", True),
        ("open_pr", "open_pr", False),
        ("open_and_merged", "open_pr", False),
        ("closed_unmerged", "unmerged", False),
        ("closed_unmerged_later_page", "unmerged", False),
        ("nondefault_merged", "base_not_default_branch", False),
        ("not_completed", "not_completed", False),
        ("not_planned", "not_completed", False),
        ("duplicate", "not_completed", False),
        ("unknown_pr_state", "pr_state_unknown", False),
        ("missing_merge_state", "pr_state_unknown", False),
        ("unknown_merged_base", "pr_base_unknown", False),
    ],
)
def test_completed_dependency_requires_no_pr_or_default_branch_merge(
    gate_env: GateEnv, case: str, status: str, satisfied: bool
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    if case == "not_completed":
        state["issues"]["13"].update(state="open", state_reason=None)
    elif case in {"not_planned", "duplicate"}:
        state["issues"]["13"]["state_reason"] = case
    elif case != "no_pr":
        related = {
            **p.pr(103, 13),
            "labels": [],
            "mergeable": True,
            "state": "closed",
        }
        if case in {
            "merged",
            "merged_and_closed_unmerged",
            "open_and_merged",
            "nondefault_merged",
            "unknown_merged_base",
        }:
            related["merged_at"] = "2026-10-18T03:00:00Z"
        if case == "nondefault_merged":
            related["base"]["ref"] = "codex/lower"
        elif case == "open_pr":
            related["state"] = "open"
        elif case == "unknown_pr_state":
            related["state"] = "unknown"
        elif case == "missing_merge_state":
            del related["merged_at"]
        elif case == "unknown_merged_base":
            related["base"] = {}
        if case == "closed_unmerged_later_page":
            state["prs"].extend(
                {**p.pr(200 + i, 999), "state": "closed"} for i in range(100)
            )
        state["prs"].append(related)
        if case in {"open_and_merged", "merged_and_closed_unmerged"}:
            state["prs"].append(
                {
                    **p.pr(104, 13),
                    "state": "open" if case == "open_and_merged" else "closed",
                    "labels": [],
                    "mergeable": True,
                }
            )
    p.save(state)
    response = g.cli("merge")
    assert response.returncode == 1, response.stderr  # The independent PRs wait for CI.
    row = g.results(response)[100]
    assert row["dependency_statuses"] == {"13": status}
    if satisfied:
        assert row["action"] == "merged"
        assert not row["reasons"]
        assert [c[2] for c in g.merge_calls()] == ["100"]
    else:
        assert f"dependency_13_{status}" in row["reasons"]
        assert row["action"] == "blocked"
        assert not g.merge_calls()
    pull_reads = [
        endpoint
        for call in p.calls()
        for endpoint in call
        if endpoint.startswith("repos/o/r/pulls?")
    ]
    assert pull_reads and all("state=all" in endpoint for endpoint in pull_reads)
    if case == "closed_unmerged_later_page":
        assert any("page=2" in endpoint for endpoint in pull_reads)


@pytest.mark.parametrize("gate_env", ["research_dependency"], indirect=True)
@pytest.mark.parametrize(
    "endpoint",
    [
        "repos/o/r/issues/13",
        "repos/o/r/pulls?state=all&per_page=100&page=1",
        "repos/o/r",
    ],
)
def test_dependency_or_repository_read_failure_never_merges(
    gate_env: GateEnv, endpoint: str
) -> None:
    g = gate_env
    response = g.cli("merge", PARENT_FAIL_GET_PATH=endpoint)
    assert response.returncode == 2
    error = json.loads(response.stderr)
    assert error["reasons"] == ["input_or_collection_error"]
    assert "injected GitHub failure" in error["error"]
    assert not g.merge_calls()


@pytest.mark.parametrize("gate_env", ["research_dependency"], indirect=True)
@pytest.mark.parametrize("state,reason", [("closed", None), ("closed", "unknown")])
def test_unknown_dependency_completion_never_merges(
    gate_env: GateEnv, state: str, reason: str | None
) -> None:
    g, p = gate_env, gate_env.review.p
    data = p.state()
    data["issues"]["13"].update(state=state, state_reason=reason)
    p.save(data)
    response = g.cli("merge")
    assert response.returncode == 2
    error = json.loads(response.stderr)
    assert error["reasons"] == ["input_or_collection_error"]
    assert "unknown issue state/reason" in error["error"]
    assert not g.merge_calls()


@pytest.mark.parametrize("gate_env", ["stacked"], indirect=True)
@pytest.mark.parametrize("merged", [False, True])
def test_reviewed_pr_with_nondefault_base_is_blocked(
    gate_env: GateEnv, merged: bool
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    if merged:
        state["prs"][0].update(state="closed", merged_at="2026-10-18T03:00:00Z")
        state["issues"]["10"].update(state="closed", state_reason="completed")
    p.save(state)
    response = g.cli("merge")
    assert response.returncode == 1, response.stderr
    row = g.results(response)[100]
    assert "base_not_default_branch" in row["reasons"]
    assert not any("base_ref_changed" in reason for reason in row["reasons"])
    assert row["action"] == ("merge_unconfirmed" if merged else "blocked")
    assert not g.merge_calls()


@pytest.mark.parametrize("gate_env", ["non_main_default"], indirect=True)
def test_repository_default_branch_is_used_instead_of_main(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    response = g.cli("merge")
    assert response.returncode == 1, response.stderr
    assert g.results(response)[100]["action"] == "merged"
    assert [c[2] for c in g.merge_calls()] == ["100"]
    assert sum("repos/o/r" in call for call in p.calls()) >= 2


@pytest.mark.parametrize("default_branch", [None, "", " ", 123])
def test_unknown_default_branch_never_merges(
    gate_env: GateEnv, default_branch: Any
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    state["default_branch"] = default_branch
    p.save(state)
    response = g.cli("merge")
    assert response.returncode == 2
    error = json.loads(response.stderr)
    assert error["reasons"] == ["input_or_collection_error"]
    assert "repository default_branch" in error["error"]
    assert not g.merge_calls()


@pytest.mark.parametrize("gate_env", ["dependency"], indirect=True)
def test_dependency_order_merges_a_before_b_while_c_waits(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    # A has the larger PR number: numeric sorting alone would merge B first.
    state["prs"][2]["statusCheckRollup"] = []
    p.save(state)
    rows = g.results(g.cli("merge", PARENT_MERGE_ADVANCE_BASE="1"))
    assert [c[2] for c in g.merge_calls()] == ["101", "100"]
    for call in g.merge_calls():
        n = int(call[2])
        assert call[call.index("--match-head-commit") + 1] == next(
            pr["head_sha"] for pr in g.review.request["prs"] if pr["number"] == n
        )
    assert rows[100]["action"] == rows[101]["action"] == "merged"
    assert rows[100]["snapshot"]["base_sha"] != g.review.request["prs"][0]["base_sha"]
    for n in (100, 101):
        original = next(s for s in g.review.request["prs"] if s["number"] == n)
        assert all(
            rows[n]["snapshot"][key] == original[key]
            for key in ("head_sha", "merge_base_sha", "diff_sha256")
        )
    assert "ci_missing_required" in rows[102]["reasons"]
    assert not any("base" in reason for reason in rows[102]["reasons"])
    assert p.state()["issues"]["1"]["state"] == "open"
    assert "#10: 完了" in p.state()["issues"]["1"]["body"]
    assert "#11: 完了" in p.state()["issues"]["1"]["body"]
    # A repeated explicit command confirms existing merges without issuing them again.
    repeated = g.results(g.cli("merge"))
    assert repeated[100]["action"] == repeated[101]["action"] == "already_merged"
    assert all(repeated[n]["snapshot"] == rows[n]["snapshot"] for n in (100, 101))
    assert len(g.merge_calls()) == 2


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
@pytest.mark.parametrize("fixed", [False, True])
def test_all_targets_merge_in_one_run_with_verified_base_advances(
    gate_env: GateEnv, fixed: bool
) -> None:
    g, r, p = gate_env, gate_env.review, gate_env.review.p
    if fixed:

        def add_finding(result: dict[str, Any]) -> None:
            result["verdict"] = "request_changes"
            result["prs"][1]["findings"] = [
                {
                    "id": "pr-101-1",
                    "severity": "major",
                    "path": "change-1.txt",
                    "line": 1,
                    "rule": "境界",
                    "detail": "回帰テストとともに修正する",
                }
            ]

        g.edit_result(add_finding)
        worktree = p.repo / ".claude/worktrees/pr-1"
        (worktree / "change-1.txt").write_text("fixed and checked\n")
        (worktree / "regression.txt").write_text("parent checked regression\n")
        run_git(worktree, "add", ".")
        run_git(worktree, "commit", "-m", "fix with regression coverage")
        head = run_git(worktree, "rev-parse", "HEAD")
        state = p.state()
        state["prs"][1]["head"]["sha"] = head
        p.save(state)
        r.cli(
            "resolve",
            "--run",
            r.request["review_run_id"],
            "--pr",
            "101",
            "--finding",
            "pr-101-1",
            "--fixed-head",
            head,
            "--evidence",
            p.spec(
                {
                    "ci": f"CI succeeded for {head}",
                    "regression_tests": "boundary regression passed",
                    "resolution": "implementation and its regression checked",
                    "reviewed_paths": ["change-1.txt", "regression.txt"],
                }
            ),
        )
    result = g.cli("merge", PARENT_MERGE_ADVANCE_BASE="1")
    assert result.returncode == 0, result.stderr
    rows = g.results(result)
    assert [c[2] for c in g.merge_calls()] == ["100", "101", "102"]
    assert all(row["action"] == "merged" for row in rows.values())
    assert len({row["snapshot"]["base_sha"] for row in rows.values()}) == 3
    for pr in p.state()["prs"]:
        assert pr["merged"] is True
        assert (
            run_git(p.repo, "rev-parse", pr["merge_commit_sha"] + "^2")
            == pr["head"]["sha"]
        )
    repeat = g.cli("merge")
    assert repeat.returncode == 0, repeat.stderr
    repeated = g.results(repeat)
    assert all(row["action"] == "already_merged" for row in repeated.values())
    assert all(repeated[n]["snapshot"] == rows[n]["snapshot"] for n in rows)
    assert len(g.merge_calls()) == 3
    assert len(r.calls()) == 1


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
def test_merged_target_head_mismatch_blocks_remaining_prs(gate_env: GateEnv) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    for pr in state["prs"][1:]:
        pr["statusCheckRollup"] = []
    p.save(state)
    assert (
        g.results(g.cli("merge", PARENT_MERGE_ADVANCE_BASE="1"))[100]["action"]
        == "merged"
    )
    state = p.state()
    state["prs"][0]["head"]["sha"] = state["prs"][1]["head"]["sha"]
    state["prs"][1]["statusCheckRollup"] = state["prs"][0]["statusCheckRollup"]
    p.save(state)
    result = g.cli("merge")
    assert result.returncode == 1, result.stderr
    rows = g.results(result)
    assert "merged_head_mismatch" in rows[100]["reasons"]
    assert "pr_100_merged_head_mismatch" in rows[101]["reasons"]
    assert [c[2] for c in g.merge_calls()] == ["100"]


@pytest.mark.parametrize("gate_env", ["independent"], indirect=True)
@pytest.mark.parametrize("change", ["base", "ci", "main_follow"])
def test_remaining_prs_are_recollected_after_each_merge(
    gate_env: GateEnv, change: str
) -> None:
    g, p = gate_env, gate_env.review.p
    state = p.state()
    state["prs"][2]["statusCheckRollup"] = []
    p.save(state)
    env = {
        "base": {"PARENT_MERGE_UNEXPLAINED_COMMIT": "1"},
        "ci": {"PARENT_MERGE_CI_PENDING": "101"},
        "main_follow": {
            "PARENT_MERGE_FOLLOW_MAIN": "101",
            "PARENT_MERGE_FOLLOW_WORKTREE": str(p.repo / ".claude/worktrees/pr-1"),
        },
    }[change]
    env["PARENT_MERGE_ADVANCE_BASE"] = "1"
    rows = g.results(g.cli("merge", **env))
    assert [c[2] for c in g.merge_calls()] == ["100"]
    expected = {
        "base": "pr_101_base_sha_changed",
        "ci": "ci_missing_required",
        "main_follow": "pr_101_merge_base_sha_changed",
    }[change]
    assert expected in rows[101]["reasons"]
    if change == "base":
        assert "base_advance_unverified" in rows[101]["reasons"]


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
    response = g.cli("merge")
    if permission in {"foreign", "quoted"}:
        assert_unclassified_agreement_comment(response, comment["id"])
        assert g.merge_calls() == [] and len(g.review.calls()) == 1
        return
    rows = g.results(response)
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
@pytest.mark.parametrize(
    "resolution", ["valid", "stale", "unpublished", "wrong_diff", "outside"]
)
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
        (worktree / "regression.txt").write_text("parent checked regression\n")
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
            "reviewed_paths": ["change.txt", "regression.txt"]
            if extra_path
            else ["change.txt"],
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
    elif resolution == "outside":
        (worktree / "outside.txt").write_text("unreviewed expansion\n")
        run_git(worktree, "add", "outside.txt")
        run_git(worktree, "commit", "-m", "change outside confirmed paths")
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
        forged_diff = b"a saved diff that differs from the actual git diff\n"
        saved["record"] = {
            **old,
            "fix_diff_sha256": hashlib.sha256(forged_diff).hexdigest(),
        }
        replacement = saved["record"]
        new_key = g.policy.digest(replacement)
        (r.run_dir / f"resolution-{new_key}.json").write_text(json.dumps(saved))
        (r.run_dir / f"resolution-{new_key}.diff").write_bytes(forged_diff)
        state = p.state()
        for comments in state["comments"].values():
            for comment in comments:
                if g.policy.records([comment], "operator") == [old]:
                    comment["body"] = g.policy.marker(replacement)
        p.save(state)
    rows = g.results(g.cli("merge"))
    if resolution == "valid":
        assert [c[2] for c in g.merge_calls()] == ["100"]
        assert rows[100]["action"] == "merged"
        assert g.merge_calls()[0][-1] == fixed
        repeated = g.results(g.cli("merge"))
        assert repeated[100]["action"] == "already_merged"
        assert repeated[100]["snapshot"] == rows[100]["snapshot"]
        assert len(g.merge_calls()) == 1
    else:
        assert g.merge_calls() == []
        assert rows[100]["decision"] == "blocked"
        assert "pr_100_unverified_head" in rows[100]["reasons"]
        if resolution == "wrong_diff":
            assert "resolution_fix_diff_mismatch" in rows[100]["reasons"]


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
