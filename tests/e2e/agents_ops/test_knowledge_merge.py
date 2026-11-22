"""Parent CLI with real Git comparisons and a fake GitHub Actions/PR boundary."""

from __future__ import annotations

import json
import zipfile
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_progression_cli import prepared, request
from tests.support.agents_ops.improvements import finish_for


def pr_with_paths(p: ParentEnv, number: int, paths: list[str]) -> dict[str, Any]:
    tree = p.worktree(f"knowledge-pr-{number}")
    for name in paths:
        file = tree / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"fixture for #{number}\n")
    run_git(tree, "add", *paths)
    run_git(tree, "commit", "-m", f"fixture PR {number}")
    head = run_git(tree, "rev-parse", "HEAD")
    pr = p.pr(number, 10, head, f"codex/knowledge-pr-{number}")
    merge = run_git(
        p.repo,
        "commit-tree",
        run_git(tree, "rev-parse", "HEAD^{tree}"),
        "-p",
        pr["base"]["sha"],
        "-p",
        head,
        "-m",
        "test merge",
    )
    pr.update(mergeable=True, merge_commit_sha=merge, labels=[])
    return pr


def successful_run(p: ParentEnv, state: dict[str, Any], pr: dict[str, Any]) -> None:
    head = pr["head"]["sha"]
    state["knowledge_runs"] = [
        {
            "id": 54,
            "run_number": 18,
            "run_attempt": 2,
            "head_sha": head,
            "head_branch": pr["head"]["ref"],
            "pull_requests": [],
            "path": ".github/workflows/knowledge.yml",
            "event": "pull_request",
            "status": "completed",
            "conclusion": "success",
        }
    ]
    archive = p.root / "artifact.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(
            "knowledge-records.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "scope": "records-only",
                    "pr_number": pr["number"],
                    "head_sha": head,
                    "base_sha": pr["base"]["sha"],
                    "merge_sha": pr["merge_commit_sha"],
                    "run_id": 54,
                    "run_attempt": 2,
                    "conclusion": "success",
                    "checks": {"kg_validate": "success", "kg_repro_paths": "success"},
                }
            ),
        )
    state["artifact_zips"] = {"88": str(archive)}
    state["knowledge_artifacts"] = {
        "54": [
            {
                "id": 88,
                "name": "knowledge-records-54-2",
                "expired": False,
                "size_in_bytes": archive.stat().st_size,
            }
        ]
    }
    # A passing PR Python check would not prove a successful main push, even
    # with a matching SHA. The fake API assigns this suite to the push on merge.
    pr["statusCheckRollup"] = [
        {
            "__typename": "CheckRun",
            "name": "Python",
            "status": "COMPLETED",
            "conclusion": "SUCCESS",
        }
    ]


def merge_request(p: ParentEnv, number: int, **extra: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            p.cli(
                "merge-knowledge",
                "--prs",
                str(number),
                "--evidence-file",
                p.note("現HEADの再現コマンドと結果を照合済み。"),
                "--wait",
                extra=extra,
            ).stdout
        ),
    )


def test_knowledge_records_are_excluded_at_both_review_entry_points(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    _, fake, calls = prepared(p)
    record = "knowledge/nodes/task/000001-result.md"
    choices = [
        [record],
        [record, "knowledge/runs/trial/repro.sh", "knowledge/runs/trial/eval.py"],
        [record, "src/implementation.py"],
        ["knowledge/webui/app/page.tsx"],
        ["knowledge/registry/deployments.yaml"],
        ["knowledge/scripts/kg_validate.py"],
        [record],
    ]
    state = p.state()
    state["prs"] = [pr_with_paths(p, 100 + i, paths) for i, paths in enumerate(choices)]
    state["prs"][-1]["merge_commit_sha"] = None
    p.save(state)
    value = json.loads(
        p.cli("review-inspect", "--prs", *map(str, range(100, 107))).stdout
    )
    assert value["review_candidates"] == [102, 103, 104, 105, 106]
    assert "test merge commit is unavailable" in value["prs"][-1]["scope"]["reason"]
    for n in (100, 101):
        result = p.cli(
            "review-request",
            "--prs",
            str(n),
            *p.review_flags(),
            "--instruction",
            "現HEADを確認",
            "--evidence-file",
            p.note(),
            "--agent-exec",
            str(fake),
            ok=False,
        )
        assert f"use merge-knowledge --prs {n}" in result.stderr
    assert all(
        "verification" not in task
        for task in json.loads(p.cli("status").stdout)["tasks"].values()
    )
    review = request(p, [102, 103, 104, 105, 106], fake, AGENT_AUTO_MERGE="0")
    assert review["review"]["stage"] == "requested"
    assert [pr["number"] for pr in review["prs"]] == [102, 103, 104, 105, 106]
    assert calls.read_text() == ""


@pytest.mark.parametrize("purpose", ["closing", "companion", "lost_response"])
def test_public_knowledge_merge_and_acceptance(
    parent_env: ParentEnv, purpose: str
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    original = numbers[0]
    state = p.state()
    n = 101
    pr = pr_with_paths(
        p, n, ["knowledge/nodes/task/000001-result.md", "knowledge/runs/trial/repro.sh"]
    )
    companion = purpose == "companion"
    if companion:
        pr["body"] = "Refs #10, #100"
        state["prs"].append(pr)
    else:
        state["prs"] = [pr]
    successful_run(p, state, pr)
    p.save(state)
    result = merge_request(
        p,
        n,
        **(
            {"PARENT_KNOWLEDGE_LOST_RESPONSE": "1"}
            if purpose == "lost_response"
            else {}
        ),
    )
    if purpose == "lost_response":
        assert result["prs"][0]["reason"] == "recovery_required"
        assert result["prs"][0]["resume_stage"] == "merge_pending"
        result = json.loads(
            p.cli(
                "merge-continue",
                "--run-id",
                result["review_run_id"],
                "--instruction-file",
                p.note("マージの応答を失ったためGitHubを再照合する。"),
                "--wait",
            ).stdout
        )
    assert result["kind"] == "knowledge_merge"
    assert result["status"] == "completed", result
    assert result["review"]["stage"] == "not_required"
    assert result["review_usage"]["confirmed_launches"] == 0
    target = result["prs"][0]
    assert target["ci"]["run"]["run_attempt"] == 2
    assert target["merge_commit"] == run_git(p.repo, "rev-parse", "main")
    assert target["main_ci"]["status"] == "success"
    assert target["main_ci"]["head"] == target["merge_commit"]
    assert target["issue_states"]["10"] == {
        "state": "open" if companion else "closed",
        "state_reason": None if companion else "completed",
    }
    assert target["issues_confirmed"] is not companion
    assert p.state()["knowledge_merges"] == [
        {"number": n, "payload": {"merge_method": "merge", "sha": pr["head"]["sha"]}}
    ]
    assert calls.read_text() == ""
    parent_status = json.loads(p.cli("status").stdout)
    assert parent_status["completion"]["knowledge_merges"][0]["status"] == "completed"
    finish = finish_for(p)
    if companion:
        incomplete = p.cli("release", "--ended", "--spec", p.spec(finish), ok=False)
        assert "close, acceptance evidence" in incomplete.stderr
        assert request(p, [original], fake)["status"] == "completed"
    p.announce_completion()
    ended = json.loads(p.cli("release", "--ended", "--spec", p.spec(finish)).stdout)
    assert ended["completion"]["record"]["prs"] == ([original, n] if companion else [n])


@pytest.mark.parametrize(
    ("location", "reason"),
    [
        ("pr", "needs-human: pulls/101"),
        ("issue", "needs-human: issues/10"),
        ("parent", "needs-human: issues/1"),
        ("stop", "target_stopped: #10"),
        ("disabled", "auto_merge_disabled"),
    ],
)
def test_knowledge_merge_obeys_human_controls(
    parent_env: ParentEnv, location: str, reason: str
) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    pr = pr_with_paths(p, 101, ["knowledge/nodes/task/000001-result.md"])
    state["prs"] = [pr]
    successful_run(p, state, pr)
    if location in {"pr", "issue", "parent"}:
        row = {
            "pr": pr,
            "issue": state["issues"]["10"],
            "parent": state["issues"]["1"],
        }[location]
        row["labels"] = [*row.get("labels", []), {"name": "needs-human"}]
    p.save(state)
    if location == "stop":
        p.cli(
            "stop",
            "--child",
            "10",
            "--spec",
            p.spec(
                {
                    key: "人の確認待ち"
                    for key in (
                        "reason",
                        "facts",
                        "impact",
                        "resume_condition",
                        "next_action",
                    )
                }
            ),
        )
    result = merge_request(
        p, 101, AGENT_AUTO_MERGE="0" if location == "disabled" else "1"
    )
    assert result["status"] == "parent_wait"
    assert result["prs"][0]["reason"] == reason
    assert "knowledge_merges" not in p.state()


def test_admitted_knowledge_pr_cannot_resume_with_a_mixed_diff(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    _, _, calls = prepared(p)
    state = p.state()
    record = "knowledge/nodes/task/000001-result.md"
    old = pr_with_paths(p, 101, [record])
    state["prs"] = [old]
    successful_run(p, state, old)
    p.save(state)
    initial = merge_request(p, 101, AGENT_AUTO_MERGE="0")
    assert initial["prs"][0]["reason"] == "auto_merge_disabled"
    mixed = pr_with_paths(p, 102, [record, "src/implementation.py"])
    state = p.state()
    pr = state["prs"][0]
    pr["head"]["sha"] = mixed["head"]["sha"]
    pr["merge_commit_sha"] = mixed["merge_commit_sha"]
    successful_run(p, state, pr)
    p.save(state)
    result = p.cli(
        "merge-continue",
        "--run-id",
        initial["review_run_id"],
        "--instruction-file",
        p.note("更新した差分の検証結果を照合した。"),
        "--wait",
        ok=False,
    )
    assert "src/implementation.py" in result.stderr
    assert "knowledge_merges" not in p.state()
    assert not calls.read_text()
