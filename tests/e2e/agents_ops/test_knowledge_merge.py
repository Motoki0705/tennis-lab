"""Parent CLI with real Git comparisons and a fake GitHub Actions/PR boundary."""

from __future__ import annotations

import fcntl
import json
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_progression_cli import prepared, request
from tests.support.agents_ops.improvements import finish_for

RECORD = "knowledge/nodes/task/000001-result.md"
SCRIPT = "knowledge/runs/trial/repro.sh"
WORKER_LOG = ".agents/logs/parents/1/workers/worker-log.md"


def pr_with_paths(
    p: ParentEnv, number: int, paths: list[str], *, message: str | None = None
) -> dict[str, Any]:
    tree = p.worktree(f"knowledge-pr-{number}")
    for name in paths:
        file = tree / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"fixture for #{number}\n")
    run_git(tree, "add", *paths)
    run_git(tree, "commit", "-m", message or f"fixture PR {number}")
    head = run_git(tree, "rev-parse", "HEAD")
    pr: dict[str, Any] = p.pr(number, 10, head, f"codex/knowledge-pr-{number}")
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
    pr.update(mergeable=True, merged=False, merge_commit_sha=merge, labels=[])
    return pr


def execution_evidence(pr: dict[str, Any], *, executed: bool = False) -> None:
    evidence = {"head_sha": pr["head"]["sha"], "status": "not_required"}
    if executed:
        evidence.update(
            status="passed",
            command=f"bash {SCRIPT}",
            exit_code=0,
            result="Expected CPU metrics and outputs observed.",
        )
    pr["body"] = pr["body"].split("\n\n## 実行証拠")[0] + (
        "\n\n## 実行証拠\n\n```json\n" + json.dumps(evidence) + "\n```\n"
    )


def acceptance_evidence(pr: dict[str, Any], **changes: object) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "prs": [
                {
                    "pr_number": pr["number"],
                    "head_sha": pr["head"]["sha"],
                    "acceptance": "verified",
                    "execution": "verified"
                    if '"status": "passed"' in pr["body"]
                    else "not_required",
                    "summary": "現HEADの受入条件を証拠と照合済み。",
                    **changes,
                }
            ],
        },
        ensure_ascii=False,
    )


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


def merge_request(p: ParentEnv, pr: dict[str, Any], **extra: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads(
            p.cli(
                "merge-knowledge",
                "--prs",
                str(pr["number"]),
                "--evidence-file",
                p.note(acceptance_evidence(pr)),
                "--wait",
                extra=extra,
            ).stdout
        ),
    )


def commit_base_change(root: Path, path: str) -> str:
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("A parallel change on main.\n", encoding="utf-8")
    run_git(root, "add", "--", path)
    run_git(root, "commit", "-m", f"Advance main with {path}")
    return run_git(root, "rev-parse", "HEAD")


def updated_test_merge(root: Path, base: str, head: str) -> str:
    return run_git(
        root,
        "commit-tree",
        run_git(root, "merge-tree", "--write-tree", base, head),
        "-p",
        base,
        "-p",
        head,
        "-m",
        "Updated test merge",
    )


def test_knowledge_records_are_excluded_at_both_review_entry_points(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    _, fake, calls = prepared(p)
    choices = [
        [RECORD],
        [RECORD, SCRIPT, "knowledge/runs/trial/eval.py"],
        [RECORD, "src/implementation.py"],
        ["knowledge/webui/app/page.tsx"],
        ["knowledge/registry/deployments.yaml"],
        ["knowledge/scripts/kg_validate.py"],
        [RECORD],
    ]
    state = p.state()
    state["prs"] = [pr_with_paths(p, 100 + i, paths) for i, paths in enumerate(choices)]
    state["prs"][-1]["merge_commit_sha"] = None
    p.save(state)
    value = json.loads(
        p.cli("review-inspect", "--prs", *map(str, range(100, 107))).stdout
    )
    assert value["review_candidates"] == [102, 103, 104, 105, 106]
    assert [pr["excluded_reason"] for pr in value["prs"][:2]] == ["records_only"] * 2
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
    pr = pr_with_paths(p, n, [RECORD, SCRIPT])
    companion = purpose == "companion"
    if companion:
        pr["body"] = "Refs #10, #100"
        state["prs"].append(pr)
    else:
        state["prs"] = [pr]
    execution_evidence(pr, executed=True)
    successful_run(p, state, pr)
    checked_base = pr["base"]["sha"]
    checked_merge = pr["merge_commit_sha"]
    observed_main = merge_base = final_base = checked_base
    if purpose == "closing":
        # The initial PR snapshot still points at the base used by CI.
        observed_main = commit_base_change(p.repo, WORKER_LOG)
        merge_base = final_base = commit_base_change(
            p.worktree("base-update"), ".agents/logs/parents/1/decision.md"
        )
        state["knowledge_advance_after_base_read"] = {
            "sha": merge_base,
            "pr": n,
            "test_merge": updated_test_merge(p.repo, merge_base, pr["head"]["sha"]),
        }
    elif companion:
        # GitHub has already regenerated the test merge since CI ran.
        observed_main = merge_base = final_base = commit_base_change(
            p.repo, "src/unrelated.py"
        )
        pr["base"]["sha"] = observed_main
        pr["merge_commit_sha"] = updated_test_merge(
            p.repo, observed_main, pr["head"]["sha"]
        )
    else:
        final_base = commit_base_change(p.worktree("base-update"), WORKER_LOG)
        state["knowledge_advance_on_merge"] = final_base
    p.save(state)
    result = merge_request(
        p,
        pr,
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
    assert target["ci"]["verified_base_sha"] == target["verified_base"] == checked_base
    assert target["ci"]["verified_merge_sha"] == checked_merge
    assert target["ci"]["observed_main_sha"] == observed_main
    assert target["ci"]["base_sha"] == pr["base"]["sha"]
    assert target["ci"]["merge_sha"] == pr["merge_commit_sha"]
    assert target["merge_base"] == merge_base
    assert target["merge_commit"] == run_git(p.repo, "rev-parse", "main")
    assert run_git(
        p.repo, "show", "-s", "--format=%P", target["merge_commit"]
    ).split() == [
        final_base,
        pr["head"]["sha"],
    ]
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
    ("phase", "path"),
    [
        ("observe", "src/tennis_scene/configs/dataset.yaml"),
        ("before_submit", "knowledge/reports/parallel.md"),
        ("during_submit", "knowledge/reports/parallel.md"),
    ],
)
def test_knowledge_main_changes_pause_the_merge(
    parent_env: ParentEnv, phase: str, path: str
) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    pr = pr_with_paths(p, 101, [RECORD])
    state["prs"] = [pr]
    execution_evidence(pr)
    successful_run(p, state, pr)
    if phase == "observe":
        commit_base_change(p.repo, path)
    else:
        tip = commit_base_change(p.worktree("base-update"), path)
        if phase == "before_submit":
            state["knowledge_advance_after_base_read"] = {
                "sha": tip,
                "pr": pr["number"],
                "test_merge": updated_test_merge(p.repo, tip, pr["head"]["sha"]),
            }
        else:
            state["knowledge_advance_on_merge"] = tip
    p.save(state)

    result = merge_request(p, pr)

    assert result["status"] == "parent_wait", result
    target = result["prs"][0]
    calls = p.calls()
    submitted = [call for call in calls if "repos/o/r/pulls/101/merge" in call]
    if phase == "during_submit":
        assert len(submitted) == 1
        assert target["resume_stage"] == "merge_pending"
        assert (
            target["error"]
            == "knowledge merge parents differ from the verified comparison"
        )
        assert "main_ci" not in target
    else:
        assert not submitted
        assert "merge_head" not in target
        assert target["resume_stage"] == "ci_wait"
        assert "main changed knowledge paths since verified CI" in target["reason"]
        assert path in target["reason"]
        assert "Merge main into the PR branch and refresh evidence" in target["reason"]
        assert target["ci"]["status"] == (
            "failure" if phase == "observe" else "success"
        )


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
    pr = pr_with_paths(p, 101, [RECORD])
    state["prs"] = [pr]
    execution_evidence(pr)
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
        p, pr, AGENT_AUTO_MERGE="0" if location == "disabled" else "1"
    )
    assert result["status"] == "parent_wait"
    assert result["prs"][0]["reason"] == reason
    assert "knowledge_merges" not in p.state()


def test_parked_knowledge_merge_can_transfer_to_formal_review(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    _, fake, calls = prepared(p)
    state = p.state()
    old = pr_with_paths(p, 101, [RECORD])
    state["prs"] = [old]
    execution_evidence(old)
    successful_run(p, state, old)
    p.save(state)
    initial = merge_request(p, old, AGENT_AUTO_MERGE="0")
    assert initial["prs"][0]["reason"] == "auto_merge_disabled"
    mixed = pr_with_paths(p, 102, [RECORD, "src/implementation.py"])
    state = p.state()
    pr = state["prs"][0]
    pr["head"]["sha"] = mixed["head"]["sha"]
    pr["merge_commit_sha"] = mixed["merge_commit_sha"]
    execution_evidence(pr)
    successful_run(p, state, pr)
    p.save(state)
    result = p.cli(
        "merge-continue",
        "--run-id",
        initial["review_run_id"],
        "--instruction-file",
        p.note(acceptance_evidence(pr)),
        "--wait",
        ok=False,
    )
    assert "src/implementation.py" in result.stderr
    assert "knowledge_merges" not in p.state()
    inspected = json.loads(p.cli("review-inspect", "--prs", "101").stdout)
    assert inspected["review_candidates"] == [101]
    formal = request(p, [101], fake)
    assert formal["kind"] == "review_and_merge"
    assert formal["status"] == "completed"
    assert formal["review_usage"]["confirmed_launches"] == 1
    assert len(calls.read_text().splitlines()) == 1
    old_status = json.loads(
        p.cli("merge-status", "--run-id", initial["review_run_id"]).stdout
    )
    assert old_status["status"] == "transferred"
    assert old_status["prs"][0]["stage"] == "transferred"
    assert "knowledge_merges" not in p.state()


@pytest.mark.parametrize("guard", ["submitted", "active", "merged"])
def test_knowledge_transfer_preserves_an_unsafe_reservation(
    parent_env: ParentEnv, guard: str
) -> None:
    p = parent_env
    _, fake, calls = prepared(p)
    state = p.state()
    pr = pr_with_paths(p, 101, [RECORD])
    state["prs"] = [pr]
    execution_evidence(pr)
    successful_run(p, state, pr)
    p.save(state)
    initial = merge_request(
        p,
        pr,
        **(
            {"PARENT_KNOWLEDGE_SEND_FAILURE": "1"}
            if guard == "submitted"
            else {"AGENT_AUTO_MERGE": "0"}
        ),
    )
    target = initial["prs"][0]
    if guard == "submitted":
        assert target["resume_stage"] == "merge_pending"
        assert target["merge_head"] == pr["head"]["sha"]
        assert any("repos/o/r/pulls/101/merge" in call for call in p.calls())
    mixed = pr_with_paths(p, 102, [RECORD, "src/implementation.py"])
    state = p.state()
    pr = state["prs"][0]
    pr["head"]["sha"] = mixed["head"]["sha"]
    pr["merge_commit_sha"] = mixed["merge_commit_sha"]
    if guard == "merged":
        pr.update(state="closed", merged=True, merged_at="2026-11-23T00:00:00Z")
    p.save(state)

    lock = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/progressions"
        / initial["review_run_id"]
        / "state.lock"
    )
    with lock.open("a") as controller:
        if guard == "active":
            fcntl.flock(controller, fcntl.LOCK_EX)
        result = p.cli(
            "review-request",
            "--prs",
            "101",
            *p.review_flags(),
            "--instruction",
            "現在の実装差分を確認する。",
            "--evidence-file",
            p.note(),
            "--agent-exec",
            str(fake),
            ok=False,
        )
    expected = {
        "submitted": "no merge intent",
        "active": "controller is already running",
        "merged": "review requires related open PRs",
    }
    assert expected[guard] in result.stderr
    current = json.loads(
        p.cli("merge-status", "--run-id", initial["review_run_id"]).stdout
    )
    assert current["prs"][0] == target
    assert not calls.read_text()


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("freeform", "requires a JSON object"),
        ("number", "identify each selected PR"),
        ("head", "does not match the PR and its current HEAD"),
        ("unstarted", "unverified or incomplete work"),
        ("command", "execution command: nonempty text required"),
        ("result", "command and result"),
        ("skip", "bundled run scripts require execution evidence"),
        ("declared", "parent execution verification disagrees"),
    ],
)
def test_knowledge_registration_requires_verified_current_execution_evidence(
    parent_env: ParentEnv, fault: str, reason: str
) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    pr = pr_with_paths(p, 101, [RECORD] if fault == "declared" else [RECORD, SCRIPT])
    execution_evidence(pr, executed=fault not in {"skip", "declared"})
    overrides: dict[str, dict[str, object]] = {
        "number": {"pr_number": 102},
        "head": {"head_sha": "0" * 40},
        "unstarted": {"summary": "実行は未着手。結果は未確認。"},
        "declared": {"execution": "verified"},
    }
    if fault == "command":
        pr["body"] = pr["body"].replace(f"bash {SCRIPT}", "")
    elif fault == "result":
        pr["body"] = pr["body"].replace(
            ', "result": "Expected CPU metrics and outputs observed."', ""
        )
    state["prs"] = [pr]
    p.save(state)
    rejected = p.cli(
        "merge-knowledge",
        "--prs",
        "101",
        "--evidence-file",
        p.note(
            "照合済みとだけ書いたメモ。"
            if fault == "freeform"
            else acceptance_evidence(pr, **overrides.get(fault, {}))
        ),
        ok=False,
    )
    assert reason in rejected.stderr
    assert "verification" not in json.loads(p.cli("status").stdout)["tasks"]["10"]
    assert not any("repos/o/r/pulls/101/merge" in call for call in p.calls())


def test_knowledge_continuation_checks_attestation_and_pr_body_at_new_head(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    old = pr_with_paths(p, 101, [RECORD])
    execution_evidence(old)
    state["prs"] = [old]
    p.save(state)
    initial = merge_request(p, old, AGENT_AUTO_MERGE="0")
    updated = pr_with_paths(p, 102, [RECORD, SCRIPT])
    state = p.state()
    pr = state["prs"][0]
    pr["head"]["sha"] = updated["head"]["sha"]
    pr["merge_commit_sha"] = updated["merge_commit_sha"]
    successful_run(p, state, pr)
    p.save(state)
    continuation = ("merge-continue", "--run-id", initial["review_run_id"])
    rejected = p.cli(
        *continuation,
        "--retry-review",
        "1",
        "--instruction-file",
        p.note(acceptance_evidence(pr)),
        ok=False,
    )
    assert "--retry-review requires a formal review run" in rejected.stderr

    for changes, reason in [
        ({"pr_number": 102}, "identify each selected PR"),
        (
            {"head_sha": old["head"]["sha"]},
            "does not match the PR and its current HEAD",
        ),
        ({"summary": "実行は未完了。結果は未確認。"}, "unverified or incomplete work"),
        ({}, "PR execution evidence does not match the current HEAD"),
    ]:
        rejected = p.cli(
            *continuation,
            "--instruction-file",
            p.note(acceptance_evidence(pr, **changes)),
            ok=False,
        )
        assert reason in rejected.stderr
    execution_evidence(pr)
    p.save(state)
    rejected = p.cli(
        *continuation, "--instruction-file", p.note(acceptance_evidence(pr)), ok=False
    )
    assert "bundled run scripts require execution evidence" in rejected.stderr
    pending = json.loads(
        p.cli("merge-status", "--run-id", initial["review_run_id"]).stdout
    )
    assert "evidence_head" not in pending["prs"][0]
    assert (
        pending["prs"][0]["verified_evidence"] == initial["prs"][0]["verified_evidence"]
    )
    assert not any("repos/o/r/pulls/101/merge" in call for call in p.calls())

    execution_evidence(pr, executed=True)
    p.save(state)
    completed = json.loads(
        p.cli(
            *continuation,
            "--instruction-file",
            p.note(acceptance_evidence(pr)),
            "--wait",
        ).stdout
    )
    assert completed["status"] == "completed"
    assert completed["prs"][0]["evidence_head"] == pr["head"]["sha"]
    assert completed["prs"][0]["merge_head"] == pr["head"]["sha"]


@pytest.mark.parametrize(
    ("closing", "refs", "reason"),
    [
        ("Closes #11\nCloses o/r#10", None, "outside the current work targets: [10]"),
        ("Closes #11\nFixes: other/repo#10", None, "another repository"),
        (
            "Closes #11\nResolves https://github.com/o/r/issues/10",
            [["o/r", 11]],
            "body only: [10]",
        ),
        ("Closes #11", [["o/r", 10], ["o/r", 11]], "GitHub only: [10]"),
    ],
)
def test_knowledge_registration_requires_all_github_closing_targets(
    parent_env: ParentEnv, closing: str, refs: list[list[Any]] | None, reason: str
) -> None:
    p = parent_env
    prepared(p, count=2)
    state = p.state()
    del state["parents"]["10"]
    pr = pr_with_paths(p, 101, [RECORD])
    pr["body"] = closing
    if refs is not None:
        pr["closing_refs"] = refs
    execution_evidence(pr)
    state["prs"] = [pr]
    p.save(state)
    rejected = p.cli(
        "merge-knowledge",
        "--prs",
        "101",
        "--evidence-file",
        p.note(acceptance_evidence(pr)),
        ok=False,
    )
    assert reason in rejected.stderr
    assert not any("repos/o/r/pulls/101/merge" in call for call in p.calls())


def test_qualified_closing_target_is_subject_to_human_controls(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    prepared(p, count=2)
    state = p.state()
    pr = pr_with_paths(p, 102, [RECORD])
    pr["body"] = "Closes #11\nCloses o/r#10"
    execution_evidence(pr)
    state["prs"] = [pr]
    state["issues"]["10"]["labels"].append({"name": "needs-human"})
    p.save(state)
    result = merge_request(p, pr)
    assert result["prs"][0]["issues"] == [10, 11]
    assert result["prs"][0]["reason"] == "needs-human: issues/10"
    assert "knowledge_merges" not in p.state()


def test_companion_cannot_close_its_implementation_issue(parent_env: ParentEnv) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    pr = pr_with_paths(p, 101, [RECORD])
    pr["body"] = "Refs #10, #100\nCloses O/R#10"
    execution_evidence(pr)
    state["prs"].append(pr)
    p.save(state)
    rejected = p.cli(
        "merge-knowledge",
        "--prs",
        "101",
        "--evidence-file",
        p.note(acceptance_evidence(pr)),
        ok=False,
    )
    assert "a companion knowledge PR cannot close Issues" in rejected.stderr
    assert not any("repos/o/r/pulls/101/merge" in call for call in p.calls())


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("closing_link", "GitHub only: [10]"),
        ("body_revision", "PR body or execution evidence changed"),
    ],
)
def test_merge_rechecks_pr_metadata_after_registration(
    parent_env: ParentEnv, mutation: str, reason: str
) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    pr = pr_with_paths(p, 101, [RECORD])
    pr["body"] = "Refs #10, #100"
    execution_evidence(pr)
    pr["after_first_closing_read"] = (
        {"closing_refs": [["o/r", 10]]}
        if mutation == "closing_link"
        else {"body": "PR body edited after attestation.\n\n" + pr["body"]}
    )
    state["prs"].append(pr)
    successful_run(p, state, pr)
    p.save(state)
    result = merge_request(p, pr)
    assert result["status"] == "parent_wait"
    assert reason in result["prs"][0]["reason"]
    assert result["prs"][0]["issues"] == []
    assert "merge_head" not in result["prs"][0]
    assert not any("repos/o/r/pulls/101/merge" in call for call in p.calls())


def test_knowledge_registration_rejects_an_undeclared_commit_closure(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    prepared(p)
    state = p.state()
    pr = pr_with_paths(
        p,
        101,
        [RECORD],
        message="Fixture record\n\nResolves: https://github.com/o/r/issues/11",
    )
    execution_evidence(pr)
    state["prs"] = [pr]
    p.save(state)
    rejected = p.cli(
        "merge-knowledge",
        "--prs",
        "101",
        "--evidence-file",
        p.note(acceptance_evidence(pr)),
        ok=False,
    )
    assert (
        "commit or PR title declares a closing target missing from the PR body"
        in rejected.stderr
    )
    assert not any("repos/o/r/pulls/101/merge" in call for call in p.calls())
