"""The public parent CLI and detached process with real git and fake HTTP/agent."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_parent_session import (
    FAKE_PARENT_REVIEWER,
    ParentEnv,
    review_setup,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import (
    parent_env as parent_env,
)


def prepared(p: ParentEnv, *, count: int = 1) -> tuple[list[int], Path, Path]:
    (p.repo / "AGENTS.md").write_text("# Test repository rules\n")
    run_git(p.repo, "add", "AGENTS.md")
    run_git(p.repo, "commit", "-m", "base review rules")
    numbers = review_setup(p, independent=True, pr_count=count)
    state = p.state()
    for pr in state["prs"]:
        pr.update(
            mergeable=True,
            labels=[],
            statusCheckRollup=[
                {
                    "__typename": "CheckRun",
                    "name": "Python",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
        )
    p.save(state)
    fake = p.root / "reviewer"
    fake.write_text(f"#!{sys.executable}\n" + FAKE_PARENT_REVIEWER)
    fake.chmod(0o755)
    output = p.root / "review-output.json"
    output.write_text(
        json.dumps({"verdict": "approve", "summary": "確認済み", "major": []})
    )
    calls = p.root / "review-calls.jsonl"
    calls.touch()
    p.env.update(
        AGENT_AUTO_MERGE="1",
        REVIEW_AGENT_CALLS=str(calls),
        REVIEW_AGENT_OUTPUT=str(output),
    )
    return numbers, fake, calls


def request(
    p: ParentEnv, numbers: list[int], fake: Path, **extra: str
) -> dict[str, Any]:
    result = p.cli(
        "review-request",
        "--prs",
        *map(str, numbers),
        *p.review_flags(),
        "--instruction-file",
        p.note("全体の要件を確認する。"),
        "--evidence-file",
        p.note("差分と受入テストを確認した。"),
        "--agent-exec",
        str(fake),
        "--wait",
        extra=extra,
    )
    return cast(dict[str, Any], json.loads(result.stdout))


@pytest.mark.parametrize("base", ["main", "trunk"])
def test_public_request_detached_review_merge_completion(
    parent_env: ParentEnv, base: str
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    # A regular package can shadow the ops review namespace in CLI processes.
    foreign = p.repo / "review"
    foreign.mkdir()
    (foreign / "__init__.py").write_text(
        "raise RuntimeError('foreign review package loaded')\n"
    )
    if base == "trunk":
        remote = p.state()
        remote["default_branch"] = base
        for pr in remote["prs"]:
            pr["base"]["ref"] = base
            pr["base"]["repo"]["default_branch"] = base
        p.save(remote)
        run_git(p.repo, "branch", "-m", "main", base)
    result = request(p, numbers, fake)
    assert result["status"] == "completed", result
    # No additional parent operation was needed between request and notification.
    assert result["prs"][0]["stage"] == "completed"
    assert len(calls.read_text().splitlines()) == 1
    merged = [c for c in p.calls() if c[:2] == ["pr", "merge"]]
    assert len(merged) == 1 and "--match-head-commit" in merged[0]
    assert p.state()["issues"]["10"]["state_reason"] == "completed"
    before = calls.read_bytes()
    status = json.loads(
        p.cli("merge-status", extra={"CLAUDE_CODE_SESSION_ID": ""}).stdout
    )
    assert status["status"] == "completed"
    assert calls.read_bytes() == before
    assert len([c for c in p.comments(100) if "broad_review" in c["body"]]) == 1


@pytest.mark.parametrize("rereview", [False, True])
def test_public_major_continue_and_history_display(
    parent_env: ParentEnv, rereview: bool
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text(
        json.dumps(
            {
                "verdict": "major",
                "summary": "修正が必要",
                "major": [
                    {"prs": numbers, "title": "誤動作", "detail": "修正して確認する。"}
                ],
            }
        )
    )
    result = request(p, numbers, fake)
    assert result["status"] == "parent_wait"
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())
    Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text(
        json.dumps({"verdict": "approve", "summary": "修正確認済み", "major": []})
    )
    config = p.root / "rereview.conf"
    config.write_bytes(
        (
            Path(__file__).resolve().parents[3] / ".agents/ops/lib/agents.conf"
        ).read_bytes()
    )
    result = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("major修正とテストを確認した。"),
            "--wait",
            *(
                [
                    "--rereview",
                    "--review-model",
                    "gpt-6.1-sol",
                    "--review-config-file",
                    str(config),
                    "--model-reason-file",
                    p.note("修正範囲が明確なためSolを選ぶ。"),
                ]
                if rereview
                else []
            ),
        ).stdout
    )
    assert result["status"] == "completed", result
    assert len(calls.read_text().splitlines()) == (2 if rereview else 1)
    if rereview:
        assert result["review"]["config_file"] == str(config)
        args = json.loads(calls.read_text().splitlines()[-1])["args"]
        assert args[args.index("--config-file") + 1] == str(config)
    inspected = json.loads(p.cli("review-inspect", "--prs", "100").stdout)
    assert inspected["prs"][0]["history"][0]["changes"]["100"]["files"] == 0


@pytest.mark.parametrize("control", ["auto", "human"])
def test_public_human_controls_and_explicit_resume(
    parent_env: ParentEnv, control: str
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    if control == "human":
        state = p.state()
        state["prs"][0]["labels"].append({"name": "needs-human"})
        p.save(state)
    result = request(
        p, numbers, fake, AGENT_AUTO_MERGE="0" if control == "auto" else "1"
    )
    assert result["status"] == "parent_wait", result
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())
    state = p.state()
    state["prs"][0]["labels"] = []
    p.save(state)
    result = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("制御を確認して続行。"),
            "--wait",
        ).stdout
    )
    assert result["status"] == "completed", result
    assert len(calls.read_text().splitlines()) == 1


def measure(p: ParentEnv, *, count: int, peers: int) -> dict[str, Any]:
    from tests.support.agents_ops.merge_gate_reads import (
        add_unrelated_parents,
        api_endpoints,
        endpoint_counts,
    )

    numbers, fake, calls = prepared(p, count=count)
    unrelated = add_unrelated_parents(p, peers)
    offset = len(p.calls())
    result = request(p, numbers, fake)
    assert result["status"] == "completed", result
    endpoints = api_endpoints(
        [
            c
            for c in p.calls()[offset:]
            if "--method" in c and c[c.index("--method") + 1] == "GET"
        ]
    )
    assert not any(e.startswith("repos/o/r/issues?") for e in endpoints)
    assert not any(
        e.partition("?")[0] == f"repos/o/r/issues/{n}"
        or e.startswith(f"repos/o/r/issues/{n}/")
        for n in unrelated
        for e in endpoints
    )
    assert len(calls.read_text().splitlines()) == 1
    return {
        "prs": count,
        "unrelated_issues": len(unrelated),
        "get_count": len(endpoints),
        "endpoint_counts": endpoint_counts(endpoints),
    }


def test_github_get_count_depends_on_targets_not_repository_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.e2e.agents_ops.test_parent_session import parent_env as factory

    measurements = []
    for count, peers in [(1, 0), (1, 30), (3, 30)]:
        root = tmp_path / f"measure-{count}-{peers}"
        root.mkdir()
        with monkeypatch.context() as patch:
            p = factory.__wrapped__(root, patch)
            measurements.append(measure(p, count=count, peers=peers))
    assert measurements[0]["get_count"] == measurements[1]["get_count"]
    assert (
        measurements[1]["get_count"]
        < measurements[2]["get_count"]
        < 3 * measurements[1]["get_count"]
    )
    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
    print(json.dumps(measurements))


@pytest.mark.parametrize("fault", ["runtime_failure", "cleanup", "output"])
def test_failed_or_incomplete_reviewer_does_not_merge(
    parent_env: ParentEnv, fault: str
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    env = {}
    if fault == "runtime_failure":
        env["REVIEW_AGENT_FINAL_EXIT"] = "3"
    elif fault == "cleanup":
        env["REVIEW_UNCONFIRMED_CLEANUP"] = "1"
    else:
        Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text("unstructured answer")
    result = request(p, numbers, fake, **env)
    assert result["status"] == "parent_wait"
    assert result["review"]["stage"] == "parent_wait"
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())
    assert len(calls.read_text().splitlines()) == 1
    repeated = request(p, numbers, fake)
    assert repeated["status"] == "parent_wait"
    assert len(calls.read_text().splitlines()) == 1


def test_effective_review_settings_are_observations(parent_env: ParentEnv) -> None:
    p = parent_env
    numbers, fake, _ = prepared(p)
    result = request(p, numbers, fake, REVIEW_EFFECTIVE_UNAVAILABLE="1")
    assert result["status"] == "completed", result
    assert (
        result["review"]["result"]["runtime"]["effective"]["model"]["status"]
        == "unavailable"
    )


@pytest.mark.parametrize(
    "change",
    [
        "owner",
        "generation",
        "revoked",
        "closing_targets",
        "default_branch",
        "unknown_default",
    ],
)
def test_progression_rechecks_owner_authority_and_target(
    parent_env: ParentEnv, change: str
) -> None:
    p = parent_env
    numbers, fake, _ = prepared(p)
    initial = request(p, numbers, fake, AGENT_AUTO_MERGE="0")
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    Store = importlib.import_module("parent.state").Store

    if change in {"owner", "generation"}:
        with Store(directory).locked() as store:
            row = store.parents["o/r#1"]
            row["session"] = "claude:other" if change == "owner" else row["session"]
            row["generation"] += 1
            store.save()
    elif change == "revoked":
        p.cli("basis-revoke", "--reason-file", p.note("実行根拠を取り消す。"))
    else:
        remote = p.state()
        if change == "closing_targets":
            remote["prs"][0]["body"] += "\nCloses #2"
        elif change == "default_branch":
            remote["prs"][0]["base"]["repo"]["default_branch"] = "trunk"
        else:
            remote["prs"][0]["base"]["repo"].pop("default_branch")
        p.save(remote)
    response = p.cli(
        "merge-continue",
        "--run-id",
        initial["review_run_id"],
        "--instruction-file",
        p.note("続行を検査する。"),
        "--wait",
        ok=None,
    )
    if change in {"closing_targets", "default_branch", "unknown_default"}:
        value = json.loads(response.stdout)
        assert (
            value["prs"][0]["reason"] == "PR has no registered creating Codex worker"
        ), value
    else:
        assert response.returncode == 2
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())


def test_machine_notifications_do_not_claim_parent_verification(
    parent_env: ParentEnv,
) -> None:
    from tests.e2e.agents_ops.test_decision_capture import captured

    p = parent_env
    numbers, fake, _ = prepared(p)
    request(p, numbers, fake)
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    entries = captured(directory)
    results = [
        r
        for r in entries
        if r["kind"] == "result" and "Automatic progression" in r["data"]["summary"]
    ]
    assert results
    assert all(r["data"]["observer"] == "machine" for r in results)


def test_controller_and_pr_leases_prevent_duplicate_effects(tmp_path: Path) -> None:
    controller_lock = importlib.import_module(
        "lib.ops_review.merge_gate_impl"
    ).controller_lock
    progression = importlib.import_module("lib.ops_review.progression")
    new_state, save = progression.new_state, progression.save
    targets = importlib.import_module("lib.ops_review.review_locks").targets

    request_value = {
        "schema_version": 4,
        "review_run_id": "parent-1-" + "1" * 32,
        "repo": "o/r",
        "parent_issue": 1,
        "prs": [{"number": 11}],
    }
    state = new_state(request_value, owner="test", generation=1)
    path = tmp_path / "progressions" / state["id"] / "state.json"
    path.parent.mkdir(parents=True)
    save(path, state)
    with (
        controller_lock(path),
        pytest.raises(ValueError, match="already running"),
        controller_lock(path),
    ):
        pytest.fail("duplicate controller admitted")
    with (
        targets(tmp_path, state),
        pytest.raises(ValueError, match="active"),
        targets(tmp_path, state),
    ):
        pytest.fail("duplicate PR admitted")
    other = {**state, "id": "parent-1-" + "2" * 32}
    with pytest.raises(ValueError, match="unfinished"), targets(tmp_path, other):
        pytest.fail("interrupted lease reused")


def test_interrupted_reviewer_never_relaunches(parent_env: ParentEnv) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    result = request(p, numbers, fake, AGENT_AUTO_MERGE="0")
    directory = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/progressions"
        / result["review_run_id"]
        / "review-1"
    )
    directory.mkdir()
    (directory / "execution.json").write_text(
        json.dumps({"status": "running", "agent_invocations": 1})
    )
    continued = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("中断から再開する。"),
            "--wait",
        ).stdout
    )
    assert "recovery_required" in continued["review"]["error"]
    assert calls.read_text() == ""
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())


def test_private_review_text_is_not_published_or_rereviewed(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text(
        json.dumps(
            {"verdict": "approve", "summary": "Checked " + str(p.repo), "major": []}
        )
    )
    result = request(p, numbers, fake)
    if result["status"] != "completed":
        result = json.loads(
            p.cli(
                "wait-merge",
                "--after",
                str(result["events"][-1]["id"]),
                "--timeout",
                "30",
            ).stdout
        )
    assert result["status"] == "completed", result
    assert result["review"]["result"]["publication_errors"]
    assert not p.comments(100)
    assert len(calls.read_text().splitlines()) == 1
