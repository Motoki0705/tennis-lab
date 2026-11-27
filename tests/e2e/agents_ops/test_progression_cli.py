"""The public parent CLI and detached process with real git and fake HTTP/agent."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import subprocess
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
from tests.support.agents_ops.capacity_admission import late_capacity_runner


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


def progression_path(p: ParentEnv, value: dict[str, Any]) -> Path:
    return (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/progressions"
        / str(value["review_run_id"])
        / "state.json"
    )


def pending_python() -> dict[str, Any]:
    return {
        "__typename": "CheckRun",
        "name": "Python",
        "status": "IN_PROGRESS",
        "conclusion": None,
    }


def complete_pending_python(p: ParentEnv) -> dict[str, str]:
    hook = p.root / "ci-clock"
    hook.mkdir()
    # Advance the fake check at the real controller's poll boundary. Saving the
    # state first lets callers verify what was persisted while CI was pending.
    (hook / "sitecustomize.py").write_text("""\
import json
import os
import sys
from pathlib import Path

if Path(sys.argv[0]).name == "merge_gate.py":
    from lib.ops_review import merge_gate_impl

    class EngineWithCIClock(merge_gate_impl.Engine):
        def __init__(self, path, backend, *, state):
            self.now = 1000.0
            super().__init__(path, backend, state=state, clock=lambda: self.now,
                             sleep=self.advance)

        def advance(self, delay):
            with self.path.with_name("pending-ci.json").open("xb") as record:
                record.write(self.path.read_bytes())
            remote = Path(os.environ["PARENT_GH_STATE"])
            state = json.loads(remote.read_text())
            (pr,) = state["prs"]
            (check,) = pr["statusCheckRollup"]
            assert check["status"] == "IN_PROGRESS"
            check.update(status="COMPLETED", conclusion="SUCCESS")
            remote.write_text(json.dumps(state))
            self.now += delay

    merge_gate_impl.Engine = EngineWithCIClock
""")
    return {"PYTHONPATH": os.pathsep.join((str(hook), p.env["PYTHONPATH"]))}


def join_controller(state_file: Path) -> None:
    with state_file.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)


def interrupt_review(p: ParentEnv, state_file: Path, *, when: str) -> None:
    # Run the production controller with a process exit at one atomic write.
    entry = """
import os
import sys
from lib.ops_review import merge_gate_impl, parent_review_impl

save = parent_review_impl.save
boundary = sys.argv[2]

def interrupt(path, value):
    match = path.parent.name == 'review-1' and path.name == boundary
    if match and boundary == 'result.json':
        os._exit(75)
    save(path, value)
    if match and boundary == 'execution.json':
        os._exit(75)

parent_review_impl.save = interrupt
sys.argv[1:] = ['--state', sys.argv[1]]
sys.exit(merge_gate_impl.main())
"""
    process = subprocess.run(
        [sys.executable, "-c", entry, str(state_file), when],
        cwd=p.repo,
        env=p.env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 75, process.stdout + process.stderr


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


def test_ready_request_launches_review_while_python_ci_is_pending(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    remote = p.state()
    (pr,) = remote["prs"]
    pr["statusCheckRollup"] = [pending_python()]
    p.save(remote)
    head = pr["head"]["sha"]
    inspected = json.loads(p.cli("review-inspect", "--prs", str(numbers[0])).stdout)
    assert inspected["review_candidates"] == numbers
    assert inspected["prs"][0]["draft"] is False
    assert "excluded_reason" not in inspected["prs"][0]

    result = request(p, numbers, fake, **complete_pending_python(p))
    assert result["status"] == "completed", result
    (line,) = calls.read_text().splitlines()
    launched = json.loads(line)
    assert launched["args"][launched["args"].index("--mode") + 1] == "read-only"
    assert launched["context"]["prs"][0]["pr"]["statusCheckRollup"] == [
        pending_python()
    ]
    assert launched["context"]["prs"][0]["pr"]["head"]["sha"] == head
    assert "change-0.txt" in launched["files"][str(numbers[0])]
    assert "+new 0" in launched["diffs"][str(numbers[0])]

    waiting = json.loads(
        progression_path(p, result).with_name("pending-ci.json").read_text()
    )
    assert waiting["review"]["stage"] == "accepted"
    target = waiting["targets"][str(numbers[0])]
    assert target["stage"] == "ci_wait"
    assert target["ci"]["status"] == "pending"
    assert target["ci_head"] == head and not target["attempts"]
    (merged,) = result["prs"]
    assert merged["stage"] == "completed" and merged["merge_head"] == head


@pytest.mark.parametrize(
    ("draft", "reason", "message"),
    [
        (True, "draft", "Mark it ready for review"),
        (None, "draft_status_unknown", "draft status is unavailable"),
    ],
)
def test_review_inspect_and_request_explain_draft_exclusion(
    parent_env: ParentEnv, draft: bool | None, reason: str, message: str
) -> None:
    p = parent_env
    (number,), fake, calls = prepared(p)
    remote = p.state()
    remote["prs"][0]["draft"] = draft
    p.save(remote)
    inspected = json.loads(p.cli("review-inspect", "--prs", str(number)).stdout)
    assert inspected["review_candidates"] == []
    (item,) = inspected["prs"]
    assert item["draft"] is draft
    assert item["excluded_reason"] == reason

    rejected = p.cli(
        "review-request",
        "--prs",
        str(number),
        *p.review_flags(),
        "--instruction",
        "受入条件を確認する。",
        "--evidence-file",
        p.note(),
        "--agent-exec",
        str(fake),
        ok=False,
    )
    assert rejected.returncode == 2
    assert f"PR #{number}" in rejected.stderr and message in rejected.stderr
    assert calls.read_text() == ""
    assert p.state()["prs"][0]["draft"] is draft
    assert p.state()["issues"]["10"]["body"] == remote["issues"]["10"]["body"]


def test_public_major_continue_and_history_display(parent_env: ParentEnv) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text(
        json.dumps(
            {
                "verdict": "major",
                "summary": "修正が必要",
                "major": [
                    {
                        "prs": [999, *numbers, 999],
                        "title": "誤動作",
                        "detail": "修正して確認する。",
                    }
                ],
            }
        )
    )
    result = request(p, numbers, fake)
    assert result["status"] == "parent_wait"
    finding = result["review"]["result"]["major"][0]
    assert finding["prs"] == numbers and finding["related_prs"] == [999]
    saved = json.loads(
        (progression_path(p, result).parent / "review-1/result.json").read_text()
    )
    assert saved["major"] == [finding]
    comment = p.comments(numbers[0])[0]["body"]
    assert f"対象 PR #{numbers[0]}。参照 PR #999。" in comment
    assert p.comments(999) == []
    assert not any(c[:2] == ["pr", "merge"] for c in p.calls())
    Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text(
        json.dumps({"verdict": "approve", "summary": "修正確認済み", "major": []})
    )
    rejected = p.cli(
        "merge-continue", "--rereview", "--instruction-file", p.note("再依頼"), ok=False
    )
    assert "unrecognized arguments" in rejected.stderr
    result = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("major修正とテストを確認した。"),
            "--wait",
        ).stdout
    )
    assert result["status"] == "completed", result
    assert len(calls.read_text().splitlines()) == 1
    inspected = json.loads(p.cli("review-inspect", "--prs", "100").stdout)
    assert inspected["prs"][0]["excluded_reason"] == "not_open"
    assert inspected["prs"][0]["history"][0]["changes"]["100"]["files"] == 0


@pytest.mark.parametrize("lease_order", ["already_waiting", "after_admission"])
def test_partial_continue_reports_a_sibling_pr_lease_conflict(
    parent_env: ParentEnv, lease_order: str
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p, count=2)
    first, peer = numbers
    remote = p.state()
    for pr in remote["prs"]:
        pr["mergeable"] = False  # Reviewable, but parked at the merge gate.
    p.save(remote)
    requested = request(p, numbers, fake)
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    state_file = directory / "progressions" / requested["review_run_id"] / "state.json"
    paused = json.loads(p.cli("wait-merge", "--after", "1", "--timeout", "10").stdout)
    with state_file.with_suffix(".lock").open("a") as controller:
        fcntl.flock(controller, fcntl.LOCK_EX)  # join the controller
    assert {pr["stage"] for pr in paused["prs"]} == {"parent_wait"}
    args = (
        "merge-continue",
        "--prs",
        str(first),
        "--wait",
        "--instruction-file",
        p.note("先に一方を続ける。"),
    )
    if lease_order == "already_waiting":
        registry = importlib.import_module("parent.state").Store(directory)
        with registry.locked():
            row = registry.parents["o/r#1"]
            task = row["tasks"]["11"]
            task.update(
                delegation_id="b" * 32,
                manual_followup={
                    "delegation_id": "b" * 32,
                    "state": "waiting",
                    "wait": {"kind": "ci", "pr": peer},
                    "owner": row["session"],
                    "generation": row["generation"],
                },
            )
            registry.save()
        lease = importlib.import_module("lib.ops_review.review_locks").pr_lease
        with lease(directory, "o/r", peer):
            result = p.cli(*args, ok=False)
        assert "#11: worker follow-up is active" in result.stderr
        assert json.loads(state_file.read_text())["decisions"] == []
    else:
        # The ordinary CLI admits the continuation. Interpose only at launch
        # so the sibling's lease is acquired after saving "running".
        entry = """
import subprocess
import sys
from pathlib import Path
from lib.ops_review import merge_gate_impl as gate, review_locks
from parent.cli_impl import main

def start(path):
    assert gate.read(path)["status"] == "running"
    with review_locks.pr_lease(path.parents[2], "o/r", int(sys.argv[1])):
        subprocess.run(
            [sys.executable, str(Path(gate.__file__).with_name("merge_gate.py")), "--state", str(path)],
            capture_output=True, check=False, timeout=5,
        )

gate.start = start
sys.exit(main(sys.argv[2:]))
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                entry,
                str(peer),
                "--repo",
                "o/r",
                "--parent",
                "1",
                *args,
            ],
            cwd=p.repo,
            env=p.env,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        )
        observed = json.loads(result.stdout)
        assert observed["status"] == "parent_wait"
        event = observed["events"][-1]
        assert event["id"] == paused["events"][-1]["id"] + 1
        assert event["kind"] == "parent_wait" and event["pr"] == peer
        assert "active progression" in event["reason"]
        assert observed["prs"][0]["stage"] == "parent_wait"
        assert observed["prs"][1] == paused["prs"][1]
        # Once the lease is released, the same partial continuation can run.
        retried = json.loads(p.cli(*args).stdout)
        assert retried["events"][-1]["pr"] == first
    assert len(calls.read_text().splitlines()) == 1
    assert not any(call[:2] == ["pr", "merge"] for call in p.calls())


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


@pytest.mark.parametrize(
    "fault", ["runtime_failure", "cleanup", "output", "launched_capacity_exit"]
)
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
    elif fault == "launched_capacity_exit":
        env.update(
            REVIEW_AGENT_FINAL_EXIT="75", REVIEW_RUNTIME_REASON="capacity_blocked"
        )
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
    if fault == "launched_capacity_exit":
        continued = json.loads(
            p.cli(
                "merge-continue",
                "--instruction-file",
                p.note("空き容量が戻った。起動済みのレビューは照合する。"),
                "--wait",
            ).stdout
        )
        assert continued["status"] == "parent_wait"
        assert continued["review_usage"]["confirmed_launches"] == 1
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


def test_capacity_refusal_preserves_the_formal_review_invocation(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    runner, low = late_capacity_runner(p.root, fake)
    refused = request(p, numbers, runner)
    assert refused["status"] == "parent_wait"
    review_dir = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/progressions"
        / refused["review_run_id"]
        / "review-1"
    )
    record = json.loads((review_dir / "agent-runtime/result.json").read_text())
    assert record["agent_spawned"] is False
    assert record["turn_end_reason"] == "capacity_blocked"
    assert calls.read_text() == ""
    assert (
        json.loads((review_dir / "execution.json").read_text())["agent_invocations"]
        == 0
    )

    low.unlink()
    continued = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("空き容量が戻った。同じレビューを続ける。"),
            "--wait",
        ).stdout
    )
    assert continued["status"] == "completed", continued
    assert continued["review_run_id"] == refused["review_run_id"]
    assert continued["review_usage"]["launch_intents"] == 2
    assert continued["review_usage"]["confirmed_launches"] == 1
    assert len(calls.read_text().splitlines()) == 1
    assert (
        json.loads((review_dir / "execution.json").read_text())["agent_invocations"]
        == 1
    )
    assert json.loads((review_dir / "agent-runtime/result.json").read_text()) == record


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


def test_saved_terminal_review_recovers_after_controller_error(
    parent_env: ParentEnv,
) -> None:
    p = parent_env
    numbers, fake, calls = prepared(p)
    Path(p.env["REVIEW_AGENT_OUTPUT"]).write_text(
        json.dumps(
            {
                "verdict": "major",
                "summary": "fix",
                "major": [{"prs": numbers, "title": "problem", "detail": "verify fix"}],
            }
        )
    )
    result = request(p, numbers, fake, AGENT_AUTO_MERGE="0")
    state_file = progression_path(p, result)
    join_controller(state_file)
    interrupt_review(p, state_file, when="result.json")
    review_dir = state_file.parent / "review-1"
    assert (
        json.loads((review_dir / "execution.json").read_text())["status"] == "running"
    )
    runtime = json.loads((review_dir / "agent-runtime/result.json").read_text())
    assert runtime["result"] == "completed" and runtime["cleanup_complete"]
    assert not (review_dir / "result.json").exists()
    assert p.comments(numbers[0]) == []
    retry = p.cli(
        "merge-continue",
        "--run-id",
        result["review_run_id"],
        "--retry-review",
        "1",
        "--instruction-file",
        p.note("recover"),
        ok=False,
    )
    assert "review runtime completed" in retry.stderr
    recovered = json.loads(
        p.cli(
            "merge-continue", "--instruction-file", p.note("recover"), "--wait"
        ).stdout
    )
    assert recovered["review"]["result"]["verdict"] == "major"
    assert (
        json.loads((review_dir / "result.json").read_text())
        == recovered["review"]["result"]
    )
    assert len(p.comments(numbers[0])) == 1
    assert len(calls.read_text().splitlines()) == 1
    assert "merge" not in [c[1] for c in p.calls() if c[:1] == ["pr"]]
    join_controller(state_file)
    adopted_state = state_file.read_bytes()
    before_adoption = json.loads(adopted_state)
    before_adoption["review"] = {"round": 1, "stage": "running"}
    state_file.write_text(json.dumps(before_adoption))
    rejected = p.cli(
        "merge-continue",
        "--run-id",
        result["review_run_id"],
        "--retry-review",
        "1",
        "--instruction-file",
        p.note("recovery after saving the result"),
        ok=False,
    )
    assert "without a saved result" in rejected.stderr
    state_file.write_bytes(adopted_state)
    final = json.loads(
        p.cli(
            "merge-continue",
            "--instruction-file",
            p.note("fixed and verified"),
            "--wait",
        ).stdout
    )
    assert final["status"] == "completed"
    replay = request(p, numbers, fake)
    assert replay["review_run_id"] == result["review_run_id"]
    assert len(calls.read_text().splitlines()) == 1
