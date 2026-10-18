"""Parallel audit provenance, fail-closed integration, partial publication and retry."""

from __future__ import annotations

import copy
import importlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops/ops-report"))
audit: Any = importlib.import_module("opsreport.audit")
agent: Any = importlib.import_module("opsreport.agent")
BASE = "a" * 40


@pytest.fixture
def scenario(agent_output: dict[str, Any]) -> dict[str, Any]:
    data = agent_output
    data["assignments"] = [
        {
            "id": key,
            "agent_id": f"native-{key}",
            "agent_type": "scout",
            "base_sha": BASE,
            "started_at": f"2026-10-18T00:00:0{i}+00:00",
            "finished_at": f"2026-10-18T01:00:0{i}+00:00",
            "status": "completed",
            "conclusion": f"{key}の調査結果",
            "evidence": ["src/model.py:12 / #50"],
            "counterevidence": [],
            "unconfirmed": [],
            "findings": [],
        }
        for i, key in enumerate(audit.AREAS)
    ]
    data["decisions"] = []
    for i, prop in enumerate(data["proposals"]):
        key = "code" if i == 0 else "governance"
        topic = f"problem-{i}"
        prop["sources"] = [f"{key}:{topic}"]
        next(r for r in data["assignments"] if r["id"] == key)["findings"].append(
            {
                "topic": topic,
                "position": "support",
                "conclusion": prop["title"],
                "evidence": prop["evidence"],
            }
        )
        data["decisions"].append(
            {
                "sources": prop["sources"],
                "outcome": "propose",
                "reason": "根拠を確認した",
                "evidence": prop["evidence"],
            }
        )
    return data


def _integrate(data: dict[str, Any], **kwargs: Any) -> Any:
    return audit.integrate(json.dumps(data), {"base_sha": BASE}, {7}, **kwargs)


def _reports(ops: Any) -> list[dict[str, Any]]:
    return [r for r in ops.state["issues"] if "ops-report" in r["labels"]]


def _state_path(ops: Any, name: str) -> Path:
    return Path(ops.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report" / name


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _latest(ops: Any) -> Path:
    paths: list[Path] = ops.run_dirs()
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def test_complete_result_links_every_proposal_to_scout_and_base(
    scenario: dict[str, Any],
) -> None:
    result = _integrate(scenario)
    assert result.complete and result.errors == []
    assert set(result.results) == set(audit.AREAS)
    assert "code:problem-0" in result.analysis.proposals[0].evidence
    assert BASE in result.analysis.proposals[0].evidence
    assert len(result.decisions) == 2


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["assignments"].pop(0),
        lambda d: d["assignments"].append(copy.deepcopy(d["assignments"][0])),
        lambda d: d["assignments"][0].update(base_sha="b" * 40),
        lambda d: d["assignments"][0].update(agent_type="default"),
        lambda d: d["assignments"][0].update(started_at="2027-01-01T00:00:00+00:00"),
        lambda d: d["assignments"][0].update(started_at="2026-10-18T00:00:00"),
        lambda d: d["assignments"][0].update(unconfirmed=["未調査"]),
        lambda d: d["assignments"][0].pop("counterevidence"),
        lambda d: d["assignments"][0]["findings"].append(
            copy.deepcopy(d["assignments"][0]["findings"][0])
        ),
        lambda d: d["assignments"].append({"id": []}),
        lambda d: d["assignments"].append(123),
        lambda d: d["assignments"][0].update(status=[]),
        lambda d: d["assignments"][1].update(agent_id="native-code"),
        lambda d: d["decisions"].pop(0),
        lambda d: d["decisions"].append(copy.deepcopy(d["decisions"][0])),
        lambda d: d["proposals"][0].update(sources=["unknown:problem"]),
        lambda d: d["proposals"][1].update(title=d["proposals"][0]["title"]),
    ],
)
def test_invalid_return_never_becomes_complete(
    scenario: dict[str, Any], mutate: Any
) -> None:
    mutate(scenario)
    result = _integrate(scenario)
    assert not result.complete and result.errors
    assert result.analysis.memory_update is None
    assert "governance" in result.results  # valid independent conclusions survive


def test_duplicate_and_conflicting_findings_are_integrated_explicitly(
    scenario: dict[str, Any],
) -> None:
    copied = copy.deepcopy(scenario["assignments"][0]["findings"][0])
    scenario["assignments"][1]["findings"] = [copied]
    scenario["decisions"][0]["sources"].append("platform:problem-0")
    scenario["decisions"][0]["outcome"] = "duplicate"
    result = _integrate(scenario)
    assert result.complete
    assert len(result.analysis.proposals) == 2
    assert "platform:problem-0" in result.analysis.proposals[0].evidence

    copied["position"] = "oppose"
    result = _integrate(scenario)
    assert not result.complete and "contradictory" in " ".join(result.errors)
    assert result.analysis.proposals == []

    scenario["decisions"][0]["outcome"] = "conflict"
    scenario["proposals"].pop(0)
    result = _integrate(scenario)
    assert not result.complete and "未解決の矛盾" in " ".join(result.errors)
    assert len(result.analysis.proposals) == 1  # independent confirmed proposal

    scenario["decisions"][0]["outcome"] = "resolved"
    # Resolve by the parent's evidence, then create exactly one proposal.
    prop = {
        **scenario["proposals"][0],
        "title": "矛盾を確認して統合",
        "sources": scenario["decisions"][0]["sources"],
    }
    scenario["proposals"].append(prop)
    assert _integrate(scenario).complete


def test_serial_results_are_incomplete_and_retry_requests_two_scouts(
    scenario: dict[str, Any],
) -> None:
    for i, row in enumerate(scenario["assignments"]):
        row.update(
            started_at=f"2026-10-18T0{i}:00:00+00:00",
            finished_at=f"2026-10-18T0{i}:30:00+00:00",
        )
    result = _integrate(scenario)
    assert not result.complete
    assert audit.requested(result.results) == ["code", "platform"]
    retry = copy.deepcopy(scenario)
    retry["assignments"] = retry["assignments"][:2]
    for row in retry["assignments"]:
        row.update(
            started_at="2026-10-19T00:00:00+00:00",
            finished_at="2026-10-19T01:00:00+00:00",
        )
    assert _integrate(retry, previous=result.results).complete


def test_no_whole_audit_elapsed_time_ceiling(
    scenario: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for row in scenario["assignments"]:
        row["finished_at"] = "2027-01-01T00:00:00+00:00"
    assert _integrate(scenario).complete
    calls = []
    out = tmp_path / "out"
    out.write_text("result")
    monkeypatch.setattr(
        agent.shell, "run_process_group", lambda cmd, **kw: calls.append((cmd, kw))
    )
    agent.execute(
        agent.AgentRun(
            "codex", tmp_path / "exec", tmp_path, tmp_path / "prompt", out, tmp_path
        )
    )
    assert calls[0][1]["timeout"] is None
    assert calls[0][0][calls[0][0].index("--profile") + 1] == "ops_report"
    unit = (
        Path(__file__).resolve().parents[3]
        / ".agents/ops/ops-report/systemd/tennis-lab-ops-report.service"
    )
    assert "TimeoutStartSec=infinity" in unit.read_text()


def test_duplicate_json_keys_are_not_silently_accepted(
    scenario: dict[str, Any],
) -> None:
    text = json.dumps(scenario).replace(
        '"agent_type": "scout"', '"agent_type": "scout", "agent_type": "default"', 1
    )
    result = audit.integrate(text, {"base_sha": BASE}, {7})
    assert not result.complete and result.results == {}
    assert "duplicate JSON key" in " ".join(result.errors)


def test_fake_parallel_run_records_overlap_and_full_inventory(ops_env: Any) -> None:
    result = ops_env.run("report", "--dry-run")
    assert result.returncode == 0, result.stderr
    run = _latest(ops_env)
    events = [
        json.loads(line)
        for line in (run / "fake_scout_events.jsonl").read_text().splitlines()
    ]
    first_end = next(i for i, event in enumerate(events) if event["event"] == "finish")
    assert sum(event["event"] == "start" for event in events[:first_end]) >= 2
    record = _json(run / "audit.json")
    assert record["complete"] and len(record["results"]) == 4
    plan = _json(run / "audit_plan.json")
    paths = [p for a in plan["assignments"] for p in a["targets"]]
    tracked = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "main"], cwd=ops_env.repo, text=True
    ).splitlines()
    assert sorted(paths) == sorted(tracked) and len(paths) == len(set(paths))
    collected = _json(run / "collected.json")
    assert not Path(collected["audit_runtime"]["checkout"]).exists()
    assert "status=complete" in result.stdout


def test_partial_publication_resumes_same_issue_with_missing_scout_only(
    ops_env: Any,
) -> None:
    failed = ops_env.run("report", FAKE_AUDIT_FAIL="platform")
    assert failed.returncode == 1, failed.stderr
    partial = _reports(ops_env)[0]
    assert partial["body"].splitlines()[1].startswith("**未完了**")
    assert "status=incomplete" in partial["body"].splitlines()[0]
    status = _json(_state_path(ops_env, "schedule.json"))
    assert (
        status["last_success"] is None and status["last_attempt"]["status"] == "failed"
    )
    assert not _state_path(ops_env, "memory.json").exists()
    initial = _json(_latest(ops_env) / "collected.json")
    assert "platform" in partial["body"] and "再開" in partial["body"]

    # The report remains pinned even when the source branch and time advance.
    (ops_env.repo / "big.py").write_text("new main content\n")
    subprocess.run(["git", "add", "big.py"], cwd=ops_env.repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "advance main",
        ],
        cwd=ops_env.repo,
        check=True,
    )
    resumed = ops_env.run("report", "--if-due", "--now", "2026-10-16T09:00:00+09:00")
    assert resumed.returncode == 0, resumed.stderr
    assert (
        len(_reports(ops_env)) == 1
        and _reports(ops_env)[0]["number"] == partial["number"]
    )
    assert "status=complete" in _reports(ops_env)[0]["body"].splitlines()[0]
    current = _json(_latest(ops_env) / "collected.json")
    assert current["audit_runtime"]["requested"] == ["platform"]
    assert current["meta"]["base_sha"] == initial["meta"]["base_sha"]
    assert current["audit_input_sha256"] == initial["audit_input_sha256"]
    assert current["meta"]["period_end"] == initial["meta"]["period_end"]
    assert (
        _json(_state_path(ops_env, "schedule.json"))["last_success"]["slot"]
        == "2026-10-12T08:00:00+09:00"
    )
    assert sum(c[:2] == ["issue", "create"] for c in ops_env.calls()) == 1
    assert sum(c[:2] == ["issue", "edit"] for c in ops_env.calls()) == 1
    following = ops_env.run("report", "--if-due", "--now", "2026-10-16T09:00:00+09:00")
    assert following.returncode == 0, following.stderr
    assert len(_reports(ops_env)) == 2
    next_input = _json(_latest(ops_env) / "collected.json")
    assert next_input["meta"]["period_start"] == initial["meta"]["period_end"]
    assert next_input["meta"]["base_sha"] != initial["meta"]["base_sha"]


@pytest.mark.parametrize(
    "problem", ["missing", "corrupt", "input", "base", "selected", "closed"]
)
def test_resume_inconsistency_fails_closed(ops_env: Any, problem: str) -> None:
    assert ops_env.run("report", FAKE_AUDIT_FAIL="platform").returncode == 1
    checkpoint = _state_path(ops_env, "audit/20261012.json")
    if problem == "missing":
        checkpoint.unlink()
    elif problem == "corrupt":
        checkpoint.write_text("{")
    elif problem == "input":
        data = _json(checkpoint)
        data["collected"]["meta"]["base_sha"] = "f" * 40
        checkpoint.write_text(json.dumps(data))
    elif problem == "base":
        state = ops_env.state
        _reports_from_state = next(
            r for r in state["issues"] if "ops-report" in r["labels"]
        )
        _reports_from_state["body"] = _reports_from_state["body"].replace(
            "base=", "base=f"
        )
        ops_env.write_state(state)
    else:
        state = ops_env.state
        issue = next(r for r in state["issues"] if "ops-report" in r["labels"])
        if problem == "closed":
            issue["state"] = "CLOSED"
        else:
            issue["body"] = issue["body"].replace("- [ ] **", "- [x] **", 1)
        ops_env.write_state(state)
    before = len(ops_env.calls())
    result = ops_env.run("report", "--if-due")
    assert result.returncode == 1
    assert not any(
        c[:2] in (["issue", "create"], ["issue", "edit"], ["pr", "create"])
        for c in ops_env.calls()[before:]
    )
    assert len(_reports(ops_env)) == 1


def test_incomplete_report_is_not_triaged(ops_env: Any) -> None:
    assert ops_env.run("report", FAKE_AUDIT_FAIL="platform").returncode == 1
    state = ops_env.state
    issue = next(r for r in state["issues"] if "ops-report" in r["labels"])
    issue["body"] = issue["body"].replace("- [ ] **", "- [x] **", 1)
    ops_env.write_state(state)
    before = len(ops_env.calls())
    result = ops_env.run("triage")
    assert result.returncode == 0, result.stderr
    assert not any(c[:2] == ["issue", "create"] for c in ops_env.calls()[before:])


def test_pinned_checkout_and_external_gitlink_scope(ops_env: Any) -> None:
    repo = ops_env.repo
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    subprocess.run(
        [
            "git",
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{base},third_party/external",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-qm",
            "pin external dependency",
        ],
        cwd=repo,
        check=True,
    )
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    (repo / "big.py").write_text("dirty caller\n")
    inventory = audit.plan(repo, base)
    assert inventory["gitlinks"][0]["path"] == "third_party/external"
    path = audit.worktree_path(repo, "test-pinned")
    with audit.checkout(repo, base, path) as snapshot:
        assert (snapshot / "big.py").read_text().startswith("x = 1")
        assert not (snapshot / "third_party/external/big.py").exists()
    assert not path.exists()
    assert (repo / "big.py").read_text() == "dirty caller\n"


def test_faulty_runner_dirty_checkout_is_preserved_for_explicit_repair(
    ops_env: Any,
) -> None:
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ops_env.repo, text=True
    ).strip()
    path = audit.worktree_path(ops_env.repo, "test-dirty")
    with (
        pytest.raises(audit.shell.CommandError, match="contains modified or untracked"),
        audit.checkout(ops_env.repo, base, path) as snapshot,
    ):
        (snapshot / "big.py").write_text("unexpected write\n")
    assert (path / "big.py").read_text() == "unexpected write\n"
    assert (ops_env.repo / "big.py").read_text().startswith("x = 1")


@pytest.mark.parametrize("stop_signal", [signal.SIGINT, signal.SIGTERM])
def test_explicit_interrupt_kills_descendants_and_publishes_incomplete(
    ops_env: Any,
    stop_signal: int,
) -> None:
    pid_file = ops_env.tmp / "child.pid"
    proc = ops_env.start("report", FAKE_AGENT_HANG="1", FAKE_AGENT_SPAWN=str(pid_file))
    try:
        deadline = time.monotonic() + 15
        while (
            not pid_file.exists()
            and proc.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert pid_file.exists(), proc.communicate(timeout=5)
        time.sleep(1.1)
        assert proc.poll() is None
        proc.send_signal(stop_signal)
        _, err = proc.communicate(timeout=20)
        assert proc.returncode == 1, err
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    child = int(pid_file.read_text())
    stat = Path(f"/proc/{child}/stat")
    assert not stat.exists() or stat.read_text().split(") ", 1)[1][0] == "Z"
    assert "KeyboardInterrupt" in _reports(ops_env)[0]["body"]
    assert _json(_state_path(ops_env, "schedule.json"))["last_success"] is None
    assert "ops-report-audit-" not in subprocess.check_output(
        ["git", "worktree", "list"], cwd=ops_env.repo, text=True
    )
