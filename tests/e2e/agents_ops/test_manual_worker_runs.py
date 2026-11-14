"""Manual worker admission, runner provenance and private registry persistence."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_human_agreement import delegation_args
from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.directory_seed import DirectorySeed

ENTRY = """
import sys
from pathlib import Path
from parent import manual_workers
manual_workers.RUNNER = Path(sys.argv[1])
from parent.cli import main
sys.exit(main(sys.argv[2:]))
"""
MANUAL_SEED = pytest.StashKey[DirectorySeed]()
WORKER_PROMPT = (
    Path(__file__).resolve().parents[3] / ".agents/ops/parent/worker_prompt.md"
)


def state_dir(p: ParentEnv) -> Path:
    return Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"


def store(p: ParentEnv) -> Any:
    return importlib.import_module("parent.state").Store(state_dir(p))


def task(p: ParentEnv) -> dict[str, Any]:
    return dict(store(p).read_data()["parents"]["o/r#1"]["tasks"]["10"])


def issue_body(p: ParentEnv, number: int) -> str:
    return str(p.state()["issues"][str(number)]["body"])


def update(p: ParentEnv, **values: Any) -> None:
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["10"].update(values)
        registry.save()


def manual(
    p: ParentEnv, *args: str, ok: bool = True, runner: Path | None = None, **extra: str
) -> subprocess.CompletedProcess[str]:
    if runner is None:
        runner = p.root / "fake-runner"
        runner.write_text(
            f"#!{sys.executable}\n"
            + Path(__file__).with_name("fake_manual_worker_runner.py").read_text()
        )
        runner.chmod(0o700)
    response = subprocess.run(
        [
            sys.executable,
            "-c",
            ENTRY,
            str(runner),
            "--repo",
            "o/r",
            "--parent",
            "1",
            *args,
        ],
        cwd=p.repo,
        env={
            **p.env,
            "MANUAL_STATE_DIR": str(state_dir(p)),
            "MANUAL_RUNNER_CALLS": str(p.root / "runner-calls.jsonl"),
            "MANUAL_RUNNER_PROMPTS": str(p.root / "runner-prompts.jsonl"),
            **extra,
        },
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert (response.returncode == 0) == ok, response.stderr + response.stdout
    return response


def run(
    p: ParentEnv, *extra: str, ok: bool = True, **env: str
) -> subprocess.CompletedProcess[str]:
    return manual(
        p,
        "run-worker",
        "--child",
        "10",
        "--prompt-file",
        p.note(),
        *extra,
        ok=ok,
        runner=None,
        **env,
    )


def latest(p: ParentEnv) -> Path:
    return Path(state_dir(p), task(p)["worker_runs"][-1]["run_dir"])


def ended(p: ParentEnv) -> None:
    p.cli(
        "record-worker-ended",
        "--child",
        "10",
        "--delegation-id",
        task(p)["delegation_id"],
        "--evidence-file",
        p.note("runnerと全子孫の終了を照合した。"),
    )


def planning_answer(
    p: ParentEnv, action: str = "approve", text: str = "提案の範囲で進める。"
) -> subprocess.CompletedProcess[str]:
    return manual(
        p,
        "decide-worker-plan",
        "--child",
        "10",
        "--action",
        action,
        "--answer-file",
        p.note(text),
    )


def resume_answer(
    p: ParentEnv, *extra: str, ok: bool = True
) -> subprocess.CompletedProcess[str]:
    return manual(
        p,
        "run-worker",
        "--child",
        "10",
        "--resume",
        "--restart-evidence-file",
        p.note("終了と同じsessionを確認した。"),
        *extra,
        ok=ok,
    )


@pytest.fixture
def reserved(parent_env: ParentEnv, request: pytest.FixtureRequest) -> ParentEnv:
    p = parent_env
    if MANUAL_SEED not in request.config.stash:
        p.agreed()
        p.child("worker")
        p.cli(*delegation_args(p.worktree("worker")))
        cache = request.getfixturevalue("tmp_path_factory").mktemp("manual-worker-seed")
        request.config.stash[MANUAL_SEED] = DirectorySeed.capture(
            p.root, cache / "snapshot"
        )
    else:
        request.config.stash[MANUAL_SEED].restore(p.root)
        run_git(p.repo, "worktree", "repair")
    return p


@pytest.mark.parametrize("version,model", [(1, None), (2, "gpt-6-astra")])
def test_launch_is_durable_private_and_preserves_registry(
    reserved: ParentEnv, version: int, model: str | None
) -> None:
    p = reserved
    assert task(p)["worker_runs"] == []
    for command in ("status", "reconcile"):
        assert "worker_runs" not in p.cli(command).stdout
    previous = {
        "delegation_id": "f" * 32,
        "run_dir": str(p.root / "private-old-run"),
        "kind": "registered",
        "recorded_at": "2026-10-29T00:00:00+00:00",
    }
    update(p, extension={"kept": [1, 2]}, worker_runs=[previous])
    before = copy.deepcopy(task(p))
    if version == 2:
        p.cli("state-migrate")
    extra: tuple[str, ...] = ()
    reason = "未解決の幾何アルゴリズムを設計するため。"
    if model:
        # A parent must supply a reason before an alternate can be admitted.
        registry = store(p).read_data()
        assert "reason" in run(p, "--model", model, ok=False).stderr
        assert store(p).read_data() == registry
        assert not (p.root / "runner-calls.jsonl").exists()
        reason_file = p.root / "model-reason.md"
        reason_file.write_text(reason)
        extra = ("--model", model, "--model-reason-file", str(reason_file))
    result = run(p, *extra)
    after = task(p)
    assert {
        k: v
        for k, v in after.items()
        if k
        not in {"worker_runs", "model", "model_reason", "preflight", "worker_ended"}
    } == {k: v for k, v in before.items() if k not in {"worker_runs", "preflight"}}
    assert after["preflight"]["phase"] == "implementing"
    assert after["worker_runs"][-1]["preflight"]["phase"] == "implementing"
    [kept, record] = after["worker_runs"]
    assert kept == previous
    assert set(record) == {
        "delegation_id",
        "run_dir",
        "kind",
        "recorded_at",
        "model",
        "model_reason",
        "preflight",
    }
    expected = model or "gpt-6.1-sol"
    assert after["model"] == record["model"] == expected
    assert after["model_reason"] == record["model_reason"]
    if model:
        assert after["model_reason"] == reason
    calls = json.loads((p.root / "runner-calls.jsonl").read_text())
    assert calls[calls.index("--model") + 1] == expected
    result_record = json.loads((latest(p) / "result.json").read_text())
    settings = result_record["contract"]["settings"]
    assert (
        settings["selected"]["model"]
        == result_record["runtime"]["effective"]["model"]["value"]
        == expected
    )
    assert settings["selected"]["effort"] == "max"
    assert settings["selection_sources"]["model"] == "task"
    assert Path(settings["source"]).name == "agents.conf" and settings["sha256"]
    collector = importlib.import_module("worker_status.collector")
    observed, availability = collector.latest_run(after, {}, 10, state_dir(p), False)
    assert availability == "recorded" and observed["directory"] == latest(p)
    assert (
        record["delegation_id"] == before["delegation_id"]
        and record["kind"] == "initial"
    )
    assert not Path(record["run_dir"]).is_absolute()
    assert datetime.fromisoformat(record["recorded_at"]).utcoffset() == timedelta(0)
    summary = json.loads(result.stdout)
    assert summary["runner_summary"]["result"] == "completed"
    assert summary["exit_code"] == 0
    assert store(p).read_data()["version"] == version
    for command in ("status", "reconcile", "sync"):
        output = p.cli(command).stdout
        assert "worker_runs" not in output and record["run_dir"] not in output
        assert reason not in output
    assert record["run_dir"] not in json.dumps(p.state())
    assert json.loads(result.stdout).get("final_file") == str(latest(p) / "final.md")
    # A second initial launch must not create a new session for this reservation.
    assert "resume" in run(p, ok=False).stderr
    assert len(task(p)["worker_runs"]) == 2


def test_run_paths_in_evidence_are_private_across_redelegation(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p)
    record = task(p)["worker_runs"][0]
    sibling = "worker-runs/sibling-private-run"
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["11"] = {
            "worker_runs": [{**record, "run_dir": sibling}]
        }
        registry.save()
    evidence = (
        f"Inspected {latest(p)}, {record['run_dir']} and {state_dir(p) / sibling}."
    )
    update(p, restart_evidence=evidence, extension={"private_evidence": [evidence]})
    for command in ("status", "reconcile"):
        output = p.cli(command).stdout
        assert record["run_dir"] not in output and "[private run]" in output
        assert sibling not in output
    assert task(p)["restart_evidence"] == evidence
    ended(p)
    response = p.cli(
        *delegation_args(Path(task(p)["worktree"])),
        "--restart-evidence-file",
        p.note(evidence),
    )
    assert "worker_runs" not in response.stdout
    assert record["run_dir"] not in response.stdout
    assert sibling not in response.stdout
    current = task(p)
    assert current["worker_runs"] == [record]
    assert "model" not in current and "model_reason" not in current
    assert current["delegation_id"] != record["delegation_id"]
    assert not any(
        run["delegation_id"] == current["delegation_id"]
        for run in current["worker_runs"]
    )
    # A new delegation can select a different model; its old run keeps its own.
    run(p, "--model", "gpt-6-astra", "--model-reason-file", p.note("統合設計のため。"))
    assert task(p)["model"] == "gpt-6-astra"
    assert [r["model"] for r in task(p)["worker_runs"]] == [
        "gpt-6.1-sol",
        "gpt-6-astra",
    ]
    for command in ("status", "reconcile"):
        assert "worker_runs" not in p.cli(command).stdout
    before = p.state()
    assert "private" in p.cli("note", "--body-file", p.note(evidence), ok=False).stderr
    assert p.state() == before


def test_legacy_reservation_gains_run_tracking_only_when_redelegated(
    reserved: ParentEnv,
) -> None:
    p = reserved
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["10"].pop("worker_runs", None)
        registry.save()
    legacy = task(p)
    assert "worker_runs" not in legacy
    for command in ("status", "reconcile"):
        assert "worker_runs" not in p.cli(command).stdout
    assert task(p) == legacy
    ended(p)
    response = p.cli(
        *delegation_args(Path(legacy["worktree"])),
        "--restart-evidence-file",
        p.note("旧委任の終了を確認し、run記録方式で再委任する。"),
    )
    current = task(p)
    assert current["worker_runs"] == []
    assert current["delegation_id"] != legacy["delegation_id"]
    assert "worker_runs" not in response.stdout


@pytest.mark.parametrize(
    "variant",
    [
        "missing",
        "other_owner",
        "inactive",
        "ended",
        "agent",
        "legacy",
        "branch",
        "stopped",
        "managed",
    ],
)
def test_launch_rejects_invalid_reservation_before_runner(
    reserved: ParentEnv, variant: str
) -> None:
    p = reserved
    if variant == "missing":
        update(p, delegated=False)
    elif variant == "other_owner":
        p.env["CLAUDE_CODE_SESSION_ID"] = "some-other-parent"
    elif variant == "inactive":
        p.cli("handoff", "--note-file", p.note())
    elif variant == "ended":
        ended(p)
    elif variant == "agent":
        update(p, agent="claude")
    elif variant == "legacy":
        update(p, delegation_id="legacy")
    elif variant == "branch":
        update(p, branch="codex/wrong")
    elif variant == "stopped":
        p.cli("stop", "--child", "10", "--spec", p.detail())
    elif variant == "requirements":
        gh = p.state()
        gh["issues"]["10"]["body"] = gh["issues"]["10"]["body"].replace(
            "動作する", "変更された動作"
        )
        p.save(gh)
    elif variant == "managed":
        with store(p).locked() as registry:
            registry.parents["o/r#1"]["dispatch"] = {"managed": True}
            registry.save()
    before = store(p).read_data()
    run(p, ok=False)
    assert not (p.root / "runner-calls.jsonl").exists()
    assert store(p).read_data() == before


def test_resume_reacquires_same_delegation_slot_with_evidence(
    reserved: ParentEnv,
) -> None:
    p = reserved
    p.cli(
        "configure-delegation",
        "--max-parallel-children",
        "1",
        "--reason-file",
        p.note(),
    )
    run(p)
    bodies = [issue_body(p, 10)]
    assert len(task(p)["preflight"]["turns"]) == 1
    assert task(p)["preflight"]["phase"] == "implementing"
    first = latest(p)
    original = task(p)["delegation_id"]
    ended(p)
    p.child("other")
    p.cli(*delegation_args(p.worktree("other"), child=11))
    assert "limit" in run(p, "--resume", ok=False).stderr
    assert (
        "limit"
        in run(p, "--resume", "--restart-evidence-file", p.note(), ok=False).stderr
    )
    assert len(task(p)["worker_runs"]) == 1 and "worker_ended" in task(p)
    with store(p).locked() as registry:
        other = registry.parents["o/r#1"]["tasks"]["11"]
    p.cli(
        "record-worker-ended",
        "--child",
        "11",
        "--delegation-id",
        other["delegation_id"],
        "--evidence-file",
        p.note(),
    )
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, branch="codex/worker")]
    p.save(gh)
    result = run(p, "--resume", "--restart-evidence-file", p.note())
    bodies.append(issue_body(p, 10))
    assert json.loads(result.stdout)["exit_code"] == 0
    assert task(p)["delegation_id"] == original and "worker_ended" in task(p)
    assert [r["kind"] for r in task(p)["worker_runs"]] == ["initial", "resume"]
    calls = [
        json.loads(line)
        for line in (p.root / "runner-calls.jsonl").read_text().splitlines()
    ]
    assert calls[-1][:3] == ["resume", "--resume-from", str(first)]
    # Both turns get the managed workers' standard rules, not just the parent's
    # file: e.g. report in the final message instead of parent-issue comments.
    prompts = [
        json.loads(line)
        for line in (p.root / "runner-prompts.jsonl").read_text().splitlines()
    ]
    assert len(prompts) == len(bodies) == 2
    for entry, body in zip(prompts, bodies, strict=True):
        assert entry["prompt_file"] == "-"
        text = entry["prompt"]
        assert text.startswith(WORKER_PROMPT.read_text(encoding="utf-8"))
        assert "親issue: #1\n担当issue: #10\n" in text
        assert f"branch: {task(p)['branch']}\n" in text
        assert f"## 担当issueの原文\n\n{body}\n" in text
        assert text.endswith("## 親の具体的な指示\n\n確認済みの証拠・次の操作。\n")
    assert (
        json.loads((first / "result.json").read_text())["runtime"]
        == json.loads((latest(p) / "result.json").read_text())["runtime"]
    )
    plan = json.loads(p.cli("status").stdout)["launch_plan"]
    assert plan["active_reservations"] == []


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6-astra"])
def test_resume_uses_latest_run_without_implicitly_releasing_slot(
    reserved: ParentEnv,
    model: str,
) -> None:
    p = reserved
    run(
        p,
        "--model",
        model,
        "--model-reason-file",
        p.note("親が難易度から選択した。"),
    )
    first = latest(p)
    # Simulate a pre-selection registry. The saved runner contract is authoritative.
    with store(p).locked() as registry:
        saved = registry.parents["o/r#1"]["tasks"]["10"]
        saved.pop("model")
        saved.pop("model_reason")
        saved["worker_runs"][-1].pop("model")
        saved["worker_runs"][-1].pop("model_reason")
        registry.save()
    run(p, "--resume")
    second = latest(p)
    run(p, "--resume", "--model", model)
    assert task(p)["model"] == model
    calls = [
        json.loads(line)
        for line in (p.root / "runner-calls.jsonl").read_text().splitlines()
    ]
    assert calls[-1][:3] == ["resume", "--resume-from", str(second)]
    for call in calls:
        assert call[call.index("--model") + 1] == model
    assert (
        json.loads((first / "result.json").read_text())["runtime"]["session_id"]
        == json.loads((latest(p) / "result.json").read_text())["runtime"]["session_id"]
    )
    opposite = "gpt-6-astra" if model == "gpt-6.1-sol" else "gpt-6.1-sol"
    before = store(p).read_data()
    before_calls = (p.root / "runner-calls.jsonl").read_bytes()
    assert "cannot change" in run(p, "--resume", "--model", opposite, ok=False).stderr
    assert store(p).read_data() == before
    assert (p.root / "runner-calls.jsonl").read_bytes() == before_calls
    assert task(p)["worker_runs"][-1]["kind"] == "resume"
    assert "worker_ended" in task(p)
    result = latest(p) / "result.json"
    record = json.loads(result.read_text())
    record["result"] = "running"
    result.write_text(json.dumps(record))
    assert "termination" in run(p, "--resume", ok=False).stderr
    assert len(task(p)["worker_runs"]) == 3
    assert first != latest(p)


@pytest.mark.parametrize("mode", ["running", "ended", "invocation_only"])
def test_register_existing_run_is_idempotent_and_keeps_external_paths_private(
    reserved: ParentEnv, mode: str
) -> None:
    p = reserved
    path = p.root / "private-existing-run"
    path.mkdir()
    contract = {
        "task_id": task(p)["delegation_id"],
        "cwd": task(p)["worktree"],
        "branch": task(p)["branch"],
        "mode": "write",
        "settings": importlib.import_module("lib.agent_settings").resolve(
            Path(__file__).resolve().parents[3] / ".agents/ops/lib/agents.conf",
            "codex",
            None,
            "gpt-6-astra" if mode == "invocation_only" else None,
            None,
            "write",
        ),
    }
    (path / "invocation.json").write_text(
        json.dumps({"schema_version": 1, "contract": contract})
    )
    if mode != "invocation_only":
        (path / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "contract": contract,
                    "result": "running" if mode == "running" else "completed",
                    "run_id": path.name,
                    "session_root": str(path),
                    "turn_end_reason": "completed",
                    "exit_code": 0,
                    "task_success": None,
                    "cleanup_complete": True,
                    "runtime": {"session_id": "offline-private-session"},
                }
            )
        )
        (path / "session.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "session_id": "offline-private-session",
                    "latest_run": str(path),
                }
            )
        )
    if mode == "ended":
        ended(p)
    args = ("record-worker-run", "--child", "10", "--run-dir", str(path))
    before = copy.deepcopy(task(p))
    assert before.pop("worker_runs") == []
    output = manual(p, *args)
    manual(p, *args)
    [record] = task(p)["worker_runs"]
    assert record["run_dir"] == str(path) and record["kind"] == "registered"
    assert {
        k: v
        for k, v in task(p).items()
        if k not in {"worker_runs", "model", "model_reason"}
    } == before
    assert (
        task(p)["model"]
        == record["model"]
        == ("gpt-6-astra" if mode == "invocation_only" else "gpt-6.1-sol")
    )
    assert task(p)["model_reason"] is record["model_reason"] is None
    assert str(path) not in output.stdout + output.stderr + p.cli("status").stdout
    assert str(path) not in json.dumps(p.state())
    if mode == "ended":
        before_registry = store(p).read_data()
        assert (
            "cannot change"
            in run(
                p,
                "--resume",
                "--model",
                "gpt-6-astra",
                "--restart-evidence-file",
                p.note(),
                ok=False,
            ).stderr
        )
        assert (
            store(p).read_data() == before_registry
        )  # The released slot stays released.
        run(
            p, "--resume", "--model", "gpt-6.1-sol", "--restart-evidence-file", p.note()
        )
        calls = json.loads((p.root / "runner-calls.jsonl").read_text())
        assert (
            "--model" not in calls
        )  # Preserve the legacy default's override provenance.
    # A settings-less legacy record must not be guessed as today's default.
    broken = p.root / "legacy-unknown-model"
    broken.mkdir()
    unknown = copy.deepcopy(contract)
    unknown.pop("settings")
    (broken / "invocation.json").write_text(
        json.dumps({"schema_version": 1, "contract": unknown})
    )
    before_registry = store(p).read_data()
    manual(p, "record-worker-run", "--child", "10", "--run-dir", str(broken), ok=False)
    assert store(p).read_data() == before_registry


@pytest.mark.parametrize(
    "field",
    ["task_id", "cwd", "branch", "conflicting_invocation", "malformed", "missing"],
)
def test_registration_rejects_wrong_contract_without_saving(
    reserved: ParentEnv, field: str
) -> None:
    p = reserved
    run(p)
    path = latest(p)
    record_path = path / "result.json"
    record = json.loads(record_path.read_text())
    if field in {"task_id", "cwd", "branch"}:
        record["contract"][field] = "wrong"
        record_path.write_text(json.dumps(record))
    elif field == "conflicting_invocation":
        invocation = json.loads((path / "invocation.json").read_text())
        invocation["contract"]["task_id"] = "wrong"
        (path / "invocation.json").write_text(json.dumps(invocation))
    elif field == "malformed":
        record_path.write_text('{"schema_version":1,"schema_version":1}')
    else:
        path = p.root / "no-record"
    before = store(p).read_data()
    result = manual(
        p, "record-worker-run", "--child", "10", "--run-dir", str(path), ok=False
    )
    assert str(path) not in result.stdout + result.stderr
    assert store(p).read_data() == before


@pytest.mark.parametrize("preflight", [False, True])
def test_runner_failure_preserves_record_and_exit_code(
    reserved: ParentEnv, preflight: bool
) -> None:
    p = reserved
    result = run(
        p,
        ok=False,
        **(
            {"MANUAL_PREFLIGHT_FAILURE": "1"}
            if preflight
            else {"MANUAL_RUNNER_EXIT": "7"}
        ),
    )
    output = json.loads(result.stdout)
    assert output["exit_code"] == result.returncode == (2 if preflight else 7)
    assert (output["runner_summary"] is None) == preflight
    if preflight:
        assert task(p)["worker_runs"] == []
        assert output["launch_state"] == "rejected_before_start"
        assert len(task(p)["launch_failures"]) == 1
        assert task(p)["preflight"]["turns"][-1]["cancelled"]
    else:
        assert len(task(p)["worker_runs"]) == 1 and "worker_ended" in task(p)
        assert output["final_file"] == str(latest(p) / "final.md")


def test_record_result_rejects_cancelled_only_history(reserved: ParentEnv) -> None:
    p = reserved
    run(p)
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, branch="codex/worker")]
    p.save(gh)
    # A retained runtime must not act as a silent fallback when its instruction
    # is absent from the non-cancelled history.
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["10"]["preflight"]["turns"][0][
            "cancelled"
        ] = True
        registry.save()
    before = store(p).read_data()
    body = issue_body(p, 10)
    rejected = manual(
        p,
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("保存済みの古いruntimeを成果に流用しない。"),
        ok=False,
    )
    assert "no non-cancelled worker turn" in rejected.stderr
    assert store(p).read_data() == before and issue_body(p, 10) == body


@pytest.mark.parametrize(
    "mode", ["missing", "mismatch", "latest_running", "earlier_running", "completed"]
)
def test_review_request_uses_current_registry_run(
    reserved: ParentEnv, mode: str
) -> None:
    p = reserved
    tree = Path(task(p)["worktree"])
    (tree / "feature.txt").write_text("implemented\n")
    run_git(tree, "add", "feature.txt")
    run_git(tree, "commit", "-m", "worker implementation")
    if mode != "missing":
        run(p)
    if mode in {"missing", "mismatch"}:
        ended(p)  # A manual end marker alone must not bypass runner inspection.
    if mode in {"latest_running", "earlier_running"}:
        first = latest(p)
        run(p, "--resume")
    if mode in {"mismatch", "latest_running", "earlier_running"}:
        path = (first if mode == "earlier_running" else latest(p)) / "result.json"
        record = json.loads(path.read_text())
        if mode == "mismatch":
            record["contract"]["task_id"] = "b" * 32
        else:
            record["result"] = "running"
        path.write_text(json.dumps(record))
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, run_git(tree, "rev-parse", "HEAD"), "codex/worker")]
    p.save(gh)
    response = manual(
        p,
        "review-request",
        *p.review_flags(),
        "--prs",
        "100",
        "--instruction-file",
        p.note(),
        "--evidence-file",
        p.note(),
        ok=mode == "completed",
    )
    if mode == "completed":
        assert json.loads(response.stdout)["review_state"] == "instructed"
        assert (
            json.loads(p.cli("status").stdout)["launch_plan"]["active_reservations"]
            == []
        )
        assert str(latest(p)) not in response.stdout + json.dumps(p.state())
    else:
        assert not (state_dir(p) / "reviews").exists()


@pytest.mark.parametrize("option", ["--effort", "--worker-runs"])
def test_removed_or_unsupported_overrides_are_not_cli_options(
    parent_env: ParentEnv, option: str
) -> None:
    command = "review-request" if option == "--worker-runs" else "run-worker"
    required = (
        ("--prs", "100", "--instruction-file", parent_env.note())
        if command == "review-request"
        else ("--child", "10", "--prompt-file", parent_env.note())
    )
    result = parent_env.cli(command, *required, option, "override", ok=False)
    assert result.returncode == 2
    assert f"unrecognized arguments: {option}" in result.stderr


@pytest.mark.parametrize("action", ["approve", "investigate"])
def test_preflight_decision_paths(reserved: ParentEnv, action: str) -> None:
    p = reserved
    headings = (
        "確認した根拠",
        "指示との不一致",
        "実装案",
        "テスト設計",
        "制約・未確認事項",
        "親に決めてほしい点",
    )
    mismatch = (
        "親の旧指示は [skip ci]、現行規則と矛盾。"
        if action == "approve"
        else "saved_train_trajectoriesの実データ形式を現行loaderが拒否。"
    )
    contents = (
        "READMEと既存loaderを確認。",
        mismatch,
        "判断後に最小変更。",
        "既存unitを拡張。",
        "GPU予算未指定、起動しない。",
        "指示・データの不一致をどう扱うか。",
    )
    body = "\n\n".join(
        f"## {h}\n\n{c}" for h, c in zip(headings, contents, strict=True)
    )
    run(p, "--planning", MANUAL_PLANNING_FINAL=body)
    first = latest(p)
    answer = (
        "現行CI規則を使い、提案の範囲で実装する。"
        if action == "approve"
        else "実データのkeyとshapeを再確認し、再提出する。"
    )
    decision_id = json.loads(planning_answer(p, action, answer).stdout)["decision_id"]
    assert (
        json.loads(planning_answer(p, action, answer).stdout)["decision_id"]
        == decision_id
    )
    assert "worker_ended" in task(p)
    extra = p.root / "additional.md"
    extra.write_bytes("\n対象外の学習は起動しない。\r\n".encode())
    resume_answer(p, "--additional-instruction-file", str(extra))
    current = task(p)
    assert current["preflight"]["phase"] == (
        "implementing" if action == "approve" else "planning"
    )
    assert current["worker_runs"][-1]["preflight"]["decision_id"] == decision_id
    assert (
        current["preflight"]["turns"][-1]["snapshot"]["previous_turn_id"]
        == current["preflight"]["turns"][0]["snapshot"]["turn_id"]
    )
    prompts = [
        json.loads(line)["prompt"]
        for line in (p.root / "runner-prompts.jsonl").read_text().splitlines()
    ]
    assert answer in prompts[-1] and "## 親の追加指示" in prompts[-1]
    assert len(prompts) == 2
    log_rows = [
        json.loads(Path(t["log"]["journal"]).read_text())
        for t in current["preflight"]["turns"]
    ]
    assert log_rows[0]["source"]["public_key"] == log_rows[1]["source"]["public_key"]
    assert log_rows[1]["files"]["instruction.md"] == answer
    assert "## 親の計画判断" not in log_rows[1]["files"]["instruction.md"]
    meta = json.loads(log_rows[1]["files"]["meta.json"])
    assert meta["instruction"]["additional_instructions"] == [
        "\n対象外の学習は起動しない。\r\n"
    ]
    assert meta["turn"]["phase"] == current["preflight"]["phase"]
    assert meta["worker"]["selected"] == {"model": "gpt-6.1-sol", "effort": "max"}
    assert meta["decision"] == {"action": action, "source_sequence": 1}
    assert (
        json.loads((first / "result.json").read_text())["runtime"]
        == json.loads((latest(p) / "result.json").read_text())["runtime"]
    )


def test_parent_can_verify_artifacts_independent_of_planning_phase(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p, "--planning")
    tree = Path(task(p)["worktree"])
    (tree / "feature.txt").write_text("implementation\n")
    run_git(tree, "add", "feature.txt")
    run_git(tree, "commit", "-m", "implementation")
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, run_git(tree, "rev-parse", "HEAD"), "codex/worker")]
    p.save(gh)
    item = json.loads(p.cli("reconcile").stdout)["tasks"][0]
    assert not item["artifacts_accepted"]
    manual(
        p,
        "record-result",
        "--child",
        "10",
        "--evidence-file",
        p.note("実差分とテストを親が確認した。"),
    )
    assert json.loads(p.cli("reconcile").stdout)["tasks"][0]["artifacts_accepted"]
    manual(p, "review-prepare", *p.review_flags(), "--prs", "100")
    assert len(task(p)["worker_runs"]) == 1


def test_readonly_status_preserves_approved_worker_and_dependency(
    reserved: ParentEnv,
) -> None:
    p = reserved
    p.child("downstream", deps=[10])
    run(p, "--planning")
    planning_answer(p)
    resume_answer(p)
    tree = Path(task(p)["worktree"])
    (tree / "feature.txt").write_text("approved implementation\n")
    run_git(tree, "add", "feature.txt")
    run_git(tree, "commit", "-m", "approved implementation")
    gh = p.state()
    gh["prs"] = [p.pr(100, 10, run_git(tree, "rev-parse", "HEAD"), "codex/worker")]
    p.save(gh)
    ended(p)
    before = store(p).read_data()
    for identity in ("", "other-viewer"):
        response = p.cli(
            "status", extra={"CLAUDE_CODE_SESSION_ID": identity, "CODEX_SESSION_ID": ""}
        )
        visible = json.loads(response.stdout)
        assert not visible["current_session_is_owner"]
        assert visible["tasks"]["10"]["worker_phase"]["phase"] == "implementing"
        downstream = next(
            item for item in visible["launch_plan"]["tasks"] if item["child"] == 11
        )
        assert "dependency_planning_unapproved" not in {
            reason["code"] for reason in downstream["reasons"]
        }
        assert downstream["stacked_on"][0]["pr"] == 100
        assert store(p).read_data() == before


def test_input_change_is_observed_without_forced_replanning(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p, "--planning")
    planning_answer(p)
    resume_answer(p)
    first = task(p)["delegation_id"]
    data = p.state()
    data["issues"]["10"]["body"] += "\n\n## 説明\n順序を整理した。\n"
    p.save(data)
    status = json.loads(p.cli("status").stdout)["tasks"]["10"]["worker_phase"]
    assert status["inputs_changed"] and status["phase"] == "implementing"
    run(p, "--resume")
    assert task(p)["delegation_id"] == first
    assert len(task(p)["worker_runs"]) == 3


@pytest.mark.parametrize("started", [False, True])
def test_legacy_phase_records_are_optional_for_continuation(
    reserved: ParentEnv, started: bool
) -> None:
    p = reserved
    if started:
        run(p)
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["10"].pop("preflight")
        registry.save()
    before = store(p).read_data()
    p.cli("status")
    assert store(p).read_data() == before
    run(p, *(["--resume"] if started else []))
    assert task(p)["preflight"]["decisions"] == []
