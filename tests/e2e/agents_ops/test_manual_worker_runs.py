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


def state_dir(p: ParentEnv) -> Path:
    return Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"


def store(p: ParentEnv) -> Any:
    return importlib.import_module("parent.state").Store(state_dir(p))


def task(p: ParentEnv) -> dict[str, Any]:
    return dict(store(p).read_data()["parents"]["o/r#1"]["tasks"]["10"])


def update(p: ParentEnv, **values: Any) -> None:
    with store(p).locked() as registry:
        registry.parents["o/r#1"]["tasks"]["10"].update(values)
        registry.save()


def manual(
    p: ParentEnv, *args: str, ok: bool = True, **extra: str
) -> subprocess.CompletedProcess[str]:
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


@pytest.mark.parametrize("version", [1, 2])
def test_launch_is_durable_private_and_preserves_registry(
    reserved: ParentEnv, version: int
) -> None:
    p = reserved
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
    result = run(p)
    after = task(p)
    assert {k: v for k, v in after.items() if k != "worker_runs"} == {
        k: v for k, v in before.items() if k != "worker_runs"
    }
    [kept, record] = after["worker_runs"]
    assert kept == previous
    assert set(record) == {"delegation_id", "run_dir", "kind", "recorded_at"}
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
    assert record["run_dir"] not in json.dumps(p.state())
    assert str(latest(p)) not in result.stdout + result.stderr
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
    assert record["run_dir"] not in response.stdout
    assert sibling not in response.stdout
    assert task(p)["worker_runs"] == [record]
    assert task(p)["delegation_id"] != record["delegation_id"]
    before = p.state()
    assert "private" in p.cli("note", "--body-file", p.note(evidence), ok=False).stderr
    assert p.state() == before


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
        "requirements",
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
    first = latest(p)
    original = task(p)["delegation_id"]
    ended(p)
    p.child("other")
    p.cli(*delegation_args(p.worktree("other"), child=11))
    assert "evidence" in run(p, "--resume", ok=False).stderr
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
    assert json.loads(result.stdout)["exit_code"] == 0
    assert task(p)["delegation_id"] == original and "worker_ended" not in task(p)
    assert [r["kind"] for r in task(p)["worker_runs"]] == ["initial", "resume"]
    calls = [
        json.loads(line)
        for line in (p.root / "runner-calls.jsonl").read_text().splitlines()
    ]
    assert calls[-1][:3] == ["resume", "--resume-from", str(first)]
    assert (
        json.loads((first / "result.json").read_text())["runtime"]
        == json.loads((latest(p) / "result.json").read_text())["runtime"]
    )
    plan = json.loads(p.cli("status").stdout)["launch_plan"]
    assert plan["active_reservations"] == [10]


def test_resume_uses_latest_run_without_implicitly_releasing_slot(
    reserved: ParentEnv,
) -> None:
    p = reserved
    run(p)
    first = latest(p)
    run(p, "--resume")
    assert task(p)["worker_runs"][-1]["kind"] == "resume"
    assert "worker_ended" not in task(p)
    result = latest(p) / "result.json"
    record = json.loads(result.read_text())
    record["result"] = "running"
    result.write_text(json.dumps(record))
    assert "termination" in run(p, "--resume", ok=False).stderr
    assert len(task(p)["worker_runs"]) == 2
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
                }
            )
        )
    if mode == "ended":
        ended(p)
    args = ("record-worker-run", "--child", "10", "--run-dir", str(path))
    before = copy.deepcopy(task(p))
    output = manual(p, *args)
    manual(p, *args)
    [record] = task(p)["worker_runs"]
    assert record["run_dir"] == str(path) and record["kind"] == "registered"
    assert {k: v for k, v in task(p).items() if k != "worker_runs"} == before
    assert str(path) not in output.stdout + output.stderr + p.cli("status").stdout
    assert str(path) not in json.dumps(p.state())


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
    assert len(task(p)["worker_runs"]) == 1 and "worker_ended" not in task(p)
    assert str(latest(p)) not in result.stdout + result.stderr


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


@pytest.mark.parametrize("option", ["--model", "--effort", "--worker-runs"])
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
