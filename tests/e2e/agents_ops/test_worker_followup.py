"""Offline CLI contracts for manual workers and their detached controller.

Reuse the real registry, manual runner boundary, and fake GitHub from
test_manual_worker_runs. The fake service launches the production controller
with a readiness socket. Queue evidence uses the published on-disk format.
Socket handshakes exercise the pending and duplicate-launch cases without
fixed sleeps in the tests. Measure setup and cases with --durations=0.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_manual_worker_runs import (
    latest,
    manual,
    run,
    state_dir,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_wait_worker import command as wait_command
from tests.e2e.agents_ops.test_wait_worker import connected, listening

HEAD = "a" * 40
DONE = 'PARENT_RESULT {"schema_version":1,"state":"done","wait":null}'
FAKE_SERVICE = r"""
import json, os, socket, subprocess, sys, tempfile
from pathlib import Path
args = sys.argv[1:]
assert all(flag in args for flag in ("--user", "--collect", "--service-type=notify"))
command = args[args.index("--") + 1:]
assert command[1:3] == ["-m", "parent.worker_monitor"]
root = Path(os.environ["FAKE_SERVICE_ROOT"])
with (root / "service-starts.jsonl").open("a") as f:
    f.write(json.dumps(command) + "\n")
env = dict(os.environ)
for flag in args:
    if flag.startswith("--setenv=") and "=" in flag.removeprefix("--setenv="):
        name, _, value = flag.removeprefix("--setenv=").partition("=")
        if name == "PYTHONPATH" and env.get(name):
            value += os.pathsep + env[name]
        env[name] = value
with tempfile.TemporaryDirectory(prefix="followup-notify-") as short:
    address = str(Path(short) / "notify.sock")
    env["NOTIFY_SOCKET"] = address
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as ready, (root / "controller.log").open("ab") as log:
        ready.bind(address)
        ready.settimeout(10)
        process = subprocess.Popen(
            [command[0], str(root / "fake-service-entry.py"), *command[3:]],
            env=env, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        )
        assert ready.recv(1024) == b"READY=1"
"""
FAKE_SERVICE_ENTRY = r"""
import os, socket, sys
from pathlib import Path
from issues.transport import HTTPClient, HTTPResponse
from parent import manual_workers, worker_monitor

manual_workers.RUNNER = Path(os.environ["FAKE_SERVICE_ROOT"]) / "fake-runner"
worker_monitor.POLL_SECONDS = 0.01
def download_log(self, url):
    assert url == "https://ci.invalid/fake-log"
    return HTTPResponse(200, {}, "compile start\nERROR: regression in fixture step")
HTTPClient.download_log = download_log
original = worker_monitor.queue_result
def queue_result(root, job):
    result = original(root, job)
    if result is None and (address := os.environ.pop("FAKE_QUEUE_PENDING", None)):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.connect(address)
            channel.sendall(job.encode())
            assert channel.recv(1) == b"C"
    return result
worker_monitor.queue_result = queue_result
sys.exit(worker_monitor.main())
"""


@pytest.fixture
def service(reserved: ParentEnv) -> ParentEnv:
    p = reserved
    path = p.root / "bin/systemd-run"
    path.write_text(f"#!{sys.executable}\n" + FAKE_SERVICE)
    path.chmod(0o755)
    (p.root / "fake-service-entry.py").write_text(FAKE_SERVICE_ENTRY)
    p.env.update(
        FAKE_SERVICE_ROOT=str(p.root), TRAINING_QUEUE_DIR=str(p.root / "queue")
    )
    return p


def waiting(kind: str, **fields: Any) -> str:
    return "報告本文\n\nPARENT_RESULT " + json.dumps(
        {"schema_version": 1, "state": "waiting", "wait": {"kind": kind, **fields}},
        ensure_ascii=False,
    )


def launch(p: ParentEnv, *reports: str, ok: bool = True) -> dict[str, Any]:
    result = run(p, ok=ok, MANUAL_FINAL_SEQUENCE=json.dumps(reports))
    return cast(dict[str, Any], json.loads(result.stdout))


def joined(p: ParentEnv, *, ok: bool = True) -> dict[str, Any]:
    result = manual(p, "wait-worker", "--child", "10", "--timeout", "15", ok=ok)
    assert result.returncode == (0 if ok else 2), result.stderr + result.stdout
    return cast(dict[str, Any], json.loads(result.stdout))


def runner_calls(p: ParentEnv) -> list[list[str]]:
    return [
        json.loads(line)
        for line in (p.root / "runner-calls.jsonl").read_text().splitlines()
    ]


def assert_same_worker(p: ParentEnv, count: int) -> None:
    runs = task(p)["worker_runs"]
    calls = runner_calls(p)
    assert len(runs) == len(calls) == count
    sessions = set()
    for i, (run_entry, args) in enumerate(zip(runs, calls, strict=True)):
        folder = state_dir(p) / run_entry["run_dir"]
        sessions.add(
            json.loads((folder / "result.json").read_text())["runtime"]["session_id"]
        )
        if i:
            assert args[:3] == [
                "resume",
                "--resume-from",
                str(state_dir(p) / runs[i - 1]["run_dir"]),
            ]
    assert len(sessions) == 1


def set_ci(p: ParentEnv, conclusion: str) -> None:
    state = p.state()
    pr = p.pr(123, 10, head=HEAD, branch="codex/worker")
    pr["statusCheckRollup"] = [
        {
            "__typename": "CheckRun",
            "name": "Python",
            "status": "COMPLETED",
            "conclusion": conclusion,
        }
    ]
    pr["labels"] = []
    p.env["MANUAL_PR_AFTER_START"] = json.dumps(pr)
    state["fake_ci_log_ids"] = ["456"]
    state["failed_ci_jobs"] = {
        "123": [
            {
                "id": 456,
                "name": "CPU tests",
                "html_url": "https://github.com/o/r/actions/runs/123/job/456",
                "steps": [{"name": "Test", "conclusion": "failure"}],
                "conclusion": "failure",
            }
        ]
    }
    p.save(state)


def set_job(p: ParentEnv, job: str, status: str, *, code: int = 0) -> None:
    root = Path(p.env["TRAINING_QUEUE_DIR"])
    path = root / status / job
    path.parent.mkdir(parents=True, exist_ok=True)
    if status == "running":
        path.touch()
        return
    initial = root / "running" / job
    if initial.exists():
        initial.rename(path)
    else:
        path.touch()
    stem = job.removesuffix(".job")
    repro = root / "repro" / stem
    repro.mkdir(parents=True, exist_ok=True)
    (repro / "output_dir.txt").write_text(f"outputs/{stem}\n")
    log = root / "logs" / f"{stem}.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text(f"exit_code={code}\n")
    terminal = root / "control/completions" / f"{stem}.json"
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.write_text(
        json.dumps(
            {
                "version": 1,
                "job": job,
                "state": status,
                "time_source": "recorded",
                "retention_started_at_ns": 100,
                "finished_at_ns": 100,
            }
        )
    )


@pytest.mark.parametrize(
    "report", ["成果のみ", waiting("unknown", job_ids=["100_1_a.job"])]
)
def test_run_worker_rejects_invalid_declaration(
    service: ParentEnv, report: str
) -> None:
    p = service
    value = launch(p, report, ok=False)
    assert value["exit_code"] == 2
    assert value["runner_summary"]["result"] == "completed"
    assert "PARENT_RESULT" in value["followup"]["reason"]
    assert not (p.root / "service-starts.jsonl").exists()
    assert joined(p, ok=False)["wait_state"] == "followup_error"


@pytest.mark.parametrize(
    ("conclusion", "follow_on"),
    [("success", False), ("success", True), ("failure", False)],
)
def test_ci_resumes_only_for_failure_or_follow_on_work(
    service: ParentEnv, conclusion: str, follow_on: bool
) -> None:
    p = service
    set_ci(p, conclusion)
    extra = {"next_step": "ドラフトを解除する。"} if follow_on else {}
    launch(
        p, waiting("ci", pr=123, head=HEAD, resume_on_success=follow_on, **extra), DONE
    )
    result = joined(p)
    resumed = follow_on or conclusion == "failure"
    assert result["wait_state"] == "ended"
    assert result["followup"]["state"] == "done"
    assert result["followup"]["auto_resumes"] == int(resumed)
    assert result["followup"]["observation"]["head"] == HEAD
    assert_same_worker(p, 1 + int(resumed))
    assert result["final_file"] == str(latest(p) / "final.md")
    if resumed:
        prompt = (latest(p) / "prompt.md").read_text()
        if follow_on:
            assert "ドラフトを解除する。" in prompt
        else:
            for evidence in (
                "CPU tests",
                '"failed_steps"',
                "ERROR: regression in fixture step",
            ):
                assert evidence in prompt
    assert len((p.root / "service-starts.jsonl").read_text().splitlines()) == 1


def test_ci_rejects_stale_head_and_shares_the_review_lease(service: ParentEnv) -> None:
    p = service
    set_ci(p, "success")
    lease = importlib.import_module("lib.ops_review.review_locks").pr_lease
    with lease(state_dir(p), "o/r", 123):
        launch(p, waiting("ci", pr=123, head=HEAD, resume_on_success=False))
        assert "active progression" in joined(p, ok=False)["followup"]["reason"]
    run(
        p,
        "--resume",
        MANUAL_FINAL_SEQUENCE=json.dumps(
            [DONE, waiting("ci", pr=123, head="b" * 40, resume_on_success=False)]
        ),
    )
    assert "reserved HEAD" in joined(p, ok=False)["followup"]["reason"]
    assert not any("check-runs?" in " ".join(call) for call in p.calls())


def test_queue_waits_for_every_job_and_duplicate_controller_cannot_resume(
    service: ParentEnv,
) -> None:
    p = service
    first, second = "100_1_first.job", "101_1_second.job"
    set_job(p, first, "done")
    set_job(p, second, "running")
    with tempfile.TemporaryDirectory(prefix="queue-pending-") as short:
        address = Path(short) / "pending.sock"
        p.env["FAKE_QUEUE_PENDING"] = str(address)
        with listening(address) as server:
            launch(p, waiting("queue", job_ids=[first, second]), DONE)
            with connected(server) as gate:
                assert gate.recv(1024).decode() == second
                assert len(runner_calls(p)) == 1
                command = json.loads((p.root / "service-starts.jsonl").read_text())
                subprocess.run(command, cwd=p.repo, env=p.env, check=True, timeout=5)
                assert len(runner_calls(p)) == 1
                status_path = Path(short) / "wait-status.sock"
                with listening(status_path) as status:
                    waiter = subprocess.Popen(
                        wait_command(
                            p.root / "fake-runner",
                            "wait-worker",
                            "--child",
                            "10",
                            "--timeout",
                            "15",
                        ),
                        cwd=p.repo,
                        env={**p.env, "MANUAL_STATUS_SOCKET": str(status_path)},
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    try:
                        # The caller has already exited, and the further wait
                        # has pinned the first run before the queue completes.
                        with connected(status) as observed:
                            assert observed.recv(1024).decode() == latest(p).name
                            set_job(p, second, "failed", code=17)
                            gate.sendall(b"C")
                            assert joined(p)["wait_state"] == "ended"
                            observed.sendall(b"C")
                        status.close()
                        out, err = waiter.communicate(timeout=10)
                        assert waiter.returncode == 0, err
                    except TimeoutError:
                        if waiter.poll() is not None:
                            out, err = waiter.communicate(timeout=1)
                            pytest.fail(out + err)
                        raise
                    finally:
                        if waiter.poll() is None:
                            waiter.kill()
                            waiter.communicate(timeout=10)
    result = json.loads(out)
    assert result["followup"]["auto_resumes"] == 1
    assert result["followup"]["state"] == "done"
    assert_same_worker(p, 2)
    assert result["final_file"] == str(latest(p) / "final.md")
    observed = result["followup"]["observation"]["jobs"]
    assert [(j["job_id"], j["state"], j["exit_code"]) for j in observed] == [
        (first, "done", 0),
        (second, "failed", 17),
    ]
    prompt = (latest(p) / "prompt.md").read_text()
    assert "outputs/100_1_first" in prompt and "outputs/101_1_second" in prompt


@pytest.mark.parametrize(
    ("repeated", "count", "reason"),
    [(False, 4, "automatic_resume_limit"), (True, 2, "event_already_consumed")],
)
def test_queue_deduplicates_events_and_stops_at_the_resume_limit(
    service: ParentEnv, repeated: bool, count: int, reason: str
) -> None:
    p = service
    jobs = [f"{i}_1_run.job" for i in range(100, 104)]
    p.env["TRAINING_QUEUE_DIR"] = str(p.repo / ".training_queue")
    for job in jobs:
        set_job(p, job, "done")
    del p.env["TRAINING_QUEUE_DIR"]
    reports = [waiting("queue", job_ids=[job]) for job in jobs]
    if repeated:
        reports[1] = reports[0]
    launch(p, *reports)
    result = joined(p)["followup"]
    assert result["state"] == "needs_parent"
    assert result["reason"] == reason
    assert result["auto_resumes"] == count - 1
    assert_same_worker(p, count)


def test_queue_rejects_unknown_job_instead_of_waiting_forever(
    service: ParentEnv,
) -> None:
    p = service
    launch(p, waiting("queue", job_ids=["100_1_unregistered.job"]))
    result = joined(p, ok=False)["followup"]
    assert "unknown training-queue job_id" in result["reason"]
    assert result["auto_resumes"] == 0
    assert_same_worker(p, 1)
