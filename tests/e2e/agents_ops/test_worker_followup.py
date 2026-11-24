"""Offline CLI contracts for manual workers and their detached controller.

Reuse the real registry, manual runner boundary, and fake GitHub from
test_manual_worker_runs. The fake service launches the production controller
with a readiness socket. Queue evidence uses the published on-disk format.
Socket handshakes exercise the pending and duplicate-launch cases without
fixed sleeps in the tests. Measure setup and cases with --durations=0.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pytest

from tests.e2e.agents_ops.test_human_agreement import delegation_args
from tests.e2e.agents_ops.test_manual_worker_runs import (
    latest,
    manual,
    run,
    state_dir,
    task,
)
from tests.e2e.agents_ops.test_manual_worker_runs import reserved as reserved
from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_wait_worker import command as wait_command
from tests.e2e.agents_ops.test_wait_worker import connected, listening

HEAD = "a" * 40
DONE = 'PARENT_RESULT {"schema_version":1,"state":"done","wait":null}'
SOURCE = Path(__file__).resolve().parents[3]
FAKE_SERVICE = r"""
import json, os, shlex, socket, subprocess, sys, tempfile
from pathlib import Path
args = sys.argv[1:]
assert all(flag in args for flag in ("--user", "--collect", "--service-type=notify"))
command = args[args.index("--") + 1:]
assert command[1:3] == ["-m", "parent.worker_monitor"]
root = Path(os.environ["FAKE_SERVICE_ROOT"])
env = dict(os.environ)
working = Path.cwd()
pre, post = [], []
for flag in args:
    if flag.startswith("--setenv=") and "=" in flag.removeprefix("--setenv="):
        name, _, value = flag.removeprefix("--setenv=").partition("=")
        env[name] = value
    elif flag.startswith("--property=WorkingDirectory=-"):
        working = Path(flag.removeprefix("--property=WorkingDirectory=-"))
    elif flag.startswith("--property=Exec"):
        name, _, value = flag.removeprefix("--property=").partition("=")
        assert value.startswith(":")
        (pre if name == "ExecStartPre" else post).append(
            [arg.replace("%%", "%") for arg in shlex.split(value.removeprefix(":"))]
        )
for argv in pre:
    subprocess.run(argv, cwd=working if working.exists() else None, check=True)
env["PYTHONPATH"] += os.pathsep + os.environ["FAKE_SERVICE_HTTP_BOOTSTRAP"]
env["FAKE_SERVICE_CLEANUP"] = json.dumps(post)
with (root / "service-starts.jsonl").open("a") as f:
    f.write(json.dumps({"command": command, "cwd": str(working), "pythonpath": env["PYTHONPATH"]}) + "\n")
with tempfile.TemporaryDirectory(prefix="followup-notify-") as short:
    address = str(Path(short) / "notify.sock")
    env["NOTIFY_SOCKET"] = address
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as ready, (root / "controller.log").open("ab") as log:
        ready.bind(address)
        ready.settimeout(10)
        process = subprocess.Popen(
            [command[0], str(root / "fake-service-entry.py"), *command[3:]],
            cwd=working, env=env, start_new_session=True,
            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        )
        assert ready.recv(1024) == b"READY=1"
"""
FAKE_SERVICE_ENTRY = r"""
import json, os, socket, subprocess, sys
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
def queue_result(root, job, **kwargs):
    if os.environ.get("FAKE_QUEUE_ELAPSED") and kwargs["missing_completion_since"]:
        kwargs["now"] += 60
    result = original(root, job, **kwargs)
    if result is None and (address := os.environ.pop("FAKE_QUEUE_PENDING", None)):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.connect(address)
            channel.sendall(job.encode())
            assert channel.recv(1) == b"C"
    return result
worker_monitor.queue_result = queue_result
original_run = manual_workers.run_worker
def run_worker(*args, **kwargs):
    try:
        return original_run(*args, **kwargs)
    except ValueError as exc:
        if "implementation reservation limit reached" in str(exc):
            if address := os.environ.pop("FAKE_CAPACITY_RETRY", None):
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
                    channel.connect(address)
                    channel.sendall(b"full")
                    assert channel.recv(1) == b"C"
        raise
manual_workers.run_worker = run_worker
try:
    sys.exit(worker_monitor.main())
finally:
    for command in json.loads(os.environ["FAKE_SERVICE_CLEANUP"]):
        subprocess.run(command, check=True)
    if address := os.environ.get("FAKE_SERVICE_STOPPED"):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            channel.connect(address)
            channel.sendall(b"removed")
"""


@pytest.fixture
def service(reserved: ParentEnv) -> ParentEnv:
    p = reserved
    (p.repo / ".venv").symlink_to(sys.prefix, target_is_directory=True)
    shutil.copytree(
        SOURCE / ".agents/ops",
        p.repo / ".agents/ops",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in (
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/utils/training_queue.py",
    ):
        destination = p.repo / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, destination)
    run_git(p.repo, "add", ".agents/ops", "src")
    run_git(p.repo, "commit", "-m", "install operational code on fixture main")
    p.env["PYTHONPATH"] = os.pathsep.join(map(str, (p.repo / ".agents/ops", p.repo)))
    path = p.root / "bin/systemd-run"
    path.write_text(f"#!{sys.executable}\n" + FAKE_SERVICE)
    path.chmod(0o755)
    (p.root / "fake-service-entry.py").write_text(FAKE_SERVICE_ENTRY)
    p.env.update(
        FAKE_SERVICE_ROOT=str(p.root),
        FAKE_SERVICE_HTTP_BOOTSTRAP=str(
            SOURCE / "tests/support/agents_ops/http_bootstrap"
        ),
        TRAINING_QUEUE_DIR=str(p.root / "queue"),
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


def service_start(p: ParentEnv) -> dict[str, Any]:
    (row,) = (p.root / "service-starts.jsonl").read_text().splitlines()
    return cast(dict[str, Any], json.loads(row))


def worker_calls(p: ParentEnv, child_task: dict[str, Any]) -> list[list[str]]:
    return [
        call
        for call in runner_calls(p)
        if call[call.index("--task-id") + 1] == child_task["delegation_id"]
    ]


def assert_same_worker(p: ParentEnv, count: int) -> None:
    child_task = task(p)
    runs = child_task["worker_runs"]
    calls = worker_calls(p, child_task)
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
                started = service_start(p)
                subprocess.run(
                    started["command"],
                    cwd=started["cwd"],
                    env={**p.env, "PYTHONPATH": started["pythonpath"]},
                    check=True,
                    timeout=5,
                )
                assert len(runner_calls(p)) == 1
                status_path = Path(short) / "wait-status.sock"
                with listening(status_path) as status:
                    waiter = subprocess.Popen(
                        wait_command(
                            p.root / "fake-runner",
                            "wait-worker",
                            "--child",
                            "10",
                            "--after-run",
                            "none",
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

    first_run = state_dir(p) / task(p)["worker_runs"][0]["run_dir"]
    second_run = latest(p)
    failed = json.loads(
        run(p, "--resume", ok=False, MANUAL_PREFLIGHT_FAILURE="1").stdout
    )
    p.cli("cancel-worker-launch", "--child", "10", "--run-dir", failed["run_dir"])
    assert task(p)["manual_followup"]["run_dirs"] == []
    assert joined(p)["followup"]["id"] == result["followup"]["id"]
    run(p, "--resume")
    assert latest(p) != second_run
    for anchor in ("none", str(first_run)):
        original = json.loads(
            manual(p, "wait-worker", "--child", "10", "--after-run", anchor).stdout
        )
        assert original["run_dir"] == str(second_run)
        assert original["followup"] == result["followup"]


@pytest.mark.parametrize(
    ("repeated", "count", "reason"),
    [(False, 11, "automatic_resume_limit"), (True, 2, "event_already_consumed")],
)
def test_queue_deduplicates_events_and_stops_at_the_resume_limit(
    service: ParentEnv, repeated: bool, count: int, reason: str
) -> None:
    p = service
    jobs = [f"{i}_1_run.job" for i in range(100, 100 + count)]
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


def test_queue_missing_completion_is_returned_by_wait_worker(
    service: ParentEnv,
) -> None:
    p = service
    job = "100_1_unpublished.job"
    root = Path(p.env["TRAINING_QUEUE_DIR"])
    (root / "failed").mkdir(parents=True)
    (root / "failed" / job).touch()
    p.env["FAKE_QUEUE_ELAPSED"] = "1"
    launch(p, waiting("queue", job_ids=[job]))
    result = joined(p, ok=False)["followup"]
    assert result["state"] == "error"
    assert result["reason"] == f"missing training-queue completion record for {job}"
    assert_same_worker(p, 1)


def test_auto_resume_retries_after_another_worker_releases_the_only_slot(
    service: ParentEnv,
) -> None:
    p = service
    p.cli(
        "configure-delegation",
        "--max-parallel-children",
        "1",
        "--reason-file",
        p.note(),
    )
    job = "100_1_slot.job"
    set_job(p, job, "running")
    with tempfile.TemporaryDirectory(prefix="queue-capacity-") as short:
        pending, busy, retry = (
            Path(short) / name for name in ("pending.sock", "busy.sock", "retry.sock")
        )
        p.env.update(FAKE_QUEUE_PENDING=str(pending), FAKE_CAPACITY_RETRY=str(retry))
        with listening(pending) as pending_server, listening(retry) as retry_server:
            launch(p, waiting("queue", job_ids=[job]), DONE)
            with connected(pending_server) as queue:
                assert queue.recv(1024).decode() == job
                p.child("other")
                p.cli(*delegation_args(p.worktree("other"), child=11))
                runner = p.root / "busy-runner"
                runner.write_text(
                    f"#!{sys.executable}\n"
                    "import os, socket, sys\n"
                    "if sys.argv[1] != 'status':\n"
                    "    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as gate:\n"
                    "        gate.connect(os.environ['OTHER_RUNNER_SOCKET'])\n"
                    "        gate.sendall(b'started')\n"
                    "        assert gate.recv(1) == b'C'\n"
                    f"runner = {str(p.root / 'fake-runner')!r}\n"
                    "os.execv(runner, [runner, *sys.argv[1:]])\n"
                )
                runner.chmod(0o700)
                with (
                    listening(busy) as busy_server,
                    subprocess.Popen(
                        wait_command(
                            runner,
                            "run-worker",
                            "--child",
                            "11",
                            "--prompt-file",
                            p.note(),
                        ),
                        cwd=p.repo,
                        env={
                            **p.env,
                            "OTHER_RUNNER_SOCKET": str(busy),
                            "MANUAL_STATE_DIR": str(state_dir(p)),
                            "MANUAL_RUNNER_CALLS": str(p.root / "runner-calls.jsonl"),
                        },
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        start_new_session=True,
                    ) as other,
                ):
                    try:
                        with connected(busy_server) as worker:
                            assert worker.recv(1024) == b"started"
                            set_job(p, job, "done")
                            queue.sendall(b"C")
                            with connected(retry_server) as full:
                                assert full.recv(1024) == b"full"
                                saved = task(p)["manual_followup"]
                                assert saved["state"] == "resuming"
                                assert saved["observation"]["jobs"][0]["job_id"] == job
                                assert not saved["resumptions"]
                                assert_same_worker(p, 1)
                                worker.sendall(b"C")
                                out, err = other.communicate(timeout=10)
                                assert other.returncode == 0, out + err
                                full.sendall(b"C")
                    finally:
                        if other.poll() is None:
                            os.killpg(other.pid, signal.SIGTERM)
    result = joined(p)["followup"]
    assert result["state"] == "done" and result["auto_resumes"] == 1
    assert_same_worker(p, 2)


def test_auto_resume_records_admission_refusal_for_parent(
    service: ParentEnv,
) -> None:
    p = service
    job = "100_1_admission.job"
    set_job(p, job, "running")
    with tempfile.TemporaryDirectory(prefix="queue-admission-") as short:
        pending = Path(short) / "pending.sock"
        p.env["FAKE_QUEUE_PENDING"] = str(pending)
        with listening(pending) as server:
            launch(p, waiting("queue", job_ids=[job]))
            with connected(server) as gate:
                assert gate.recv(1024).decode() == job
                remote = p.state()
                remote["prs"] = [p.pr(123, 10, branch="codex/another")]
                p.save(remote)
                set_job(p, job, "done")
                gate.sendall(b"C")
    result = joined(p)["followup"]
    assert result["state"] == "needs_parent"
    assert "continuation requires a unique open or merged PR" in result["reason"]
    assert result["auto_resumes"] == 0
    assert result["observation"]["jobs"][0]["job_id"] == job
    assert_same_worker(p, 1)


def test_queue_monitor_pins_main_code_without_holding_checkout_lock(
    service: ParentEnv,
) -> None:
    p = service
    job = "100_1_pin.job"
    set_job(p, job, "running")
    head = run_git(p.repo, "rev-parse", "HEAD")
    relative = ".agents/ops/parent/worker_prompt.md"
    marker = "更新後のmainだけにある追加指示。"
    with tempfile.TemporaryDirectory(prefix="queue-pin-") as short:
        pending, stopped = Path(short) / "pending.sock", Path(short) / "stopped.sock"
        p.env.update(FAKE_QUEUE_PENDING=str(pending), FAKE_SERVICE_STOPPED=str(stopped))
        with listening(pending) as server, listening(stopped) as cleanup:
            launch(p, waiting("queue", job_ids=[job]), DONE)
            with connected(server) as gate:
                assert gate.recv(1024).decode() == job
                checkout = Path(service_start(p)["cwd"])
                assert run_git(checkout, "rev-parse", "HEAD") == head
                assert (checkout / relative).read_bytes() == (
                    p.repo / relative
                ).read_bytes()
                with (p.repo / ".git/ops-code.lock").open("a") as updater:
                    fcntl.flock(updater, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with (p.repo / relative).open("a", encoding="utf-8") as source:
                        source.write(f"\n{marker}\n")
                    run_git(p.repo, "add", relative)
                    run_git(p.repo, "commit", "-m", "update monitor source on main")
                assert run_git(p.repo, "rev-parse", "HEAD") != head
                assert run_git(checkout, "rev-parse", "HEAD") == head
                set_job(p, job, "done")
                gate.sendall(b"C")
            assert joined(p)["followup"]["state"] == "done"
            assert_same_worker(p, 2)
            assert marker not in (latest(p) / "prompt.md").read_text()
            with connected(cleanup) as notice:
                assert notice.recv(1024) == b"removed"
            assert not checkout.exists()
            assert str(checkout) not in run_git(
                p.repo, "worktree", "list", "--porcelain"
            )
