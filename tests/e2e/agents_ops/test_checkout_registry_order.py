"""Real dispatcher, registry and runner paths share checkout-before-registry order."""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_git_sync import wait_file, wait_for_flock
from tests.e2e.agents_ops.test_ops_control import ControlEnv
from tests.e2e.agents_ops.test_ops_control import control_factory as control_factory
from tests.support.agents_ops.checkout_cli import ROOT, line


@dataclass
class World:
    control: ControlEnv
    parent: Path
    schedule: Path
    runtime: Path

    @property
    def repo(self) -> Path:
        return Path(self.control.repo)

    @property
    def env(self) -> dict[str, str]:
        return {
            **self.control.environment,
            "PYTHONPATH": str(self.repo / ".agents/ops"),
        }

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, *arguments],
            cwd=self.repo,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result


@pytest.fixture
def world(control_factory: Any) -> World:
    control: ControlEnv = control_factory("real-dispatcher")
    shutil.copytree(
        ROOT / ".agents/ops",
        control.repo / ".agents/ops",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (control.repo / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)
    base = Path(control.environment["XDG_STATE_HOME"]) / "tennis-lab-agents"
    result = World(
        control,
        base / "parent-sessions",
        base / "dispatcher",
        base / "judgement/agent-runtime",
    )
    result.run(
        "-m",
        "dispatcher",
        "init",
        "--repo",
        "o/r",
        "--cwd",
        str(control.repo),
        "--agent-exec",
        str(control.repo / ".agents/ops/lib/agent_exec.sh"),
    )
    fake = control.repo.parent / "bin/codex"
    shutil.copyfile(Path(__file__).with_name("fake_agent_cli.py"), fake)
    fake.chmod(0o700)
    control.environment.update(
        FAKE_ARGS=str(control.repo.parent / "args"),
        FAKE_CWD=str(control.repo.parent / "cwd"),
        CODEX_HOME=str(control.repo.parent / "codex-home"),
    )
    prompt = control.repo.parent / "prompt.md"
    prompt.write_text("offline control fixture\n")
    result.run(
        str(control.repo / ".agents/ops/lib/agent_runtime.py"),
        "--agent",
        "codex",
        "--cwd",
        str(control.repo),
        "--prompt-file",
        str(prompt),
        "--run-dir",
        str(result.runtime),
    )
    path = result.schedule / "schedule.json"
    state = json.loads(path.read_text())
    state["active"] = "interrupted"
    state["runs"].append(
        {
            "id": "interrupted",
            "slot": state["cursor"],
            "explicit": True,
            "phase": "recovery_required",
            "controller": None,
            "directory": str(result.runtime.parent),
            "diagnostic": "interrupted controller",
            "candidates": {},
        }
    )
    path.write_text(json.dumps(state))
    return result


UPDATER = """
import sys
from pathlib import Path
from lib.checkout_lock import checkout_lock
from git_sync.runtime import Maintenance
from git_sync.git import Skip
root = Path(sys.argv[1])
with checkout_lock(root, exclusive=True) as fd:
    maintenance = Maintenance(root, Path(sys.argv[2]), lambda *a, **k: None)
    maintenance.control.checkout_fd = fd
    print('updater-locked', flush=True)
    sys.stdin.readline()
    try:
        maintenance.pause()
    except Skip as error:
        print(error.reason, flush=True)
    else:
        raise AssertionError('interrupted judgement must defer maintenance')
    sys.stdin.readline()
"""

API = """
import sys
from dispatcher.cli_impl import main
print('api-imported', flush=True)
sys.stdin.readline()
if sys.argv[1] == 'consistent-snapshot':
    from pathlib import Path
    from parent.state import Store
    with Store(Path(sys.argv[2])).consistent_snapshot():
        print('snapshot-read', flush=True)
    raise SystemExit(0)
if sys.argv[1] == 'ownership':
    from pathlib import Path
    from parent.state import Store
    store = Store(Path(sys.argv[2]))
    with store.ownership(), store.locked():
        print('ownership-read', flush=True)
    raise SystemExit(0)
raise SystemExit(main())
"""


def stop(process: subprocess.Popen[str], *, group: bool = False) -> None:
    if process.poll() is None:
        if group:
            assert os.getpgid(process.pid) == process.pid
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    try:
        process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


@pytest.mark.parametrize(
    ("entry", "operation"),
    [
        ("cli", "shutdown"),
        ("api", "shutdown"),
        ("cli", "recover-judgement"),
        ("api", "recover-judgement"),
        ("api", "consistent-snapshot"),
        ("api", "ownership"),
    ],
)
def test_update_first_never_waits_for_a_registry_owner_waiting_for_checkout(
    world: World,
    entry: str,
    operation: str,
) -> None:
    arguments = [operation]
    if operation in {"consistent-snapshot", "ownership"}:
        arguments.append(str(world.parent))
    if operation == "recover-judgement":
        evidence = world.repo.parent / "recovery.txt"
        evidence.write_text("verified controller end\n")
        arguments += ["--evidence-file", str(evidence)]
    manager: subprocess.Popen[str] | None = None
    if entry == "api":
        manager = subprocess.Popen(
            [sys.executable, "-c", API, *arguments],
            cwd=world.repo,
            env=world.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert line(manager) == "api-imported"
    with subprocess.Popen(
        [
            sys.executable,
            "-c",
            UPDATER,
            str(world.repo),
            str(world.repo.parent / "maintenance"),
        ],
        cwd=world.repo,
        env=world.env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as updater:
        try:
            assert line(updater) == "updater-locked"
            if manager is None:
                manager = subprocess.Popen(
                    [sys.executable, "-m", "dispatcher", *arguments],
                    cwd=world.repo,
                    env=world.env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                assert manager.stdin is not None
                manager.stdin.write("execute\n")
                manager.stdin.flush()
            wait_for_flock(manager)
            with (world.parent / "registry.lock").open("a") as probe:
                try:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    registry_available = True
                except BlockingIOError:
                    registry_available = False
            if operation == "ownership":
                with (world.parent / "ownership.lock").open("a") as probe:
                    try:
                        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        ownership_available = True
                    except BlockingIOError:
                        ownership_available = False
                assert ownership_available, "ownership acquired before checkout"
            assert updater.stdin is not None
            updater.stdin.write("pause\n")
            updater.stdin.flush()
            # Actual Maintenance.pause -> dispatcher status -> actual registry.
            # Reversing registry/checkout order makes this handshake time out.
            assert line(updater) == "managed_jobs_running"
            assert registry_available
            assert manager.poll() is None
            updater.communicate(input="release\n", timeout=8)
            assert updater.returncode == 0
            stdout, stderr = manager.communicate(timeout=8)
            assert manager.returncode == 0, stdout + stderr
            state = json.loads((world.schedule / "schedule.json").read_text())
            assert state["runs"][0]["phase"] in {"failed", "recovery_required"}
            assert json.loads((world.runtime / "result.json").read_text())[
                "cleanup_complete"
            ]
        finally:
            stop(updater, group=True)
            if manager is not None:
                stop(manager)


LIFECYCLE = """
import json,sys
from pathlib import Path
from lib.checkout_lock import guard_interactive_cli
guard_interactive_cli(allow_exclusive=True)
from parent.state import Store
from dispatcher.runtime import Runtime
with Store(Path(sys.argv[1])).locked():
    result = Runtime(Path(sys.argv[2])).inspect(Path(sys.argv[3]), stop=sys.argv[4]=='stop', timeout=5)
    print(json.dumps(result['summary']), flush=True)
"""


def test_resident_skips_cycle_when_source_is_updating_a_different_job_checkout(
    world: World,
) -> None:
    job_checkout = world.repo.parent / "separate-job-checkout"
    subprocess.run(["git", "init", "-q", str(job_checkout)], check=True)
    schedule = world.schedule / "schedule.json"
    state = json.loads(schedule.read_text())
    state["config"]["cwd"] = str(job_checkout)
    schedule.write_text(json.dumps(state))
    driver = """
import sys
from dispatcher import cli_impl
cycles = 0
def wait_between_cycles(seconds):
    global cycles
    cycles += 1
    print('cycle-complete', flush=True)
    if cycles == 1:
        sys.stdin.readline()
    else:
        raise SystemExit(0)
cli_impl.time.sleep = wait_between_cycles
raise SystemExit(cli_impl.main(['supervise'], _supervisor_launcher=True))
"""
    with subprocess.Popen(
        [sys.executable, "-c", driver],
        cwd=world.repo,
        env=world.env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as resident:
        try:
            # The real startup and first cycle have completed before updating.
            assert line(resident) == "cycle-complete"
            with (world.repo / ".git/ops-code.lock").open("a") as updater:
                fcntl.flock(updater, fcntl.LOCK_EX)
                stdout, stderr = resident.communicate(input="cycle\n", timeout=8)
                assert resident.returncode == 0, stdout + stderr
                assert stdout.strip() == "cycle-complete"
            assert json.loads(schedule.read_text()) == state
        finally:
            stop(resident)


def test_board_request_during_update_does_not_block_graceful_resident_stop(
    world: World,
) -> None:
    driver = """
import os,socket
import uvicorn
from board.web import create_app
listener = socket.socket()
listener.bind(('127.0.0.1', 0))
listener.listen()
class Server(uvicorn.Server):
    async def startup(self, sockets=None):
        await super().startup(sockets=sockets)
        print(listener.getsockname()[1], flush=True)
Server(uvicorn.Config(create_app(os.environ), log_level='warning', access_log=False)).run(sockets=[listener])
"""
    with subprocess.Popen(
        [sys.executable, "-c", driver],
        cwd=world.repo,
        env=world.env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as resident:
        try:
            port = int(line(resident))
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                connection.request("GET", "/")
                page = connection.getresponse()
                assert page.status == 200
                token = page.getheader("X-CSRF-Token")
                assert token is not None
                page.read()
                with (world.repo / ".git/ops-code.lock").open("a") as updater:
                    fcntl.flock(updater, fcntl.LOCK_EX)
                    connection.request(
                        "POST",
                        "/api/posts",
                        body=json.dumps(
                            {
                                "subject": "Question during maintenance",
                                "body": "Temporary offline request",
                                "kind": "question",
                                "to": [],
                            }
                        ),
                        headers={
                            "Content-Type": "application/json",
                            "Origin": f"http://127.0.0.1:{port}",
                            "X-CSRF-Token": token,
                        },
                    )
                    response = connection.getresponse()
                    assert response.status == 503
                    assert "maintenance" in json.loads(response.read())["detail"]
                    connection.close()
                    resident.terminate()
                    stdout, stderr = resident.communicate(timeout=5)
                    assert resident.returncode in {0, -signal.SIGTERM}, stdout + stderr
            finally:
                connection.close()
        finally:
            stop(resident, group=True)


@pytest.mark.parametrize("operation", ["status", "stop"])
def test_maintenance_lifecycle_under_registry_reuses_exclusive_lock_without_downgrade(
    world: World,
    operation: str,
) -> None:
    with (world.repo / ".git/ops-code.lock").open("a") as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX)
        environment = {
            **world.env,
            "TENNIS_LAB_CHECKOUT_PARENT_FD": str(exclusive.fileno()),
            "TENNIS_LAB_CHECKOUT_PARENT_DIRECTORY": str(world.repo / ".git"),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                LIFECYCLE,
                str(world.parent),
                str(world.repo / ".agents/ops/lib/agent_exec.sh"),
                str(world.runtime),
                operation,
            ],
            cwd=world.repo,
            env=environment,
            pass_fds=(exclusive.fileno(),),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout)["result"] == "completed"
        with (
            (world.repo / ".git/ops-code.lock").open("a") as probe,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)


def test_maintenance_can_stop_a_running_worktree_runner_during_main_update(
    world: World,
) -> None:
    directory = world.runtime.parent / "legacy-runtime"
    child = world.repo.parent / "legacy-child"
    environment = {
        **world.env,
        "FAKE_SCENARIO": "hang",
        "FAKE_CHILD": str(child),
        "FAKE_SESSION_ID": "legacy-codex-session",
    }
    with subprocess.Popen(
        [
            "bash",
            str(ROOT / ".agents/ops/lib/agent_exec.sh"),
            "--agent",
            "codex",
            "--cwd",
            str(world.repo),
            "--prompt-file",
            str(world.repo.parent / "prompt.md"),
            "--run-dir",
            str(directory),
        ],
        cwd=world.repo,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as runner:
        try:
            wait_file(child)
            with (world.repo / ".git/ops-code.lock").open("a") as exclusive:
                fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        LIFECYCLE,
                        str(world.parent),
                        str(world.repo / ".agents/ops/lib/agent_exec.sh"),
                        str(directory),
                        "stop",
                    ],
                    cwd=world.repo,
                    env={
                        **world.env,
                        "TENNIS_LAB_CHECKOUT_PARENT_FD": str(exclusive.fileno()),
                        "TENNIS_LAB_CHECKOUT_PARENT_DIRECTORY": str(
                            world.repo / ".git"
                        ),
                    },
                    pass_fds=(exclusive.fileno(),),
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert json.loads(result.stdout)["exit_code"] == 143
                record = json.loads((directory / "result.json").read_text())
                assert record["cleanup_complete"] is True
                runner.communicate(timeout=5)
                assert runner.returncode == 143
        finally:
            stop(runner)


@pytest.mark.parametrize("case", ["unlocked-descriptor", "exclusive-run"])
def test_borrowed_checkout_requires_real_flock_and_lifecycle_command(
    world: World, case: str
) -> None:
    with (world.repo / ".git/ops-code.lock").open("a") as descriptor:
        if case == "exclusive-run":
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        result = subprocess.run(
            [
                sys.executable,
                str(world.repo / ".agents/ops/lib/agent_runtime.py"),
                "run" if case == "exclusive-run" else "status",
                "--run-dir",
                str(world.runtime),
            ],
            cwd=world.repo,
            env={
                **world.env,
                "TENNIS_LAB_CHECKOUT_PARENT_FD": str(descriptor.fileno()),
                "TENNIS_LAB_CHECKOUT_PARENT_DIRECTORY": str(world.repo / ".git"),
            },
            pass_fds=(descriptor.fileno(),),
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        assert result.returncode == 75
        assert "checkout CLI refused" in result.stderr
