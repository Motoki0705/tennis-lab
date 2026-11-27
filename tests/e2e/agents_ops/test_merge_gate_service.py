"""Production merge launch with local Git, fake GitHub and a fake user manager."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import OPS, ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_progression_cli import prepared, progression_path
from tests.e2e.agents_ops.test_wait_worker import connected, listening

ROOT = OPS.parents[1]
OPS_PATHS = [
    ".agents/ops",
    "src/__init__.py",
    "src/utils/__init__.py",
    "src/utils/training_queue.py",
]
POLL_PROBES = """
import socket
from types import SimpleNamespace

def _test_wait(address):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
        channel.settimeout(12)
        channel.connect(address)
        channel.sendall(b"waiting")
        assert channel.recv(1) == b"C"

if (address := os.environ.get("MERGE_GATE_TEST_WAITER")):
    time = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: _test_wait(address))

if (engine_socket := os.environ.get("MERGE_GATE_TEST_POLL")):
    class TestEngine(Engine):
        def __init__(self, *args, **kwargs):
            self.elapsed = 0
            super().__init__(
                *args, **kwargs, clock=lambda: time.time() + self.elapsed, sleep=self.pause
            )

        def pause(self, seconds):
            _test_wait(engine_socket)
            self.elapsed += seconds

    Engine = TestEngine
"""
FAKE_CODEX = r"""
import json, os, socket, sys
from pathlib import Path

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
    channel.settimeout(12)
    channel.connect(os.environ["MERGE_GATE_TEST_REVIEW"])
    channel.sendall(b"reviewing")
    assert channel.recv(1) == b"C"
result = json.dumps({"verdict": "approve", "summary": "Passed", "major": []})
Path(sys.argv[sys.argv.index("--output-last-message") + 1]).write_text(result)
session = "review-fixture"
context = {
    "turn_id": "test-turn",
    "cwd": str(Path.cwd()),
    "model": sys.argv[sys.argv.index("--model") + 1],
    "effort": next(
        json.loads(arg.removeprefix("model_reasoning_effort="))
        for arg in sys.argv if arg.startswith("model_reasoning_effort=")
    ),
}
transcript = Path(os.environ["CODEX_HOME"]) / "sessions" / (session + ".jsonl")
transcript.parent.mkdir(parents=True)
transcript.write_text("".join(json.dumps(event) + "\n" for event in (
    {"type": "session_meta", "payload": {"id": session}},
    {"type": "turn_context", "payload": context},
)))
for event in (
    {"type": "thread.started", "thread_id": session},
    {"type": "item.completed", "item": {"type": "agent_message", "text": result}},
    {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
):
    print(json.dumps(event), flush=True)
"""


def executable(path: Path, code: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + code)
    path.chmod(0o700)


@pytest.fixture
def primary(parent_env: ParentEnv) -> ParentEnv:
    p = parent_env
    p.env.pop("AGENTS_TEST_DIRECT_MERGE_GATE")
    p.env.update(
        PYTHONPATH=str(p.repo / ".agents/ops"),
        CODEX_HOME=str(p.root / "codex-home"),
        GH_CONFIG_DIR=str(p.root / "gh-config"),
        XDG_CONFIG_HOME=str(p.root / "xdg-config"),
        TRAINING_QUEUE_DIR=str(p.root / "queue"),
        PARENT_LOCAL_JOBS_DIR=str(p.root / "local-jobs"),
        MERGE_GATE_TEST_LAUNCHES=str(p.root / "service-starts.jsonl"),
        MERGE_GATE_TEST_BOOTSTRAP=str(ROOT / "tests/support/agents_ops/http_bootstrap"),
    )
    shutil.copytree(
        OPS, p.repo / ".agents/ops", ignore=shutil.ignore_patterns("__pycache__")
    )
    for source in OPS_PATHS[1:]:
        target = p.repo / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    (p.repo / ".venv").symlink_to(Path(sys.prefix).resolve(), target_is_directory=True)
    impl = p.repo / ".agents/ops/review/merge_gate_impl.py"
    with impl.open("a") as stream:
        stream.write(POLL_PROBES)
    run_git(p.repo, "add", "-f", "--", *OPS_PATHS)
    run_git(p.repo, "commit", "-m", "install controller and poll probes")
    executable(
        p.root / "bin/systemd-run",
        Path(__file__).with_name("fake_merge_gate_service.py").read_text(),
    )
    executable(
        p.root / "bin/systemctl",
        """import sys
assert sys.argv[1:3] == ["--user", "show"]
if "--value" in sys.argv:
    assert sys.argv[3:] == ["--property=Version", "--value"]
    print("255")
else:
    assert sys.argv[4:] == ["--property=LoadState,ActiveState"]
    print("LoadState=not-found")
    print("ActiveState=inactive")
""",
    )
    return p


def launches(p: ParentEnv) -> list[dict[str, Any]]:
    path = Path(p.env["MERGE_GATE_TEST_LAUNCHES"])
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text().splitlines()]


def review_args(p: ParentEnv, runner: Path) -> tuple[str, ...]:
    return (
        "review-request",
        "--prs",
        "100",
        *p.review_flags(),
        "--instruction-file",
        p.note("変更をレビューする。"),
        "--evidence-file",
        p.note("現HEADの受入結果を確認した。"),
        "--agent-exec",
        str(runner),
    )


def test_git_sync_during_merge_wait_preserves_controller_and_waiter_can_rejoin(
    primary: ParentEnv,
) -> None:
    p = primary
    prepared(p)
    executable(p.root / "bin/codex", FAKE_CODEX)
    pending = p.state()
    pending["prs"][0]["statusCheckRollup"][0].update(status="QUEUED", conclusion="")
    p.save(pending)
    old = run_git(p.repo, "rev-parse", "HEAD")
    origin, producer = p.root / "origin.git", p.root / "producer"
    run_git(p.repo, "clone", "--bare", ".", str(origin))
    run_git(p.repo, "remote", "add", "origin", str(origin))
    run_git(p.repo, "clone", str(origin), str(producer))
    prompt = ".agents/ops/review/prompt_template.md"
    with (producer / prompt).open("a") as template:
        template.write("\nNew controller source on main.\n")
    run_git(producer, "add", "--", prompt)
    run_git(
        producer,
        "-c",
        "user.name=Producer",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "update controller source",
    )
    new = run_git(producer, "rev-parse", "HEAD")
    run_git(producer, "push", "origin", "main")

    with tempfile.TemporaryDirectory(prefix="merge-gate-test-") as short:
        review, poll, waiter, stopped = (
            Path(short) / name
            for name in (
                "review",
                "poll",
                "waiter",
                "stopped",
            )
        )
        p.env.update(
            MERGE_GATE_TEST_REVIEW=str(review),
            MERGE_GATE_TEST_POLL=str(poll),
            MERGE_GATE_TEST_STOPPED=str(stopped),
        )
        with (
            listening(review) as reviewing,
            listening(poll) as polling,
            listening(waiter) as waiting,
            listening(stopped) as done,
        ):
            # Reuse a saved reference to main's standard runner. Its process
            # must still load the pinned agent_runtime.py during the review.
            result = json.loads(
                p.cli(*review_args(p, p.repo / ".agents/ops/lib/agent_exec.sh")).stdout
            )
            path = progression_path(p, result)
            (launch,) = launches(p)
            checkout = Path(launch["checkout"])
            assert checkout.parent == path.parents[2] / "merge-gate-checkouts"
            assert run_git(checkout, "rev-parse", "HEAD") == old
            assert run_git(checkout, "branch", "--show-current") == ""
            assert (checkout / ".venv").resolve() == Path(sys.prefix).resolve()
            assert (checkout / "src/utils/training_queue.py").is_file()
            assert not (checkout / "input.txt").exists()
            assert json.loads(path.read_text())["checkout"] == str(p.repo)
            assert launch["env"]["PYTHONPATH"].split(os.pathsep) == [
                str(checkout / ".agents/ops"),
                str(checkout),
            ]
            for env, value in launch["env"].items():
                if env != "PYTHONPATH":
                    assert value == p.env[env]
            with connected(reviewing) as channel:
                assert channel.recv(64) == b"reviewing"
                with (p.repo / ".git/ops-code.lock").open("a") as updater:
                    fcntl.flock(updater, fcntl.LOCK_EX | fcntl.LOCK_NB)
                channel.sendall(b"C")
            with (
                connected(polling) as controller,
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "parent",
                        "--repo",
                        "o/r",
                        "--parent",
                        "1",
                        "wait-merge",
                        "--run-id",
                        result["review_run_id"],
                        "--timeout",
                        "12",
                    ],
                    cwd=p.repo,
                    env={**p.env, "MERGE_GATE_TEST_WAITER": str(waiter)},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ) as process,
            ):
                try:
                    assert controller.recv(64) == b"waiting"
                    with connected(waiting) as channel:
                        assert channel.recv(64) == b"waiting"
                        synced = subprocess.run(
                            [
                                sys.executable,
                                str(p.repo / ".agents/ops/git_sync/sync.py"),
                                "--repo",
                                str(p.repo),
                            ],
                            cwd=p.repo,
                            env=p.env,
                            text=True,
                            capture_output=True,
                            timeout=10,
                            check=False,
                        )
                        assert synced.returncode == 0, synced.stdout + synced.stderr
                        assert json.loads(synced.stdout)["head_after"] == new
                        channel.sendall(b"C")
                    stdout, stderr = process.communicate(timeout=8)
                    assert process.returncode == 75, stdout + stderr
                    assert "checkout HEAD changed during wait" in stderr
                    assert run_git(checkout, "rev-parse", "HEAD") == old
                    assert (checkout / prompt).read_bytes() != (
                        p.repo / prompt
                    ).read_bytes()
                    with (
                        pytest.raises(BlockingIOError),
                        path.with_suffix(".lock").open("a") as lock,
                    ):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    ci = p.state()
                    ci["prs"][0]["statusCheckRollup"][0].update(
                        status="COMPLETED", conclusion="SUCCESS"
                    )
                    p.save(ci)
                    controller.sendall(b"C")
                finally:
                    if process.poll() is None:
                        process.kill()
            joined = json.loads(
                p.cli(
                    "wait-merge", "--run-id", result["review_run_id"], "--timeout", "10"
                ).stdout
            )
            assert joined["status"] == "completed", joined
            assert joined["review_usage"]["confirmed_launches"] == 1
            assert len([call for call in p.calls() if call[:2] == ["pr", "merge"]]) == 1
            assert "checkout_interrupted" not in path.read_text()
            with connected(done) as notice:
                assert notice.recv(64) == b"0"
            assert not checkout.exists()
            assert str(checkout) not in run_git(
                p.repo, "worktree", "list", "--porcelain"
            )


@pytest.mark.parametrize("failure", ["dirty", "restore"])
def test_start_failure_is_visible_and_continue_launches_a_fresh_checkout(
    primary: ParentEnv, failure: str
) -> None:
    p = primary
    _, fake, calls = prepared(p)
    dirty = p.repo / "src/utils/training_queue.py"
    if failure == "dirty":
        dirty.write_text("# uncommitted source\n")
    else:
        p.env["MERGE_GATE_TEST_FAIL_SETUP"] = "1"
    result = p.cli(*review_args(p, fake), ok=False)
    assert result.returncode == 2
    assert (
        "commit the merge controller source"
        if failure == "dirty"
        else "injected restore failure"
    ) in result.stderr
    paused = json.loads(p.cli("wait-merge", "--timeout", "2").stdout)
    assert paused["status"] == "parent_wait"
    assert paused["events"][-1]["reason"] == "controller_start_failed"
    assert not calls.read_text()
    first = launches(p)
    if failure == "dirty":
        assert first == []
        run_git(p.repo, "restore", "--", str(dirty))
    else:
        assert len(first) == 1 and not Path(first[0]["checkout"]).exists()
        assert first[0]["checkout"] not in run_git(
            p.repo, "worktree", "list", "--porcelain"
        )
        del p.env["MERGE_GATE_TEST_FAIL_SETUP"]

    continued = json.loads(
        p.cli(
            "merge-continue",
            "--run-id",
            paused["review_run_id"],
            "--instruction-file",
            p.note("実行環境を修復した。"),
            "--wait",
        ).stdout
    )
    assert continued["status"] == "completed", continued
    assert continued["review_run_id"] == paused["review_run_id"]
    assert len(calls.read_text().splitlines()) == 1
    resumed = launches(p)[-1]
    assert all(
        resumed[key] != row[key] for row in first for key in ("checkout", "unit")
    )


def test_service_requires_auto_merge_opt_in_from_its_caller(
    primary: ParentEnv,
) -> None:
    p = primary
    _, fake, calls = prepared(p)
    del p.env["AGENT_AUTO_MERGE"]
    paused = json.loads(p.cli(*review_args(p, fake), "--wait").stdout)
    assert paused["events"][-1]["reason"] == "auto_merge_disabled", paused
    assert not calls.read_text()
    (launch,) = launches(p)
    assert launch["env"]["AGENT_AUTO_MERGE"] is None
