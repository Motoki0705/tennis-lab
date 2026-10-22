"""Persistent runner contract against offline CLIs, disposable git and real processes."""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / ".agents/ops/lib/agent_exec.sh"
CONFIG = ROOT / ".agents/ops/lib/agents.conf"
FAKE = Path(__file__).with_name("fake_agent_cli.py")


class Harness:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.work = root / "work"
        self.work.mkdir()
        self.env = {
            **os.environ,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "CODEX_HOME": str(root / "codex-home"),
            "FAKE_ARGS": str(root / "args.txt"),
            "FAKE_CWD": str(root / "cwd.txt"),
            "FAKE_ENV": str(root / "env.json"),
            "FAKE_CHILD": str(root / "child.pid"),
        }
        self.git("init", "-q", "-b", "test-branch")
        self.git(
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        )
        bin_dir = root / "bin"
        bin_dir.mkdir()
        for name in ("claude", "codex"):
            path = bin_dir / name
            path.write_bytes(FAKE.read_bytes())
            path.chmod(0o700)
        self.env["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]

    def git(self, *args: str) -> None:
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=self.work,
            env=self.env,
            check=True,
            capture_output=True,
        )

    def cmd(
        self,
        name: str = "initial",
        *,
        agent: str = "codex",
        action: str = "run",
        prompt: str = "implementation task\n",
        extra: tuple[str, ...] = (),
    ) -> list[str]:
        prompt_path = self.root / f"{name}-prompt.md"
        prompt_path.write_text(prompt)
        return [
            "bash",
            str(RUNNER),
            action,
            "--agent",
            agent,
            "--cwd",
            str(self.work),
            "--prompt-file",
            str(prompt_path),
            "--run-dir",
            str(self.root / name),
            "--task-id",
            "task-key",
            *extra,
        ]

    def run(
        self, name: str = "initial", *, scenario: str = "success", **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.cmd(name, **kwargs),
            env={**self.env, "FAKE_SCENARIO": scenario},
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )

    def result(self, name: str = "initial") -> dict[str, Any]:
        value: dict[str, Any] = json.loads(
            (self.root / name / "result.json").read_text()
        )
        return value

    def args(self) -> list[str]:
        return (self.root / "args.txt").read_text().splitlines()

    def query(
        self, action: str, name: str = "initial"
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(RUNNER), action, "--run-dir", str(self.root / name)],
            env=self.env,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.mark.parametrize(
    ("agent", "extra", "model", "effort"),
    [
        ("codex", (), "gpt-6.1-sol", "max"),
        ("codex", ("--model", "gpt-6-astra"), "gpt-6-astra", "max"),
        ("claude", (), "claude-opus-5-5", "xhigh"),
        ("claude", ("--effort", "high"), "claude-opus-5-5", "high"),
    ],
)
def test_role_pins_and_runtime_evidence(
    h: Harness, agent: str, extra: tuple[str, ...], model: str, effort: str
) -> None:
    h.env.update(CODEX_MODEL="wrong-inherited-default", CLAUDE_EFFORT="low")
    proc = h.run(agent=agent, extra=extra)
    assert proc.returncode == 0, proc.stderr
    result = h.result()
    settings = result["contract"]["settings"]
    assert settings["selected"]["model"] == model
    assert settings["selected"]["effort"] == effort
    assert settings["defaults"]["model"] == (
        "gpt-6.1-sol" if agent == "codex" else "claude-opus-5-5"
    )
    assert settings["selection_sources"]["model"] == (
        "task" if "--model" in extra else agent.upper() + "_MODEL"
    )
    assert result["runtime"]["effective"]["model"]["value"] == model
    if agent == "codex":
        assert h.args()[:4] == ["--disable", "daemon_auto_start", "exec", "--cd"]
        assert result["runtime"]["effective"]["effort"]["value"] == effort
        assert result["runtime"]["effective"]["service_tier"]["status"] == "unavailable"
        assert result["runtime"]["turn_id"]
    else:
        assert result["runtime"]["turn_id"] is None
        assert result["runtime"]["turn_id_unavailable_reason"]
        assert result["runtime"]["effective"]["effort"]["reason"]
    assert result["runtime"]["usage"]["value"]["input_tokens"] == 10
    assert result["task_success"] is None
    assert result["result"] == "completed" and result["turn_end_reason"] == "completed"
    assert result["contract"]["branch"] == "test-branch"
    assert result["prompt_sha256"]
    for path in (h.root / "initial").iterdir():
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_parent_identity_removed_and_public_output_contains_no_local_identity(
    h: Harness, agent: str
) -> None:
    h.env.update(
        CLAUDE_CODE_SESSION_ID="private-parent-session",
        CLAUDECODE="1",
        CODEX_THREAD_ID="private-parent-thread",
        CODEX_SESSION_ID="private-root-session",
    )
    proc = h.run(agent=agent)
    assert proc.returncode == 0, proc.stderr
    environment = json.loads((h.root / "env.json").read_text())
    assert (
        not {
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDECODE",
            "CODEX_THREAD_ID",
            "CODEX_SESSION_ID",
        }
        & environment.keys()
    )
    assert environment["AGENT_RUNTIME_TASK_ID"] == "task-key"
    assert environment["AGENT_RUNTIME_RUN_DIR"] == str(h.root / "initial")
    public = proc.stdout + h.query("status").stdout
    assert "private-" not in public and str(h.root) not in public
    assert "session_id" not in public and "runtime_errors" not in public


@pytest.mark.parametrize("agent", ["codex", "claude"])
@pytest.mark.parametrize("mode", ["read-only", "write"])
def test_resume_preserves_identity_settings_and_new_prompt(
    h: Harness, agent: str, mode: str
) -> None:
    extra = (
        ("--mode", mode, "--model", "gpt-6-astra")
        if agent == "codex"
        else ("--mode", mode)
    )
    first = h.run(agent=agent, extra=extra)
    assert first.returncode == 0, first.stderr
    initial = h.result()
    resumed = h.run(
        "continued",
        agent=agent,
        action="resume",
        prompt="fix the implementation\n",
        extra=("--resume-from", str(h.root / "initial"), "--mode", mode),
    )
    assert resumed.returncode == 0, resumed.stderr
    second = h.result("continued")
    assert second["runtime"]["session_id"] == initial["runtime"]["session_id"]
    assert second["contract"] == initial["contract"]
    assert second["prompt_sha256"] != initial["prompt_sha256"]
    assert Path(second["final_path"]).read_text() == "fix the implementation\n"
    assert ("resume" if agent == "codex" else "--resume") in h.args()
    assert initial["runtime"]["session_id"] in h.args()
    if agent == "codex":
        assert second["runtime"]["turn_id"] != initial["runtime"]["turn_id"]
        assert h.args()[h.args().index("--model") + 1] == "gpt-6-astra"
        if mode == "read-only":
            assert h.args().index("--sandbox") < h.args().index("resume")
    stale = h.run(
        "stale",
        agent=agent,
        action="resume",
        extra=("--resume-from", str(h.root / "initial"), "--mode", mode),
    )
    assert stale.returncode == 2 and "not the latest" in stale.stderr


@pytest.mark.parametrize(
    "change",
    ["task", "worktree", "branch", "settings", "mode", "model", "external_turn"],
)
def test_resume_refuses_changed_contract_before_launch(h: Harness, change: str) -> None:
    assert h.run().returncode == 0
    extra: tuple[str, ...] = ("--resume-from", str(h.root / "initial"))
    if change == "task":
        extra += ("--task-id", "other-task")
    elif change == "worktree":
        other = h.root / "other"
        h.git("worktree", "add", "--detach", str(other), "HEAD")
        extra += ("--cwd", str(other))
    elif change == "branch":
        h.git("checkout", "-qb", "other-branch")
    elif change == "settings":
        config = h.root / "other.conf"
        config.write_bytes(CONFIG.read_bytes() + b"\n# config changed\n")
        extra += ("--config-file", str(config))
    elif change == "mode":
        extra += ("--mode", "write")
    elif change == "model":
        extra += ("--model", "gpt-6-astra")
    else:
        transcript = next((h.root / "codex-home/sessions").rglob("*.jsonl"))
        with transcript.open("a") as stream:
            stream.write(
                json.dumps(
                    {"type": "turn_context", "payload": {"turn_id": "external-turn"}}
                )
                + "\n"
            )
    (h.root / "args.txt").unlink()
    proc = h.run("rejected", action="resume", extra=extra)
    assert proc.returncode == 2, proc.stderr
    assert not (h.root / "args.txt").exists()
    assert not (h.root / "rejected").exists()


@pytest.mark.parametrize(
    "extra",
    [
        ("--model", "unconfigured-model"),
        ("--effort", "low"),
        ("--effort", ""),
        ("--timeout", "nan"),
        ("--timeout", "0"),
        ("--task-id", ""),
        ("--branch", "other-branch"),
        ("--profile", "review", "--model", "gpt-6-astra"),
    ],
)
def test_invalid_selection_does_not_launch(h: Harness, extra: tuple[str, ...]) -> None:
    proc = h.run(extra=extra)
    assert proc.returncode == 2
    assert not (h.root / "args.txt").exists()
    assert not (h.root / "initial").exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "empty", "bad_effort", "duplicate", "shell"]
)
def test_configuration_is_literal_complete_and_validated_before_overrides(
    h: Harness, mutation: str
) -> None:
    text = CONFIG.read_text()
    if mutation == "missing":
        text = text.replace('CODEX_MODEL="gpt-6.1-sol"', "")
    elif mutation == "empty":
        text = text.replace('CODEX_SERVICE_TIER="default"', 'CODEX_SERVICE_TIER=""', 1)
    elif mutation == "bad_effort":
        text = text.replace('CODEX_EFFORT="max"', 'CODEX_EFFORT="invalid"', 1)
    elif mutation == "duplicate":
        text += '\nCODEX_MODEL="gpt-6-astra"\n'
    else:
        text += f"\ntouch {h.root / 'shell-executed'}\n"
    config = h.root / "invalid.conf"
    config.write_text(text)
    proc = h.run(extra=("--config-file", str(config), "--model", "gpt-6-astra"))
    assert proc.returncode == 2
    assert (
        not (h.root / "args.txt").exists() and not (h.root / "shell-executed").exists()
    )


@pytest.mark.parametrize(
    ("scenario", "reason", "code"),
    [
        ("malformed", "invalid_output", 1),
        ("duplicate_keys", "invalid_output", 1),
        ("missing_completion", "invalid_output", 1),
        ("different_final", "invalid_output", 1),
        ("quota", "usage_limit", 1),
        ("exit_error", "cli_error", 7),
        ("wrong_model", "metadata_error", 1),
        ("wrong_effort", "metadata_error", 1),
        ("missing_transcript", "metadata_error", 1),
        ("corrupt_transcript", "metadata_error", 1),
        ("invalid_usage", "invalid_output", 1),
        ("leftover", "descendants_reaped", 1),
        ("detached_leftover", "descendants_reaped", 1),
    ],
)
def test_failed_turns_preserve_evidence_and_never_claim_task_success(
    h: Harness, scenario: str, reason: str, code: int
) -> None:
    proc = h.run(scenario=scenario)
    assert proc.returncode == code, proc.stderr
    result = h.result()
    assert result["turn_end_reason"] == reason
    assert result["result"] == "failed" and result["task_success"] is None
    assert (h.root / "initial/events.jsonl").exists()
    assert (h.root / "initial/stderr.log").exists()
    assert result["runtime"]["session_id"]
    if scenario in {"leftover", "detached_leftover"}:
        assert not process_alive(int((h.root / "child.pid").read_text()))


@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_whitespace_final_is_not_a_completed_turn(h: Harness, agent: str) -> None:
    proc = h.run(agent=agent, prompt=" \n")
    assert proc.returncode == 1 and "empty final message" in proc.stderr
    assert h.result()["turn_end_reason"] == "invalid_output"


def test_resumed_runtime_must_report_the_same_session(h: Harness) -> None:
    assert h.run().returncode == 0
    proc = h.run(
        "bad-resume",
        action="resume",
        scenario="wrong_session",
        extra=("--resume-from", str(h.root / "initial")),
    )
    assert proc.returncode == 1
    assert "session differs" in h.result("bad-resume")["error"]
    assert h.result("bad-resume")["result"] == "failed"


def test_stdin_and_role_entry(h: Harness) -> None:
    cmd = h.cmd()
    cmd[cmd.index("--agent") : cmd.index("--agent") + 2] = ["--role", "implementation"]
    cmd[cmd.index("--prompt-file") + 1] = "-"
    proc = subprocess.run(
        cmd,
        env=h.env,
        text=True,
        input="stdin task\n",
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert Path(h.result()["final_path"]).read_text() == "stdin task\n"


def process_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2 :].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def wait_child(h: Harness) -> int:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if (h.root / "child.pid").exists():
            return int((h.root / "child.pid").read_text())
        time.sleep(0.05)
    raise AssertionError("fake child did not start")


@pytest.mark.parametrize("scenario", ["hang", "closed_stdout", "detached"])
def test_timeout_reaps_uncooperative_descendants(h: Harness, scenario: str) -> None:
    proc = h.run(scenario=scenario, extra=("--timeout", "1"))
    assert proc.returncode == 124, proc.stderr
    assert h.result()["turn_end_reason"] == "timeout"
    assert not process_alive(int((h.root / "child.pid").read_text()))


@pytest.mark.parametrize("operation", ["stop", "signal", "lost_runner"])
def test_stop_signal_and_lost_runner_recovery(h: Harness, operation: str) -> None:
    proc = subprocess.Popen(
        h.cmd(extra=("--timeout", "15")),
        env={**h.env, "FAKE_SCENARIO": "hang"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        child = wait_child(h)
        running = h.query("status")
        assert running.returncode == 0
        assert json.loads(running.stdout)["agent_alive"]
        if operation == "signal":
            proc.send_signal(signal.SIGINT)
        else:
            if operation == "lost_runner":
                proc.kill()
                proc.wait(timeout=5)
            stopped = h.query("stop")
            assert stopped.returncode == 0, stopped.stderr
        proc.communicate(timeout=10)
        assert not process_alive(child)
        result = h.result()
        assert result["turn_end_reason"] == "stopped" and result["result"] == "failed"
        assert result["task_success"] is None
        resumed = h.run(
            "continued",
            action="resume",
            extra=("--resume-from", str(h.root / "initial")),
        )
        assert resumed.returncode == 0, resumed.stderr
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.communicate(timeout=10)


def test_live_session_lock_rejects_concurrent_resume(h: Harness) -> None:
    proc = subprocess.Popen(
        h.cmd(extra=("--timeout", "15")),
        env={**h.env, "FAKE_SCENARIO": "hang"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_child(h)
        rejected = h.run(
            "concurrent",
            action="resume",
            extra=("--resume-from", str(h.root / "initial")),
        )
        assert rejected.returncode == 2 and "live runner" in rejected.stderr
        assert not (h.root / "concurrent").exists()
    finally:
        proc.terminate()
        proc.communicate(timeout=10)


@pytest.mark.parametrize("filename", ["result.json", "session.json"])
@pytest.mark.parametrize(
    "content",
    ['{"schema_version":999}', "{broken", '{"schema_version":1,"schema_version":1}'],
)
def test_corrupt_or_unknown_records_are_not_replaced(
    h: Harness, filename: str, content: str
) -> None:
    assert h.run().returncode == 0
    path = h.root / "initial" / filename
    path.write_text(content)
    proc = h.run(
        "rejected", action="resume", extra=("--resume-from", str(h.root / "initial"))
    )
    assert proc.returncode == 2
    assert path.read_text() == content
    assert not (h.root / "rejected").exists()


@pytest.mark.parametrize("agent", ["codex", "claude"])
def test_missing_usage_is_explicit_and_not_zero(h: Harness, agent: str) -> None:
    proc = h.run(agent=agent, scenario="no_usage")
    assert proc.returncode == 0, proc.stderr
    usage = h.result()["runtime"]["usage"]
    assert (
        usage["status"] == "unavailable" and usage["value"] is None and usage["reason"]
    )


def test_final_records_survive_reusing_an_external_output_path(h: Harness) -> None:
    alias = h.root / "latest.md"
    assert h.run(extra=("--out", str(alias))).returncode == 0
    initial = h.result()
    h.git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--allow-empty",
        "-qm",
        "implementation",
    )
    proc = h.run(
        "continued",
        action="resume",
        prompt="follow-up\n",
        extra=("--resume-from", str(h.root / "initial"), "--out", str(alias)),
    )
    assert proc.returncode == 0, proc.stderr
    assert alias.read_text() == "follow-up\n"
    assert Path(initial["final_path"]).read_text() == "implementation task\n"
    assert Path(h.result("continued")["final_path"]).read_text() == "follow-up\n"


def test_resume_cannot_overwrite_a_previous_record(h: Harness) -> None:
    assert h.run().returncode == 0
    original = (h.root / "initial/result.json").read_bytes()
    proc = h.run(
        "continued",
        action="resume",
        extra=(
            "--resume-from",
            str(h.root / "initial"),
            "--out",
            str(h.root / "initial/result.json"),
        ),
    )
    assert proc.returncode == 2
    assert (h.root / "initial/result.json").read_bytes() == original


def test_missing_executable_saves_a_failed_attempt(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(ROOT / ".agents/ops/lib"))
    runtime: Any = importlib.import_module("agent_runtime")
    monkeypatch.setattr(runtime, "enable_child_subreaper", lambda: None)
    monkeypatch.setattr(
        runtime, "command", lambda *args: [str(h.root / "missing-executable")]
    )
    result = runtime.run(runtime.parser().parse_args(h.cmd()[2:]))
    assert result["exit_code"] == 127 and result["turn_end_reason"] == "startup_failed"
    assert result["runtime"]["session_id"] is None
    assert h.result()["result"] == "failed"


@pytest.mark.parametrize("operation", ["resume", "stop_reused_pid"])
def test_existing_or_reused_process_identity_prevents_unsafe_actions(
    h: Harness, operation: str
) -> None:
    sys.path.insert(0, str(ROOT / ".agents/ops/lib"))
    runtime: Any = importlib.import_module("agent_runtime")
    assert h.run().returncode == 0
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time;time.sleep(20)"], start_new_session=True
    )
    try:
        result = h.result()
        result["process"] = runtime.process_identity(sleeper.pid)
        if operation == "stop_reused_pid":
            result["result"] = "running"
            result["process"]["start_ticks"] = "0"
        (h.root / "initial/result.json").write_text(json.dumps(result))
        if operation == "resume":
            proc = h.run(
                "rejected",
                action="resume",
                extra=("--resume-from", str(h.root / "initial")),
            )
        else:
            proc = h.query("stop")
        assert proc.returncode == 2, proc.stderr
        assert sleeper.poll() is None
    finally:
        sleeper.terminate()
        sleeper.wait(timeout=5)
