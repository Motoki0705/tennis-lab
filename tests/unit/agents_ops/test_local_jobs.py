"""Unit contracts for detached local command execution and observation."""

from __future__ import annotations

import importlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
JOB = "20261124T010203Z_0123456789ab_sample"


@pytest.fixture
def local_jobs(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("parent.local_jobs")


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("parent.worker_wait_sources")


def _write_meta(
    local_jobs: Any,
    root: Path,
    job_id: str,
    *,
    cwd: Path,
    command: list[str],
    name: str = "sample",
) -> Path:
    job_dir = root / job_id
    job_dir.mkdir(parents=True, mode=0o700)
    local_jobs.write_json_atomic(
        job_dir / "meta.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "name": name,
            "cwd": str(cwd),
            "command": command,
            "started_at": "2026-11-24T01:02:03+00:00",
            "unit": f"tennis-lab-local-job-{job_id}",
            "log_path": str(job_dir / "output.log"),
            "runner_pid": None,
            "process": None,
        },
    )
    return job_dir


def test_jobs_root_matches_state_directory_and_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_jobs: Any
) -> None:
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_STATE_HOME", str(xdg))
    monkeypatch.delenv("PARENT_LOCAL_JOBS_DIR", raising=False)
    assert (
        local_jobs.jobs_root() == xdg / "tennis-lab-agents/parent-sessions/local-jobs"
    )

    monkeypatch.setenv("PARENT_LOCAL_JOBS_DIR", "")
    assert (
        local_jobs.jobs_root() == xdg / "tennis-lab-agents/parent-sessions/local-jobs"
    )

    custom = tmp_path / "custom-jobs"
    monkeypatch.setenv("PARENT_LOCAL_JOBS_DIR", str(custom))
    assert local_jobs.jobs_root() == custom

    monkeypatch.setenv("PARENT_LOCAL_JOBS_DIR", "relative/path")
    with pytest.raises(ValueError, match="must be absolute"):
        local_jobs.jobs_root()


@pytest.mark.parametrize(
    ("exit_code", "expected_state", "output_script"),
    [
        (
            0,
            "done",
            "import sys; print('line 1', flush=True); print('all ok', file=sys.stderr)",
        ),
        (
            7,
            "failed",
            "import sys; print('starting check'); print('ERROR: test failed on line 9', file=sys.stderr); sys.exit(7)",
        ),
    ],
)
def test_run_job_records_completion_and_log_excerpt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_jobs: Any,
    sources: Any,
    exit_code: int,
    expected_state: str,
    output_script: str,
) -> None:
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    cwd = tmp_path / "work"
    cwd.mkdir()
    job_dir = _write_meta(
        local_jobs,
        root,
        JOB,
        cwd=cwd,
        command=[sys.executable, "-c", output_script],
    )

    with tempfile.TemporaryDirectory(prefix="notify-") as short:
        notify_path = str(Path(short) / "notify.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_sock:
            notify_sock.bind(notify_path)
            notify_sock.settimeout(5)
            monkeypatch.setenv("NOTIFY_SOCKET", notify_path)
            assert local_jobs.run_job(root, JOB) == 0
            assert notify_sock.recv(1024) == b"READY=1"

    completion = json.loads((job_dir / "completion.json").read_text(encoding="utf-8"))
    assert completion["schema_version"] == 1
    assert completion["job_id"] == JOB
    assert completion["name"] == "sample"
    assert completion["state"] == expected_state
    assert completion["exit_code"] == exit_code
    assert completion["duration_seconds"] >= 0.0
    assert completion["log_path"] == str(job_dir / "output.log")

    observed = sources.local_result(root, JOB, now=10.0, missing_completion_since={})
    assert observed is not None
    assert observed["state"] == expected_state
    assert observed["exit_code"] == exit_code
    assert observed["log"] == str(job_dir / "output.log")
    if exit_code == 0:
        assert observed["log_excerpt"] == "line 1\nall ok"
    else:
        assert "ERROR: test failed on line 9" in observed["log_excerpt"]


def test_run_job_missing_executable_records_failed_completion(
    tmp_path: Path, local_jobs: Any, sources: Any
) -> None:
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    job_dir = _write_meta(
        local_jobs,
        root,
        JOB,
        cwd=tmp_path,
        command=[str(tmp_path / "no-such-binary")],
    )
    assert local_jobs.run_job(root, JOB) == 0
    assert (job_dir / "completion.json").is_file()
    observed = sources.local_result(root, JOB, now=0.0, missing_completion_since={})
    assert observed is not None
    assert observed["state"] == "failed"
    assert observed["exit_code"] == 127
    assert "failed to execute command" in observed["log_excerpt"]


def test_run_job_sigterm_cancels_child_and_records_cancelled(
    tmp_path: Path, local_jobs: Any, sources: Any
) -> None:
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    with tempfile.TemporaryDirectory(prefix="local-cancel-") as short:
        notify_path = Path(short) / "notify.sock"
        gate_path = Path(short) / "gate.sock"
        child_script = (
            "import socket, sys, time; "
            f"s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); "
            f"s.connect({str(gate_path)!r}); "
            "s.sendall(b'child_ready'); "
            "s.recv(1)"
        )
        job_dir = _write_meta(
            local_jobs,
            root,
            JOB,
            cwd=tmp_path,
            command=[sys.executable, "-c", child_script],
        )
        with (
            socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as notify_sock,
            socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as gate_server,
        ):
            notify_sock.bind(str(notify_path))
            notify_sock.settimeout(5)
            gate_server.bind(str(gate_path))
            gate_server.listen(1)
            gate_server.settimeout(5)
            wrapper = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "parent.local_jobs",
                    "_run",
                    str(root),
                    JOB,
                ],
                env={
                    **os.environ,
                    "PYTHONPATH": str(OPS),
                    "NOTIFY_SOCKET": str(notify_path),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert notify_sock.recv(1024) == b"READY=1"
                conn, _ = gate_server.accept()
                with conn:
                    assert conn.recv(64) == b"child_ready"
                    wrapper.send_signal(signal.SIGTERM)
                    out, err = wrapper.communicate(timeout=10)
                    assert wrapper.returncode == 0, out + err
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.communicate(timeout=5)
    observed = sources.local_result(root, JOB, now=0.0, missing_completion_since={})
    assert observed is not None
    assert observed["state"] == "cancelled"
    assert observed["exit_code"] == 128 + signal.SIGTERM
    assert (job_dir / "completion.json").is_file()


def test_start_invokes_systemd_run_and_validates_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_jobs: Any
) -> None:
    root = tmp_path / "jobs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_log = tmp_path / "systemd-calls.jsonl"
    fake_systemd = bin_dir / "systemd-run"
    fake_systemd.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"with Path({str(calls_log)!r}).open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if os.environ.get('FAIL_SYSTEMD'):\n"
        "    print('unit launch refused', file=sys.stderr)\n"
        "    sys.exit(1)\n",
        encoding="utf-8",
    )
    fake_systemd.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PARENT_LOCAL_JOBS_DIR", str(root))

    started = local_jobs.start_job(
        name="pytest-unit",
        positional=[".venv/bin/pytest", "-q", "tests/unit/agents_ops"],
        cwd=str(tmp_path),
    )
    job_id = started["job_id"]
    assert started["unit"] == f"tennis-lab-local-job-{job_id}"
    assert Path(started["log_path"]) == root / job_id / "output.log"
    job_dir = root / job_id
    assert (job_dir.stat().st_mode & 0o777) == 0o700
    assert (job_dir / "meta.json").is_file()
    meta = json.loads((job_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["command"] == [".venv/bin/pytest", "-q", "tests/unit/agents_ops"]
    assert meta["cwd"] == str(tmp_path.resolve())

    args = json.loads(calls_log.read_text(encoding="utf-8").splitlines()[0])
    assert "--user" in args
    assert "--quiet" in args
    assert "--collect" in args
    assert f"--unit=tennis-lab-local-job-{job_id}" in args
    assert "--service-type=notify" in args
    assert "--property=TimeoutStartSec=30s" in args
    assert "--property=UMask=0077" in args
    assert f"--property=WorkingDirectory={tmp_path.resolve()}" in args
    assert any(flag.startswith("--setenv=PYTHONPATH=") for flag in args)
    assert args[args.index("--") + 2 :] == [
        "-m",
        "parent.local_jobs",
        "_run",
        str(root),
        job_id,
    ]

    shell_started = local_jobs.start_job(
        name="pipe_check",
        positional=[],
        shell="echo ok | wc -l",
        cwd=str(tmp_path),
    )
    shell_meta = json.loads(
        (root / shell_started["job_id"] / "meta.json").read_text(encoding="utf-8")
    )
    assert shell_meta["command"] == ["bash", "-lc", "echo ok | wc -l"]

    monkeypatch.setenv("FAIL_SYSTEMD", "1")
    with pytest.raises(ValueError, match="unit launch refused"):
        local_jobs.start_job(name="fail-launch", positional=["echo", "hi"])
    failed_dirs = [p for p in root.iterdir() if p.name.endswith("_fail-launch")]
    assert len(failed_dirs) == 1
    failed_completion = json.loads((failed_dirs[0] / "completion.json").read_text())
    assert failed_completion["state"] == "failed"
    assert failed_completion["exit_code"] == 1

    for kwargs in (
        {"name": "../bad", "positional": ["echo"]},
        {"name": "-bad", "positional": ["echo"]},
        {"name": "bad name", "positional": ["echo"]},
        {"name": "ok", "positional": []},
        {"name": "ok", "positional": ["echo"], "shell": "echo"},
        {"name": "ok", "positional": [], "shell": "   "},
        {"name": "ok", "positional": ["echo"], "cwd": str(tmp_path / "missing")},
    ):
        with pytest.raises(ValueError):
            local_jobs.start_job(**kwargs)


def test_local_result_live_race_grace_and_corruption(
    tmp_path: Path, local_jobs: Any, sources: Any
) -> None:
    runtime = importlib.import_module("dispatcher.runtime")
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)

    with pytest.raises(ValueError, match="unknown local job_id"):
        sources.local_result(root, JOB, now=0.0, missing_completion_since={})
    with pytest.raises(ValueError, match="unknown local job_id"):
        sources.local_result(root, f"../{JOB}", now=0.0, missing_completion_since={})

    job_dir = _write_meta(local_jobs, root, JOB, cwd=tmp_path, command=["sleep"])
    live_proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        identity = runtime.process_identity(live_proc.pid)
        assert identity is not None
        local_jobs.write_json_atomic(
            job_dir / "running.json",
            {
                "schema_version": 1,
                "job_id": JOB,
                "runner_pid": live_proc.pid,
                "process": identity,
                "started_at": "2026-11-24T01:02:03+00:00",
            },
        )
        missing: dict[str, float] = {"stale": 1.0, JOB: 2.0}
        assert (
            sources.local_result(root, JOB, now=5.0, missing_completion_since=missing)
            is None
        )
        assert JOB not in missing
    finally:
        assert live_proc.stdin is not None
        live_proc.stdin.close()
        live_proc.wait(timeout=5)

    missing = {}
    assert (
        sources.local_result(root, JOB, now=100.0, missing_completion_since=missing)
        is None
    )
    assert (
        sources.local_result(root, JOB, now=109.9, missing_completion_since=missing)
        is None
    )
    with pytest.raises(
        ValueError, match=f"local job terminated without completion record: {JOB}"
    ):
        sources.local_result(root, JOB, now=110.0, missing_completion_since=missing)

    # Race guard: completion appears just before re-check after dead running.json.
    calls = 0
    original_read_completion = sources.read_completion

    def delayed_completion(passed_root: Path, passed_job: str) -> dict[str, Any] | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            return None
        return {
            "schema_version": 1,
            "job_id": passed_job,
            "state": "done",
            "exit_code": 0,
            "duration_seconds": 1.25,
        }

    try:
        sources.read_completion = delayed_completion
        race_res = sources.local_result(
            root, JOB, now=200.0, missing_completion_since=missing
        )
        assert race_res is not None
        assert race_res["state"] == "done" and race_res["exit_code"] == 0
    finally:
        sources.read_completion = original_read_completion

    # Corrupt completion.json is rejected strictly.
    (job_dir / "completion.json").write_text("{corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid local job completion record"):
        sources.local_result(root, JOB, now=300.0, missing_completion_since={})

    (job_dir / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": JOB,
                "state": "done",
                "exit_code": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid local job completion record"):
        sources.local_result(root, JOB, now=300.0, missing_completion_since={})


def test_cli_start_and_status_inspect_live_and_completed_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, local_jobs: Any, capsys: Any
) -> None:
    root = tmp_path / "jobs"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_systemd = bin_dir / "systemd-run"
    fake_systemd.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "cmd = args[args.index('--') + 1:]\n"
        "env = dict(os.environ)\n"
        "for flag in args:\n"
        "    if flag.startswith('--setenv=') and '=' in flag.removeprefix('--setenv='):\n"
        "        k, _, v = flag.removeprefix('--setenv=').partition('=')\n"
        "        env[k] = v\n"
        "subprocess.run(cmd, env=env, check=True)\n",
        encoding="utf-8",
    )
    fake_systemd.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("PARENT_LOCAL_JOBS_DIR", str(root))

    assert (
        local_jobs.main(
            [
                "start",
                "--name",
                "echo_ok",
                "--cwd",
                str(tmp_path),
                "--",
                sys.executable,
                "-c",
                "print('cli output ok')",
            ]
        )
        == 0
    )
    started = json.loads(capsys.readouterr().out)
    job_id = started["job_id"]

    assert local_jobs.main(["status", job_id]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["job_id"] == job_id
    assert status["state"] == "done"
    assert status["exit_code"] == 0
    assert status["runner_alive"] is False
    assert status["log_excerpt"] == "cli output ok"

    assert local_jobs.main(["status", "20261124T010203Z_0123456789ab_missing"]) == 2
    assert "unknown local job_id" in capsys.readouterr().err
