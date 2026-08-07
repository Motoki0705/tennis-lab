from __future__ import annotations

import subprocess
from pathlib import Path

from pytest import MonkeyPatch

from src.automation.chatgpt_mcp.jobs import (
    DockerSandbox,
    SandboxSpec,
    TrainingQueueManager,
)
from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore


def _settings(tmp_path: Path) -> GatewaySettings:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    uv_root = tmp_path / "uv-python"
    uv_root.mkdir()
    return GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        public_base_url="https://mcp.example.test",
        uv_python_root=uv_root,
    )


def test_sandbox_command_mounts_only_repo_and_uv_runtime(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    spec = SandboxSpec(
        job_id="job-0123456789abcdef",
        command="python -m pytest",
        workspace=str(settings.repo_root),
        use_gpu=True,
        network_access=False,
        timeout_seconds=60,
    )

    command = DockerSandbox(settings).command(spec, detached=True)
    joined = " ".join(command)
    assert "--gpus all" in joined
    assert "--network none" in joined
    assert "--cap-drop ALL" in joined
    assert "/var/run/docker.sock" not in joined
    assert str(settings.repo_root) in joined
    volumes = [
        command[index + 1] for index, value in enumerate(command) if value == "--volume"
    ]
    assert volumes == [
        f"{settings.repo_root}:{settings.repo_root}",
        f"{settings.uv_python_root}:{settings.uv_python_root}:ro",
    ]
    assert spec.command == command[-1]


def test_training_status_preserves_queue_result(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    store = SqliteStore(settings.state_dir / "gateway.sqlite3")
    manager = TrainingQueueManager(settings, store)
    queue_file = "123_cuda-smoke.job"
    (manager.queue_dir / "done").mkdir(parents=True)
    (manager.queue_dir / "done" / queue_file).touch()
    store.put(
        "training_jobs",
        "train-0123456789abcdef",
        {
            "job_id": "train-0123456789abcdef",
            "queue_file": queue_file,
        },
    )
    monkeypatch.setattr(
        manager.sandbox,
        "inspect",
        lambda _job_id: {
            "status": "exited",
            "running": False,
            "exit_code": 0,
            "started_at": "start",
            "finished_at": "finish",
            "error": None,
        },
    )

    result = manager.status("train-0123456789abcdef")

    assert result["status"] == "succeeded"
    assert result["container_status"] == "exited"
    assert result["exit_code"] == 0


def test_private_tunnel_training_runner_does_not_require_public_url(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    original = _settings(tmp_path)
    settings = GatewaySettings(
        repo_root=original.repo_root,
        state_dir=original.state_dir,
        public_base_url=None,
        uv_python_root=original.uv_python_root,
    )
    settings.ensure_state()
    manager = TrainingQueueManager(
        settings, SqliteStore(settings.state_dir / "gateway.sqlite3")
    )
    queue_commands: list[list[str]] = []
    monkeypatch.setattr(
        manager.sandbox.workspaces,
        "resolve_workspace",
        lambda _workspace: settings.repo_root,
    )

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        queue_commands.append(command)
        stdout = "queued: 123_private-training.job\n" if "add" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr("src.automation.chatgpt_mcp.jobs.subprocess.run", fake_run)

    result = manager.enqueue(
        name="private-training",
        command="python -c 'print(1)'",
        workspace=str(settings.repo_root),
        issue=None,
        session="pytest",
        network_access=False,
        timeout_seconds=60,
    )

    runner = queue_commands[0][3]
    assert result["status"] == "queued"
    assert "TENNIS_MCP_PUBLIC_BASE_URL" not in runner
    assert "src.automation.chatgpt_mcp.sandbox_exec" in runner
