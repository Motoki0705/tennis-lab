from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from src.automation.chatgpt_mcp.jobs import (
    DockerSandbox,
    JobError,
    JobManager,
    SandboxSpec,
    TrainingQueueManager,
    _redact_secrets,
)
from src.automation.chatgpt_mcp.settings import GatewaySettings
from src.automation.chatgpt_mcp.storage import SqliteStore
from src.automation.chatgpt_mcp.workspace import RevisionWorkspace

_REVISION = "1" * 40
_WORKSPACE_ID = "rev-0123456789abcdef"


class StubWorkspaces:
    def __init__(self, source: Path) -> None:
        self.source = source

    def assert_execution_ready(
        self, *, workspace_id: str, expected_sha: str
    ) -> RevisionWorkspace:
        assert workspace_id == _WORKSPACE_ID
        assert expected_sha == _REVISION
        return RevisionWorkspace(
            workspace_id=workspace_id,
            path=self.source,
            branch="feature/test",
            revision=expected_sha,
        )


def _settings(tmp_path: Path) -> tuple[GatewaySettings, StubWorkspaces]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".venv/bin").mkdir(parents=True)
    uv_root = tmp_path / "uv-python"
    uv_root.mkdir()
    settings = GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        public_base_url=None,
        uv_python_root=uv_root,
    )
    settings.ensure_state()
    source = settings.revision_workspace_dir / _WORKSPACE_ID
    source.mkdir(parents=True)
    (source / "example.py").write_text("print('ok')\n", encoding="utf-8")
    return settings, StubWorkspaces(source)


def test_sandbox_mounts_only_read_only_source_and_private_job_copy(
    tmp_path: Path,
) -> None:
    settings, workspaces = _settings(tmp_path)
    spec = SandboxSpec(
        job_id="job-0123456789abcdef",
        command="python -m pytest",
        workspace_id=_WORKSPACE_ID,
        expected_sha=_REVISION,
        use_gpu=False,
        timeout_seconds=60,
    )

    command = DockerSandbox(settings, workspaces).command(spec, detached=True)
    joined = " ".join(command)
    mounts = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--mount"
    ]

    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pull never" in joined
    assert "--gpus all" not in joined
    assert "/var/run/docker.sock" not in joined
    assert "/mnt/c" not in joined
    assert any("dst=/source,readonly" in mount for mount in mounts)
    assert any("dst=/workspace" in mount and "readonly" not in mount for mount in mounts)
    assert any("dst=/source/.git,readonly" in mount for mount in mounts)
    assert not any(
        mount.startswith(f"type=bind,src={settings.repo_root},") for mount in mounts
    )
    assert command[-1] == spec.command


def test_gpu_flag_is_available_only_to_queue_specs(tmp_path: Path) -> None:
    settings, workspaces = _settings(tmp_path)
    spec = SandboxSpec(
        job_id="train-0123456789abcdef",
        command="python -c 'import torch; print(torch.cuda.is_available())'",
        workspace_id=_WORKSPACE_ID,
        expected_sha=_REVISION,
        use_gpu=True,
        timeout_seconds=60,
    )

    command = DockerSandbox(settings, workspaces).command(spec, detached=False)

    assert "--gpus" in command
    assert command[command.index("--gpus") + 1] == "all"
    assert command[command.index("--network") + 1] == "none"


def test_direct_commands_are_cpu_only_and_time_bounded(tmp_path: Path) -> None:
    settings, workspaces = _settings(tmp_path)
    manager = JobManager(
        settings,
        SqliteStore(settings.database_path),
        workspaces,
    )

    with pytest.raises(JobError, match="direct commands are limited"):
        manager.start(
            command="sleep 3600",
            workspace_id=_WORKSPACE_ID,
            expected_sha=_REVISION,
            timeout_seconds=1801,
        )


def test_training_queue_uses_generated_safe_metadata_and_isolated_bootstrap(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    settings, workspaces = _settings(tmp_path)
    store = SqliteStore(settings.database_path)
    manager = TrainingQueueManager(settings, store, workspaces)
    queue_commands: list[tuple[list[str], Path]] = []

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        cwd = Path(str(kwargs["cwd"]))
        queue_commands.append((command, cwd))
        stdout = "queued: 123_private-training.job\n" if "add" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr("src.automation.chatgpt_mcp.jobs.subprocess.run", fake_run)

    result = manager.enqueue(
        name="private-training",
        command="python -c 'print(1)'",
        workspace_id=_WORKSPACE_ID,
        expected_sha=_REVISION,
        issue=716,
        timeout_seconds=60,
    )

    add_command, add_cwd = queue_commands[0]
    runner = add_command[3]
    session = add_command[add_command.index("--session") + 1]
    spec_path = next(settings.job_specs_dir.glob("train-*.json"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    assert result["status"] == "queued"
    assert add_cwd == workspaces.source
    assert session.startswith("train-")
    assert "\n" not in session and "\r" not in session
    assert " -I " in runner
    assert "src.automation.chatgpt_mcp.sandbox_exec" in runner
    assert "TENNIS_MCP_PUBLIC_BASE_URL" not in runner
    assert spec["workspace_id"] == _WORKSPACE_ID
    assert spec["expected_sha"] == _REVISION
    assert spec["use_gpu"] is True
    assert "network_access" not in spec


def test_secret_redaction_covers_common_runtime_tokens() -> None:
    value = "token sk-example_12345678901234567890 and Bearer abcdefghijklmnop"

    redacted = _redact_secrets(value)

    assert "sk-example" not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert redacted.count("[REDACTED]") == 2
