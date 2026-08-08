from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError
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
    control = tmp_path / "control"
    versioned_venv = control / "venvs/test-runtime/bin"
    versioned_venv.mkdir(parents=True)
    (versioned_venv / "python").write_text("", encoding="utf-8")
    (control / "venv").symlink_to("venvs/test-runtime", target_is_directory=True)
    runtime = control / "current"
    (runtime / "src/automation/chatgpt_mcp").mkdir(parents=True)
    queue = control / "bin/training_queue.sh"
    queue.parent.mkdir(parents=True)
    queue.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    queue.chmod(0o700)
    uv_root = tmp_path / "uv-python"
    uv_root.mkdir()
    settings = GatewaySettings(
        repo_root=repo,
        state_dir=tmp_path / "state",
        control_dir=control,
        public_base_url=None,
        uv_python_root=uv_root,
    )
    settings.ensure_state()
    source = settings.revision_workspace_dir / _WORKSPACE_ID
    source.mkdir(parents=True)
    (source / "example.py").write_text("print('ok')\n", encoding="utf-8")
    (source / ".git").write_text("gitdir: /trusted/worktrees/test\n", encoding="utf-8")
    return settings, StubWorkspaces(source)


def _spec(
    *,
    job_id: str = "job-0123456789abcdef",
    use_gpu: bool = False,
    execution_root: str = "revision",
) -> SandboxSpec:
    return SandboxSpec(
        job_id=job_id,
        command="python -m pytest --disable-warnings",
        workspace_id=_WORKSPACE_ID,
        expected_sha=_REVISION,
        execution_root=execution_root,
        working_directory="tests",
        use_gpu=use_gpu,
        timeout_seconds=60,
    )


def test_sandbox_exposes_full_project_rw_but_keeps_control_plane_unmounted(
    tmp_path: Path,
) -> None:
    settings, workspaces = _settings(tmp_path)
    spec = _spec()

    command = DockerSandbox(settings, workspaces).command(spec, detached=True)
    joined = " ".join(command)
    mounts = [
        command[index + 1] for index, value in enumerate(command) if value == "--mount"
    ]
    command_path = settings.sandbox_jobs_dir / spec.job_id / "command"

    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pull never" in joined
    assert "--init" in command
    assert command[command.index("--memory") + 1] == "24g"
    assert command[command.index("--shm-size") + 1] == "4g"
    assert "--gpus all" not in joined
    assert "/var/run/docker.sock" not in joined
    assert "/mnt/c" not in joined
    assert any("dst=/source,readonly" in mount for mount in mounts)
    assert any(
        "dst=/workspace" in mount and "readonly" not in mount for mount in mounts
    )
    assert any(
        f"src={settings.repo_root},dst=/tennis-lab" in mount and "readonly" not in mount
        for mount in mounts
    )
    assert any("dst=/source/.git,readonly" in mount for mount in mounts)
    assert any("dst=/tennis-lab/.git,readonly" in mount for mount in mounts)
    assert any("dst=/run/tennis-mcp-command,readonly" in mount for mount in mounts)
    assert any(
        f"src={settings.runtime_venv_root.resolve()},dst={settings.runtime_venv_root}"
        in mount
        and "readonly" in mount
        for mount in mounts
    )
    assert any(
        f"src={settings.runtime_venv_root.resolve()},"
        f"dst={settings.runtime_venv_root.resolve()}"
        in mount
        and "readonly" in mount
        for mount in mounts
    )
    assert not any(str(settings.control_dir / "current") in mount for mount in mounts)
    assert spec.command not in joined
    assert command_path.read_text(encoding="utf-8") == spec.command
    assert stat.S_IMODE(command_path.stat().st_mode) == 0o600
    wrapper = command[-1]
    for root in ("data", "outputs", "ckpt", "artifacts", ".cache", "third_party"):
        assert root in wrapper
    assert "cd -- tests" in wrapper


def test_project_execution_root_runs_from_full_rw_project(tmp_path: Path) -> None:
    settings, workspaces = _settings(tmp_path)
    spec = _spec(execution_root="project")

    command = DockerSandbox(settings, workspaces).command(spec, detached=False)

    assert "cd /tennis-lab" in command[-1]
    assert "cd -- tests" in command[-1]


def test_working_directory_cannot_escape_selected_root() -> None:
    with pytest.raises(ValidationError, match="may not escape"):
        SandboxSpec(
            job_id="job-0123456789abcdef",
            command="true",
            workspace_id=_WORKSPACE_ID,
            expected_sha=_REVISION,
            working_directory="../outside",
        )


def test_started_container_does_not_retain_host_command_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    settings, workspaces = _settings(tmp_path)
    sandbox = DockerSandbox(settings, workspaces)
    spec = _spec()

    monkeypatch.setattr(
        "src.automation.chatgpt_mcp.jobs.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, "container-id\n", ""
        ),
    )

    assert sandbox.start(spec) == "container-id"
    assert not (settings.sandbox_jobs_dir / spec.job_id / "command").exists()


def test_gpu_flag_is_available_to_queue_specs_without_network(tmp_path: Path) -> None:
    settings, workspaces = _settings(tmp_path)
    spec = _spec(job_id="train-0123456789abcdef", use_gpu=True)

    command = DockerSandbox(settings, workspaces).command(spec, detached=False)

    assert command[command.index("--gpus") + 1] == "all"
    assert command[command.index("--memory") + 1] == "48g"
    assert command[command.index("--shm-size") + 1] == "8g"
    assert command[command.index("--network") + 1] == "none"
    assert spec.command not in " ".join(command)


def test_direct_commands_are_cpu_only_but_allow_long_local_validation(
    tmp_path: Path,
) -> None:
    settings, workspaces = _settings(tmp_path)
    manager = JobManager(
        settings,
        SqliteStore(settings.database_path),
        workspaces,
    )

    with pytest.raises(JobError, match="direct commands are limited"):
        manager.start(
            command="sleep 90000",
            workspace_id=_WORKSPACE_ID,
            expected_sha=_REVISION,
            execution_root="project",
            working_directory=".",
            timeout_seconds=24 * 3600 + 1,
        )


def test_training_queue_uses_external_runner_and_safe_private_spec(
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
    user_command = "python -c 'print(1)'"

    result = manager.enqueue(
        name="private-training",
        command=user_command,
        workspace_id=_WORKSPACE_ID,
        expected_sha=_REVISION,
        execution_root="project",
        working_directory="src/tasks",
        issue=716,
        timeout_seconds=60,
    )

    add_command, add_cwd = queue_commands[0]
    runner = add_command[3]
    session = add_command[add_command.index("--session") + 1]
    spec_path = next(settings.job_specs_dir.glob("train-*.json"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    assert result["status"] == "queued"
    assert add_command[1] == str(settings.trusted_queue_script)
    assert add_cwd == workspaces.source
    assert session.startswith("train-")
    assert "\n" not in session and "\r" not in session
    assert " -I " in runner
    assert str(settings.runtime_current_dir) in runner
    assert "src.automation.chatgpt_mcp.sandbox_exec" in runner
    assert user_command not in runner
    assert spec["workspace_id"] == _WORKSPACE_ID
    assert spec["expected_sha"] == _REVISION
    assert spec["use_gpu"] is True
    assert spec["execution_root"] == "project"
    assert spec["working_directory"] == "src/tasks"
    add_environment = queue_commands[0][0]
    assert add_environment[0] == "bash"
    assert manager._queue_environment()["TRAINING_QUEUE_LOCK_FILE"] == str(
        settings.gpu_lock_file
    )


def test_cancel_queued_training_moves_only_external_queue_job(
    tmp_path: Path,
) -> None:
    settings, workspaces = _settings(tmp_path)
    store = SqliteStore(settings.database_path)
    manager = TrainingQueueManager(settings, store, workspaces)
    job_id = "train-0123456789abcdef"
    queue_file = "123_cancel-me.job"
    queued = settings.trusted_queue_dir / "jobs" / queue_file
    queued.parent.mkdir(parents=True)
    queued.write_text("trusted bootstrap\n", encoding="utf-8")
    spec = settings.job_specs_dir / f"{job_id}.json"
    spec.write_text("{}\n", encoding="utf-8")
    store.put(
        "training_jobs",
        job_id,
        {
            "job_id": job_id,
            "queue_file": queue_file,
            "created_at": 1.0,
        },
    )

    result = manager.cancel(job_id)

    assert result == {"job_id": job_id, "status": "cancelled"}
    assert not queued.exists()
    assert (settings.trusted_queue_dir / "cancelled" / queue_file).is_file()
    assert not spec.exists()
    assert not any(settings.repo_root.rglob(queue_file))


def test_execution_layout_documents_destructive_project_boundary(
    tmp_path: Path,
) -> None:
    settings, workspaces = _settings(tmp_path)

    layout = DockerSandbox(settings, workspaces).execution_layout()

    assert layout["project_root"] == "/tennis-lab"
    assert "read-write" in layout["project_access"]
    assert layout["network"] == "disabled"
    assert layout["direct_memory_limit_gb"] == 24
    assert layout["queued_memory_limit_gb"] == 48
    assert layout["direct_concurrency"] == 2
    assert "Docker socket is not mounted" in layout["host_boundaries"]


def test_secret_redaction_covers_common_runtime_tokens() -> None:
    value = "token sk-example_12345678901234567890 and Bearer abcdefghijklmnop"

    redacted = _redact_secrets(value)

    assert "sk-example" not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert redacted.count("[REDACTED]") == 2
