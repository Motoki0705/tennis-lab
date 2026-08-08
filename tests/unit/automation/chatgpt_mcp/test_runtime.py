from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from src.automation.chatgpt_mcp.runtime import RuntimeInstaller, RuntimeInstallError
from src.automation.chatgpt_mcp.settings import GatewaySettings


def _run(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(arguments),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def _source_checkout(tmp_path: Path) -> tuple[Path, Path, str]:
    source = tmp_path / "reviewed-source"
    _run("git", "init", "-q", "-b", "main", str(source))
    _run("git", "config", "user.email", "test@example.com", cwd=source)
    _run("git", "config", "user.name", "Test", cwd=source)

    package = source / "src/automation/chatgpt_mcp"
    package.mkdir(parents=True)
    (source / "src/__init__.py").write_text("", encoding="utf-8")
    (source / "src/automation/__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    configuration = source / "src/utils/configuration"
    configuration.mkdir(parents=True)
    for module_name in ("errors.py", "schema.py", "paths.py"):
        (configuration / module_name).write_text("VALUE = 1\n", encoding="utf-8")
    queue = source / ".agents/skills/training-queue/scripts/training_queue.sh"
    queue.parent.mkdir(parents=True)
    queue.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    os.chmod(queue, 0o700)

    _run("git", "add", "src", ".agents", cwd=source)
    _run("git", "commit", "-qm", "runtime source", cwd=source)
    revision = _run("git", "rev-parse", "HEAD", cwd=source)
    remote = tmp_path / "origin.git"
    _run("git", "clone", "-q", "--bare", str(source), str(remote))
    _run("git", "remote", "add", "origin", str(remote), cwd=source)
    return source, remote, revision


def _settings(tmp_path: Path, remote: Path) -> GatewaySettings:
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    settings = GatewaySettings(
        repo_root=project,
        state_dir=tmp_path / "state",
        control_dir=tmp_path / "control",
        public_base_url=None,
        origin_url=str(remote),
        gpu_lock_file=tmp_path / "gpu.lock",
        uv_python_root=tmp_path / "uv",
    )
    settings.uv_python_root.mkdir()
    trusted_python = settings.runtime_venv_root / "bin/python"
    trusted_python.parent.mkdir(parents=True)
    trusted_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    os.chmod(trusted_python, 0o700)
    return settings


def test_installer_externalizes_clean_exact_runtime_venv_queue_and_git_mirror(
    tmp_path: Path,
) -> None:
    source, remote, revision = _source_checkout(tmp_path)
    settings = _settings(tmp_path, remote)

    first = RuntimeInstaller(settings).install(source, expected_sha=revision)
    second = RuntimeInstaller(settings).install(source, expected_sha=revision)

    assert first.revision == revision
    assert second.revision == first.revision
    assert settings.project_venv_link.is_symlink()
    assert settings.project_venv_link.resolve() == settings.runtime_venv_root
    assert first.python_executable == settings.runtime_venv_root / "bin/python"
    assert settings.runtime_current_dir.resolve() == first.release_dir
    assert first.release_dir.joinpath("src/utils/configuration/paths.py").is_file()
    assert settings.trusted_queue_script.is_file()
    assert os.access(settings.trusted_queue_script, os.X_OK)
    assert (
        _run(
            "git",
            "--git-dir",
            str(settings.trusted_git_dir),
            "rev-parse",
            "--is-bare-repository",
        )
        == "true"
    )
    assert settings.runtime_version_path.read_text(encoding="utf-8").strip() == revision


def test_installer_rejects_canonical_project_as_runtime_source(tmp_path: Path) -> None:
    source, remote, revision = _source_checkout(tmp_path)
    settings = _settings(tmp_path, remote)
    settings = GatewaySettings(
        repo_root=source,
        state_dir=settings.state_dir,
        control_dir=settings.control_dir,
        public_base_url=None,
        origin_url=str(remote),
        gpu_lock_file=tmp_path / "gpu.lock",
        uv_python_root=settings.uv_python_root,
    )

    with pytest.raises(RuntimeInstallError, match="separate clean reviewed checkout"):
        RuntimeInstaller(settings).install(source, expected_sha=revision)


def test_installer_rejects_dirty_or_wrong_revision_source(tmp_path: Path) -> None:
    source, remote, revision = _source_checkout(tmp_path)
    settings = _settings(tmp_path, remote)

    with pytest.raises(RuntimeInstallError, match="expected"):
        RuntimeInstaller(settings).install(source, expected_sha="f" * 40)

    (source / "untracked.txt").write_text("unsafe\n", encoding="utf-8")
    with pytest.raises(RuntimeInstallError, match="completely clean"):
        RuntimeInstaller(settings).install(source, expected_sha=revision)
