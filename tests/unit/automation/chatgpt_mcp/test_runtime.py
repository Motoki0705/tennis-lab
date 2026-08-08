from __future__ import annotations

import os
import subprocess
from pathlib import Path

from src.automation.chatgpt_mcp.runtime import RuntimeInstaller
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


def _source_checkout(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
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

    venv_python = source / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    os.chmod(venv_python, 0o700)

    _run("git", "add", "src", ".agents", cwd=source)
    _run("git", "commit", "-qm", "runtime source", cwd=source)
    remote = tmp_path / "origin.git"
    _run("git", "clone", "-q", "--bare", str(source), str(remote))
    return source, remote


def test_installer_externalizes_runtime_venv_queue_and_git_mirror(
    tmp_path: Path,
) -> None:
    source, remote = _source_checkout(tmp_path)
    settings = GatewaySettings(
        repo_root=source,
        state_dir=tmp_path / "state",
        control_dir=tmp_path / "control",
        public_base_url=None,
        origin_url=str(remote),
        uv_python_root=tmp_path / "uv",
    )
    settings.uv_python_root.mkdir()
    trusted_python = settings.runtime_venv_root / "bin/python"
    trusted_python.parent.mkdir(parents=True)
    trusted_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    os.chmod(trusted_python, 0o700)

    first = RuntimeInstaller(settings).install(source)
    second = RuntimeInstaller(settings).install(source)

    assert first.revision == _run("git", "rev-parse", "HEAD", cwd=source)
    assert second.revision == first.revision
    assert source.joinpath(".venv").is_symlink()
    assert source.joinpath(".venv").resolve() == settings.runtime_venv_root
    assert first.python_executable == settings.runtime_venv_root / "bin/python"
    assert first.python_executable.is_file()
    assert settings.runtime_current_dir.is_symlink()
    assert settings.runtime_current_dir.resolve() == first.release_dir
    assert first.release_dir.joinpath("src/utils/configuration/paths.py").is_file()
    assert (first.release_dir / "src/automation/chatgpt_mcp/example.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"
    assert settings.trusted_queue_script.is_file()
    assert os.access(settings.trusted_queue_script, os.X_OK)
    bare = _run(
        "git",
        "--git-dir",
        str(settings.trusted_git_dir),
        "rev-parse",
        "--is-bare-repository",
    )
    assert bare == "true"
    assert settings.runtime_version_path.read_text(encoding="utf-8").strip() == (
        first.revision
    )
