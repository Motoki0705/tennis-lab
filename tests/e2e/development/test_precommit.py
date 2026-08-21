"""Command-level tests for the repository's pre-commit environment wrapper."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = REPO_ROOT / "scripts" / "run_in_repo_venv.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _environment_with_fake_git(tmp_path: Path, git_common_dir: Path) -> dict[str, str]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "git",
        "#!/bin/sh\n"
        'test "$*" = "rev-parse --path-format=absolute --git-common-dir" || exit 64\n'
        f"printf '%s\\n' '{git_common_dir}'\n",
    )
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}",
    }


def test_wrapper_resolves_the_main_checkout_virtualenv(tmp_path: Path) -> None:
    main_checkout = tmp_path / "main-checkout"
    git_common_dir = main_checkout / ".git"
    git_common_dir.mkdir(parents=True)
    venv_bin = main_checkout / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_executable(
        venv_bin / "probe",
        "#!/bin/sh\n"
        "printf 'path=%s\\n' \"$PATH\"\n"
        "printf 'arg=<%s>\\n' \"$@\"\n",
    )
    environment = _environment_with_fake_git(tmp_path, git_common_dir)

    result = subprocess.run(
        [WRAPPER, "probe", "first argument", "second"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].split(os.pathsep, maxsplit=1)[0] == f"path={venv_bin}"
    assert lines[1:] == ["arg=<first argument>", "arg=<second>"]


def test_wrapper_fails_loudly_when_shared_tool_is_missing(tmp_path: Path) -> None:
    main_checkout = tmp_path / "main-checkout"
    git_common_dir = main_checkout / ".git"
    git_common_dir.mkdir(parents=True)
    environment = _environment_with_fake_git(tmp_path, git_common_dir)

    result = subprocess.run(
        [WRAPPER, "ruff"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode != 0
    assert str(main_checkout / ".venv" / "bin" / "ruff") in result.stderr
    assert "uv sync --locked" in result.stderr


def test_all_system_hooks_use_the_repository_virtualenv_wrapper() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )

    system_hooks = [
        hook
        for repository in config["repos"]
        for hook in repository["hooks"]
        if hook["language"] == "system"
    ]
    assert system_hooks
    for hook in system_hooks:
        assert hook["entry"].startswith("./scripts/run_in_repo_venv.sh ")
