"""Command-level tests for the repository's :mod:`spin` workflow."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_spin(
    *args: str,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "spin", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment) if environment is not None else None,
    )


def test_help_lists_project_workflows() -> None:
    result = _run_spin("--help")

    assert result.returncode == 0, result.stderr
    for command in (
        "doctor",
        "lint",
        "setup",
        "setup-nht",
        "test",
        "typecheck",
    ):
        assert command in result.stdout


def _write_fake_command(path: Path, *, log_path: Path) -> None:
    path.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$0 $*\" >> {log_path}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_fake_uv(
    path: Path,
    *,
    log_path: Path,
    tool_bin: Path,
    tool_directory: Path,
) -> None:
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$*" = "tool dir --bin" ]; then\n'
        f"  printf '%s\\n' {tool_bin}\n"
        "  exit 0\n"
        "fi\n"
        'if [ "$*" = "tool dir" ]; then\n'
        f"  printf '%s\\n' {tool_directory}\n"
        "  exit 0\n"
        "fi\n"
        f"printf '%s\\n' \"$0 $*\" >> {log_path}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.parametrize("render_only", [False, True])
def test_setup_nht_builds_requested_isolated_runtimes(
    tmp_path: Path,
    render_only: bool,
) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    log_path = tmp_path / "commands.log"
    tool_directory = tmp_path / "tools"
    for command in ("git", "nht-reconstruct", "nht-render"):
        _write_fake_command(bin_directory / command, log_path=log_path)
    _write_fake_uv(
        bin_directory / "uv",
        log_path=log_path,
        tool_bin=bin_directory,
        tool_directory=tool_directory,
    )
    environment = {
        **os.environ,
        "PATH": f"{bin_directory}{os.pathsep}{os.environ.get('PATH', '')}",
    }

    arguments = ("setup-nht", "--render-only") if render_only else ("setup-nht",)
    result = _run_spin(*arguments, environment=environment)

    assert result.returncode == 0, result.stdout + result.stderr
    commands = log_path.read_text(encoding="utf-8")
    assert (
        "git submodule update --init --recursive --checkout third_party/nht" in commands
    )
    assert "uv tool install --force --python 3.11 --editable" in commands
    assert "--python 3.11" in commands
    gsplat_build = (
        f"uv pip install --python {tool_directory}/nht/bin/python "
        "--no-build-isolation --no-cache --reinstall-package gsplat "
        "--torch-backend cu130 torch==2.9.1 --editable third_party/nht/gsplat"
    )
    assert gsplat_build in commands
    assert "--with-editable third_party/nht/gsplat" not in commands
    assert "--with torch==2.9.1 --with torchvision==0.24.1" in commands
    assert "third_party/nht[aov]" in commands
    assert "setuptools<81" in commands
    assert "setuptools<81 wheel ninja rich" in commands
    assert "tinycudann @ git+https://github.com/NVlabs/tiny-cuda-nn/" in commands
    native_probe = (
        "from gsplat import csrc; "
        "from gsplat.nht.deferred_shader import DeferredShaderModule"
    )
    assert native_probe in commands
    assert commands.index("--with torch==") < commands.index(gsplat_build)
    assert commands.index(gsplat_build) < commands.index(native_probe)
    assert "NHT public CLI is ready" in result.stdout
    if render_only:
        assert ".trainer-venv" not in commands
        assert "examples/requirements.txt" not in commands
        assert "nht_adapter.py probe" not in commands
        assert "trainer setup was not requested" in result.stdout
    else:
        assert "uv venv --clear --python 3.11 third_party/nht/.trainer-venv" in commands
        assert "torch==2.9.1 torchvision==0.24.1" in commands
        assert "third_party/nht/gsplat/examples/requirements.txt" in commands
        assert "third_party/nht/nht_pipeline/nht_adapter.py probe" in commands
        assert "NHT trainer runtime is ready" in result.stdout


def test_render_only_setup_rejects_sfm_backend_before_installation() -> None:
    result = _run_spin("setup-nht", "--render-only", "--with-sfm-learned")
    assert result.returncode != 0
    assert "cannot be combined" in result.stderr
    assert "$ git" not in result.stdout


def test_lint_can_check_an_explicit_path() -> None:
    result = _run_spin("lint", ".spin/cmds.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "All checks passed" in result.stdout


def test_typecheck_can_check_an_explicit_path() -> None:
    result = _run_spin("typecheck", ".spin/cmds.py")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success: no issues found" in result.stdout


def test_changed_mode_rejects_an_unknown_git_base() -> None:
    result = _run_spin(
        "lint",
        "--changed",
        "--base",
        "refs/heads/definitely-not-a-real-branch",
    )

    assert result.returncode != 0
    assert "was not found" in result.stderr


def test_test_command_forwards_pytest_arguments() -> None:
    result = _run_spin(
        "test",
        "--all",
        "--serial",
        "tests/unit/utils/geometry/test_angles.py",
        "-q",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
