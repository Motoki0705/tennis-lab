"""Real Git helper canaries in disposable repos; never invoke the real agy."""

from __future__ import annotations

import importlib
import os
import pty
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops/lib"))
policy: Any = importlib.import_module("agy_permissions")


@pytest.fixture
def helper_repo(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    work = tmp_path / "workspace"
    work.mkdir()
    clean = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    clean.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_SYSTEM="/dev/null",
    )

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=work,
            env=clean,
            check=True,
            capture_output=True,
            timeout=10,
        )

    git("init", "--quiet", "--template=")
    git("config", "user.name", "Helper test")
    git("config", "user.email", "helper@example.invalid")
    (work / "a").write_text("needle\n")
    (work / ".gitattributes").write_text("a diff=probe filter=probe\n")
    git("add", "a", ".gitattributes")
    git("-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "needle")
    # An invalid signature is enough for showSignature to invoke gpg.program.
    raw = subprocess.run(
        ["git", "cat-file", "commit", "HEAD"],
        cwd=work,
        env=clean,
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout
    header, message = raw.split(b"\n\n", 1)
    signed = (
        header
        + b"\ngpgsig -----BEGIN PGP SIGNATURE-----\n invalid\n -----END PGP SIGNATURE-----\n\n"
        + message
    )
    commit = (
        subprocess.run(
            ["git", "hash-object", "-t", "commit", "-w", "--stdin"],
            input=signed,
            cwd=work,
            env=clean,
            check=True,
            capture_output=True,
            timeout=10,
        )
        .stdout.decode()
        .strip()
    )
    git("update-ref", "HEAD", commit)
    (work / "a").write_text("needle\nchanged\n")
    marker = tmp_path / "helper-ran"
    helper = tmp_path / "helper"
    helper.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\nprint('needle')\n"
    )
    helper.chmod(0o700)
    for key, value in {
        "core.fsmonitor": str(helper),
        "core.pager": str(helper),
        "pager.status": "true",
        "diff.external": str(helper),
        "diff.probe.command": str(helper),
        "diff.probe.textconv": str(helper),
        "log.showSignature": "true",
        "gpg.program": str(helper),
        "filter.probe.clean": str(helper),
        "filter.probe.smudge": str(helper),
        "filter.probe.required": "true",
    }.items():
        git("config", key, value)
    return work, marker, clean


def test_allowed_git_forms_do_not_execute_configured_helpers(
    helper_repo: tuple[Path, Path, dict[str, str]], tmp_path: Path
) -> None:
    work, marker, clean = helper_repo
    # Prove the canary is connected, rather than passing because no helper can run.
    for command in (
        "git status --short",
        "git log --format=%G? -1",
        "git diff --ext-diff",
        "git show --textconv HEAD:a",
        "git diff --no-ext-diff --no-textconv",
    ):
        subprocess.run(
            shlex.split(command), cwd=work, env=clean, capture_output=True, timeout=10
        )
        assert marker.exists(), command
        marker.unlink()
    master, slave = pty.openpty()
    try:
        subprocess.run(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "filter.probe.clean=",
                "-c",
                "filter.probe.required=false",
                "--paginate",
                "status",
                "--short",
            ],
            cwd=work,
            env=clean,
            stdin=subprocess.DEVNULL,
            stdout=slave,
            stderr=subprocess.PIPE,
            timeout=10,
            check=True,
        )
    finally:
        os.close(slave)
        os.close(master)
    assert marker.exists(), "pager control"
    marker.unlink()
    subprocess.run(
        ["git", "config", "filter.probe.process", str(tmp_path / "helper")],
        cwd=work,
        env=clean,
        check=True,
        capture_output=True,
        timeout=10,
    )
    hostile = {
        **clean,
        "GIT_EXTERNAL_DIFF": str(tmp_path / "helper"),
        "GIT_CONFIG_PARAMETERS": "'core.fsmonitor'='" + str(tmp_path / "helper") + "'",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "log.showSignature",
        "GIT_CONFIG_VALUE_0": "true",
        "GIT_DIR": str(tmp_path / "wrong-repo"),
        "GIT_WORK_TREE": str(tmp_path),
        "GIT_PAGER": str(tmp_path / "helper"),
    }
    safe = policy.readonly_environment(hostile, cwd=work)
    commands = (
        "git status -sb --porcelain=v2",
        "git ls-files --stage --cached",
        "git grep -ni --no-color needle -- a",
        "git log -1 --format='%h %s' --oneline",
        "git log --all --full-history -- a",
        "git log --no-ext-diff --no-textconv -p --stat -n1",
        "git diff --no-ext-diff --no-textconv -- a",
        "git show --no-ext-diff --no-textconv HEAD:a",
    )
    for command in commands:
        assert policy.command_allowed(command), command
        # A terminal is required to exercise the pager path. Output is a tiny
        # fixture (<4 KiB), so it cannot fill the PTY while we wait for Git.
        master, slave = pty.openpty()
        try:
            completed = subprocess.run(
                shlex.split(command),
                cwd=work,
                env=safe,
                stdin=subprocess.DEVNULL,
                stdout=slave,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            assert completed.returncode == 0, (command, completed.stderr)
        finally:
            os.close(slave)
            os.close(master)
        assert not marker.exists(), command


def test_unreadable_filter_configuration_is_not_silently_ignored(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/config").write_text("[invalid Git configuration\n")
    (tmp_path / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / ".git/objects").mkdir()
    (tmp_path / ".git/refs").mkdir()
    with pytest.raises(ValueError, match="cannot inspect workspace Git filters"):
        policy.readonly_environment(os.environ, cwd=tmp_path)
