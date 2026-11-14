"""CLI integration with the checkout launcher and real registered worktrees."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Cli:
    repo: Path
    environment: dict[str, str]

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], env=self.environment, text=True
        ).strip()

    def run(
        self, *args: str, launcher: bool = True
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable]
        if launcher:
            command.extend(
                [
                    str(self.repo / ".agents/ops/checkout_job.py"),
                    "--repo",
                    str(self.repo),
                    "--",
                    sys.executable,
                ]
            )
        command.extend(["-m", "storage_tier", *args])
        return subprocess.run(
            command,
            cwd=self.repo,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
        )


@pytest.fixture
def cli(tmp_path: Path) -> Cli:
    repo = tmp_path / "repo"
    ops = repo / ".agents/ops"
    (ops / "lib").mkdir(parents=True)
    shutil.copy(ROOT / ".agents/ops/checkout_job.py", ops)
    shutil.copy(ROOT / ".agents/ops/lib/checkout_lock.py", ops / "lib")
    shutil.copytree(
        ROOT / ".agents/ops/storage_tier",
        ops / "storage_tier",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(ops),
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    result = Cli(repo, environment)
    result.git("init", "-q")
    (repo / "README.md").write_text("fixture\n")
    result.git("add", "README.md")
    result.git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "fixture",
    )
    for name in ("data", "outputs", "ckpt", ".venv"):
        (repo / name).mkdir()
    (repo / "data/input.bin").write_bytes(b"source bytes")
    return result


def test_prepare_worktree_creates_d_outputs_and_supports_actual_hardlink(
    cli: Cli,
) -> None:
    worktree = cli.repo / ".claude/worktrees/new-worker"
    cli.git("worktree", "add", "-qb", "worker", str(worktree))
    preview = cli.run("prepare-worktree", str(worktree), "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert not (worktree / "outputs").exists() and not (worktree / "data").exists()
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 0, result.stderr
    assert (worktree / "outputs").is_dir() and not (worktree / "outputs").is_symlink()
    for name in ("data", "ckpt", ".venv"):
        assert (worktree / name).is_symlink() and (
            worktree / name
        ).resolve() == cli.repo / name
    os.link(worktree / "data/input.bin", worktree / "outputs/copy.bin")
    assert (worktree / "outputs/copy.bin").stat().st_ino == (
        cli.repo / "data/input.bin"
    ).stat().st_ino
    assert cli.run("prepare-worktree", str(worktree)).returncode == 0


def test_prepare_refuses_existing_e_symlink_without_changing_it(
    cli: Cli, tmp_path: Path
) -> None:
    worktree = cli.repo / ".claude/worktrees/existing-worker"
    cli.git("worktree", "add", "-qb", "existing", str(worktree))
    archive = tmp_path / "archive"
    archive.mkdir()
    (worktree / "outputs").symlink_to(archive, target_is_directory=True)
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 1 and "real directory required" in result.stderr
    assert (worktree / "outputs").is_symlink() and (
        worktree / "outputs"
    ).resolve() == archive
    assert not (worktree / "data").exists()


def test_cli_requires_launcher_and_missing_mount_fails_even_for_dry_run(
    cli: Cli,
) -> None:
    direct = cli.run("scan", launcher=False)
    assert direct.returncode == 75 and "checkout_job.py" in direct.stderr
    assert cli.run("hydrate").returncode == 2
    assert cli.run("index", "--dry-run").returncode == 2
    unactivated = cli.run("cycle")
    assert unactivated.returncode == 1 and "requires activate" in unactivated.stderr
    assert not (cli.repo / ".storage-tier").exists()
    config = cli.repo / "storage.toml"
    config.write_text(
        'mirror_mount = "/not-a-tennis-mount"\ndataset_containers = ["data/task"]\ndataset_units = []\nsettle_seconds = 3600\n'
    )
    result = cli.run("--config", str(config), "cycle", "--dry-run")
    assert result.returncode == 1
    assert "not mounted" in json.loads(result.stderr)["error"]
    assert not (cli.repo / ".storage-tier").exists()
    assert (cli.repo / "data/input.bin").read_bytes() == b"source bytes"
