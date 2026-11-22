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
    # The default D: layout; E:-resident tests opt in through their own config.
    config = ops / "storage_tier/config.toml"
    config.write_text(
        "\n".join(
            line
            for line in config.read_text().splitlines()
            if not line.startswith("worktree_outputs_on_mirror")
        )
        + "\n"
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
    persistent = cli.repo / ".worktree-outputs/new-worker"
    assert (worktree / "outputs").is_symlink()
    assert (worktree / "outputs").resolve() == persistent
    for name in ("data", "ckpt", ".venv"):
        assert (worktree / name).is_symlink() and (
            worktree / name
        ).resolve() == cli.repo / name
    os.link(worktree / "data/input.bin", worktree / "outputs/copy.bin")
    assert (worktree / "outputs/copy.bin").stat().st_ino == (
        cli.repo / "data/input.bin"
    ).stat().st_ino
    assert cli.run("prepare-worktree", str(worktree)).returncode == 0
    # Exercise the same FD-based remover used by expiry; only the link is removed.
    code = (
        "from pathlib import Path; from worktree_cleanup.inventory import identity, remove_tree; "
        f"p=Path({str(worktree)!r}); remove_tree(p, identity(p))"
    )
    environment = {**cli.environment, "PYTHONPATH": str(ROOT / ".agents/ops")}
    removed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert removed.returncode == 0, removed.stderr
    assert not worktree.exists()
    assert (persistent / "copy.bin").read_bytes() == b"source bytes"


def test_prepare_refuses_existing_e_symlink_without_changing_it(
    cli: Cli, tmp_path: Path
) -> None:
    worktree = cli.repo / ".claude/worktrees/existing-worker"
    cli.git("worktree", "add", "-qb", "existing", str(worktree))
    archive = tmp_path / "archive"
    archive.mkdir()
    (worktree / "outputs").symlink_to(archive, target_is_directory=True)
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 1 and "outputs must point" in result.stderr
    assert (worktree / "outputs").is_symlink() and (
        worktree / "outputs"
    ).resolve() == archive
    assert not (worktree / "data").exists()


def test_prepare_keeps_main_output_alias_without_independent_storage(cli: Cli) -> None:
    worktree = cli.repo / ".claude/worktrees/main-alias"
    cli.git("worktree", "add", "-qb", "alias", str(worktree))
    (worktree / "outputs").symlink_to(cli.repo / "outputs")
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["output_root"] == str(cli.repo / "outputs")
    assert not (cli.repo / ".worktree-outputs/main-alias").exists()
    assert (worktree / "outputs").resolve() == cli.repo / "outputs"


def test_prepare_dry_run_refuses_to_read_old_symlink_target(cli: Cli) -> None:
    worktree = cli.repo / ".claude/worktrees/old-alias"
    cli.git("worktree", "add", "-qb", "old-alias", str(worktree))
    output = worktree / "outputs"
    output.symlink_to(cli.repo / "outputs")
    os.utime(output, ns=(1, 1), follow_symlinks=False)
    before = output.lstat()
    result = cli.run("prepare-worktree", str(worktree), "--dry-run")
    assert result.returncode == 1 and "dry-run cannot inspect" in result.stderr
    assert output.lstat() == before
    assert not (worktree / "data").exists()
    assert not (cli.repo / ".worktree-outputs").exists()


def test_cli_requires_launcher_and_missing_mount_fails_even_for_dry_run(
    cli: Cli,
) -> None:
    direct = cli.run("scan", launcher=False)
    assert direct.returncode == 75 and "checkout_job.py" in direct.stderr
    assert cli.run("hydrate").returncode == 2
    assert cli.run("index", "--dry-run").returncode == 2
    assert cli.run("--queue-dir", "/unused", "scan").returncode == 2
    unactivated = cli.run("cycle")
    assert unactivated.returncode == 1 and "requires activate" in unactivated.stderr
    assert not (cli.repo / ".storage-tier").exists()
    config = cli.repo / "storage.toml"
    config.write_text(
        'mirror_mount = "/not-a-tennis-mount"\ndataset_containers = ["data/task"]\ndataset_units = []\n'
    )
    result = cli.run("--config", str(config), "cycle", "--dry-run")
    assert result.returncode == 1
    assert "not mounted" in json.loads(result.stderr)["error"]
    assert not (cli.repo / ".storage-tier").exists()
    assert (cli.repo / "data/input.bin").read_bytes() == b"source bytes"


def test_linked_checkout_cli_uses_its_config_with_explicit_main_repo(cli: Cli) -> None:
    worktree = cli.repo / ".claude/worktrees/storage-operator"
    cli.git("worktree", "add", "-qb", "operator", str(worktree))
    ops = worktree / ".agents/ops"
    shutil.copytree(cli.repo / ".agents/ops", ops)
    mount = cli.repo.parent / "E"
    mount.mkdir()
    (ops / "storage_tier/config.toml").write_text(
        f"mirror_mount = {json.dumps(str(mount))}\n"
        "dataset_containers = []\ndataset_units = []\n"
        'resident_units = ["data/smplx"]\nunmanaged_data_links = []\n'
    )
    before_head = cli.git("rev-parse", "HEAD")
    before = (cli.repo / "README.md").read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            str(ops / "checkout_job.py"),
            "--repo",
            str(worktree),
            "--",
            sys.executable,
            "-m",
            "storage_tier",
            "--repo",
            str(cli.repo),
            "prepare-worktree",
            str(worktree),
            "--dry-run",
        ],
        cwd=worktree,
        env={**cli.environment, "PYTHONPATH": str(ops)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["mirror_output_root"] == str(
        mount / cli.repo.relative_to("/") / ".worktree-outputs/storage-operator"
    )
    assert cli.git("rev-parse", "HEAD") == before_head
    assert (cli.repo / "README.md").read_bytes() == before
    assert not (worktree / "outputs").exists()


def _on_mirror(cli: Cli, tmp_path: Path, *, link: bool = True) -> Path:
    """Opt the fixture into E:-resident outputs with a real mount directory."""
    mount = tmp_path / "E"
    mount.mkdir()
    config = cli.repo / ".agents/ops/storage_tier/config.toml"
    lines = [
        f'mirror_mount = "{mount}"' if line.startswith("mirror_mount") else line
        for line in config.read_text().splitlines()
    ]
    config.write_text("\n".join([*lines, "worktree_outputs_on_mirror = true"]) + "\n")
    root = mount / cli.repo.relative_to("/") / ".worktree-outputs"
    root.mkdir(parents=True)
    if link:
        (cli.repo / ".worktree-outputs").symlink_to(root, target_is_directory=True)
    return root


def test_prepare_worktree_creates_e_resident_outputs_through_exact_link(
    cli: Cli, tmp_path: Path
) -> None:
    root = _on_mirror(cli, tmp_path)
    worktree = cli.repo / ".claude/worktrees/e-worker"
    cli.git("worktree", "add", "-qb", "e-worker", str(worktree))
    preview = cli.run("prepare-worktree", str(worktree), "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert not (root / "e-worker").exists()
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["output_root"] == str(root / "e-worker")
    assert report["mirror_output_root"] is None
    # The worktree link keeps naming the main checkout path, so existing
    # worktrees and the later D: layout agree on the same link text.
    assert os.readlink(worktree / "outputs") == str(
        cli.repo / ".worktree-outputs/e-worker"
    )
    assert (worktree / "outputs").resolve() == root / "e-worker"
    assert (root / "e-worker").is_dir() and not (root / "e-worker").is_symlink()
    assert cli.run("prepare-worktree", str(worktree)).returncode == 0


@pytest.mark.parametrize("shape", ["real-directory", "wrong-target", "missing"])
def test_e_resident_outputs_refuse_any_other_root_shape(
    cli: Cli, tmp_path: Path, shape: str
) -> None:
    root = _on_mirror(cli, tmp_path, link=False)
    if shape == "real-directory":
        (cli.repo / ".worktree-outputs").mkdir()
    elif shape == "wrong-target":
        other = tmp_path / "elsewhere"
        other.mkdir()
        (cli.repo / ".worktree-outputs").symlink_to(other, target_is_directory=True)
    worktree = cli.repo / ".claude/worktrees/refused"
    cli.git("worktree", "add", "-qb", "refused", str(worktree))
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 1, result.stdout
    assert "worktree" in result.stderr and "outputs" in result.stderr
    assert not os.path.lexists(worktree / "outputs")
    assert not (worktree / "data").exists()
    assert list(root.iterdir()) == []


def test_default_layout_still_refuses_a_linked_outputs_root(
    cli: Cli, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (cli.repo / ".worktree-outputs").symlink_to(elsewhere, target_is_directory=True)
    worktree = cli.repo / ".claude/worktrees/default"
    cli.git("worktree", "add", "-qb", "default", str(worktree))
    result = cli.run("prepare-worktree", str(worktree))
    assert result.returncode == 1 and "real directory required" in result.stderr
    assert not os.path.lexists(worktree / "outputs")
    assert list(elsewhere.iterdir()) == []
