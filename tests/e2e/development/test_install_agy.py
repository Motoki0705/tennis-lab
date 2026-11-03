"""Install local fake archives into private directories; no root or real agy."""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
INSTALLER = ROOT / "scripts/github_actions/install_agy.sh"


@pytest.fixture
def installation(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    archive = tmp_path / "fake.tar.gz"
    # Running this fixture would fail: the installer must verify bytes without
    # starting either a real agy or the archive's executable.
    content = b"#!/bin/sh\necho 'fixture must not run' >&2\nexit 77\n"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("antigravity")
        member.size = len(content)
        member.mode = 0o755
        handle.addfile(member, io.BytesIO(content))
    config = tmp_path / "agents.conf"
    config.write_text(
        f'AGY_VERSION="1.4.11"\nAGY_URL="{archive.as_uri()}"\n'
        f'AGY_SHA256="{hashlib.sha256(archive.read_bytes()).hexdigest()}"\n'
        'ISSUE_REWRITE_AGY_MODEL="gemini-3.8-flash-high"\n'
    )
    return archive, config, tmp_path / "install root/bin", content


def install(config: Path, destination: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(INSTALLER),
            "--config-file",
            str(config),
            "--bin-dir",
            str(destination),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )


def test_verified_install_is_executable_and_idempotent(
    installation: tuple[Path, Path, Path, bytes],
) -> None:
    archive, config, destination, content = installation
    first = install(config, destination)
    assert first.returncode == 0, first.stderr
    target = destination / "agy"
    assert target.read_bytes() == content
    assert target.stat().st_mode & 0o777 == 0o755
    stat = target.stat()
    assert "sudo -u tennis-actions" in first.stdout
    assert "AGY_CLI_DISABLE_AUTO_UPDATE=true" in first.stdout
    archive.unlink()  # a same-version repeat must not even download the archive
    second = install(config, destination)
    assert second.returncode == 0, second.stderr
    assert "already installed" in second.stdout
    assert target.stat().st_mtime_ns == stat.st_mtime_ns
    assert target.stat().st_ino == stat.st_ino


@pytest.mark.parametrize("existing", [False, True])
def test_checksum_mismatch_never_creates_or_replaces_the_installation(
    installation: tuple[Path, Path, Path, bytes],
    existing: bool,
) -> None:
    _, config, destination, _ = installation
    if existing:
        destination.mkdir(parents=True)
        (destination / "agy").write_text("old installation")
    config.write_text(
        re.sub(
            r'AGY_SHA256="[0-9a-f]+"', f'AGY_SHA256="{"0" * 64}"', config.read_text()
        )
    )
    proc = install(config, destination)
    assert proc.returncode != 0
    assert "sha256 mismatch" in proc.stderr
    if existing:
        assert (destination / "agy").read_text() == "old installation"
        assert sorted(path.name for path in destination.iterdir()) == ["agy"]
    else:
        assert not destination.exists()


def test_modified_installed_bytes_cannot_reuse_a_receipt(
    installation: tuple[Path, Path, Path, bytes],
) -> None:
    _, config, destination, content = installation
    assert install(config, destination).returncode == 0
    (destination / "agy").write_text("self-updated or modified")
    restored = install(config, destination)
    assert restored.returncode == 0, restored.stderr
    assert (destination / "agy").read_bytes() == content
    assert "already installed" not in restored.stdout


def test_smoke_workflow_requires_owner_manual_main_and_read_permissions() -> None:
    # BaseLoader keeps the Actions 'on' key as a string (YAML 1.1 treats it as bool).
    workflow = yaml.load(
        (ROOT / ".github/workflows/agy-smoke.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {
        "contents": "read",
        "issues": "read",
        "pull-requests": "read",
    }
    job = workflow["jobs"]["smoke"]
    assert "github.actor == github.repository_owner" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert job["runs-on"] == ["self-hosted", "linux", "x64", "tennis-lab"]
    assert job["env"]["HOME"] == "/var/lib/tennis-lab-actions/home"
    checkout = job["steps"][0]["with"]
    assert checkout["persist-credentials"] == "false"
    assert checkout["ref"] == "main"
    assert "scripts/github_actions/agy_smoke.py" in job["steps"][1]["run"]
    assert "--permissions-probe --timeout 120" in job["steps"][2]["run"]
    assert int(job["timeout-minutes"]) >= 75


def test_runner_install_gives_every_created_home_directory_to_the_runner() -> None:
    # The runner path needs root, so check the script: install -d applies -o/-g only
    # to named directories, and a root-owned ~/.local blocked uv in runner jobs.
    script = INSTALLER.read_text()
    assert 'runner_local="/var/lib/tennis-lab-actions/home/.local"' in script
    assert 'bin_dir="$runner_local/bin"' in script
    owner_calls = [
        line.strip()
        for line in script.splitlines()
        if line.strip().startswith("install -d -o tennis-actions")
    ]
    assert owner_calls == [
        'install -d -o tennis-actions -g tennis-actions -m 0755 "$runner_local" "$bin_dir"'
    ]
    # Both the fresh install and the unchanged-receipt path repair ownership.
    assert script.count("      own_runner_dirs\n") == 1
    assert script.count("  own_runner_dirs\nelse") == 1
