"""Exercise actual rclone copy/hash/move against an isolated local alias."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def drive_cli(tmp_path: Path) -> tuple[Callable[..., dict[str, Any]], Path]:
    rclone = shutil.which("rclone")
    if rclone is None:
        pytest.skip("rclone executable required for isolated local integration")
    assert rclone is not None
    executable: str = rclone
    remote = tmp_path / "remote"
    project = remote / "tennis_lab"
    project.mkdir(parents=True)
    config = tmp_path / "rclone.conf"
    config.write_text(f"[test]\ntype = alias\nremote = {remote}\n")
    script = (
        Path(__file__).resolve().parents[3]
        / ".agents/skills/tennis-drive/scripts/drive.py"
    )
    env = {**os.environ, "RCLONE_CONFIG": str(config)}

    def invoke(
        *args: str, expected: int = 0, rclone_bin: Path | None = None
    ) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--remote-root",
                "test:tennis_lab",
                "--rclone-bin",
                str(rclone_bin) if rclone_bin else executable,
                *args,
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        payload: dict[str, Any] = json.loads(result.stdout)
        assert payload["schema_version"] == 1
        return payload

    return invoke, project


def test_transfer_verify_relocate_and_inventory(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    source = tmp_path / "input"
    source.mkdir()
    (source / "empty-dir").mkdir()
    (source / "quoted ' name.txt").write_text("tennis\n")
    (source / "Copy of x: v2.txt").write_text("colon\n")
    (source / "back\\slash.txt").write_text("backslash\n")
    planned = call("upload", str(source), "data/fixture-v1", "--dry-run", "--verify")
    assert planned["status"] == "planned" and not (remote / "data/fixture-v1").exists()
    uploaded = call("upload", str(source), "data/fixture-v1", "--verify")
    assert uploaded["verified"]
    assert (remote / "data/fixture-v1/empty-dir").is_dir()
    call("upload", str(source), "data/fixture-v1", expected=1)
    assert call("inspect", "data/fixture-v1")["file_count"] == 3
    assert call("verify", str(source), "data/fixture-v1")["matches"]
    copied = call("copy", "data/fixture-v1", "staging/copy")
    assert copied["verified"]
    assert (remote / "data/fixture-v1").exists()
    call("move", "staging/copy", "staging/renamed")
    assert not (remote / "staging/copy").exists()
    assert (remote / "staging/renamed/quoted ' name.txt").read_text() == "tennis\n"
    assert (remote / "staging/renamed/Copy of x: v2.txt").read_text() == "colon\n"
    assert (remote / "staging/renamed/back\\slash.txt").read_text() == "backslash\n"
    recovered = tmp_path / "recovered"
    assert call("download", "staging/renamed", str(recovered), "--verify")["verified"]
    (recovered / "quoted ' name.txt").write_text("corrupt")
    assert call("verify", str(recovered), "staging/renamed", expected=3)["changed"]
    inventory = tmp_path / "inventory.json"
    call("inventory", "--path", "data", "--output", str(inventory))
    document = json.loads(inventory.read_text())
    assert document["root_id"] is None  # The local alias has no Drive IDs.
    records = document["entries"]
    assert "fixture-v1/Copy of x: v2.txt" in {item["path"] for item in records}
    for item in records:
        if item["type"] == "file":
            assert {name.lower().replace("-", "") for name in item["hashes"]} == {
                "md5",
                "sha1",
                "sha256",
            }
    call("inventory", "--path", "data", "--output", str(inventory), expected=1)
    call("trash", "staging/renamed", "--recursive", expected=1)
    assert (remote / "staging/renamed").exists()


def test_overwrite_is_copy_not_deleting_sync(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    source = tmp_path / "input"
    source.mkdir()
    (source / "current").write_text("current")
    destination = remote / "data"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    assert call("upload", str(source), "data", "--overwrite", "--verify")["verified"]
    assert (destination / "keep").read_text() == "keep"
    recovered = tmp_path / "recovered"
    recovered.mkdir()
    (recovered / "local-only").write_text("local-only")
    assert call("download", "data", str(recovered), "--overwrite", "--verify")[
        "verified"
    ]
    assert (recovered / "current").read_text() == "current"
    assert (recovered / "keep").read_text() == "keep"
    assert (recovered / "local-only").read_text() == "local-only"
    call("copy", "data", "data/recursive", expected=1)
    call("move", "data", ".", expected=1)
    assert (destination / "current").exists()


def test_single_file_transfer_and_rename(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
    tmp_path: Path,
) -> None:
    call, remote = drive_cli
    source = tmp_path / "weights.bin"
    source.write_bytes(bytes(range(256)))
    assert call("upload", str(source), "staging/weights.bin", "--verify")["verified"]
    assert call("copy", "staging/weights.bin", "staging/copy.bin")["verified"]
    assert call("move", "staging/copy.bin", "staging/renamed.bin")["verified"]
    assert (remote / "staging/renamed.bin").read_bytes() == source.read_bytes()
    assert not (remote / "staging/copy.bin").exists()
    assert set(call("inspect", "staging/weights.bin", "--checksum")["hashes"]) == {
        "md5",
        "sha1",
        "sha256",
    }


def test_download_verification_rejects_untransferred_source_file(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path], tmp_path: Path
) -> None:
    call, remote = drive_cli
    source = remote / "data"
    source.mkdir()
    (source / "copied.txt").write_text("copied")
    (source / "missing.txt").write_text("missing")
    destination = tmp_path / "recovered"
    destination.mkdir()
    (destination / "local-only.txt").write_text("keep")
    rclone = shutil.which("rclone")
    assert rclone is not None
    wrapper = tmp_path / "partial-rclone"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import os\nimport sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'copy':\n"
        "    args += ['--exclude', 'missing.txt']\n"
        f"os.execv({rclone!r}, [{rclone!r}, *args])\n"
    )
    wrapper.chmod(0o700)
    result = call(
        "download",
        "data",
        str(destination),
        "--overwrite",
        "--verify",
        expected=1,
        rclone_bin=wrapper,
    )
    assert not result["ok"]
    assert result["verification"]["extra_on_drive"] == ["missing.txt"]
    assert result["verification"]["missing_on_drive"] == ["local-only.txt"]
    assert (destination / "copied.txt").read_text() == "copied"
    assert (destination / "local-only.txt").read_text() == "keep"
    assert not (destination / "missing.txt").exists()


def test_mkdir_dry_run_existing_directory_and_file_conflict(
    drive_cli: tuple[Callable[..., dict[str, Any]], Path],
) -> None:
    call, remote = drive_cli
    path = "staging/new/empty"
    assert call("mkdir", path, "--dry-run")["status"] == "planned"
    assert not (remote / "staging").exists()
    assert call("mkdir", path)["status"] == "completed"
    assert (remote / path).is_dir()
    (remote / path / "keep").write_text("keep")
    call("mkdir", path)
    assert (remote / path / "keep").read_text() == "keep"
    assert not call("mkdir", path + "/keep", expected=1)["ok"]
    assert not call("mkdir", ".", expected=1)["ok"]
