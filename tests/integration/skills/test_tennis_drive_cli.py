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

    def invoke(*args: str, expected: int = 0) -> dict[str, Any]:
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--remote-root",
                "test:tennis_lab",
                "--rclone-bin",
                executable,
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
    planned = call("upload", str(source), "data/fixture-v1", "--dry-run", "--verify")
    assert planned["status"] == "planned" and not (remote / "data/fixture-v1").exists()
    uploaded = call("upload", str(source), "data/fixture-v1", "--verify")
    assert uploaded["verified"]
    assert (remote / "data/fixture-v1/empty-dir").is_dir()
    call("upload", str(source), "data/fixture-v1", expected=1)
    assert call("inspect", "data/fixture-v1")["file_count"] == 1
    assert call("verify", str(source), "data/fixture-v1")["matches"]
    copied = call("copy", "data/fixture-v1", "staging/copy")
    assert copied["verified"]
    assert (remote / "data/fixture-v1").exists()
    call("move", "staging/copy", "staging/renamed")
    assert not (remote / "staging/copy").exists()
    assert (remote / "staging/renamed/quoted ' name.txt").read_text() == "tennis\n"
    recovered = tmp_path / "recovered"
    assert call("download", "staging/renamed", str(recovered), "--verify")["verified"]
    (recovered / "quoted ' name.txt").write_text("corrupt")
    assert call("verify", str(recovered), "staging/renamed", expected=3)["changed"]
    inventory = tmp_path / "inventory.json"
    call("inventory", "--path", "data", "--output", str(inventory))
    records = json.loads(inventory.read_text())["entries"]
    assert any(item["hashes"] for item in records if item["type"] == "file")
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
