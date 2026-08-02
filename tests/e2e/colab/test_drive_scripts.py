"""Command-level tests for the local rclone Drive utility scripts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPOSITORY_ROOT = Path(__file__).parents[3]
SCRIPTS_DIR = REPOSITORY_ROOT / "scripts/colab/scripts"


def _run_script(
    script_name: str,
    *arguments: str,
    drive_root: Path,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["RCLONE_CONFIG_MOCK_TYPE"] = "local"
    environment["TENNIS_LAB_DRIVE_REMOTE"] = f"mock:{drive_root}"
    if environment_overrides:
        environment.update(environment_overrides)
    return subprocess.run(
        [str(SCRIPTS_DIR / script_name), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _json_output(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)
    return parsed


def test_list_uses_rclone_and_respects_depth_and_limit(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    (drive_root / "data/nested").mkdir(parents=True)
    (drive_root / "data/sample.txt").write_text("sample", encoding="utf-8")
    (drive_root / "data/nested/deep.txt").write_text("deep", encoding="utf-8")

    depth_result = _run_script(
        "list_drive.sh",
        "--max-depth",
        "1",
        "--format",
        "json",
        drive_root=drive_root,
    )
    payload = _json_output(depth_result)
    entries = payload["entries"]
    assert isinstance(entries, list)
    assert [entry["path"] for entry in entries] == ["data"]

    limited_result = _run_script(
        "list_drive.sh", "--limit", "1", "--format", "json", drive_root=drive_root
    )
    limited = _json_output(limited_result)
    assert limited["count"] == 1
    assert limited["truncated"] is True


def test_search_filters_rclone_listing_by_name_type_and_path(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    (drive_root / "models/archive.ckpt").mkdir(parents=True)
    (drive_root / "models/first.ckpt").write_bytes(b"one")
    (drive_root / "models/second.pt").write_bytes(b"two")
    (drive_root / "other.ckpt").write_bytes(b"other")

    result = _run_script(
        "search_drive.sh",
        "--path",
        "models",
        "--name",
        "*.ckpt",
        "--type",
        "file",
        "--format",
        "json",
        drive_root=drive_root,
    )

    payload = _json_output(result)
    entries = payload["entries"]
    assert isinstance(entries, list)
    assert [entry["path"] for entry in entries] == ["models/first.ckpt"]


@pytest.mark.parametrize("unsafe_path", ["../outside", "/absolute", "other:path"])
def test_drive_relative_paths_cannot_escape_remote_root(
    tmp_path: Path, unsafe_path: str
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()

    result = _run_script("inspect_file.sh", unsafe_path, drive_root=drive_root)

    assert result.returncode == 1
    assert "Unsafe Drive-relative path" in result.stderr


def test_upload_refuses_symbolic_link_source(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    local_file = tmp_path / "actual.txt"
    local_file.write_text("content", encoding="utf-8")
    local_link = tmp_path / "link.txt"
    local_link.symlink_to(local_file)

    result = _run_script(
        "upload_to_drive.sh", str(local_link), "link.txt", drive_root=drive_root
    )

    assert result.returncode == 1
    assert "cannot be symbolic links" in result.stderr
    assert not (drive_root / "link.txt").exists()


def test_upload_dry_run_copy_verify_and_overwrite_guard(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    local_file = tmp_path / "local file.txt"
    local_file.write_text("version one", encoding="utf-8")
    destination = drive_root / "uploads/remote file.txt"

    dry_run = _run_script(
        "upload_to_drive.sh",
        str(local_file),
        "uploads/remote file.txt",
        "--dry-run",
        "--format",
        "json",
        drive_root=drive_root,
    )
    assert _json_output(dry_run)["status"] == "planned"
    assert not destination.exists()

    uploaded = _run_script(
        "upload_to_drive.sh",
        str(local_file),
        "uploads/remote file.txt",
        "--verify",
        "--format",
        "json",
        drive_root=drive_root,
    )
    uploaded_payload = _json_output(uploaded)
    assert uploaded_payload["verified"] is True
    assert destination.read_text(encoding="utf-8") == "version one"

    refused = _run_script(
        "upload_to_drive.sh",
        str(local_file),
        "uploads/remote file.txt",
        drive_root=drive_root,
    )
    assert refused.returncode == 1
    assert "--overwrite" in refused.stderr

    local_file.write_text("version two", encoding="utf-8")
    replaced = _run_script(
        "upload_to_drive.sh",
        str(local_file),
        "uploads/remote file.txt",
        "--overwrite",
        drive_root=drive_root,
    )
    assert replaced.returncode == 0, replaced.stderr
    assert destination.read_text(encoding="utf-8") == "version two"


def test_directory_overwrite_updates_conflicts_without_deleting_extras(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "drive"
    destination = drive_root / "datasets/sample"
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("preserved", encoding="utf-8")
    local_directory = tmp_path / "dataset"
    (local_directory / "nested").mkdir(parents=True)
    (local_directory / "nested/item.txt").write_text("new", encoding="utf-8")

    result = _run_script(
        "upload_to_drive.sh",
        str(local_directory),
        "datasets/sample",
        "--overwrite",
        "--verify",
        drive_root=drive_root,
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "stale.txt").read_text(encoding="utf-8") == "preserved"
    assert (destination / "nested/item.txt").read_text(encoding="utf-8") == "new"


def test_download_copies_directory_and_refuses_implicit_overwrite(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "drive"
    source = drive_root / "datasets/sample"
    source.mkdir(parents=True)
    (source / "item.txt").write_text("from drive", encoding="utf-8")
    local_destination = tmp_path / "downloads/sample"

    downloaded = _run_script(
        "download_from_drive.sh",
        "datasets/sample",
        str(local_destination),
        "--verify",
        drive_root=drive_root,
    )
    assert downloaded.returncode == 0, downloaded.stderr
    assert (local_destination / "item.txt").read_text(encoding="utf-8") == "from drive"

    repeated = _run_script(
        "download_from_drive.sh",
        "datasets/sample",
        str(local_destination),
        drive_root=drive_root,
    )
    assert repeated.returncode == 1
    assert "--overwrite" in repeated.stderr

    (source / "item.txt").write_text("updated", encoding="utf-8")
    (local_destination / "local-only.txt").write_text("keep", encoding="utf-8")
    updated = _run_script(
        "download_from_drive.sh",
        "datasets/sample",
        str(local_destination),
        "--overwrite",
        "--verify",
        drive_root=drive_root,
    )
    assert updated.returncode == 0, updated.stderr
    assert (local_destination / "item.txt").read_text(encoding="utf-8") == "updated"
    assert (local_destination / "local-only.txt").read_text(encoding="utf-8") == "keep"


def test_inspect_reports_rclone_metadata_and_hashes(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    content = b"checkpoint-data"
    (drive_root / "model.ckpt").write_bytes(content)

    result = _run_script(
        "inspect_file.sh",
        "model.ckpt",
        "--checksum",
        "--format",
        "json",
        drive_root=drive_root,
    )

    payload = _json_output(result)
    assert payload["type"] == "file"
    assert payload["size_bytes"] == len(content)
    hashes = payload["hashes"]
    assert isinstance(hashes, dict)
    assert hashes["md5"]
    assert hashes["sha256"]


def test_verify_returns_zero_for_match_and_three_for_difference(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    local_file = tmp_path / "model.ckpt"
    local_file.write_bytes(b"same")
    drive_file = drive_root / "model.ckpt"
    drive_file.write_bytes(b"same")

    matching = _run_script(
        "verify_transfer.sh",
        str(local_file),
        "model.ckpt",
        "--format",
        "json",
        drive_root=drive_root,
    )
    assert _json_output(matching)["matches"] is True

    drive_file.write_bytes(b"different")
    different = _run_script(
        "verify_transfer.sh",
        str(local_file),
        "model.ckpt",
        "--format",
        "json",
        drive_root=drive_root,
    )
    assert different.returncode == 3
    payload = json.loads(different.stdout)
    assert payload["matches"] is False
    assert payload["changed"] == ["."]


def test_verify_detects_empty_directory_structure_difference(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_tree = drive_root / "tree"
    drive_tree.mkdir(parents=True)
    local_tree = tmp_path / "tree"
    (local_tree / "empty").mkdir(parents=True)

    result = _run_script(
        "verify_transfer.sh",
        str(local_tree),
        "tree",
        "--format",
        "json",
        drive_root=drive_root,
    )

    assert result.returncode == 3
    payload = json.loads(result.stdout)
    assert payload["missing_on_drive"] == ["empty"]


def test_missing_rclone_fails_with_actionable_message(tmp_path: Path) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()

    result = _run_script(
        "list_drive.sh",
        drive_root=drive_root,
        environment_overrides={"RCLONE_BIN": "/missing/rclone"},
    )

    assert result.returncode == 1
    assert "Install rclone first" in result.stderr
