"""Command-level tests for the Colab Drive utility scripts."""

from __future__ import annotations

import hashlib
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
    working_directory: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["TENNIS_LAB_DRIVE_ROOT"] = str(drive_root)
    return subprocess.run(
        [str(SCRIPTS_DIR / script_name), *arguments],
        cwd=working_directory or REPOSITORY_ROOT,
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


def test_list_respects_depth_limit_and_reports_truncation(tmp_path: Path) -> None:
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


def test_search_filters_by_glob_type_and_start_path(tmp_path: Path) -> None:
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


def test_drive_path_cannot_escape_root_directly_or_through_symlink(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (drive_root / "escape").symlink_to(outside, target_is_directory=True)

    direct = _run_script(
        "inspect_file.sh", "../outside/secret.txt", drive_root=drive_root
    )
    indirect = _run_script(
        "inspect_file.sh", "escape/secret.txt", drive_root=drive_root
    )
    normalized_inside = _run_script(
        "inspect_file.sh", "directory/../file.txt", drive_root=drive_root
    )

    assert direct.returncode == 1
    assert "cannot contain '..'" in direct.stderr
    assert indirect.returncode == 1
    assert "symbolic links" in indirect.stderr
    assert normalized_inside.returncode == 1
    assert "cannot contain '..'" in normalized_inside.stderr


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
    destination = drive_root / "uploads/remote file.txt"
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
    assert replaced.returncode == 0
    assert destination.read_text(encoding="utf-8") == "version two"


def test_directory_upload_replaces_tree_without_leaving_stale_files(
    tmp_path: Path,
) -> None:
    drive_root = tmp_path / "drive"
    destination = drive_root / "datasets/sample"
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("stale", encoding="utf-8")
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
    assert not (destination / "stale.txt").exists()
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


def test_inspect_reports_metadata_and_checksum(tmp_path: Path) -> None:
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
    assert payload["sha256"] == hashlib.sha256(content).hexdigest()
    assert payload["file_count"] == 1


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


def test_missing_drive_mount_fails_with_actionable_message(tmp_path: Path) -> None:
    missing_root = tmp_path / "not-mounted"

    result = _run_script("list_drive.sh", drive_root=missing_root)

    assert result.returncode == 1
    assert "Mount Google Drive first" in result.stderr
