"""The notebook/archive entry point accepts tar prefixes and rejects ambiguity."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts/colab/setup/prepare_meiji_archive.py"
)


def _add(bundle: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    bundle.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize(
    "prefix",
    ["", "meiji_3cam/dataset/", "data/tennis_multivew/processed/meiji_3cam/dataset/"],
)
def test_archive_root_detection_and_reuse(tmp_path: Path, prefix: str) -> None:
    archive = tmp_path / "data.tar"
    destination = tmp_path / "dataset"
    with tarfile.open(archive, "w") as bundle:
        _add(
            bundle,
            prefix + "dataset.json",
            json.dumps({"version": 1, "clips": [{"clip_id": "r/c"}]}).encode(),
        )
        _add(bundle, prefix + "clips/r/c/clip.json", b"{}")
    command = [
        sys.executable,
        str(SCRIPT),
        "--archive",
        str(archive),
        "--destination",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run(command, check=True, capture_output=True)
    assert (destination / "clips/r/c/clip.json").is_file()
    with archive.open("ab") as handle:
        handle.write(b"different archive")
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    assert result.returncode != 0 and "not from this archive" in result.stderr


@pytest.mark.parametrize("name", ["../escape", "/absolute"])
def test_archive_rejects_paths_outside_destination(tmp_path: Path, name: str) -> None:
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as bundle:
        _add(bundle, name, b"bad")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--archive",
            str(archive),
            "--destination",
            str(tmp_path / "dataset"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "Unsupported archive member" in result.stderr
    assert not (tmp_path / "dataset").exists()
