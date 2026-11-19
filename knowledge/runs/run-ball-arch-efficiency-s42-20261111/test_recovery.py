"""Recovery must reject corrupt or ambiguous input artifacts before training."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from recover import copy_immutable
from restore_inputs import copy_file, copy_snapshot

from src.utils.checksum import dual_sha256


def test_measurement_entrypoints_import_outside_pytest_namespace():
    directory = Path(__file__).resolve().parent
    code = f"import sys; sys.path.insert(0, {str(directory)!r}); import measure, summarize; print('direct entrypoints imported')"
    result = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert result.strip() == "direct entrypoints imported"


def test_frozen_input_copy_checks_hash_and_refuses_corrupt_existing_file(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"fixed pretrained weights")
    destination = tmp_path / "local" / "checkpoint.pt"
    expected = dict(size_bytes=source.stat().st_size, sha256=dual_sha256(source))
    copy_file(source, destination, expected)
    copy_file(source, destination, expected)
    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="no automatic overwrite"):
        copy_file(source, destination, expected)
    assert destination.read_bytes() == b"corrupt"


def test_snapshot_rejects_escaping_member_and_wrong_pinned_manifest(tmp_path):
    source = tmp_path / "snapshot"
    source.mkdir()
    marker = source / "_SNAPSHOT.json"
    marker.write_text(
        json.dumps(
            dict(
                schema="verified_artifact_snapshot.v1",
                files={"../outside": {"size_bytes": 1, "sha256": "0" * 64}},
            )
        )
    )
    with pytest.raises(RuntimeError, match="Snapshot marker differs"):
        copy_snapshot(source, tmp_path / "wrong-marker", "0" * 64)
    with pytest.raises(RuntimeError, match="Invalid snapshot path"):
        copy_snapshot(source, tmp_path / "outside")
    assert not (tmp_path / "outside").exists()


def test_recovery_metadata_requires_identical_existing_bytes(tmp_path):
    source = tmp_path / "verified"
    source.write_bytes(b"durable checkpoint pointer")
    destination = tmp_path / "restored" / "state.json"
    copy_immutable(source, destination)
    copy_immutable(source, destination)
    destination.write_bytes(b"unverified local pointer")
    with pytest.raises(RuntimeError, match="metadata differs"):
        copy_immutable(source, destination)
    assert destination.read_bytes() == b"unverified local pointer"
