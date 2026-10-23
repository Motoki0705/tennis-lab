"""An interrupted download may leave incomplete targets and rclone temporaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location("colab_verify", Path(__file__).with_name("colab_verify.py"))
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_manifest(root: Path) -> dict[str, Any]:
    (root / "shards").mkdir()
    (root / "shards/a.bin").write_bytes(b"complete")
    manifest = dict(files=[dict(path="shards/a.bin", size_bytes=8, sha256=hashlib.sha256(b"complete").hexdigest())])
    (root / "distribution.json").write_text(json.dumps(manifest))
    return manifest


def test_known_temporary_is_removed_after_the_final_target_matches(tmp_path: Path) -> None:
    manifest = fixture_manifest(tmp_path)
    temporary = tmp_path / "shards/a.bin.transfer.partial"
    temporary.write_bytes(b"incomplete")
    partial = [dict(path="shards/a.bin.transfer.partial", size_bytes=10,
                    sha256=hashlib.sha256(b"incomplete").hexdigest())]
    result = module.reconcile_partial(tmp_path, partial, manifest)
    assert result["removed_known_incomplete_temporaries"] == partial
    assert not temporary.exists()
    assert (tmp_path / "shards/a.bin").read_bytes() == b"complete"


def test_incomplete_regular_target_can_be_restarted(tmp_path: Path) -> None:
    manifest = fixture_manifest(tmp_path)
    partial = [dict(path="shards/a.bin", size_bytes=3, sha256=hashlib.sha256(b"com").hexdigest())]
    result = module.reconcile_partial(tmp_path, partial, manifest)
    assert result["restarted_incomplete_files"] == 1
    assert result["retained_complete_files"] == 0


def test_changed_temporary_is_preserved_for_inspection(tmp_path: Path) -> None:
    manifest = fixture_manifest(tmp_path)
    temporary = tmp_path / "shards/a.bin.transfer.partial"
    temporary.write_bytes(b"changed")
    partial = [dict(path="shards/a.bin.transfer.partial", size_bytes=10,
                    sha256=hashlib.sha256(b"incomplete").hexdigest())]
    with pytest.raises(ValueError, match="refuse automatic cleanup"):
        module.reconcile_partial(tmp_path, partial, manifest)
    assert temporary.exists()
