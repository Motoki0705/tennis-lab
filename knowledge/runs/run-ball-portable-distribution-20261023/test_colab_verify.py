"""An interrupted download may leave incomplete targets and rclone temporaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from src.tasks.ball_detection.data.distribution import (
    prepare_distribution,
    sample_windows,
)
from src.tasks.ball_detection.data.distribution.manifest import distribution_files
from src.tasks.ball_detection.data.store import BallFrameStore
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip

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


def existing_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    source, staging, destination = (tmp_path / name for name in ("source", "stage", "distribution"))
    clip_id = "tracknet/train/clip"
    write_store_clip(source, clip_id, [frame(0, ball()), frame(1)])
    manifest = prepare_distribution(source, staging, version="portable-v1", clip_ids=[clip_id], inputs={})
    for name, path in distribution_files(staging, source_directory=source).items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    shutil.copyfile(staging / "distribution.json", destination / "distribution.json")
    windows = [(clip_id, 0, 2)]
    (tmp_path / "windows.json").write_text(json.dumps(windows))
    (tmp_path / "reference.json").write_text(json.dumps(sample_windows(BallFrameStore(source), windows)))
    (tmp_path / "uv.lock").write_text("fixture lock")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RCLONE_CONFIG", "/managed-test-config-not-read")

    def no_download(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Verification-only mode must not perform another transfer")

    def git_head(argv: list[str], **kwargs: Any) -> str:
        assert argv == ["git", "rev-parse", "HEAD"]
        return "unit-test-commit\n"

    monkeypatch.setattr(module, "download", no_download)
    monkeypatch.setattr(module.subprocess, "check_output", git_head)
    return Namespace(destination=Path("distribution"), receipts=Path("receipts"),
                     windows=Path("windows.json"), reference=Path("reference.json"),
                     manifest_sha256=dual_sha256(destination / "distribution.json"),
                     expected_size_bytes=sum(e["size_bytes"] for e in manifest["files"])
                     + (destination / "distribution.json").stat().st_size,
                     hash_workers=1, resume_existing=False, verify_existing=True, interrupt_download=False)


def test_existing_copy_is_fully_verified_without_another_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = existing_copy(tmp_path, monkeypatch)
    module.run(args)
    summary = json.loads((args.receipts / "summary.json").read_text())
    assert summary["mode"] == "verify_existing" and summary["download_seconds"] == 0
    assert summary["file_verification"]["status"] == "verified"
    assert summary["samples_equal"] and summary["sampled_windows"] == 1
    assert json.loads((args.receipts / "samples.json").read_text()) == json.loads(args.reference.read_text())


def reject_changed_inputs(args: Namespace, damage: str) -> None:
    if damage == "manifest_identity":
        args.manifest_sha256 = "0" * 64
    else:
        shard = next((args.destination / "shards").iterdir())
        data = shard.read_bytes()
        shard.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError):
        module.run(args)
    assert not (args.receipts / "summary.json").exists()


def test_existing_copy_still_pins_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reject_changed_inputs(existing_copy(tmp_path, monkeypatch), "manifest_identity")


def test_existing_copy_still_hashes_every_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reject_changed_inputs(existing_copy(tmp_path, monkeypatch), "payload")
