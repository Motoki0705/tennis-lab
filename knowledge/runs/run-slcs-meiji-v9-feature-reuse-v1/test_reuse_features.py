"""Small CPU fixtures for strict feature reuse and failure preservation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from src.tasks.slcs.data.dino_tokens import DinoTokenSpec, dino_dir, write_dino_tokens
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.utils.checksum import dual_sha256


@pytest.fixture
def driver() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v9_reuse_test", Path(__file__).with_name("reuse_features.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_clip(root: Path) -> ClipManifest:
    root.mkdir(parents=True)
    (root / "media").mkdir()
    (root / "media/cam0.mp4").write_bytes(b"identical video fixture")
    payload = {
        "version": 2,
        "dataset_id": "fixture",
        "clip_id": "video_000/clip_000",
        "video_id": "video_000",
        "clip_name": "clip_000",
        "fps": 30.0,
        "num_frames": 3,
        "width": 16,
        "height": 16,
        "global_start_sec": 0.0,
        "global_end_sec": 0.1,
        "camera_ids": ["cam0"],
        "video_paths": ["media/cam0.mp4"],
        "cameras": [],
        "sync_source": "fixture",
        "exported_at": "fixture",
    }
    (root / "clip.json").write_text(json.dumps(payload))
    return ClipManifest.load(root)


@pytest.fixture
def clips(
    tmp_path: Path, driver: ModuleType
) -> tuple[ClipManifest, ClipManifest, ClipManifest, DinoTokenSpec, dict[str, Any]]:
    original, source, target = [
        make_clip(tmp_path / name) for name in ("original", "source", "target")
    ]
    spec = DinoTokenSpec("fixture", 16, 16, 16, 2, 2)
    identity = {
        "script": "src/tennis_scene/scripts/build_slcs_dataset.py",
        "checkpoint_sha256": driver.PIN,
        "video_sha256": {"cam0": dual_sha256(original.media_path("cam0"))},
    }
    write_dino_tokens(
        source,
        {"cam0": (np.ones((2, 1, 2), np.float16), np.array([0, 2], np.int64))},
        spec,
        generator=identity,
    )
    return original, source, target, spec, identity


def test_atomic_feature_only_reuse_and_existing_validation(
    driver: ModuleType, clips: tuple[Any, ...]
) -> None:
    original, source, target, spec, identity = clips
    report: dict[str, Any] = {"errors": []}
    hashes = driver.HashAudit(report)
    legacy = driver.legacy_module()
    _, expected = driver.prepare_clip(original, source, target, spec, hashes, legacy)
    teacher = source.clip_dir / "annotations/teacher-secret.bin"
    teacher.write_bytes(b"must not copy")
    result = legacy.migrate_clip(original, target, source.clip_dir, spec, driver.PIN)
    assert result["status"] == "linked"
    assert driver.validate_cache(target, spec, identity, hashes, legacy) == expected
    assert not (target.clip_dir / "annotations/teacher-secret.bin").exists()
    assert (dino_dir(source.clip_dir) / "cam0.npz").stat().st_ino == (
        dino_dir(target.clip_dir) / "cam0.npz"
    ).stat().st_ino
    assert (
        legacy.migrate_clip(original, target, source.clip_dir, spec, driver.PIN)[
            "status"
        ]
        == "already_valid"
    )
    hashes.finish()
    assert not report["errors"]


@pytest.mark.parametrize(
    "damage",
    [
        "producer",
        "missing_marker",
        "sampling",
        "nonfinite",
        "dtype",
        "manifest",
        "media",
    ],
)
def test_bad_inputs_rejected_before_publication(
    driver: ModuleType, clips: tuple[Any, ...], damage: str
) -> None:
    original, source, target, spec, _ = clips
    directory = dino_dir(source.clip_dir)
    if damage == "producer":
        path = directory / "annotation.json"
        marker = json.loads(path.read_text())
        marker["generator"]["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(marker))
    elif damage == "missing_marker":
        (directory / "annotation.json").unlink()
    elif damage in {"sampling", "nonfinite", "dtype"}:
        tokens = np.ones(
            (2, 1, 2), dtype=np.float32 if damage == "dtype" else np.float16
        )
        if damage == "nonfinite":
            tokens[0, 0, 0] = np.nan
        indices = np.array([0, 1] if damage == "sampling" else [0, 2], np.int64)
        np.savez_compressed(directory / "cam0.npz", tokens=tokens, frame_idx=indices)
    elif damage == "manifest":
        path = target.clip_dir / "clip.json"
        data = json.loads(path.read_text())
        data["exported_at"] = "changed"
        path.write_text(json.dumps(data))
    else:
        target.media_path("cam0").write_bytes(b"different video")
    with pytest.raises(ValueError):
        driver.prepare_clip(
            original,
            source,
            target,
            spec,
            driver.HashAudit({"errors": []}),
            driver.legacy_module(),
        )
    assert not dino_dir(target.clip_dir).exists()


def test_changed_bytes_remain_failure_evidence(
    driver: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "input"
    path.write_bytes(b"before")
    report: dict[str, Any] = {"errors": []}
    audit = driver.HashAudit(report)
    before = audit.capture(path)
    path.write_bytes(b"after")
    audit.finish("prepublication_hash")
    path.write_bytes(b"before")
    audit.finish()
    assert report["errors"] and report["errors"][0]["stage"] == "prepublication_hash"
    assert report["input_sha256_prepublication"][str(path)] != before
    assert report["input_sha256_after"][str(path)] == before


def test_wrong_dataset_version_rejected(driver: ModuleType, tmp_path: Path) -> None:
    original = tmp_path / "processed/meiji_3cam/dataset"
    source, target = tmp_path / "meiji_rgb_v7", tmp_path / "meiji_rgb_v9"
    for root in (original, source):
        root.mkdir(parents=True)
    with pytest.raises(ValueError, match="source v8"):
        driver.validate_versions(original, source, target)


def test_failed_cli_receipt_and_no_overwrite(
    driver: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    output = tmp_path / "audit"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "reuse",
            "--original",
            str(tmp_path / "wrong"),
            "--source",
            str(tmp_path / "v7"),
            "--target",
            str(tmp_path / "v9"),
            "--checkpoint",
            str(tmp_path / "absent"),
            "--output-dir",
            str(output),
        ],
    )
    assert driver.main() == 1
    receipt = output / "audit.json"
    data = receipt.read_bytes()
    assert json.loads(data)["status"] == "failed"
    with pytest.raises(FileExistsError):
        driver.main()
    assert receipt.read_bytes() == data


def test_skip_is_failure_not_completion(
    driver: ModuleType, clips: tuple[Any, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    original, source, target, spec, identity = clips
    report: dict[str, Any] = {"errors": [], "clips": {}}
    hashes = driver.HashAudit(report)
    legacy = driver.legacy_module()
    _, expected = driver.prepare_clip(original, source, target, spec, hashes, legacy)
    monkeypatch.setattr(
        legacy,
        "migrate_clip",
        lambda *args: {"status": "skipped", "reason": "missing marker"},
    )
    with pytest.raises(ValueError, match="reuse incomplete"):
        driver.publish_clip(
            original, source, target, spec, identity, expected, hashes, legacy, report
        )
    assert report["clips"][target.clip_id]["status"] == "skipped"
    assert not dino_dir(target.clip_dir).exists()


def test_materialization_creates_only_manifest_media_and_rejects_changed_target(
    tmp_path: Path,
) -> None:
    from src.tennis_scene.dataset_pipeline.build import materialize_dataset
    from src.tennis_scene.generate_dataset.manifest import register_exported_clip

    original_root = tmp_path / "original"
    relative = Path("videos/video_000/clips/clip_000")
    original = make_clip(original_root / relative)
    register_exported_clip(original_root, original.clip_dir / "clip.json")
    (original.clip_dir / "annotations").mkdir()
    (original.clip_dir / "annotations/teacher.bin").write_bytes(b"teacher")
    target_root = tmp_path / "target"
    materialize_dataset(original_root, target_root, clip_ids=(original.clip_id,))
    assert (
        target_root / relative / "media/cam0.mp4"
    ).stat().st_ino == original.media_path("cam0").stat().st_ino
    assert not (target_root / relative / "annotations").exists()
    (target_root / relative / "clip.json").write_text("changed")
    with pytest.raises(ValueError, match="Input manifest changed"):
        materialize_dataset(original_root, target_root, clip_ids=(original.clip_id,))
