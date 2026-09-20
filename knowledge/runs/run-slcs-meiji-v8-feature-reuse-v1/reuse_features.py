"""Explicit CPU-only migration of verified Meiji v7 RGB tokens to v8.

Run with the repository's Python and a NEW --output-dir. Each invocation writes
its own receipt; reruns validate already published target caches. No teacher
annotation, old receipt, or feature marker is rewritten. Hardlinks require the
source and target to share a filesystem and both to remain immutable afterward.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.tasks.slcs.data.dino_tokens import DinoTokenSpec
    from src.tennis_scene.generate_dataset.manifest import ClipManifest


def _inventory(directory: Path, cameras: tuple[str, ...]) -> list[Path]:
    """Reject aliases, traversal, and files outside the exact feature inventory."""
    from src.tennis_scene.generate_dataset.manifest import validate_id_component

    if directory.resolve() != directory or not directory.is_dir():
        raise ValueError(f"Feature directory is missing or uses a symlink: {directory}")
    for camera in cameras:
        validate_id_component(camera, field_name="camera_id")
    names = {"annotation.json", *(f"{camera}.npz" for camera in cameras)}
    actual = {entry.name for entry in directory.iterdir()}
    if actual != names:
        raise ValueError(
            f"Unexpected feature inventory: missing={names - actual}, extra={actual - names}"
        )
    files = [directory / name for name in sorted(names)]
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ValueError(f"Feature inventory must contain regular files: {directory}")
    marker = json.loads((directory / "annotation.json").read_text())
    entries = marker["cameras"]
    if not isinstance(entries, dict) or set(entries) != set(cameras):
        raise ValueError("Feature marker camera inventory differs from clip")
    for camera in cameras:
        if entries[camera]["file"] != f"{camera}.npz":
            raise ValueError(f"Unsafe or unexpected feature filename for {camera}")
    return files


def _hash_inventory(files: list[Path]) -> dict[str, str]:
    from src.utils.checksum import dual_sha256

    return {path.name: dual_sha256(path) for path in files}


def _require_hashes(files: list[Path], expected: dict[str, str]) -> None:
    from src.utils.checksum import FileIntegrityError

    actual = _hash_inventory(files)
    if actual != expected:
        raise FileIntegrityError(
            "Feature bytes changed during validation/publication",
            details={
                "paths": [str(path) for path in files],
                "expected": expected,
                "actual": actual,
            },
        )


def _rename_no_replace(source: Path, target: Path) -> None:
    """Linux atomic directory publication that cannot replace a concurrent target."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2  # Explicit failure if this platform lacks renameat2.
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(target))


def _match_clip(original: ClipManifest, other: ClipManifest) -> dict[str, str]:
    from src.utils.checksum import dual_sha256

    if (
        original.digest() != other.digest()
        or original.camera_ids != other.camera_ids
        or original.num_frames != other.num_frames
        or original.video_paths != other.video_paths
    ):
        raise ValueError(
            f"Clip manifest/camera/frame identity mismatch: {other.clip_dir}"
        )
    expected = {
        camera: dual_sha256(original.media_path(camera))
        for camera in original.camera_ids
    }
    actual = {
        camera: dual_sha256(other.media_path(camera)) for camera in other.camera_ids
    }
    if actual != expected:
        raise ValueError(
            f"Source media differ: {other.clip_dir}; expected={expected}; actual={actual}"
        )
    return expected


def migrate_clip(
    original: ClipManifest,
    target: ClipManifest,
    source_clip_dir: Path,
    spec: DinoTokenSpec,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Verify one cache and publish only its unchanged feature files, or skip it."""
    from zipfile import BadZipFile

    from src.tasks.slcs.data.dino_tokens import dino_dir
    from src.tennis_scene.dataset_pipeline.features import validated_feature_cache
    from src.tennis_scene.generate_dataset.manifest import ClipManifest
    from src.utils.checksum import FileIntegrityError

    result: dict[str, Any] = {
        "source": str(source_clip_dir),
        "target": str(target.clip_dir),
    }
    media = _match_clip(original, target)
    identity: dict[str, object] = {
        "script": "src/tennis_scene/scripts/build_slcs_dataset.py",
        "checkpoint_sha256": checkpoint_sha256,
        "video_sha256": media,
    }
    result["generator"] = identity
    destination = dino_dir(target.clip_dir)
    if destination.exists() or destination.is_symlink():
        files = _inventory(destination, target.camera_ids)
        hashes = _hash_inventory(files)
        if not validated_feature_cache(target, spec, identity):
            raise ValueError(f"Existing target has no completion marker: {destination}")
        _require_hashes(files, hashes)
        return {**result, "status": "already_valid", "feature_sha256": hashes}

    try:
        source = ClipManifest.load(source_clip_dir)
        _match_clip(original, source)
        directory = dino_dir(source.clip_dir)
        if not (directory / "annotation.json").is_file():
            return {
                **result,
                "status": "skipped",
                "reason": "Source RGB completion marker missing",
            }
        files = _inventory(directory, source.camera_ids)
        hashes = _hash_inventory(files)
        if not validated_feature_cache(source, spec, identity):
            raise ValueError("Source RGB completion marker missing")
        _require_hashes(files, hashes)
    except FileIntegrityError:
        raise
    except (ValueError, TypeError, KeyError, OSError, EOFError, BadZipFile) as error:
        return {
            **result,
            "status": "skipped",
            "reason": f"{type(error).__name__}: {error}",
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve() != destination.parent:
        raise ValueError(f"Target feature parent uses a symlink: {destination.parent}")
    staging = Path(
        tempfile.mkdtemp(prefix=".dino_v3-migration-", dir=destination.parent)
    )
    try:
        for file in files:
            os.link(file, staging / file.name, follow_symlinks=False)
        staged = _inventory(staging, target.camera_ids)
        _require_hashes(files, hashes)
        _require_hashes(staged, hashes)
        _rename_no_replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {**result, "status": "linked", "feature_sha256": hashes}


def _main_repo(repo: Path) -> Path:
    common = Path(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            text=True,
        ).strip()
    )
    return common.parent


def run(receipt: dict[str, Any], receipt_path: Path, repo: Path) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from src.tasks.slcs.data.dino_tokens import DinoTokenSpec
    from src.tennis_scene.dataset_pipeline.build import materialize_dataset
    from src.tennis_scene.dataset_pipeline.configuration import DatasetBuildConfig
    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        load_dataset_manifest,
    )
    from src.utils.checksum import FileIntegrityError, dual_sha256
    from src.utils.io import save_json_atomic

    main_repo = _main_repo(repo)
    with initialize_config_dir(
        version_base="1.3", config_dir=str(repo / "src/tennis_scene/configs")
    ):
        cfg = compose(
            config_name="build_slcs_dataset",
            overrides=[
                f"paths.project_root={main_repo}",
                f"paths.external_asset_root={main_repo / 'third_party'}",
                "device=cpu",
                "stage=features",
            ],
        )
    runtime = DatasetBuildConfig.from_config(cfg)
    source_root = runtime.resolver.roots.data_root / "slcs/meiji_rgb_v7"
    target_root = runtime.resolver.roots.data_root / "slcs/meiji_rgb_v8"
    if runtime.destination != target_root:
        raise ValueError(f"Meiji config no longer targets v8: {runtime.destination}")
    with initialize_config_dir(
        version_base="1.3", config_dir=str(repo / "src/tasks/slcs/configs")
    ):
        settings = compose(config_name="precompute_dino_tokens")
    raw_spec = OmegaConf.to_container(settings.data.dino, resolve=True)
    if not isinstance(raw_spec, dict):
        raise TypeError("SLCS DINO precompute spec must be a mapping")
    spec = DinoTokenSpec.from_dict({str(key): value for key, value in raw_spec.items()})
    expected_checkpoint = str(cfg.checkpoint_sha256.dinov3)
    receipt.update(
        {
            "source": str(source_root),
            "target": str(target_root),
            "original_dataset": str(runtime.source),
            "spec": asdict(spec),
            "checkpoint": str(runtime.feature_checkpoint),
            "checkpoint_sha256": expected_checkpoint,
            "excluded_clips": dict(cfg.excluded_clips),
            "selected_clips": list(runtime.dataset_clip_ids),
        }
    )
    save_json_atomic(receipt, receipt_path)
    actual_checkpoint = dual_sha256(runtime.feature_checkpoint)
    if actual_checkpoint != expected_checkpoint:
        raise FileIntegrityError(
            "Fixed DINOv3 checkpoint identity mismatch",
            details={
                "path": str(runtime.feature_checkpoint),
                "expected": expected_checkpoint,
                "actual": actual_checkpoint,
            },
        )
    receipt["checkpoint_verified"] = True
    save_json_atomic(receipt, receipt_path)
    materialize_dataset(runtime.source, target_root, clip_ids=runtime.dataset_clip_ids)
    original_manifest = load_dataset_manifest(runtime.source)
    target_manifest = load_dataset_manifest(target_root)
    for clip_id in runtime.dataset_clip_ids:
        record = original_manifest.clips[clip_id]
        receipt["current_clip"] = clip_id
        receipt["clips"][clip_id] = {
            "status": "validating",
            "source": str(source_root / record.path),
            "target": str(target_root / record.path),
        }
        save_json_atomic(receipt, receipt_path)
        original = ClipManifest.load(runtime.source / record.path)
        target = ClipManifest.load(target_root / target_manifest.clips[clip_id].path)
        receipt["clips"][clip_id] = migrate_clip(
            original, target, source_root / record.path, spec, expected_checkpoint
        )
        receipt["counts"] = dict(
            Counter(item["status"] for item in receipt["clips"].values())
        )
        save_json_atomic(receipt, receipt_path)
        print(f"{clip_id}: {receipt['clips'][clip_id]['status']}", flush=True)
    receipt.pop("current_clip", None)
    receipt["status"] = "complete"
    save_json_atomic(receipt, receipt_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for this invocation's migration receipt",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from src.utils.io import save_json_atomic, utc_now_iso

    output = args.output_dir.resolve()
    if output.is_relative_to((_main_repo(repo) / "data").resolve()):
        raise ValueError("Migration receipts must live outside the immutable data tree")
    output.mkdir(parents=True, exist_ok=False)
    receipt_path = output / "migration_receipt.json"
    receipt: dict[str, Any] = {
        "script": str(Path(__file__).resolve().relative_to(repo)),
        "started_at": utc_now_iso(),
        "status": "running",
        "clips": {},
        "counts": {},
    }
    save_json_atomic(receipt, receipt_path)
    try:
        run(receipt, receipt_path, repo)
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
        if "current_clip" in receipt:
            receipt["clips"][receipt["current_clip"]].update(
                status="failed", reason=receipt["error"]
            )
        receipt["counts"] = dict(
            Counter(item["status"] for item in receipt["clips"].values())
        )
        save_json_atomic(receipt, receipt_path)
        raise


if __name__ == "__main__":
    main()
