"""Immutable, image-free Court scene inputs for remote NHT rendering."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from src.synthetic_data_generation.alignment.validation import load_alignment_result
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
    validate_standard_scene_export,
)

ASSET_SCHEMA = "court_scene_assets_v1"
MANIFEST_FILE = "assets.json"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def file_sha256(path: Path) -> str:
    """Stream one ordinary file; never follow an asset symlink."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an ordinary asset file: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid asset-relative path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or path.as_posix() != value
        or path == PurePosixPath(".")
    ):
        raise ValueError(f"Invalid asset-relative path: {value!r}")
    return path


def _contained(root: Path, value: Any) -> Path:
    relative = _relative_path(value)
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset reference escapes its root: {value}")
    if any(
        parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent
    ):
        raise ValueError(f"Asset references must not use symlinks: {value}")
    return path


def _json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _record(root: Path, relative: str) -> dict[str, Any]:
    path = _contained(root, relative)
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _scene_inputs(workspace: Path) -> tuple[StandardSceneExport, tuple[str, ...]]:
    export = workspace / "reconstruction/export"
    scene = validate_standard_scene_export(
        export / "scene.json", without_source_images=True
    )
    alignment = load_alignment_result(workspace / "alignment/alignment.json")
    if not set(
        (*alignment.partitions.fit_camera_ids, *alignment.partitions.holdout_camera_ids)
    ).issubset(scene.camera_ids):
        raise ValueError("Alignment camera partitions disagree with public export.")
    relative_files = (
        "reconstruction/export/scene.json",
        "reconstruction/export/cameras.json",
        "reconstruction/export/points_scene.npy",
        scene.checkpoint_path.relative_to(workspace).as_posix(),
        scene.runtime_config_path.relative_to(workspace).as_posix(),
        "alignment/alignment.json",
    )
    for relative in relative_files:
        path = _contained(workspace, relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected an ordinary asset input: {path}")
    return scene, relative_files


def pack_scene_assets(
    source_root: Path,
    destination: Path,
    *,
    scene_ids: Sequence[str],
    repository_commit: str,
    nht_commit: str,
) -> dict[str, Any]:
    """Copy the exact render/label dependency closure into a new asset version.

    Alignment is parsed for semantics, then copied byte for byte. Evidence and
    prior dataset descriptors remain digest references to the source workspace.
    No reconstruction, alignment fitting, image conversion or GPU work occurs.
    """
    source_root = source_root.resolve(strict=True)
    destination = destination.absolute()
    ids = tuple(scene_ids)
    if not ids or len(ids) != len(set(ids)) or any(not _ID.fullmatch(s) for s in ids):
        raise ValueError("scene_ids must be nonempty, unique portable identifiers.")
    if not _COMMIT.fullmatch(repository_commit) or not _COMMIT.fullmatch(nht_commit):
        raise ValueError("Both code versions must be full Git commit hashes.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Asset destination already exists: {destination}")
    if destination.resolve().is_relative_to(source_root):
        raise ValueError("Asset destination must be outside the source workspaces.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    manifest: dict[str, Any] = {
        "schema": ASSET_SCHEMA,
        "repository_commit": repository_commit,
        "nht_commit": nht_commit,
        "source_images": "omitted",
        "alignment_validation": "stored_result_geometry",
        "scenes": [],
        "files": [],
    }
    try:
        for scene_id in sorted(ids):
            workspace = _contained(source_root, scene_id)
            scene, inputs = _scene_inputs(workspace)
            if scene.scene_id != scene_id:
                raise ValueError("Workspace scene ID disagrees with public export.")
            references = []
            for relative in (
                "run.json",
                "resolved-config.yaml",
                "datasets/court/dataset.json",
                "datasets/court/samples/index.npz",
                "alignment/manual-confirmation.json",
                "alignment/ground-line-map.npz",
            ):
                if (workspace / relative).exists():
                    references.append(_record(workspace, relative))
            manifest["scenes"].append(
                {
                    "scene_id": scene_id,
                    "workspace": f"synthetic_data_generation/scenes/{scene_id}",
                    "source_references": references,
                }
            )
            for relative in inputs:
                source_record = _record(workspace, relative)
                asset_relative = f"scenes/{scene_id}/{relative}"
                output = staging / asset_relative
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(workspace / relative, output)
                record = _record(staging, asset_relative)
                if record["sha256"] != source_record["sha256"]:
                    raise ValueError(f"Asset changed while copying: {relative}")
                manifest["files"].append(record)
        manifest["files"].sort(key=lambda item: item["path"])
        (staging / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        verify_scene_assets(staging)
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return asset_receipt(destination, manifest)


def _check_record(record: Any) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise ValueError("Invalid asset file record.")
    _relative_path(record["path"])
    if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
        raise ValueError("Invalid asset file size.")
    if not isinstance(record["sha256"], str) or not _DIGEST.fullmatch(record["sha256"]):
        raise ValueError("Invalid asset SHA-256.")


def verify_scene_assets(
    root: Path, *, manifest_sha256: str | None = None
) -> dict[str, Any]:
    """Verify exact inventory, content and stored geometry after any transfer."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Asset root must be an ordinary directory.")
    root = root.resolve(strict=True)
    manifest_path = root / MANIFEST_FILE
    digest = file_sha256(manifest_path)
    if manifest_sha256 is not None and digest != manifest_sha256:
        raise ValueError("Asset manifest SHA-256 mismatch.")
    manifest = _json(manifest_path)
    if (
        set(manifest)
        != {
            "schema",
            "repository_commit",
            "nht_commit",
            "source_images",
            "alignment_validation",
            "scenes",
            "files",
        }
        or manifest["schema"] != ASSET_SCHEMA
    ):
        raise ValueError("Unsupported Court scene asset manifest schema.")
    if (
        manifest["source_images"] != "omitted"
        or manifest["alignment_validation"] != "stored_result_geometry"
    ):
        raise ValueError("Unsupported asset validation policy.")
    for key in ("repository_commit", "nht_commit"):
        if not isinstance(manifest[key], str) or not _COMMIT.fullmatch(manifest[key]):
            raise ValueError("Invalid asset producer commit.")
    if not isinstance(manifest["files"], list) or not isinstance(
        manifest["scenes"], list
    ):
        raise ValueError("Asset scenes/files must be arrays.")
    expected = {MANIFEST_FILE}
    for record in manifest["files"]:
        _check_record(record)
        if record["path"] in expected:
            raise ValueError("Duplicate asset file path.")
        expected.add(record["path"])
        if _record(root, record["path"]) != record:
            raise ValueError(f"Asset content mismatch: {record['path']}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Asset symlink is forbidden: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("Asset inventory mismatch.")
    closure = {MANIFEST_FILE}
    ids: set[str] = set()
    for item in manifest["scenes"]:
        if not isinstance(item, dict) or set(item) != {
            "scene_id",
            "workspace",
            "source_references",
        }:
            raise ValueError("Invalid asset scene record.")
        scene_id = item["scene_id"]
        if (
            not isinstance(scene_id, str)
            or not _ID.fullmatch(scene_id)
            or scene_id in ids
        ):
            raise ValueError("Invalid or duplicate asset scene ID.")
        ids.add(scene_id)
        if item["workspace"] != f"synthetic_data_generation/scenes/{scene_id}":
            raise ValueError("Invalid source workspace reference.")
        if not isinstance(item["source_references"], list):
            raise ValueError("Invalid source provenance references.")
        for reference in item["source_references"]:
            _check_record(reference)
        scene, inputs = _scene_inputs(_contained(root, f"scenes/{scene_id}"))
        if scene.scene_id != scene_id:
            raise ValueError("Asset scene ID disagrees with public export.")
        closure.update(f"scenes/{scene_id}/{relative}" for relative in inputs)
    if not ids or closure != expected:
        raise ValueError(
            "Asset inventory does not equal the rendering dependency closure."
        )
    return asset_receipt(root, manifest)


def asset_receipt(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a small receipt; the asset manifest remains the content authority."""
    return {
        "schema": "court_scene_assets_receipt_v1",
        "manifest_sha256": file_sha256(root / MANIFEST_FILE),
        "scene_ids": [item["scene_id"] for item in manifest["scenes"]],
        "file_count": len(manifest["files"]),
        "size_bytes": sum(item["size_bytes"] for item in manifest["files"]),
    }
