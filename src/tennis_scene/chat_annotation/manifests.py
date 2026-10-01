"""Receipt-verified index of prepared clip manifests under one output root."""

from __future__ import annotations

from pathlib import Path

from .runtime.contracts import (
    ClipManifest,
    annotation_clip_id,
    read_json,
    sha256_file,
)


def require_regular_path(path: Path, root: Path) -> None:
    """Reject paths outside ``root`` or reached through a symlink below it."""
    if not path.is_relative_to(root):
        raise ValueError("path outside output root")
    if any(
        part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)
    ):
        raise ValueError(f"symlink is forbidden: {path}")


def load_prepared_manifests(root: Path) -> dict[str, ClipManifest]:
    """Map annotation clip IDs to manifests whose ready receipt matches exactly.

    ``root`` must already be resolved. In-progress ``.building-*`` directories
    are ignored; any receipt mismatch or duplicate clip identity is an error.
    """
    if not (root / "_preparation").is_dir():
        raise ValueError("output root must contain _preparation manifests")
    manifests: dict[str, ClipManifest] = {}
    for path in sorted((root / "_preparation").glob("*/*/clips/*/clip_manifest.json")):
        if path.parent.name.startswith(".building-"):
            continue
        require_regular_path(path, root)
        manifest = ClipManifest.model_validate(read_json(path))
        ready_path = path.parents[2] / "ready" / f"{path.parent.name}.json"
        require_regular_path(ready_path, root)
        ready = read_json(ready_path)
        if ready["files"] != {
            "clip_manifest.json": sha256_file(path),
            manifest.filename: manifest.sha256,
        }:
            raise ValueError(f"preparation receipt mismatch: {path}")
        clip_id = annotation_clip_id(manifest)
        if clip_id in manifests:
            raise ValueError(f"duplicate clip identity: {clip_id}")
        manifests[clip_id] = manifest
    return manifests
