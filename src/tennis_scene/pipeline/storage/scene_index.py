"""Resolve a published scene without guessing among component revisions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.checksum import dual_sha256


def indexed_scene_path(index_path: Path) -> Path:
    document = json.loads(index_path.read_text())
    if document.get("schema") != "tennis_scene_index_v1":
        raise ValueError("Unsupported scene index schema")
    export = document.get("exports", {}).get("scene")
    if not isinstance(export, dict):
        raise ValueError("Clip has no completed scene export")
    root = index_path.parent.resolve()
    paths: dict[str, Path] = {}
    for role in ("scene", "metadata"):
        record = export["files"][role]
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or dual_sha256(path) != record["sha256"]:
            raise ValueError(f"Scene export {role} path/checksum mismatch")
        paths[role] = path
    if paths["scene"].with_suffix(".metadata.json") != paths["metadata"]:
        raise ValueError("Scene export sidecar mismatch")
    return paths["scene"]


def annotation_scene_path(directory: Path, marker: dict[str, Any]) -> Path:
    relative = marker.get("scene_result")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Annotation marker requires an explicit scene_result")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or path.suffix != ".npz":
        raise ValueError("Annotation scene path escapes its publication or has an invalid suffix")
    if "scene_index" in marker and (marker["scene_index"] != "scene.json" or indexed_scene_path(directory / "scene.json") != path):
        raise ValueError("Annotation marker and scene index disagree")
    return path
