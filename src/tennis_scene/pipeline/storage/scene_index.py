"""Resolve a published scene without guessing among component revisions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.checksum import dual_sha256


def read_component_descriptor(root: Path, reference: Mapping[str, Any], *, node: str | None, source_sha256: str) -> dict[str, Any]:
    """Read one component descriptor after checking its path, checksum and index entry.

    ``node=None`` skips the node-name check (a store reads by reference only).
    """
    path = (root / reference["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or dual_sha256(path) != reference["sha256"]:
        raise ValueError(f"Component descriptor path/checksum mismatch: {node or reference['path']}")
    descriptor = json.loads(path.read_text())
    if not isinstance(descriptor, dict):
        raise ValueError("Component descriptor must be an object")
    expected = (reference["artifact_id"], reference["schema"], reference["version"], source_sha256)
    actual = (descriptor.get("artifact_id"), descriptor.get("output_schema"), descriptor.get("output_version"), descriptor.get("source_sha256"))
    if actual != expected or (node is not None and descriptor.get("node") != node):
        raise ValueError(f"Component descriptor disagrees with the scene index: {node or reference['path']}")
    return descriptor


def assert_current_component_lineage(document: dict[str, Any], root: Path, inputs: dict[str, Any]) -> None:
    """Reject an old export even if its file checksum still matches."""
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(inputs, dict) or not inputs:
        raise ValueError("Scene export has no component input lineage")
    checked: set[str] = set()
    visiting: set[str] = set()

    def visit(node: str) -> None:
        if node in checked:
            return
        if node in visiting:
            raise ValueError("Scene export contains cyclic component dependencies")
        visiting.add(node)
        reference = artifacts.get(node)
        if not isinstance(reference, dict):
            raise ValueError(f"Scene export missing active component: {node}")
        descriptor = read_component_descriptor(root, reference, node=node, source_sha256=str(document.get("source_sha256")))
        dependencies = descriptor.get("dependencies")
        if not isinstance(dependencies, dict):
            raise ValueError(f"Scene export component has invalid dependencies: {node}")
        for dependency in dependencies.values():
            producers = [name for name, active in artifacts.items() if active == dependency]
            if len(producers) != 1:
                raise ValueError(f"Scene export references a superseded component dependency: {node}")
            visit(producers[0])
        visiting.remove(node)
        checked.add(node)

    for node, reference in inputs.items():
        if artifacts.get(node) != reference:
            raise ValueError(f"Scene export references a superseded component: {node}")
        visit(node)


def indexed_scene_path(index_path: Path) -> Path:
    document = json.loads(index_path.read_text())
    if document.get("schema") != "tennis_scene_index_v1":
        raise ValueError("Unsupported scene index schema")
    export = document.get("exports", {}).get("scene")
    if not isinstance(export, dict):
        raise ValueError("Clip has no completed scene export")
    root = index_path.parent.resolve()
    inputs = export.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Scene export has no component input lineage")
    assert_current_component_lineage(document, root, inputs)
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
