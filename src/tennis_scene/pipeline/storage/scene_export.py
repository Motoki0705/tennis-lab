"""Immutable integrated-scene exports recorded in a clip store index."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import SceneResult
from src.utils.inference_recording import measured, temporary_directory


def validate_scene_export(scene: SceneResult, path: Path) -> None:
    """An immutable export must still contain the scene being published."""
    saved = load_scene_result(path)
    for field in fields(SceneResult):
        expected, actual = getattr(scene, field.name), getattr(saved, field.name)
        if isinstance(expected, np.ndarray):
            equal = isinstance(actual, np.ndarray) and expected.dtype == actual.dtype and np.array_equal(expected, actual)
        elif field.name == "metadata":
            equal = json_value(expected) == json_value(actual)
        else:
            equal = expected == actual
        if not equal:
            raise ValueError(f"Scene export disagrees with reconstructed field {field.name}")


def write_scene_export(scene: SceneResult, destination: Path) -> None:
    """Create an archive directory once under the caller's publication lock."""
    destination.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=destination.parent))
    try:
        with temporary_directory(temporary):
            save_scene_result(scene, temporary / "scene.npz")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@measured("save")
def export_scene(store: ClipStore, scene: SceneResult, reference: ArtifactRef) -> Path:
    """Write ``exports/<artifact ID>/scene.npz`` once and adopt it in ``scene.json``.

    ``reference`` is the adopted ``scene_assembly`` artifact that produced
    ``scene``. An existing export directory for the same artifact is immutable
    and validated before adoption. Creation and adoption share the clip lock.
    """
    exports = store.root / "exports"
    destination = exports / reference.artifact_id
    scene_path: Path = destination / "scene.npz"

    def prepare() -> None:
        if destination.exists():
            current = json.loads(store.index_path.read_text())["exports"].get("scene")
            if current is not None and current.get("inputs") == json_value({"scene_assembly": reference}):
                # Never bless changed bytes by replacing the previously recorded checksum.
                indexed_scene_path(store.index_path)
            validate_scene_export(scene, scene_path)
            return
        write_scene_export(scene, destination)
    store.record_export("scene", {"scene": scene_path, "metadata": destination / "scene.metadata.json"},
        {"scene_assembly": reference}, prepare=prepare)
    return scene_path
