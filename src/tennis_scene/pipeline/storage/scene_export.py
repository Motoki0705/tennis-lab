"""Immutable integrated-scene exports recorded in a clip store index."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path

import numpy as np

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.event_detection import EVENT_DETECTION
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path
from src.tennis_scene.schema import SceneResult
from src.utils.inference_recording import measured, temporary_directory


def scene_export_node(version: int) -> str:
    """Return the producer for a declared SceneResult schema version."""
    if type(version) is not int or version not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported SceneResult version: {version}")
    node: str = EVENT_DETECTION if version in (3, 4) else "scene_assembly"
    return node


def validate_scene_export(scene: SceneResult, path: Path) -> None:
    """An immutable export must still contain the scene being published."""
    saved = load_scene_result(path)
    for field in fields(SceneResult):
        expected, actual = getattr(scene, field.name), getattr(saved, field.name)
        if isinstance(expected, np.ndarray):
            equal = (
                isinstance(actual, np.ndarray)
                and expected.dtype == actual.dtype
                and np.array_equal(
                    expected.view(f"V{expected.itemsize}"), actual.view(f"V{actual.itemsize}"),
                )
            )
        elif field.name == "metadata":
            equal = json_value(expected) == json_value(actual)
        else:
            equal = expected == actual
        if not equal:
            raise ValueError(f"Scene export disagrees with reconstructed field {field.name}")


def write_scene_export(
    scene: SceneResult, destination: Path, *,
    save: Callable[[SceneResult, Path], None] | None = None,
) -> None:
    """Create an archive directory once under the caller's publication lock."""
    destination.parent.mkdir(exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=destination.parent))
    writer = save_scene_result if save is None else save
    try:
        with temporary_directory(temporary):
            writer(scene, temporary / "scene.npz")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@measured("save")
def export_scene(
    store: ClipStore, scene: SceneResult, reference: ArtifactRef, *,
    save: Callable[[SceneResult, Path], None] | None = None,
) -> Path:
    """Write ``exports/<artifact ID>/scene.npz`` once and adopt it in ``scene.json``.

    ``reference`` identifies the component that produced ``scene``.
    An existing export directory for the same artifact is immutable
    and validated before adoption. Creation and adoption share the clip lock.
    """
    destination = store.root / "exports" / reference.artifact_id
    scene_path: Path = destination / "scene.npz"
    inputs = {scene_export_node(scene.schema_version): reference}

    def prepare() -> None:
        if destination.exists():
            current = json.loads(store.index_path.read_text())["exports"].get("scene")
            if current is not None and current.get("inputs") == json_value(inputs):
                # Never bless changed bytes by replacing the previously recorded checksum.
                indexed_scene_path(store.index_path)
            validate_scene_export(scene, scene_path)
            return
        write_scene_export(scene, destination, save=save)
    store.record_export("scene", {"scene": scene_path, "metadata": destination / "scene.metadata.json"},
        inputs, prepare=prepare)
    return scene_path
