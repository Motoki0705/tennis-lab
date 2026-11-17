"""Immutable integrated-scene exports recorded in a clip store index."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import fields
from pathlib import Path

import numpy as np

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.schema import SceneResult
from src.utils.inference_recording import measured, temporary_directory


def validate_scene_export(scene: SceneResult, path: Path) -> None:
    """An immutable export must still contain the scene being published."""
    saved = load_scene_result(path)
    for field in fields(SceneResult):
        expected, actual = getattr(scene, field.name), getattr(saved, field.name)
        if isinstance(expected, np.ndarray):
            equal = isinstance(actual, np.ndarray) and expected.dtype == actual.dtype and np.array_equal(expected, actual)
        else:
            equal = expected == actual
        if not equal:
            raise ValueError(f"Scene export disagrees with reconstructed field {field.name}")


@measured("save")
def export_scene(store: ClipStore, scene: SceneResult, reference: ArtifactRef) -> Path:
    """Write ``exports/<artifact ID>/scene.npz`` once and adopt it in ``scene.json``.

    ``reference`` is the adopted ``scene_assembly`` artifact that produced
    ``scene``. An existing export directory for the same artifact is immutable
    and reused as-is.
    """
    exports = store.root / "exports"
    exports.mkdir(exist_ok=True)
    destination = exports / reference.artifact_id
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=exports))
        try:
            with temporary_directory(temporary):
                save_scene_result(scene, temporary / "scene.npz")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    scene_path: Path = destination / "scene.npz"
    validate_scene_export(scene, scene_path)
    store.record_export("scene", {"scene": scene_path, "metadata": destination / "scene.metadata.json"},
        {"scene_assembly": reference})
    return scene_path
