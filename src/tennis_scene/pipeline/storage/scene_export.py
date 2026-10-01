"""Immutable integrated-scene exports recorded in a clip store index."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.schema import SceneResult


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
            save_scene_result(scene, temporary / "scene.npz")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    scene_path: Path = destination / "scene.npz"
    store.record_export("scene", {"scene": scene_path, "metadata": destination / "scene.metadata.json"},
        {"scene_assembly": reference})
    return scene_path
