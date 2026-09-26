"""Write tennis_scene publications for reader tests without running models."""

from __future__ import annotations

from pathlib import Path

from src.tennis_scene.archive import save_scene_result
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    ANNOTATION_RELATIVE_DIR,
    ANNOTATION_SCHEMA_VERSION,
    scene_array_manifest,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SceneResult
from src.utils.checksum import dual_sha256
from src.utils.io import save_json_atomic


def write_historical_v1_annotation(clip_dir: Path, clip_id: str, scene: SceneResult) -> Path:
    """Write the pre-component v1 layout that published datasets on disk still contain.

    ``annotation.json`` names ``scene.npz`` directly and has no ``scene_index``.
    The production generator no longer writes this layout; readers must keep
    accepting it.
    """
    if scene.schema_version != 1:
        raise ValueError("The historical layout only ever held v1 scenes")
    destination = clip_dir / ANNOTATION_RELATIVE_DIR
    destination.mkdir(parents=True)
    save_scene_result(scene, destination / "scene.npz")
    marker = {
        "version": ANNOTATION_SCHEMA_VERSION, "clip_id": clip_id, "generator": "src.tennis_scene",
        "scene_result": "scene.npz", "clip_manifest_sha256": dual_sha256(clip_dir / "clip.json"),
        "arrays": scene_array_manifest(scene), "scene_schema_version": 1,
    }
    marker_path: Path = save_json_atomic(marker, destination / "annotation.json")
    return marker_path


def publish_scene_to_clip_store(clip_dir: Path, clip_id: str, scene: SceneResult) -> Path:
    """Adopt ``scene`` as the clip store's ``scene_assembly`` artifact and export it.

    This is the state the declared pipeline leaves behind before
    ``generate_pseudo_annotations`` publishes the completion marker.
    """
    store = ClipStore(clip_dir / ANNOTATION_RELATIVE_DIR, {"clip_id": clip_id})
    reference = store.publish("scene_assembly", scene, ArtifactCodec(SceneResult), schema="scene_result", version=2,
        identity={"fixture": clip_id}, dependencies={}, provenance={"origin": "test_fixture"})
    return export_scene(store, scene, reference)
