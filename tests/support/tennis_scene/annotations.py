"""Write tennis_scene publications for reader tests without running models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.tennis_scene.ball_event_schema import ball_event_metadata
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    ANNOTATION_RELATIVE_DIR,
    generate_pseudo_annotations,
)
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SceneResult

FIXTURE_PIPELINE_CONFIG_YAML = "fixture: tests.support.tennis_scene.annotations\n"
FIXTURE_PUBLICATION_IDENTITY: Mapping[str, object] = {"fixture": "tests.support.tennis_scene.annotations"}


def publish_scene_to_clip_store(clip_dir: Path, clip_id: str, scene: SceneResult) -> Path:
    """Adopt and export a fixture scene through the clip store.

    This is the state the declared pipeline leaves behind before
    ``generate_pseudo_annotations`` publishes the completion marker.
    """
    store = ClipStore(clip_dir / ANNOTATION_RELATIVE_DIR, {"clip_id": clip_id})
    name = "event_detection" if scene.schema_version == 3 else "scene_assembly"
    reference = store.publish(name, scene, ArtifactCodec(SceneResult), schema="scene_result", version=scene.schema_version,
        identity={"fixture": clip_id}, dependencies={}, provenance={"origin": "test_fixture"})
    path: Path = export_scene(store, scene, reference)
    return path


def with_fixture_ball_events(scene: SceneResult) -> SceneResult:
    """Add one out-of-point bounce and an independent event coverage mask."""
    metadata = ball_event_metadata("annotation", {
        "code_commit": "a" * 40,
        "annotation_sha256": "b" * 64,
        "labeler": "fixture",
        "labeler_original": "fixture",
    })
    coverage: np.ndarray = np.ones(scene.num_frames, dtype=bool)
    coverage[0] = False
    return replace(scene,
        metadata={**scene.metadata, "scene_schema_version": 3, "ball_events": metadata},
        ball_event_frame=np.array([scene.num_frames - 1], np.int32),
        ball_event_kind=np.array([1], np.uint8),
        ball_event_coverage=coverage,
        ball_event_status=np.array([0], np.uint8),
        ball_event_frame_uncertainty=np.array([2], np.uint8),
        ball_event_in_point=np.array([False]),
        ball_event_subtype=np.array([7], np.uint8),
        ball_event_side=np.array([0], np.int8),
    )


def publish_dataset_annotations(dataset_root: Path, scenes: Mapping[str, SceneResult], *, overwrite: bool = False) -> None:
    """Publish ``scenes`` exactly as the production generator does, without models.

    Each clip gets a component store holding the scene export, and the
    generator's own marker writer validates the scene and records it.
    """
    def runner(video_paths: Sequence[Path], camera_ids: Sequence[str], clip_dir: Path) -> SceneResult:
        clip_id = ClipManifest.load(clip_dir).clip_id
        publish_scene_to_clip_store(clip_dir, clip_id, scenes[clip_id])
        return scenes[clip_id]

    outcomes = generate_pseudo_annotations(
        dataset_root.resolve(), runner, pipeline_config_yaml=FIXTURE_PIPELINE_CONFIG_YAML,
        publication_identity=FIXTURE_PUBLICATION_IDENTITY, clip_ids=sorted(scenes), overwrite=overwrite,
        continue_on_error=False,
    )
    failed = [outcome for outcome in outcomes if outcome.status != "generated"]
    if failed:
        raise ValueError(f"Fixture publication failed: {failed}")
