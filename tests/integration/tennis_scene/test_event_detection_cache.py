"""Event cache and publication identity through the standard pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.tasks.ball_refiner_3d.model_io.contracts import RefinerPrediction
from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.ball_event_schema import EventSource
from src.tennis_scene.ball_events import annotation as meiji_annotation
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline import orchestrator, source_identity
from src.tennis_scene.pipeline.clip_inputs import (
    DatasetClipInputs,
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.components.event_detection import (
    EVENT_DETECTION,
    EventDetectionConfig,
)
from src.tennis_scene.pipeline.dataset_publication import (
    ANNOTATION_RELATIVE_DIR,
    completed_annotation,
    publish_annotation,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import ComponentRunner
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT
from tests.support.tennis_scene.video_dataset import add_clip
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    setup_pipeline,
)


class _Refiner:
    def predict(self, coordinates: torch.Tensor, missing: torch.Tensor, **settings: Any) -> RefinerPrediction:
        probability = torch.zeros((*coordinates.shape[:2], 3))
        probability[:, 4, 1] = 0.8
        return RefinerPrediction(coordinates, probability)


def _event_config(
    method: EventSource, cfg: PipelineRuntimeConfig, clip: DatasetClipInputs, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> EventDetectionConfig:
    if method == "rule":
        return EventDetectionConfig(method)
    if method == "refiner":
        checkpoint = cfg.roots.checkpoint_root / "refiner.ckpt"
        monkeypatch.setattr(
            "src.tasks.ball_refiner_3d.inference.predictor.RefinerPredictor.from_checkpoint",
            lambda path, *, device: _Refiner(),
        )
        return EventDetectionConfig(method, checkpoint=checkpoint, checkpoint_sha256=sha256(b"asset").hexdigest(),
                                    fps_tolerance=0.06)

    record = clip.record
    manifest = tmp_path / "meiji-v2.json"
    timeline = {"clip_id": record.clip_id, "fps": record.fps, "frames": record.num_frames}
    manifest.write_text(json.dumps({
        "benchmark_id": "meiji_4cam/multiview-v2",
        "clips": [{**timeline, "cameras": [
            {"camera_id": camera, "sha256": digest} for camera, digest in clip.media_sha256.items()
        ]}],
    }))
    monkeypatch.setattr(meiji_annotation, "MEIJI_V2_MANIFEST_SHA256", sha256(manifest.read_bytes()).hexdigest())
    (tmp_path / meiji_annotation.ANNOTATION_SCHEMA_FILENAME).write_text("{}")
    meiji_annotation.annotation_path(tmp_path, record.clip_id).write_text(json.dumps({
        **timeline, "schema": "meiji_event_labels.v1", "benchmark": "multiview-v2",
        "labeler": "ChatGPT", "unlabelable": [],
        "events": [{"kind": "bounce", "id": "one", "frame": 4, "frame_uncertainty": 1,
                    "status": "certain", "in_point": True}],
    }))
    return EventDetectionConfig(method, annotation_directory=tmp_path, annotation_benchmark_manifest=manifest)


def _changed_event_config(config: EventDetectionConfig, clip_id: str) -> EventDetectionConfig:
    if config.method == "rule":
        return replace(config, thresholds=replace(config.thresholds, bounce_max_height_m=0.31))
    if config.method == "refiner":
        return replace(config, fps_tolerance=0.12)
    assert config.annotation_directory is not None
    path = meiji_annotation.annotation_path(config.annotation_directory, clip_id)
    annotation = json.loads(path.read_text())
    annotation["events"][0]["frame"] = 6
    path.write_text(json.dumps(annotation))
    return config


@pytest.mark.parametrize(("method", "event_source"), [
    ("rule", "tennis_scene/ball_events/rules.py"),
    ("refiner", "tasks/ball_refiner_3d/inference/scene.py"),
    ("annotation", "tennis_scene/ball_events/annotation.py"),
], ids=["rule", "refiner", "annotation"])
def test_event_cache_tracks_sources_and_settings_and_keeps_computation_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: EventSource, event_source: str,
) -> None:
    project = tmp_path / "checkout"
    source = project / "src" / event_source
    source.parent.mkdir(parents=True)
    source.write_bytes((PROJECT_ROOT / "src" / event_source).read_bytes())
    monkeypatch.setattr(orchestrator, "PROJECT_ROOT", project)
    first_commit, second_commit = "a" * 40, "b" * 40
    checkout_commit = first_commit
    monkeypatch.setattr(source_identity, "subprocess", SimpleNamespace(check_output=lambda *a, **kw: checkout_commit))

    fixture, _, overrides = setup_pipeline(tmp_path, monkeypatch)
    dataset, clip_id = tmp_path / "dataset", "video_000/clip_000"
    add_clip(dataset, clip_id)
    record, = select_dataset_clips(dataset)
    bound = resolve_dataset_clip(dataset, record)
    store_root = bound.clip_dir / ANNOTATION_RELATIVE_DIR
    event_config = _event_config(method, fixture.config, bound, tmp_path, monkeypatch)
    config = replace(fixture.config, event_detection=event_config)
    materialize_assets(enabled_model_assets(config).values(), tmp_path, court=config.court_kp)
    yaml = f"event_detection:\n  method: {method}\n"

    def new_pipeline(cfg: PipelineRuntimeConfig) -> TennisSceneOrchestrator:
        return TennisSceneOrchestrator(cfg, components=overrides)

    def run(pipeline: TennisSceneOrchestrator) -> tuple[SceneResult, ComponentRunner]:
        scene = pipeline.run(bound.video_paths, video_role=PathRole.DATA, camera_ids=bound.camera_ids,
                             store_root=store_root, clip_id=clip_id)
        assert pipeline.last_runner is not None
        return scene, pipeline.last_runner

    def load_events(cfg: PipelineRuntimeConfig) -> PipelineRuntimeConfig:
        return replace(cfg, component_sources={**cfg.component_sources, EVENT_DETECTION: "load"})

    first = new_pipeline(config)
    publication = first.publication_identity()
    scene, initial = run(first)
    assert initial.statuses[EVENT_DETECTION] == "executed"
    assert scene.metadata["ball_events"]["provenance"]["code_commit"] == first_commit
    archive = first.completed_scene_export()
    marker = publish_annotation(bound, scene, pipeline_config_yaml=yaml, publication_identity=publication,
                                expected_scene=archive)
    checkout_commit = second_commit

    for cfg, status in ((config, "cached"), (load_events(config), "loaded")):
        pipeline = new_pipeline(cfg)
        scene, runner = run(pipeline)
        assert runner.statuses[EVENT_DETECTION] == status
        assert runner.references == initial.references
        assert pipeline.completed_scene_export() == archive
        assert scene.metadata["ball_events"]["provenance"]["code_commit"] == first_commit
    archived = load_scene_result(store_root)
    assert archived.schema_version == 3
    assert archived.metadata["ball_events"]["provenance"]["code_commit"] == first_commit
    same_publication = new_pipeline(config).publication_identity()
    assert same_publication == publication
    assert completed_annotation(bound, same_publication) == marker

    with source.open("a", encoding="utf-8") as stream:
        stream.write("\n# edit only the checkout's event source\n")
    changed_code = new_pipeline(config)
    changed_publication = changed_code.publication_identity()
    assert changed_publication["code_sha256"] != publication["code_sha256"]
    published_marker = marker.read_bytes()
    with pytest.raises(ValueError, match="Published event_detection implementation differs"):
        publish_annotation(bound, scene, pipeline_config_yaml=yaml, publication_identity=changed_publication,
                           expected_scene=archive)
    assert marker.read_bytes() == published_marker
    # Hold the assembly reference fixed to isolate the event's own implementation
    # check from the pipeline's broader source fingerprint.
    event_node = initial.nodes[EVENT_DETECTION]
    nodes = standard_definition(load_events(config), event_node.context.source,
                                code_identity=changed_code.code_identity, overrides=overrides)
    plan = ComponentRunner(nodes, initial.store).plan
    plan.references[(clip_id, "scene_assembly")] = initial.references["scene_assembly"]
    with pytest.raises(ValueError, match=f"Loaded component identity changed: {EVENT_DETECTION}"):
        plan.decide((clip_id, EVENT_DETECTION))

    scene, reexecuted = run(changed_code)
    assert reexecuted.statuses[EVENT_DETECTION] == "executed"
    assert reexecuted.references[EVENT_DETECTION] != initial.references[EVENT_DETECTION]
    assert changed_code.completed_scene_export() != archive
    assert scene.metadata["ball_events"]["provenance"]["code_commit"] == second_commit
    publish_annotation(bound, scene, pipeline_config_yaml=yaml, publication_identity=changed_publication,
                       expected_scene=changed_code.completed_scene_export())

    changed_settings = replace(config, event_detection=_changed_event_config(event_config, clip_id))
    changed = new_pipeline(changed_settings)
    with pytest.raises(ValueError, match="Stale annotation inputs/model/settings"):
        completed_annotation(bound, changed.publication_identity())
    with pytest.raises(ValueError, match=f"Loaded component identity changed: {EVENT_DETECTION}"):
        run(new_pipeline(load_events(changed_settings)))
    _, recomputed = run(changed)
    assert recomputed.statuses["scene_assembly"] == "cached"
    assert recomputed.statuses[EVENT_DETECTION] == "executed"
    assert recomputed.references[EVENT_DETECTION] != reexecuted.references[EVENT_DETECTION]
    assert changed.completed_scene_export() != changed_code.completed_scene_export()
