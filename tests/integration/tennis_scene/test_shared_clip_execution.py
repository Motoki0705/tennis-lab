"""Both existing entries share plans, immutable exports and dataset publication."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.tennis_scene.generate_dataset.manifest import register_exported_clip
from src.tennis_scene.generate_dataset.pseudo_annotation import (
    generate_pseudo_annotations,
)
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.clip_inputs import (
    resolve_dataset_clip,
    select_dataset_clips,
)
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionModule
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesModule,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ComponentIO
from src.tennis_scene.pipeline.dataset_publication import (
    completed_annotation,
    publish_annotation,
)
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
    standard_model_components,
)
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ComponentNode,
    ExecutionPlan,
)
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path
from src.tennis_scene.pipeline.video_batch import run_video_batch
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole
from src.utils.inference_recording import InferenceRecorder
from src.utils.io import save_json_atomic
from src.utils.video.reader import OpenCVVideoFrameReader, probe_video_decoder
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
    setup_pipeline,
)


class EmptyInput:
    version = 1

    def assemble(self, context: AssemblyContext, artifacts: Any) -> dict[str, Any]:
        return {}


class FailingScene:
    def process(self, inputs: dict[str, Any]) -> SceneResult:
        raise ValueError("clip b failure")


def _register_clip(tmp_path: Path, paths: tuple[Path, ...]) -> tuple[Path, Path, tuple[Path, ...]]:
    clip_id = "video_000/clip_000"
    dataset = tmp_path / "dataset"
    clip = dataset / "videos/video_000/clips/clip_000"
    clip.mkdir(parents=True)
    # The fake predictors use real per-clip stores; only media probing and model
    # predictions are replaced by the existing deterministic fixture.
    for path in paths:
        (clip / path.name).write_bytes(path.read_bytes())
    paths = tuple(clip / path.name for path in paths)
    save_json_atomic(dict(version=2, dataset_id="shared-plan", clip_id=clip_id, video_id="video_000", clip_name="clip_000",
        fps=30., num_frames=24, width=1280, height=720, global_start_sec=0., global_end_sec=.8,
        camera_ids=["cam0", "cam1", "cam2"], video_paths=[p.name for p in paths], cameras=[],
        sync_source="clip_studio", exported_at="2026-11-16T00:00:00Z"), clip / "clip.json")
    register_exported_clip(dataset, clip / "clip.json")
    return dataset, clip, paths


def test_existing_entries_and_multi_clip_plan_publish_the_same_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    dataset, clip, paths = _register_clip(tmp_path, paths)
    clip_id = "video_000/clip_000"
    root = clip / "annotations/tennis_scene"
    expected = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=root, clip_id=clip_id)
    assert pipeline.last_runner is not None and pipeline.last_store is not None
    before = json.loads((root / "scene.json").read_text())
    spec = pipeline.last_runner.plan.clips[clip_id]
    source_b = replace(spec.nodes[0].context.source, clip_id="b")
    failing = FailingScene()
    fail_node = ComponentNode("scene_assembly", failing, ComponentIO("scene_assembly", dict, SceneResult, {}, "scene_result", 2),
                              EmptyInput(), {}, AssemblyContext(source_b), {}, "test")
    failed_store = ClipStore(tmp_path / "failed", json_value(source_b))
    plan = ExecutionPlan([spec, ClipExecution("b", [fail_node], failed_store)])
    executor = PlanExecutor(plan)
    completed: list[str] = []
    def publish_completed(ready_clip: str) -> None:
        assert ready_clip == clip_id
        assert not plan.clip_complete("b")
        executor.finalize_clip(ready_clip)
        completed.append(ready_clip)
    with pytest.raises(ValueError, match="clip b failure"):
        executor.run(on_clip_complete=publish_completed)
    assert completed == [clip_id]
    actual, archive = executor.finalize_clip(clip_id)
    assert archive.is_file()
    finalized = (root / "scene.json").read_bytes()
    executor.finalize_clip(clip_id)
    assert (root / "scene.json").read_bytes() == finalized
    assert not list((failed_store.root / "exports").glob("*/scene.npz"))
    with pytest.raises(ValueError, match="incomplete"):
        executor.finalize_clip("b")
    for field in fields(SceneResult):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b
    assert plan.statuses[("b", "scene_assembly")] == "failed"
    assert {status for (clip_key, _), status in plan.statuses.items() if clip_key == clip_id} == {"cached"}
    after = json.loads((root / "scene.json").read_text())
    assert after["artifacts"] == before["artifacts"] and after["exports"] == before["exports"]

    calls: list[int] = []
    def run(videos: Sequence[Path], cameras: Sequence[str], directory: Path) -> SceneResult:
        calls.append(1)
        return pipeline.run(videos, video_role=PathRole.DATA, camera_ids=cameras,
                            store_root=directory / "annotations/tennis_scene", clip_id=clip_id)
    # publication_identity normally hashes enabled assets; all component
    # overrides are fake here, so this integration uses their fixed identity.
    first = generate_pseudo_annotations(dataset, run, pipeline_config_yaml="device: cpu\n", publication_identity={"fixture": "fixed"})
    assert first[0].status == "generated" and first[0].annotation_path is not None
    marker = json.loads(first[0].annotation_path.read_text())
    assert marker["scene_result"] == str(archive.relative_to(root)) and marker["scene_index"] == "scene.json"
    assert marker["arrays"]["ball_3d"] == {"shape": [24, 3], "dtype": "float32"}
    assert json.loads((root / "scene.json").read_text())["artifacts"] == before["artifacts"]
    assert all(stage.calls == 1 for stage in stages.values())
    second = generate_pseudo_annotations(dataset, run, pipeline_config_yaml="device: cpu\n", publication_identity={"fixture": "fixed"})
    assert second[0].status == "skipped" and calls == [1]


@pytest.mark.parametrize("entry", ["single", "batch"])
def test_alternative_people_publication_preserves_legacy_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str,
) -> None:
    fixture, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    dataset, clip, _ = _register_clip(tmp_path, paths)
    record, = select_dataset_clips(dataset)
    bound = resolve_dataset_clip(dataset, record)
    root = clip / "annotations/tennis_scene"
    scoped = root / "people-models/rfdetr-l+rtmpose-x"
    alternate_config = runtime(tmp_path, overrides=("people_models.detector=rfdetr-l", "people_models.pose=rtmpose-x"))
    for config in (fixture.config, alternate_config):
        materialize_assets(enabled_model_assets(config).values(), tmp_path, court=config.court_kp)
    monkeypatch.setattr("src.tennis_scene.pipeline.model_resources.require_people_checkpoint", lambda *_: None)
    monkeypatch.setattr("src.submodules.models.rfdetr.person_detector.require_rf_detr_version", lambda: None)

    def detect(component: PersonDetectionModule, request: Any, models: Any) -> Any:
        value = stages[f"person_detection/{request.video.camera_id}"].result
        if component.config.detector == "rfdetr-l":
            return replace(value, confidence=value.confidence * np.float32(.75))
        return value

    def pose(component: PersonPoseFeaturesModule, request: Any, models: Any) -> Any:
        value = stages[f"person_pose_features/{request.video.camera_id}"].result
        return replace(value, model_identity=models.identity("pose"),
                       preprocessing_identity=document_digest(component.cache_identity()))

    monkeypatch.setattr(PersonDetectionModule, "process_with_models", detect)
    monkeypatch.setattr(PersonPoseFeaturesModule, "process_with_models", pose)
    overrides = {name: stage for name, stage in stages.items()
                 if name.split("/", 1)[0] not in {"person_detection", "person_pose_features"}}

    def run_single(pipeline: TennisSceneOrchestrator, identity: Mapping[str, object]) -> Path:
        def run(videos: Sequence[Path], cameras: Sequence[str], directory: Path) -> SceneResult:
            return pipeline.run(videos, video_role=PathRole.DATA, camera_ids=cameras,
                                store_root=directory / "annotations/tennis_scene", clip_id=record.clip_id)
        outcome, = generate_pseudo_annotations(
            dataset, run, pipeline_config_yaml="fixture: true\n", publication_identity=identity,
            continue_on_error=False, scene_export=pipeline.completed_scene_export,
        )
        marker = outcome.annotation_path
        assert outcome.status == "generated" and isinstance(marker, Path)
        return marker

    legacy = TennisSceneOrchestrator(fixture.config, components=overrides)
    legacy_identity = legacy.publication_identity()
    legacy_marker = run_single(legacy, legacy_identity)
    assert legacy_marker == root / "annotation.json"
    old_marker, old_index = legacy_marker.read_bytes(), (root / "scene.json").read_bytes()
    old_archive = annotation_scene_path(root, json.loads(old_marker))

    alternate = TennisSceneOrchestrator(alternate_config, components=overrides)
    alternate_identity = alternate.publication_identity()
    if entry == "batch":
        batch = run_video_batch(dataset, "video_000", alternate, pipeline_config_yaml="fixture: true\n",
                                continue_on_error=False, synchronize_cuda=False)
        assert batch.exit_code == 0
        alternate_marker = scoped / "annotation.json"
    else:
        alternate_marker = run_single(alternate, alternate_identity)
        alternate.run(bound.video_paths, video_role=PathRole.DATA, camera_ids=bound.camera_ids,
                      store_root=None, clip_id=record.clip_id)
        assert legacy.last_store is not None and alternate.last_store is not None
        cache = alternate_config.cache_directory / legacy.last_store.source_key[:20]
        assert alternate.last_store.root == cache / "people-models/rfdetr-l+rtmpose-x"
        assert alternate.completed_scene_export().is_relative_to(alternate.last_store.root)
    assert alternate_config.output_path.parent == tmp_path / "run/people-models/rfdetr-l+rtmpose-x"
    assert alternate_marker == scoped / "annotation.json"
    assert completed_annotation(bound, alternate_identity) == alternate_marker
    assert completed_annotation(bound, legacy_identity) == legacy_marker
    assert legacy_marker.read_bytes() == old_marker
    assert (root / "scene.json").read_bytes() == old_index
    assert annotation_scene_path(root, json.loads(old_marker)) == old_archive
    assert annotation_scene_path(scoped, json.loads(alternate_marker.read_bytes())) != old_archive


def test_recording_preserves_dataset_publication_identity_and_scene_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path, court=pipeline.config.court_kp)
    dataset, clip, _ = _register_clip(tmp_path, paths)
    clip_id = "video_000/clip_000"
    store = clip / "annotations/tennis_scene"

    def run(videos: Sequence[Path], cameras: Sequence[str], directory: Path) -> SceneResult:
        return pipeline.run(videos, video_role=PathRole.DATA, camera_ids=cameras,
                            store_root=directory / "annotations/tennis_scene", clip_id=clip_id)

    def publish() -> tuple[str, bytes, bytes]:
        outcome, = generate_pseudo_annotations(
            dataset, run, pipeline_config_yaml="device: cpu\n",
            publication_identity=pipeline.publication_identity(), continue_on_error=False,
            scene_export=pipeline.completed_scene_export,
        )
        assert outcome.status == "generated" and outcome.annotation_path is not None
        marker = json.loads(outcome.annotation_path.read_text())
        scene = annotation_scene_path(store, marker)
        return str(marker["publication_identity_sha256"]), scene.read_bytes(), scene.with_suffix(".metadata.json").read_bytes()

    baseline = publish()
    store.rename(clip / "annotations/plain")
    trace = tmp_path / "recording.json"
    with InferenceRecorder(trace, synchronize_cuda=False):
        recorded = publish()

    assert recorded == baseline
    assert all(stage.calls == 2 for stage in stages.values())
    data = json.loads(trace.read_text())
    assert data["status"] == "complete"
    assert any(event["stage"] == "publication" for event in data["events"])


@pytest.mark.parametrize("change", ["backend", "capture_property", "capture_options", "crop"])
def test_clip_appearance_conditions_invalidate_cached_association_and_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    _, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    dataset, clip, paths = _register_clip(tmp_path, paths)
    bound = resolve_dataset_clip(dataset, select_dataset_clips(dataset)[0])
    cfg = runtime(tmp_path, overrides=("player_association.reuse_clip_embeddings=true",))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    fixture_associate = stages["player_association"].process
    association = standard_model_components(cfg, bound.camera_ids)["player_association"]
    monkeypatch.setattr(association, "process_with_models", lambda request, models: fixture_associate(request))
    pipeline = TennisSceneOrchestrator(cfg, components={**stages, "player_association": association})
    backend, orientation = "FFMPEG", 1.

    class Capture:
        def __init__(self, path: str) -> None:
            self.first_camera = Path(path) == paths[0]

        def getBackendName(self) -> str:
            return backend if self.first_camera else "FFMPEG"

        def get(self, prop: int) -> float:
            if prop == cv2.CAP_PROP_ORIENTATION_AUTO:
                return orientation if self.first_camera else 1.
            return 0.

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(cv2, "useOptimized", lambda: True)
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    root = clip / "annotations/tennis_scene"
    clip_id = bound.record.clip_id
    baseline = pipeline.publication_identity()
    result = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=bound.camera_ids, store_root=root, clip_id=clip_id)
    marker = publish_annotation(bound, result, pipeline_config_yaml="fixture: true\n", publication_identity=baseline)
    assert completed_annotation(bound, baseline) == marker
    assert pipeline.last_runner is not None and pipeline.last_store is not None
    previous = pipeline.last_runner
    source = previous.plan.clips[clip_id].nodes[0].context.source
    store = pipeline.last_store
    original_marker = marker.read_bytes()

    if change == "backend":
        backend = "GSTREAMER"
    elif change == "capture_property":
        orientation = 0.
    elif change == "capture_options":
        monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "video_codec;h264")
    else:
        monkeypatch.setattr(cv2, "useOptimized", lambda: False)

    publication = pipeline.publication_identity()
    assert (publication != baseline) == (change == "crop")
    with pytest.raises(ValueError, match="Stale annotation"):
        completed_annotation(bound, publication)
    with pytest.raises(ValueError, match="appearance decoder/preprocessing identity"):
        publish_annotation(bound, result, pipeline_config_yaml="fixture: true\n", publication_identity=publication)
    assert marker.read_bytes() == original_marker

    nodes = standard_definition(cfg, source, code_identity=pipeline.code_identity, overrides=pipeline.components)
    # Hold the previously saved dependencies fixed. A changed upstream ref
    # must not be what invalidates the association's own cache key.
    key = (clip_id, "player_association")
    plan = ExecutionPlan([ClipExecution(clip_id, nodes, store, targets=[key[1]])])
    plan.references.update({(clip_id, name): ref for name, ref in previous.references.items() if name != key[1]})
    assert plan.decide(key).action == "execute"
    loaded = tuple(replace(node, source="load") if node.name == key[1] else node for node in nodes)
    load_plan = ExecutionPlan([ClipExecution(clip_id, loaded, store, targets=[key[1]])])
    load_plan.references.update(plan.references)
    with pytest.raises(ValueError, match="appearance decoder/preprocessing identity"):
        load_plan.decide(key)

    new = pipeline.run(paths, video_role=PathRole.DATA, camera_ids=bound.camera_ids, store_root=root, clip_id=clip_id)
    assert pipeline.last_runner is not None
    assert pipeline.last_runner.statuses["person_appearance_features/cam0"] == "executed"
    assert pipeline.last_runner.statuses["person_appearance_features/cam1"] == ("cached" if change in {"backend", "capture_property"} else "executed")
    assert pipeline.last_runner.statuses["player_association"] == "executed"
    assert publish_annotation(bound, new, pipeline_config_yaml="fixture: true\n", publication_identity=publication) == marker
    assert completed_annotation(bound, publication) == marker


def test_decoder_change_after_planning_stops_before_reading_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = "FFMPEG"
    released: list[str] = []

    class Capture:
        def isOpened(self) -> bool:
            return True

        def getBackendName(self) -> str:
            return backend

        def get(self, prop: int) -> float:
            return 0.

        def release(self) -> None:
            released.append(backend)

        def read(self) -> None:
            pytest.fail("unexpected frame read with a different decoder")

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    path = tmp_path / "cam0.mp4"
    expected = probe_video_decoder(path)
    backend = "GSTREAMER"
    with pytest.raises(ValueError, match="Video decoder changed after planning"):
        list(OpenCVVideoFrameReader(path, expected_decoder=expected))
    assert released == ["FFMPEG", "GSTREAMER"]


def test_standalone_input_change_prevents_scene_finalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, paths, stages = setup_pipeline(tmp_path, monkeypatch)
    stage = stages["ball_detection/cam0"]
    process = stage.process
    def changed(inputs: Any) -> Any:
        result = process(inputs)
        paths[0].write_bytes(b"changed during inference")
        return result
    monkeypatch.setattr(stage, "process", changed)
    root = tmp_path / "store"
    with pytest.raises(ValueError, match="inputs changed during reconstruction"):
        pipeline.run(paths, video_role=PathRole.DATA, camera_ids=("cam0", "cam1", "cam2"), store_root=root)
    assert json.loads((root / "scene.json").read_text())["exports"] == {}
    assert pipeline.last_receipt["status"] == "failed"
