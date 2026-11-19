"""Publication of completion markers over the clip store's immutable scene export."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from src.tennis_scene.generate_dataset.pseudo_annotation import (
    SceneRunner,
    generate_pseudo_annotations,
)
from src.tennis_scene.schema import SceneResult
from src.utils.io import load_json
from tests.support.tennis_scene.annotations import publish_scene_to_clip_store

CLIP_ID = "video_000/clip_000"
IDENTITY = {"checkpoint": "sha-one"}


def _body_scene(clip: Path, scene: SceneResult, mode: str = 'current'):
    import numpy as np

    from src.tennis_scene.pipeline.artifacts import json_value
    from src.tennis_scene.pipeline.body_types import BodyParameters, BodyRecoveryRequest
    from src.tennis_scene.pipeline.components.body_placement import BodyPlacementOutput
    from src.tennis_scene.pipeline.components.body_view_selection import (
        BodySelection,
        BodyViewSelectionOutput,
    )
    from src.tennis_scene.pipeline.components.gvhmr import (
        GVHMROutput,
        RecoveredBody,
        RecoveredBodySegment,
    )
    from src.tennis_scene.pipeline.components.hmr2_features import (
        HMR2BodyFeatures,
        HMR2FeatureSegment,
        HMR2FeaturesOutput,
    )
    from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
    from src.tennis_scene.pipeline.storage.clip_store import ClipStore
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    from src.tennis_scene.pipeline.storage.scene_export import export_scene
    from src.utils.checksum import dual_sha256

    video = clip / 'media/cam0.mp4'
    source = ClipSource(CLIP_ID, (SourceVideo('cam0', video, dual_sha256(video), scene.num_frames, scene.fps, scene.width, scene.height),))
    store = ClipStore(clip / 'annotations/tennis_scene', json_value(source), memory_entries=0)
    frames = np.array([0, 2], np.int64)
    request = BodyRecoveryRequest(video, frames, np.zeros((2, 17, 3), np.float32), np.ones((2, 3), np.float32),
        source.size, np.eye(3))
    selected = BodyViewSelectionOutput((BodySelection(7, 'cam0', (request,), (2,)),))
    def publish(name, value, schema, version, dependencies):
        return store.publish(name, value, ArtifactCodec(type(value)), schema=schema, version=version,
            identity={'fixture': name}, dependencies=dependencies, provenance={'origin': 'component'})
    selection = publish('body_view_selection', selected, 'body_view_selection', 1, {})
    feature_selection = publish('other_selection', selected, 'body_view_selection', 1, {}) if mode == 'wrong_selection' else selection
    features = HMR2FeaturesOutput(CLIP_ID, (HMR2BodyFeatures(7, 'cam0', source.video('cam0').sha256,
        (HMR2FeatureSegment(0, frames, np.zeros((2, 1024), np.float32)),)),))
    images = publish('hmr2_features', features, 'hmr2_features', 1, {'selection': feature_selection})
    parameters = BodyParameters(*(np.zeros((2, n), np.float32) for n in (63, 3, 10, 3)))
    recovered = GVHMROutput((RecoveredBody(7, 'cam0', (RecoveredBodySegment(frames, parameters, 2),)),))
    dependencies = {'selection': selection}
    if mode not in ('legacy', 'missing_features'):
        dependencies['features'] = images
    mesh = publish('gvhmr', recovered, 'body_parameters', 1 if mode == 'legacy' else 2, dependencies)
    placement = publish('body_placement', BodyPlacementOutput(None), 'placed_bodies', 1, {'recovered': mesh})
    ref = publish('scene_assembly', scene, 'scene_result', 2, {'placement': placement})
    archive = export_scene(store, scene, ref)
    if mode == 'corrupt_features':
        path = next((store.root / images.path).parent.glob('*.npy'))
        with path.open('ab') as stream:
            stream.write(b'corrupt')
    return store, archive


@pytest.mark.parametrize('mode,message', [('legacy', 'body_parameters v2'), ('missing_features', 'features'),
    ('wrong_selection', 'selection'), ('corrupt_features', 'checksum')])
def test_current_marker_rejects_incompatible_body_lineage(
    structured_dataset: Path, valid_scene_result: SceneResult, mode: str, message: str,
) -> None:
    def runner(_videos, _cameras, directory):
        _body_scene(directory, valid_scene_result, mode)
        return valid_scene_result
    outcome = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml='test: true',
        publication_identity={'schema': 'declared_component_pipeline_v1'})[0]
    assert outcome.status == 'failed' and message in outcome.error
    assert not (_clip(structured_dataset) / 'annotations/tennis_scene/annotation.json').exists()


def _scene_receipt(store, archive):
    from src.tennis_scene.pipeline.direct_annotation import (
        DirectAnnotationConfig,
        scene_receipt,
    )
    from tests.unit.tennis_scene.pipeline.config_factories import make_submodule_runtime

    config = DirectAnnotationConfig(store.root / 'request.json', 'request', store.root / 'bundle.json', 'bundle',
        'stored_artifact', False, 'publication')
    runtime = make_submodule_runtime()
    scene_receipt(store, config, 'request', archive, people_runtime=runtime)
    return config, runtime


@pytest.mark.parametrize('mode', ['legacy', 'missing_features'])
def test_publication_only_resume_rejects_incompatible_body_lineage(
    structured_dataset: Path, valid_scene_result: SceneResult, mode: str,
) -> None:
    from src.tennis_scene.pipeline.direct_annotation import saved_scene

    store, archive = _body_scene(_clip(structured_dataset), valid_scene_result, mode)
    config, runtime = _scene_receipt(store, archive)
    with pytest.raises(ValueError, match='body_parameters v2|features'):
        saved_scene(store, config, 'request', people_runtime=runtime)


def test_current_body_publication_can_resume_and_reuse_without_inference(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    from src.tennis_scene.pipeline.direct_annotation import saved_scene

    store, archive = _body_scene(_clip(structured_dataset), valid_scene_result)
    config, runtime = _scene_receipt(store, archive)
    recovered, _ = saved_scene(store, config, 'request', people_runtime=runtime)
    assert recovered.num_frames == valid_scene_result.num_frames
    identity = {'schema': 'declared_component_pipeline_v1'}
    calls = []
    def runner(*_):
        calls.append(1)
        return recovered
    first = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml='test: true', publication_identity=identity)[0]
    second = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml='test: true', publication_identity=identity)[0]
    assert [first.status, second.status] == ['generated', 'skipped'] and calls == [1]
    # A completed marker cannot hide later damage to its saved feature ancestry.
    images = store.references()['hmr2_features']
    path = next((store.root / images.path).parent.glob('*.npy'))
    with path.open('ab') as stream:
        stream.write(b'corrupt')
    damaged = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml='test: true', publication_identity=identity)[0]
    assert damaged.status == 'failed' and 'checksum' in str(damaged.error) and calls == [1]


def _clip(dataset: Path) -> Path:
    return dataset / "videos/video_000/clips/clip_000"


def _publishing_runner(scene: SceneResult, calls: list[int] | None = None) -> SceneRunner:
    """Stand in for the orchestrator: leave the export in the clip store, return the scene."""
    def runner(video_paths: Sequence[Path], camera_ids: Sequence[str], clip_dir: Path) -> SceneResult:
        assert [path.name for path in video_paths] == ["cam0.mp4"] and list(camera_ids) == ["cam0"]
        if calls is not None:
            calls.append(1)
        publish_scene_to_clip_store(clip_dir, CLIP_ID, scene)
        return scene
    return runner


def test_marker_points_at_the_export_and_is_reused_for_the_same_identity(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    calls: list[int] = []
    runner = _publishing_runner(valid_scene_result, calls)
    first = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)
    assert first[0].status == "generated" and first[0].annotation_path is not None
    marker = load_json(first[0].annotation_path)
    assert marker["clip_id"] == CLIP_ID and marker["scene_index"] == "scene.json"
    assert marker["scene_result"].startswith("exports/") and marker["arrays"]["ball_3d"]["shape"] == [3, 3]
    second = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)
    assert second[0].status == "skipped" and len(calls) == 1


def test_config_is_content_addressed_and_the_export_stays_immutable(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    outcome = generate_pseudo_annotations(structured_dataset, _publishing_runner(valid_scene_result),
        pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)[0]
    assert outcome.annotation_path is not None
    marker = load_json(outcome.annotation_path)
    annotations = outcome.annotation_path.parent
    export = (annotations / marker["scene_result"]).parent
    assert sorted(path.name for path in export.iterdir()) == ["scene.metadata.json", "scene.npz"]
    config = annotations / marker["pipeline_config"]
    assert config.parent.name == "configs" and config.read_text() == "device: cpu\n"


def test_completed_annotation_cannot_hide_changed_weights(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    calls: list[int] = []
    runner = _publishing_runner(valid_scene_result, calls)
    assert generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity=IDENTITY)[0].status == "generated"
    changed = generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="test: true", publication_identity={"checkpoint": "sha-two"})
    assert changed[0].status == "failed" and "Stale annotation" in str(changed[0].error)
    assert len(calls) == 1


def test_historical_marker_without_identity_is_never_silently_reused(
    structured_dataset: Path, valid_scene_result: SceneResult
) -> None:
    from src.utils.io import save_json_atomic
    save_json_atomic({"version": 1, "scene_result": "scene.npz"}, _clip(structured_dataset) / "annotations/tennis_scene/annotation.json")
    outcome = generate_pseudo_annotations(structured_dataset, _publishing_runner(valid_scene_result),
        pipeline_config_yaml="x: 1", publication_identity=IDENTITY)[0]
    assert outcome.status == "failed" and "use overwrite=true" in str(outcome.error)


@pytest.mark.parametrize("change,message", [
    (lambda s: setattr(s, "fps", 25.0), "fps 25.0 != 30.0"),
    (lambda s: s.metadata.update(pipeline_contract="legacy_stages"), "declared_components_v1"),
    (lambda s: setattr(s, "ball_3d", None), "Invalid v2 ball_3d"),
])
def test_contract_violation_records_failure_without_completion_marker(
    structured_dataset: Path, valid_scene_result: SceneResult, change, message: str
) -> None:
    change(valid_scene_result)
    outcomes = generate_pseudo_annotations(structured_dataset, lambda *_: valid_scene_result,
        pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)
    assert outcomes[0].status == "failed" and message in str(outcomes[0].error)
    annotations = _clip(structured_dataset) / "annotations"
    assert not (annotations / "tennis_scene/annotation.json").exists()
    assert message in load_json(annotations / "tennis_scene.failure.json")["error"]


def test_integrity_failure_propagates_without_completion_marker(structured_dataset: Path) -> None:
    from src.utils.checksum import FileIntegrityError

    error = FileIntegrityError("providers disagree", details={"path": "checkpoint"})

    def runner(_videos: Sequence[Path], _cameras: Sequence[str], _clip: Path) -> SceneResult:
        raise error

    with pytest.raises(FileIntegrityError) as caught:
        generate_pseudo_annotations(structured_dataset, runner, pipeline_config_yaml="device: cpu\n",
            publication_identity=IDENTITY, continue_on_error=False)
    assert caught.value is error
    annotations = _clip(structured_dataset) / "annotations"
    assert not (annotations / "tennis_scene/annotation.json").exists()
    assert "FileIntegrityError" in load_json(annotations / "tennis_scene.failure.json")["error"]


def test_publication_preserves_the_store_and_slcs_follows_the_index(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    from src.tasks.slcs.data.annotation import load_slcs_annotation
    from src.tennis_scene.generate_dataset.manifest import ClipManifest
    clip = _clip(structured_dataset)
    export = publish_scene_to_clip_store(clip, CLIP_ID, valid_scene_result)
    index_before = (clip / "annotations/tennis_scene/scene.json").read_bytes()
    outcomes = generate_pseudo_annotations(structured_dataset, lambda *_: valid_scene_result,
        pipeline_config_yaml="device: cpu", publication_identity=IDENTITY)
    assert outcomes[0].status == "generated" and outcomes[0].annotation_path is not None
    assert (clip / "annotations/tennis_scene/scene.json").read_bytes() == index_before
    marker = load_json(outcomes[0].annotation_path)
    assert marker["scene_result"] == str(export.relative_to(clip / "annotations/tennis_scene"))
    loaded = load_slcs_annotation(ClipManifest.load(clip))
    assert loaded.player_position.shape == valid_scene_result.player_position.shape


@pytest.mark.parametrize("changed", ["clip.json", "media/cam0.mp4"])
def test_input_change_during_run_cannot_publish_completion_marker(
    structured_dataset: Path, valid_scene_result: SceneResult, changed: str,
) -> None:
    def run(videos: Sequence[Path], cameras: Sequence[str], directory: Path) -> SceneResult:
        publish_scene_to_clip_store(directory, CLIP_ID, valid_scene_result)
        path = directory / changed
        path.write_bytes(path.read_bytes() + b" ")
        return valid_scene_result
    result = generate_pseudo_annotations(structured_dataset, run, pipeline_config_yaml="device: cpu\n", publication_identity=IDENTITY)[0]
    assert result.status == "failed" and "inputs changed during reconstruction" in str(result.error)
    assert not (_clip(structured_dataset) / "annotations/tennis_scene/annotation.json").exists()
    assert (_clip(structured_dataset) / "annotations/tennis_scene.failure.json").is_file()


def test_failure_and_overwrite_markers_are_clip_local(
    structured_dataset: Path, valid_scene_result: SceneResult,
) -> None:
    import shutil

    from src.tennis_scene.generate_dataset.manifest import (
        ClipManifest,
        register_exported_clip,
    )
    from src.utils.io import save_json_atomic

    other = _clip(structured_dataset).with_name("clip_001")
    shutil.copytree(_clip(structured_dataset), other)
    manifest = load_json(other / "clip.json")
    manifest.update(clip_name="clip_001", clip_id="video_000/clip_001")
    save_json_atomic(manifest, other / "clip.json")
    register_exported_clip(structured_dataset, other / "clip.json")
    fail = True
    def run(videos: Sequence[Path], cameras: Sequence[str], directory: Path) -> SceneResult:
        clip_id = ClipManifest.load(directory).clip_id
        if fail and clip_id == CLIP_ID:
            raise ValueError("first clip failure")
        publish_scene_to_clip_store(directory, clip_id, valid_scene_result)
        return valid_scene_result
    first = generate_pseudo_annotations(structured_dataset, run, pipeline_config_yaml="test: true", publication_identity=IDENTITY)
    assert [(r.clip_id, r.status) for r in first] == [(CLIP_ID, "failed"), ("video_000/clip_001", "generated")]
    failed_marker = _clip(structured_dataset) / "annotations/tennis_scene.failure.json"
    assert failed_marker.is_file()
    assert not (_clip(structured_dataset) / "annotations/tennis_scene/annotation.json").exists()
    assert not (other / "annotations/tennis_scene.failure.json").exists()
    fail = False
    resumed = generate_pseudo_annotations(structured_dataset, run, pipeline_config_yaml="test: true", publication_identity=IDENTITY,
                                         clip_ids=[CLIP_ID], overwrite=True)
    assert resumed[0].status == "generated" and not failed_marker.exists()
    assert (other / "annotations/tennis_scene/annotation.json").is_file()


@pytest.mark.parametrize("field,value", [("num_frames", 4), ("fps", 29.), ("width", 65)])
def test_manifest_timeline_mismatch_fails_before_runner(
    structured_dataset: Path, field: str, value: object,
) -> None:
    from src.utils.io import save_json_atomic

    path = _clip(structured_dataset) / "clip.json"
    manifest = load_json(path)
    manifest[field] = value
    save_json_atomic(manifest, path)
    calls: list[int] = []
    def forbidden(*args: object) -> SceneResult:
        calls.append(1)
        raise AssertionError("invalid inputs reached reconstruction")
    outcome = generate_pseudo_annotations(structured_dataset, forbidden, pipeline_config_yaml="test: true", publication_identity=IDENTITY)[0]
    assert outcome.status == "failed" and "timeline disagrees" in str(outcome.error)
    assert not calls
