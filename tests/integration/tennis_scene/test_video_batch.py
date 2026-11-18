"""Video entry through real plans, runtime, stores and per-clip publication."""

from __future__ import annotations

import json
import multiprocessing
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.archive import load_scene_result
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path
from src.tennis_scene.pipeline.video_batch import run_video_batch
from src.tennis_scene.pipeline.video_court import VideoCourtConfig
from src.tennis_scene.schema import SceneResult
from src.utils.configuration import PathRole
from src.utils.inference_recording import model_event, span
from src.utils.io import save_json_atomic
from src.utils.video import VideoReadError
from tests.integration.tennis_scene.test_video_court import (
    CLIPS as SHARED_CLIPS,
)
from tests.integration.tennis_scene.test_video_court import shared_fixture
from tests.support.tennis_scene.video_dataset import add_clip
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    FixedStage,
    setup_pipeline,
)


class SharedStage(InferenceComponent[Any, Any]):
    def __init__(self, stage: FixedStage, spec: ModelSpec[Any]) -> None:
        self.stage, self.spec, self.io = stage, spec, stage.io
        self.failure: BaseException | None = None
        self.before: Callable[[Any], None] = lambda _: None

    def model_specs(self) -> Mapping[str, ModelSpec[Any]]:
        return {"shared": self.spec}

    def process_with_models(self, inputs: Any, models: ModelBorrower) -> Any:
        models.borrow("shared")
        self.before(inputs)
        # Fake inference retains clip/camera scope and records a partial tail.
        for size in (16, 8):
            with span("inference", model=self.spec.kind, batch_shape=[size, 3, 4, 6]):
                if self.failure is not None:
                    raise self.failure
        return self.stage.process(inputs)


def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int = 2):
    pipeline, _, stages = setup_pipeline(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "publication_identity", lambda: {"fixture": "video-batch-v1"})
    dataset = tmp_path / "dataset"
    clips = [add_clip(dataset, f"video_000/clip_{i:03d}") for i in range(count)]
    events = []

    def factory(name: str) -> object:
        with model_event("load", f"lower_{name}", weights={}, device="cpu", dtype="float32", instance=name):
            events.append(("load", name))
        return object()

    def release(name: str, _: object) -> None:
        events.append(("unload", name))

    for stage_name, kind in (("ball_detection", "fake_ball"), ("pose_estimation", "fake_pose")):
        spec = ModelSpec(kind, {}, kind, "cpu", "float32", {}, partial(factory, kind), release=partial(release, kind))
        for camera in range(3):
            key = f"{stage_name}/cam{camera}"
            stages[key] = SharedStage(stages[key], spec)
    return pipeline, dataset, clips, stages, events


def execute(pipeline, dataset: Path, **options):
    return run_video_batch(dataset, "video_000", pipeline, pipeline_config_yaml="fixture: video-batch-v1\n", **options)


def scene(clip: Path) -> SceneResult:
    directory = clip / "annotations/tennis_scene"
    return load_scene_result(annotation_scene_path(directory, json.loads((directory / "annotation.json").read_text())))


def shared_batch_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from src.tennis_scene.pipeline.definition import enabled_model_assets
    from tests.unit.tennis_scene.pipeline.test_auto_pipeline import materialize_assets

    pipeline, paths, _, _, _, artifact = shared_fixture(tmp_path, monkeypatch)
    dataset = tmp_path / "dataset"
    clips = [add_clip(dataset, clip_id) for clip_id in SHARED_CLIPS]
    for clip in clips:
        for path in paths:
            (clip / path.name).write_bytes(path.read_bytes())
    for path in (dataset / "dataset.json", *(clip / "clip.json" for clip in clips)):
        document = json.loads(path.read_text())
        document["dataset_id"] = "fixture"
        save_json_atomic(document, path)
    materialize_assets(enabled_model_assets(pipeline.config).values(), tmp_path)
    return pipeline, dataset, clips, artifact


@pytest.mark.parametrize("action", ["build", "load"])
def test_shared_court_is_acquired_once_before_the_plan_and_matches_single_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    import src.tennis_scene.pipeline.video_court as court_api
    from src.utils.checksum import dual_sha256

    pipeline, dataset, clips, artifact = shared_batch_fixture(tmp_path, monkeypatch)
    expected = pipeline.run(tuple(clips[0] / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
        camera_ids=("cam0", "cam1", "cam2"), store_root=tmp_path / "single", clip_id=SHARED_CLIPS[0])
    request = tmp_path / "build-request.json"
    request.write_text('{"fixture": "frozen build boundary"}')
    if action == "build":
        pipeline.config = replace(pipeline.config, video_court=VideoCourtConfig(
            "build", artifact.root, "fixture", "video_000", request, dual_sha256(request), None, False))
    pipeline.video_court, pipeline._video_court_key = None, None
    prepare = court_api.prepare_video_court
    acquired = []
    stages = pipeline.components
    assert stages is not None
    for stage in stages.values():
        stage.calls = 0

    def acquire(config, resolver, *, code_identity):
        acquired.append(config.action)
        assert all(stage.calls == 0 for stage in stages.values())
        assert not any((clip / "annotations/tennis_scene/scene.json").exists() for clip in clips)
        # Numerical build/white-stripe fitting is covered by the shared-court
        # tests. Substitute that boundary while executing the actual load path.
        return artifact if action == "build" else prepare(config, resolver, code_identity=code_identity)

    monkeypatch.setattr(court_api, "prepare_video_court", acquire)
    result = execute(pipeline, dataset)
    assert result.exit_code == 0 and acquired == [action]
    assert tuple(result.plan.clips) == SHARED_CLIPS
    assert all(stage.calls == 2 for stage in stages.values())
    np.testing.assert_array_equal(scene(clips[0]).court_kp, expected.court_kp)
    np.testing.assert_array_equal(scene(clips[0]).ball_3d, expected.ball_3d)
    progress = json.loads(result.progress_path.read_text())
    assert progress["video_court"]["artifact_digest"] == artifact.digest
    assert progress["publication_identity"]["video_court_cameras"] == artifact.manifest["cameras"]
    assert not progress["models"]


@pytest.mark.parametrize("scope", [{"dataset_id": "foreign"}, {"video_id": "video_001"}])
def test_shared_court_scope_is_rejected_before_acquisition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: dict) -> None:
    import src.tennis_scene.pipeline.video_court as court_api

    pipeline, dataset, clips, _ = shared_batch_fixture(tmp_path, monkeypatch)
    pipeline.config = replace(pipeline.config, video_court=replace(pipeline.config.video_court, **scope))
    monkeypatch.setattr(court_api, "prepare_video_court", lambda *a, **kw: pytest.fail("foreign scope must not load/build"))
    with pytest.raises(ValueError, match="another dataset/video"):
        execute(pipeline, dataset)
    assert not any((clip / "annotations/tennis_scene/scene.json").exists() for clip in clips)


@pytest.mark.parametrize("count", [1, 3])
def test_one_plan_reuses_each_model_once_and_records_real_batch_sizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    pipeline, dataset, clips, stages, events = harness(tmp_path, monkeypatch, count)
    monkeypatch.setattr(pipeline, "run", lambda *a, **kw: pytest.fail("batch must build one shared plan"))
    result = execute(pipeline, dataset)
    assert result.exit_code == 0 and [o.status for o in result.outcomes] == ["generated"] * count
    assert set(result.plan.clips) == {f"video_000/clip_{i:03d}" for i in range(count)}
    assert set(result.plan.statuses.values()) == {"executed"}
    assert all(clip.store.memory_entries == 0 for clip in result.plan.clips.values())
    assert all((stage.stage if isinstance(stage, SharedStage) else stage).calls == count for stage in stages.values())
    assert Counter(events) == {("load", "fake_ball"): 1, ("unload", "fake_ball"): 1,
                               ("load", "fake_pose"): 1, ("unload", "fake_pose"): 1}
    progress = json.loads(result.progress_path.read_text())
    assert progress["status"] == "complete" and progress["active_node"] is None
    assert Counter(e["attributes"]["model"] for e in progress["models"] if e["stage"] == "model_load") == {"fake_ball": 1, "fake_pose": 1}
    assert progress["batch_shapes"]["fake_ball"] == {
        "[16, 3, 4, 6]": {"succeeded": 3 * count, "failed": 0, "items": 16},
        "[8, 3, 4, 6]": {"succeeded": 3 * count, "failed": 0, "items": 8}}
    assert all(scene(clip).ball_3d_valid.all() for clip in clips)
    events.clear()
    second = execute(pipeline, dataset)
    assert [o.status for o in second.outcomes] == ["skipped"] * count and not second.plan.nodes and not events


def test_batch_matches_single_entry_and_reuses_mixed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, dataset, clips, _, events = harness(tmp_path, monkeypatch)
    expected = []
    for clip in clips:
        expected.append(pipeline.run(tuple(clip / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
            camera_ids=("cam0", "cam1", "cam2"), store_root=clip / "annotations/tennis_scene",
            clip_id=f"video_000/{clip.name}"))
    # Completed nodes exist without dataset completion markers. Keep a non-prefix
    # cache mixture so resumption must inspect each node through the real plan.
    index_path = clips[1] / "annotations/tennis_scene/scene.json"
    index = json.loads(index_path.read_text())
    index["artifacts"].pop("ball_detection/cam1")
    index["exports"] = {}
    index_path.write_text(json.dumps(index))
    events.clear()
    result = execute(pipeline, dataset)
    assert result.exit_code == 0
    assert result.plan.statuses[("video_000/clip_000", "ball_detection/cam1")] == "cached"
    assert result.plan.statuses[("video_000/clip_001", "ball_detection/cam1")] == "executed"
    assert events == [("load", "fake_ball"), ("unload", "fake_ball")]
    for clip, original in zip(clips, expected, strict=True):
        actual = scene(clip)
        for field in fields(SceneResult):
            a, b = getattr(actual, field.name), getattr(original, field.name)
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                assert a == b


def test_batch_keeps_the_single_entry_data_path_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    pipeline, dataset, clips, _, _ = harness(data, monkeypatch, 1)
    foreign = tmp_path / "outside.mp4"
    foreign.write_bytes(b"foreign media")
    media = clips[0] / "cam1.mp4"
    media.unlink()
    media.symlink_to(foreign)
    with pytest.raises(ValueError, match="outside|root"):
        execute(pipeline, dataset)
    assert not list(dataset.rglob("annotations"))


def test_clip_too_long_isolated_and_other_clip_publishes_immediately(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.tennis_scene.pipeline.video_batch as batch
    from src.tennis_scene.pipeline.components.court_side import CourtSideModule

    pipeline, dataset, clips, _, _ = harness(tmp_path, monkeypatch, 3)
    process = CourtSideModule.process

    def fail_one(module, inputs):
        if inputs.source.clip_id.endswith("clip_001"):
            raise ReconstructionUnavailable("clip_too_long", "2108 frames exceeds 1024")
        return process(module, inputs)

    publish = batch.publish_annotation
    published = []
    def publish_one(inputs, *args, **kwargs):
        if inputs.record.clip_id.endswith("clip_000"):
            assert not (clips[2] / "annotations/tennis_scene/annotation.json").exists()
        published.append(inputs.record.clip_id)
        return publish(inputs, *args, **kwargs)

    monkeypatch.setattr(CourtSideModule, "process", fail_one)
    monkeypatch.setattr(batch, "publish_annotation", publish_one)
    result = execute(pipeline, dataset)
    assert result.exit_code == 1 and [o.status for o in result.outcomes] == ["generated", "failed", "generated"]
    assert result.plan.statuses[("video_000/clip_001", "court_side")] == "failed"
    assert result.plan.statuses[("video_000/clip_001", "scene_assembly")] == "blocked"
    assert not (clips[1] / "annotations/tennis_scene/annotation.json").exists()
    assert (clips[1] / "annotations/tennis_scene.failure.json").is_file()
    assert published == ["video_000/clip_000", "video_000/clip_002"]
    assert json.loads(result.progress_path.read_text())["clips"]["video_000/clip_001"]["reason"] == "clip_too_long"


@pytest.mark.parametrize("failure_type", [ValueError, VideoReadError])
def test_clip_failure_keeps_the_resident_model_for_other_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             failure_type: type[Exception]) -> None:
    pipeline, dataset, clips, stages, events = harness(tmp_path, monkeypatch)
    stage = stages["ball_detection/cam0"]
    def fail(inputs):
        if inputs.video.path.parent == clips[0]:
            raise failure_type("malformed clip observation")
    stage.before = fail
    result = execute(pipeline, dataset)
    assert [o.status for o in result.outcomes] == ["failed", "generated"]
    assert events.count(("load", "fake_ball")) == events.count(("unload", "fake_ball")) == 1
    assert not (clips[0] / "annotations/tennis_scene/annotation.json").exists()


@pytest.mark.parametrize("failure,continue_on_error", [
    (KeyboardInterrupt("stop"), True), (RuntimeError("CUDA device-side assert"), True), (ValueError("bad clip"), False)])
def test_interruption_and_device_failure_stop_without_partial_markers_then_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                 failure: BaseException, continue_on_error: bool) -> None:
    pipeline, dataset, clips, stages, events = harness(tmp_path, monkeypatch)
    stages["ball_detection/cam1"].failure = failure
    with pytest.raises(type(failure), match=str(failure)):
        execute(pipeline, dataset, continue_on_error=continue_on_error)
    assert not list(dataset.rglob("annotation.json"))
    assert events.count(("load", "fake_ball")) == events.count(("unload", "fake_ball")) == 1
    receipt = json.loads(next(dataset.rglob("progress.json")).read_text())
    assert receipt["status"] == ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
    assert receipt["nodes"]["video_000/clip_000"]["ball_detection/cam1"]["status"] == "failed"
    assert receipt["nodes"]["video_000/clip_001"]["scene_assembly"]["status"] == "blocked"
    stages["ball_detection/cam1"].failure = None
    events.clear()
    resumed = execute(pipeline, dataset)
    assert resumed.exit_code == 0
    assert resumed.plan.statuses[("video_000/clip_000", "ball_detection/cam0")] == "cached"
    assert resumed.plan.statuses[("video_000/clip_000", "ball_detection/cam1")] == "executed"
    assert all(scene(clip).schema_version == 2 for clip in clips)


def test_shared_model_load_failure_is_fatal_even_when_it_is_a_value_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, dataset, _, stages, _ = harness(tmp_path, monkeypatch)
    def broken():
        raise ValueError("shared weights invalid")
    for camera in range(3):
        old = stages[f"ball_detection/cam{camera}"].spec
        stages[f"ball_detection/cam{camera}"].spec = replace(old, factory=broken)
    with pytest.raises(ValueError, match="shared weights invalid"):
        execute(pipeline, dataset)
    assert not list(dataset.rglob("annotation.json"))
    receipt = json.loads(next(dataset.rglob("progress.json")).read_text())
    assert [c["status"] for c in receipt["clips"].values()] == ["failed", "blocked"]


def test_corrupt_cache_and_completed_export_cannot_be_reported_as_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, dataset, clips, stages, events = harness(tmp_path, monkeypatch)
    stages["ball_detection/cam1"].failure = KeyboardInterrupt("stop")
    with pytest.raises(KeyboardInterrupt):
        execute(pipeline, dataset)
    stage_dir = clips[0] / "annotations/tennis_scene"
    index = json.loads((stage_dir / "scene.json").read_text())
    reference = index["artifacts"]["ball_detection/cam0"]
    (stage_dir / reference["path"]).parent.joinpath("array_0000.npy").write_bytes(b"broken")
    stages["ball_detection/cam1"].failure = None
    events.clear()
    result = execute(pipeline, dataset)
    assert [o.status for o in result.outcomes] == ["failed", "generated"]
    assert "checksum" in result.outcomes[0].error
    assert not (stage_dir / "annotation.json").exists()
    good = clips[1] / "annotations/tennis_scene"
    marker = json.loads((good / "annotation.json").read_text())
    (good / marker["scene_result"]).write_bytes(b"broken export")
    again = execute(pipeline, dataset, clip_ids=("video_000/clip_001",))
    assert again.exit_code == 1 and again.outcomes[0].status == "failed"


@pytest.mark.parametrize("entry", ["batch", "single"])
def test_single_entry_concurrent_export_cannot_mix_batch_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str) -> None:
    import src.tennis_scene.generate_dataset.pseudo_annotation as single
    import src.tennis_scene.pipeline.video_batch as batch
    from src.tennis_scene.pipeline.storage.scene_export import validate_scene_export
    from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path

    pipeline, dataset, clips, _, _ = harness(tmp_path, monkeypatch, 1)
    clip = clips[0]
    context = multiprocessing.get_context("fork")
    start, done = context.Event(), context.Event()

    def single_writer():
        assert start.wait(10)
        pipeline.config = replace(pipeline.config, cache_overwrite=True)
        # Same numeric outputs, distinct component lineage. Array equality
        # alone cannot authorize a marker for the original writer's settings.
        pipeline.code_identity = "another-writer"
        pipeline.run(tuple(clip / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
            camera_ids=("cam0", "cam1", "cam2"), store_root=clip / "annotations/tennis_scene", clip_id="video_000/clip_000")
        done.set()

    writer = context.Process(target=single_writer)
    writer.start()
    module = batch if entry == "batch" else single
    publish = module.publish_annotation

    def concurrent(*args, **kwargs):
        start.set()
        assert done.wait(10)
        current = indexed_scene_path(clip / "annotations/tennis_scene/scene.json")
        assert kwargs["expected_scene"] != current
        validate_scene_export(args[1], current)
        return publish(*args, **kwargs)

    monkeypatch.setattr(module, "publish_annotation", concurrent)
    try:
        if entry == "batch":
            result = execute(pipeline, dataset)
            assert result.exit_code == 1
            outcomes = result.outcomes
        else:
            def reconstruct(paths, cameras, directory):
                return pipeline.run(paths, video_role=PathRole.DATA, camera_ids=cameras,
                    store_root=directory / "annotations/tennis_scene", clip_id="video_000/clip_000")
            outcomes = single.generate_pseudo_annotations(dataset, reconstruct, pipeline_config_yaml="fixture: video-batch-v1\n",
                publication_identity=pipeline.publication_identity(), scene_export=pipeline.completed_scene_export)
        assert outcomes[0].status == "failed" and "superseded" in outcomes[0].error
        assert not list(dataset.rglob("annotation.json"))
    finally:
        start.set()
        writer.join(10)
        if writer.is_alive():
            writer.terminate()
            writer.join(10)
    assert writer.exitcode == 0


@pytest.mark.parametrize("action", ["build", "load"])
def test_shared_side_is_acquired_once_before_plan_and_publishes_each_clip(tmp_path, monkeypatch, action):
    import src.tennis_scene.pipeline.video_court_side as side_api
    from src.tennis_scene.pipeline.artifacts import json_value
    from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source
    from src.tennis_scene.pipeline.definition import standard_definition
    from src.tennis_scene.pipeline.runner import ComponentRunner
    from src.tennis_scene.pipeline.storage.clip_store import ClipStore
    from src.tennis_scene.pipeline.video_artifact import VideoArtifactConfig
    from src.utils.checksum import dual_sha256

    pipeline, dataset, clips, court = shared_batch_fixture(tmp_path, monkeypatch)
    pipeline.config = replace(pipeline.config, court_side=replace(pipeline.config.court_side, min_frames=30))
    pins = {}
    for i, clip in enumerate(clips):
        source = prepare_clip_source(tuple(clip / f"cam{c}.mp4" for c in range(3)),
                                     ("cam0", "cam1", "cam2"), clip_id=SHARED_CLIPS[i]).source
        store = ClipStore(tmp_path / f"prefix-{i}", json_value(source))
        nodes = standard_definition(pipeline.config, source, code_identity=pipeline.code_identity,
                                    overrides=pipeline.components, video_court=court)
        ComponentRunner(nodes, store).run(["court_calibration", *(f"ball_points/cam{c}" for c in range(3))])
        pins[source.clip_id] = side_api.PinnedSideInputs.from_store(store.root)
    artifact = side_api.build_video_court_side(pins, court=court, directory=tmp_path / "side",
        thresholds=pipeline.config.court_side, max_frames=pipeline.config.sampling_max_frames,
        code_identity=pipeline.code_identity)
    request = tmp_path / "side-inputs.json"
    save_json_atomic({"schema": "video_court_side_inputs_v1", "evidence_clip_ids": None,
        "clips": {clip_id: {"role": "artifact", "directory": pin.directory.name, "inputs_digest": pin.digest}
                  for clip_id, pin in pins.items()}}, request)
    pipeline.config = replace(pipeline.config, video_court_side=VideoArtifactConfig(
        action=action, directory=tmp_path / "side", dataset_id="fixture", video_id="video_000", rebuild=False,
        request=request if action == "build" else None,
        request_sha256=dual_sha256(request) if action == "build" else None,
        artifact_digest=artifact.digest if action == "load" else None))
    expected = pipeline.run(tuple(clips[0] / f"cam{i}.mp4" for i in range(3)), video_role=PathRole.DATA,
        camera_ids=("cam0", "cam1", "cam2"), store_root=tmp_path / "single-side", clip_id=SHARED_CLIPS[0])
    pipeline.video_court_side, pipeline._video_court_side_key = None, None
    stages = pipeline.components
    for stage in stages.values():
        stage.calls = 0
    acquired = []
    prepare = side_api.prepare_video_court_side

    def acquire(*args, **kwargs):
        assert all(stage.calls == 0 for stage in stages.values())
        assert not any((clip / "annotations/tennis_scene/scene.json").exists() for clip in clips)
        acquired.append(args[0].action)
        return prepare(*args, **kwargs)

    monkeypatch.setattr(side_api, "prepare_video_court_side", acquire)
    result = execute(pipeline, dataset)
    assert result.exit_code == 0 and acquired == [action]
    assert tuple(result.plan.clips) == SHARED_CLIPS
    assert [outcome.status for outcome in result.outcomes] == ["generated", "generated"]
    np.testing.assert_array_equal(scene(clips[0]).court_kp, expected.court_kp)
    np.testing.assert_array_equal(scene(clips[0]).ball_3d, expected.ball_3d)
    assert all(scene(clip).metadata["court_reference"]["view_half_turns"] == [False, False, True] for clip in clips)
    progress = json.loads(result.progress_path.read_text())
    side = progress["video_court_side"]
    assert side["artifact_digest"] == artifact.digest
    assert progress["publication_identity"]["video_court_side_digest"] == artifact.digest
    assert all(report["status"] == "warning" and report["reason"] == "court_side_insufficient_frames"
               for report in side["clips"].values())


def test_shared_side_scope_is_rejected_before_acquisition(tmp_path, monkeypatch):
    import src.tennis_scene.pipeline.video_court_side as side_api
    from src.tennis_scene.pipeline.video_artifact import VideoArtifactConfig

    pipeline, dataset, clips, _ = shared_batch_fixture(tmp_path, monkeypatch)
    pipeline.config = replace(pipeline.config, video_court_side=VideoArtifactConfig(
        action="load", directory=tmp_path / "side", dataset_id="foreign", video_id="video_000",
        rebuild=False, request=None, request_sha256=None, artifact_digest="a" * 64))
    monkeypatch.setattr(side_api, "prepare_video_court_side", lambda *a, **kw: pytest.fail("foreign scope must stop"))
    with pytest.raises(ValueError, match="another dataset/video"):
        execute(pipeline, dataset)
    assert not any((clip / "annotations/tennis_scene/annotation.json").exists() for clip in clips)
