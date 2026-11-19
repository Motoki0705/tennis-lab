"""Run real stage adapters through a multi-clip plan and artifact store on CPU."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.submodules.models import PersonDetectionRequest, PersonDetectionResult
from src.tasks.ball_detection.model_io.contracts import (
    BallCandidateConfig,
    BallPrediction,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.batching import (
    BatchAdapter,
    BatchConfig,
    BatchJob,
    BatchMemoryLimit,
    BatchSizes,
    SourceMemory,
    WorkingMemory,
    drain_batch_jobs,
)
from src.tennis_scene.pipeline.components import (
    ball_detection,
    hmr2_features,
    person_detection,
    person_features,
)
from src.tennis_scene.pipeline.components.ball_detection import (
    BallDetectionInput,
    BallDetectionModule,
)
from src.tennis_scene.pipeline.components.hmr2_features import HMR2FeaturesModule
from src.tennis_scene.pipeline.components.person_detection import (
    PersonDetectionModule,
    PersonDetectionOutput,
)
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ComponentNode,
    ExecutionPlan,
)
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceRuntime,
    ModelSpec,
    NodeKey,
    model_identities,
)
from src.tennis_scene.pipeline.input_assembly.preprocessing import (
    PersonDetectionInputAssembler,
)
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.inference_recording import InferenceRecorder
from src.utils.video import FramePacket, VideoReadError
from tests.unit.tennis_scene.pipeline.components.test_ball_detection import _prediction
from tests.unit.tennis_scene.pipeline.components.test_hmr2_features import (
    asymmetric_selection,
)
from tests.unit.tennis_scene.pipeline.components.test_person_features import (
    inputs as feature_inputs,
)
from tests.unit.tennis_scene.pipeline.config_factories import (
    fake_encoder_spec,
    make_ball_config,
    make_people_config,
)


def _drain(adapter: BatchAdapter, model: Any, inputs: dict[NodeKey, Any], size: int) -> dict[NodeKey, Any]:
    config = BatchConfig(enabled=True, sizes=replace(BatchSizes(), **{adapter.batch_model: size}))
    budget = WorkingMemory(config.max_working_bytes)
    spec = ModelSpec(adapter.batch_model, {}, "fake", "cpu", "float32", {}, lambda: model)
    outputs: dict[NodeKey, Any] = {}

    def fail(_: NodeKey, error: Exception) -> None:
        raise error

    with InferenceRuntime({key: {adapter.batch_alias: spec} for key in inputs}) as runtime, runtime.batch(tuple(inputs)) as borrowers:
        jobs = []
        for key, item in inputs.items():
            memory = SourceMemory(budget)
            jobs.append(BatchJob(key, None, adapter, borrowers[key], adapter.open_batch(key[0], item, borrowers[key], memory), memory))
        drain_batch_jobs(jobs, config, budget, complete=outputs.__setitem__, fail=fail)
    assert budget.used == 0 and all(job.source is None for job in jobs)
    assert budget.peak <= config.max_working_bytes
    return outputs


def test_dino_batches_only_compatible_sources_and_preserves_every_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    videos = {}
    packets = {}
    for clip, frames, shape, values in (
        ("a", 3, (20, 30), ((11, 0, 13), (21, 22, 0))),
        ("b", 2, (40, 60), ((31, 0), (41, 42))),
        ("c", 1, (16, 32), ((51,), (61,))),
    ):
        height, width = shape
        videos[clip] = tuple(SourceVideo(f"cam{camera}", tmp_path / f"{clip}-{camera}.mp4",
                                       f"{clip}-{camera}", frames, 30., width, height)
                             for camera in range(2))
        for video, codes in zip(videos[clip], values, strict=True):
            packets[video.path] = [SimpleNamespace(index=i, frame=np.full((*shape, 3), code, np.uint8))
                                   for i, code in enumerate(codes)]
    monkeypatch.setattr(person_detection, "OpenCVVideoFrameReader", lambda path, **kw: iter(packets[path]))
    batches: list[list[int]] = []

    class Detector:
        def predict_batch(self, requests: Sequence[PersonDetectionRequest], *, batch_size: int) -> list[PersonDetectionResult]:
            assert len(requests) == batch_size
            codes = [int(item.frame_bgr[0, 0, 0]) for item in requests]
            batches.append(codes)
            return [PersonDetectionResult(
                np.array([[code, code + 1, code + 2, code + 3]], np.float32) if code else np.empty((0, 4), np.float32),
                np.array([.8], np.float32) if code else np.empty(0, np.float32)) for code in codes]

    class Component(PersonDetectionModule):
        def model_specs(self) -> dict[str, ModelSpec[Detector]]:
            return {"detector": ModelSpec("dino", {}, "fake", "cpu", "float32", {}, Detector)}

    component = Component(make_people_config(tmp_path))
    settings = {"models": model_identities(component.model_specs(), lambda path: {})}

    errors: list[tuple[str, Exception]] = []

    def run(batch_size: int, *, max_bytes: int = 1024**3, isolate: bool = False) -> dict:
        executions = []
        for clip_id, cameras in videos.items():
            source = ClipSource(clip_id, cameras)
            assembler = PersonDetectionInputAssembler()
            nodes = tuple(ComponentNode(f"person_detection/{video.camera_id}", component, component.io,
                       assembler, {}, AssemblyContext(source, video.camera_id), settings, "test") for video in cameras)
            executions.append(ClipExecution(clip_id, nodes,
                ClipStore(tmp_path / f"{batch_size}-{max_bytes}" / clip_id, json_value(source)), overwrite=True))
        config = BatchConfig(enabled=True, sizes=replace(BatchSizes(), dino=batch_size),
                             max_sources=6, max_working_bytes=max_bytes)
        executor = PlanExecutor(ExecutionPlan(executions), batching=config)
        try:
            recording = tmp_path / f"batch-{batch_size}-{max_bytes}.json"
            with InferenceRecorder(recording):
                executor.run(continue_on_error=isolate, on_clip_error=lambda clip, error: errors.append((clip, error)))
            if batch_size == 3 and not isolate:
                import json
                dispatch = json.loads(recording.read_text())["batch_dispatches"]["dino"]
                assert dispatch["batches"] == 5 and dispatch["items"] == 12
                assert dispatch["cross_clip_batches"] > 0
                assert dispatch["fill_fraction"] == 12 / 15
                assert dispatch["peak_active_sources"] == 6 and dispatch["peak_queued_items"] == 5
            return {key: json_value(executor.output(key)) for key in executor.plan.references}
        finally:
            assert executor.working_memory.used == 0
            assert executor.working_memory.peak <= max_bytes

    reference = run(1)
    assert all(len(batch) == 1 for batch in batches)
    batches.clear()
    assert run(3) == reference
    assert any(11 in batch and 31 in batch for batch in batches)
    assert any(11 in batch and 21 in batch for batch in batches)
    assert {len(batch) for batch in batches} == {1, 2, 3}
    assert all(not (set(batch) & {51, 61}) or set(batch) <= {51, 61} for batch in batches)
    assert reference[("a", "person_detection/cam0")]["frame_offsets"] == [0, 1, 1, 2]
    assert reference[("a", "person_detection/cam0")]["source_rows"] == [0, 1]
    assert reference[("b", "person_detection/cam0")]["boxes_xyxy"] == [[31., 32., 33., 34.]]
    with pytest.raises(BatchMemoryLimit, match="max_working_bytes"):
        run(3, max_bytes=32)
    assert [(clip, type(error)) for clip, error in errors] == [("a", BatchMemoryLimit)]
    errors.clear()

    def broken_reader(path: Path, **_: Any) -> Iterator[SimpleNamespace]:
        if path.stem == "b-0":
            yield packets[path][0]
            raise VideoReadError("missing next frame")
        yield from packets[path]

    monkeypatch.setattr(person_detection, "OpenCVVideoFrameReader", broken_reader)
    assert run(3, isolate=True) == {key: value for key, value in reference.items() if key[0] != "b"}
    assert [(clip, type(error)) for clip, error in errors] == [("b", VideoReadError)]

    def oom(*_: Any, **__: Any) -> Any:
        batches.append([])
        raise torch.OutOfMemoryError("fake CUDA OOM")

    monkeypatch.setattr(Detector, "predict_batch", oom)
    batches.clear()
    with pytest.raises(torch.OutOfMemoryError, match="fake CUDA OOM"):
        run(3, isolate=True)
    assert batches == [[]]


def test_court_reads_frame_zero_and_writes_source_camera_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tasks.court_detection.geometry.quality import CourtQuality
    from src.tasks.court_detection.inference import regions
    from src.tennis_scene.pipeline.components import court_kp
    from tests.unit.tasks.court_detection.inference.test_unified_predictor import (
        geometry_prediction,
    )
    from tests.unit.tennis_scene.pipeline.config_factories import (
        make_court_kp_config,
        materialize_model_weights,
    )

    sources = {}
    for clip, frames, size in (("a", 41, (30, 20)), ("b", 19, (60, 40))):
        video = SourceVideo("cam0", tmp_path / f"{clip}.mp4", clip, frames, 30., *size)
        sources[clip, "cam0"] = court_kp.CourtDetectionInput(video)
    videos = {inputs.video.path: inputs.video for inputs in sources.values()}
    reads: list[tuple[str, int]] = []
    forwards: list[int] = []
    prediction = geometry_prediction()

    def read(path: Path, frame: int) -> FramePacket:
        video = videos[path]
        reads.append((path.stem, frame))
        return FramePacket(frame, np.full((video.height, video.width, 3), 18, np.uint8), (video.width, video.height))

    class Court:
        adapter = SimpleNamespace(spec=SimpleNamespace(target_bundle=SimpleNamespace(
            targets={"kp": SimpleNamespace(output_channels=14)})))

        def batch_image_size(self, height: int, width: int) -> tuple[int, int]:
            return 32, 48

        def prepare_batch_image(self, rgb: np.ndarray) -> Any:
            return SimpleNamespace(images=torch.zeros(1, 3, *self.batch_image_size(*rgb.shape[:2])))

        def predict_prepared(self, images: list[Any], **kwargs: Any) -> list[Any]:
            forwards.append(len(images))
            return [prediction for _ in images]

    monkeypatch.setattr(court_kp, "read_video_frame", read)
    monkeypatch.setattr(regions, "assess_prediction_quality", lambda *a: CourtQuality(True, .25, (), {}))
    module = materialize_model_weights(court_kp.CourtKPModule(make_court_kp_config(tmp_path)))
    result = _drain(module, Court(), sources, 2)
    assert reads == [("a", 0), ("b", 0)] and forwards == [2]
    for key, item in result.items():
        video = sources[key].video
        assert item.frame_indices.tolist() == [0]
        assert item.diagnostics["source_frame_count"] == video.num_frames
        assert item.diagnostics["cameras"][0]["video_path"] == str(video.path)
        points, _ = prediction.downstream_keypoints()
        np.testing.assert_allclose(item.keypoints[0, 0], points / [video.width - 1, video.height - 1], rtol=1e-6)
        np.testing.assert_array_equal(item.visibility[0, 0], [False] + [True] * 13)


def test_pose_and_clip_keep_sparse_rows_and_invalid_crop_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = feature_inputs(tmp_path)
    boxes = a.detections.boxes_xyxy + np.array([12, 0, 12, 0], np.float32)
    # A positive-area box outside the image remains a pose row and has no CLIP embedding.
    boxes[1] = [66, 4, 78, 34]
    b = replace(a, video=replace(a.video, path=tmp_path / "b.mp4"),
                detections=replace(a.detections, boxes_xyxy=boxes))
    empty = replace(a, detections=PersonDetectionOutput(
        a.video.camera_id, np.zeros(6, np.int64), np.empty((0, 4), np.float32),
        np.empty(0, np.float32), np.empty(0, np.int64)))
    sources = {("a", "cam"): a, ("b", "cam"): b, ("empty", "cam"): empty}
    reads: Counter[Path] = Counter()

    def reader(path: Path, *, max_frames: int) -> Iterator[SimpleNamespace]:
        reads[path] += 1
        for frame in range(max_frames):
            image: np.ndarray = np.full((48, 64, 3), 30 + frame + 60 * (path == b.video.path), np.uint8)
            yield SimpleNamespace(index=frame, frame=image)

    monkeypatch.setattr(person_features, "OpenCVVideoFrameReader", reader)
    pose_batches: list[int] = []
    clip_batches: list[int] = []

    class Pose:
        def predict_crops(self, pixels: torch.Tensor, boxes: torch.Tensor) -> Any:
            pose_batches.append(len(pixels))
            return SimpleNamespace(keypoints=boxes[:, None].expand(-1, 17, -1).clone())

    class Encoder:
        name = person_features.CLIP_ENCODER
        input_size = (8, 4)

        def embed(self, pixels: torch.Tensor) -> torch.Tensor:
            clip_batches.append(len(pixels))
            embeddings = torch.zeros((len(pixels), 1280))
            embeddings[:, 0] = pixels[:, 0, 0, 0]
            embeddings[:, 1] = 1.
            return torch.nn.functional.normalize(embeddings, dim=1)

    pose_module = person_features.PersonPoseFeaturesModule(make_people_config(tmp_path))
    appearance = person_features.PersonAppearanceFeaturesModule(fake_encoder_spec(Encoder))
    apps = {key: person_features.PersonAppearanceFeaturesInput(value.video, value.detections) for key, value in sources.items()}
    poses = _drain(pose_module, Pose(), sources, 1)
    embeddings = _drain(appearance, Encoder(), apps, 1)
    pose_batches.clear()
    clip_batches.clear()
    more_poses = _drain(pose_module, Pose(), sources, 4)
    more_embeddings = _drain(appearance, Encoder(), apps, 4)
    for key in sources:
        assert json_value(more_poses[key]) == json_value(poses[key])
        assert json_value(more_embeddings[key]) == json_value(embeddings[key])
    assert pose_batches == [4, 4] and clip_batches == [4, 3]
    assert reads == {a.video.path: 4, b.video.path: 4}
    assert poses[("a", "cam")].source_rows.tolist() == [3, 8, 11, 19]
    assert poses[("a", "cam")].keypoints[0, 0, 0] == 10
    assert poses[("b", "cam")].keypoints[0, 0, 0] == 22
    assert embeddings[("b", "cam")].appearance_valid.tolist() == [True, False, True, True]
    assert not embeddings[("b", "cam")].embeddings[1].any()


def test_hmr2_does_one_read_per_camera_and_scatters_segments_per_clip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = {("clip-a", "hmr2"): asymmetric_selection(tmp_path),
               ("clip-b", "hmr2"): asymmetric_selection(tmp_path, "clip-b")}
    reads: Counter[Path] = Counter()
    forward: list[int] = []

    def reader(path: Path, *, max_frames: int) -> Iterator[SimpleNamespace]:
        reads[path] += 1
        for i in range(max_frames):
            code = 17 + i + 35 * (path.stem == "cam1") + 80 * (path.parent.name == "clip-b")
            yield SimpleNamespace(index=i, frame=np.full((480, 640, 3), code, np.uint8))

    class HMR:
        def predict_crops(self, pixels: torch.Tensor) -> Any:
            forward.append(len(pixels))
            return SimpleNamespace(features=pixels[:, 0, 128, 128, None].expand(-1, 1024))

    monkeypatch.setattr(hmr2_features, "OpenCVVideoFrameReader", reader)
    module = HMR2FeaturesModule(make_people_config(tmp_path))
    baseline = _drain(module, HMR(), sources, 1)
    forward.clear()
    reads.clear()
    batched = _drain(module, HMR(), sources, 4)
    assert forward == [4, 4, 4, 4, 2]
    assert len(reads) == 4 and set(reads.values()) == {1}
    for key, clip in batched.items():
        assert json_value(clip) == json_value(baseline[key])
        assert clip.clip_id == key[0]
        body = clip.bodies[0]
        assert (body.person_id, body.camera_id, body.video_sha256) == (9, "cam1", f"{key[0]}-cam1-sha")
        assert body.segments[1].source_frames.tolist() == [10, 14]
        red = (17 + 35 + 10 + 80 * (key[0] == "clip-b")) / 255.
        assert body.segments[1].features[0, 0] == pytest.approx((red - .485) / .229)


@pytest.mark.parametrize("aggregation", ["last_window_wins", "max_score", "nearest_window_centre_then_earlier_start"])
def test_ball_batches_only_the_window_axis_and_keeps_selected_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aggregation: str,
) -> None:
    codes = {"a0": [11, 12, 13, 14, 15, 16], "a1": [21, 22, 23, 24, 25], "b0": [31, 32, 33]}
    addresses = {"a0": ("a", "cam0"), "a1": ("a", "cam1"), "b0": ("b", "cam0")}
    sources = {addresses[key]: BallDetectionInput(
        SourceVideo(addresses[key][1], tmp_path / f"{key}.mp4", key, len(frames), 30., 6, 4))
        for key, frames in codes.items()}

    def reader(path: Path, *, max_frames: int) -> Iterator[FramePacket]:
        for index, code in enumerate(codes[path.stem][:max_frames]):
            yield FramePacket(index, np.full((4, 6, 3), code, np.uint8), (6, 4))

    seen: list[list[list[int]]] = []

    class Ball:
        configured_frames = 3

        def predict(self, pixels: torch.Tensor, *, candidate_config: BallCandidateConfig) -> BallPrediction:
            actual = (pixels[:, :, 0, 0, 0] * 255).round().to(torch.int64).tolist()
            seen.append(actual)
            maps = torch.full((len(pixels), 3, 4, 6), .01)
            for b, window in enumerate(actual):
                for t, value in enumerate(window):
                    maps[b, t, 1, value % 5] = .5 + t / 10
            return _prediction(maps, candidate_config)

    monkeypatch.setattr(ball_detection, "OpenCVVideoFrameReader", reader)
    config = replace(make_ball_config(tmp_path), num_frames=3, image_size=(4, 6), window_stride=2,
                     overlap_aggregation=aggregation)
    module = BallDetectionModule(config)
    base = _drain(module, Ball(), sources, 1)
    seen.clear()
    outputs = _drain(module, Ball(), sources, 4)
    assert json_value([outputs[key] for key in sources]) == json_value([base[key] for key in sources])
    assert seen == [[[11, 12, 13], [21, 22, 23], [31, 32, 33], [13, 14, 15]], [[23, 24, 25], [14, 15, 16]]]
    expected = {
        "last_window_wins": [0, 0, 2, 3, 3, 3],
        "max_score": [0, 0, 0, 2, 2, 3],
        "nearest_window_centre_then_earlier_start": [0, 0, 0, 2, 3, 3],
    }[aggregation]
    assert outputs[("a", "cam0")].evidence.selected_window_start.tolist() == expected
    assert outputs[("b", "cam0")].evidence.selected_time_index.tolist() == [0, 1, 2]
