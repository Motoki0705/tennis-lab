"""Source-row alignment and bounded, separate model passes on CPU fakes."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.components import person_features as module
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import (
    fake_encoder_spec,
    make_people_config,
    materialize_model_weights,
)


def inputs(tmp_path: Path) -> module.PersonFeaturesInput:
    # Empty middle and trailing frames; kept detector rows are deliberately sparse.
    video = SourceVideo('cam', tmp_path / 'source.mp4', 'media', 5, 30., 64, 48)
    detections = PersonDetectionOutput('cam', np.array([0, 2, 2, 3, 4, 4], np.int64),
        np.array([[2, 3, 18, 35], [31, 5, 55, 27], [4, 7, 26, 43], [32, 9, 60, 45]], np.float32),
        np.array([.8, .9, .6, .7], np.float32), np.array([3, 8, 11, 19], np.int64))
    return module.PersonFeaturesInput(video, detections)


class Encoder:
    name = 'clipreid_vitb16_market1501'
    input_size = (8, 4)
    calls: list[int] = []

    def embed(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append(len(pixels))
        result = torch.zeros((len(pixels), 1280))
        # The first channel identifies the source frame, the second its box.
        result[:, :2] = pixels[:, :2, 0, 0]
        return torch.nn.functional.normalize(result, dim=1)


def reader(*args: Any, **kwargs: Any) -> Any:
    for frame in range(5):
        image: np.ndarray = np.empty((48, 64, 3), np.uint8)
        image[..., 0] = 7
        image[..., 1] = np.arange(64)[None] + 20
        image[..., 2] = 50 + frame * 20
        yield SimpleNamespace(index=frame, frame=image)


def test_pose_and_appearance_keep_rows_empty_frames_and_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = inputs(tmp_path)
    seen: list[int] = []
    class Pose:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs['precision'] == 'fp32'
            assert kwargs['flip_test'] == make_people_config(tmp_path).runtime.vitpose.flip_test
        def load(self) -> None: pass
        def unload(self) -> None: pass
        def predict(self, request: Any) -> Any:
            seen.extend(f for f, _ in request.frames_bgr)
            values = torch.zeros((len(request.bbx_xys), 17, 3))
            values[..., :2] = request.bbx_xys[:, None, :2]
            values[..., 2] = 1.4  # raw peak, never clipped to a probability
            return SimpleNamespace(keypoints=values)
    monkeypatch.setattr(module, 'ViTPosePose2D', Pose)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', reader)
    pose = materialize_model_weights(module.PersonPoseFeaturesModule(make_people_config(tmp_path)))
    poses = pose.process(request)
    assert seen == list(range(5))
    np.testing.assert_array_equal(poses.keypoints[:, 0], np.array([[10, 19, 1.4], [43, 16, 1.4], [15, 25, 1.4], [46, 27, 1.4]], np.float32))
    Encoder.calls = []
    appearance = materialize_model_weights(module.PersonAppearanceFeaturesModule(fake_encoder_spec(Encoder), batch_size=3))
    features = appearance.process(module.PersonAppearanceFeaturesInput(request.video, request.detections, poses))
    assert Encoder.calls == [3, 1]  # crosses frames, including an incomplete final batch
    for artifact in (poses, features):
        np.testing.assert_array_equal(artifact.frame_offsets, [0, 2, 2, 3, 4, 4])
        np.testing.assert_array_equal(artifact.source_rows, [3, 8, 11, 19])
    expected = np.array([[50, 24], [50, 53], [90, 26], [110, 55]], np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(features.embeddings[:, :2], expected, atol=1e-7)
    assert features.appearance_valid.all()


@pytest.mark.parametrize('stage', ['pose', 'appearance'])
def test_no_detections_preserves_timeline_without_loading_models(tmp_path: Path, stage: str) -> None:
    request = inputs(tmp_path)
    empty = replace(request.detections, frame_offsets=np.zeros(6, np.int64), boxes_xyxy=np.empty((0, 4), np.float32),
                    confidence=np.empty(0, np.float32), source_rows=np.empty(0, np.int64))
    component = (module.PersonPoseFeaturesModule(make_people_config(tmp_path)) if stage == 'pose' else
                 module.PersonAppearanceFeaturesModule(fake_encoder_spec(lambda: pytest.fail('must not load'))))
    # No assets or actual video are needed for an empty detection timeline.
    pose_input = replace(request, detections=empty)
    pose_output = module.PersonPoseFeaturesModule(make_people_config(tmp_path)).process(pose_input)
    result = component.process(pose_input if stage == "pose" else module.PersonAppearanceFeaturesInput(request.video, empty, pose_output))
    assert len(result.source_rows) == 0 and result.frame_offsets.tolist() == [0] * 6


@pytest.mark.parametrize('bad_reader', ['short', 'reordered', 'size'])
def test_feature_pass_rejects_decode_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_reader: str) -> None:
    packets = list(reader())
    if bad_reader == 'short':
        packets.pop()
    elif bad_reader == 'reordered':
        packets[1], packets[2] = packets[2], packets[1]
    else:
        packets[2].frame = packets[2].frame[:20]
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **kw: packets)
    component = materialize_model_weights(module.PersonAppearanceFeaturesModule(fake_encoder_spec(Encoder), batch_size=3))
    with pytest.raises(ValueError, match='timeline|order|size'):
        request = inputs(tmp_path)
        poses = module.PersonPoseFeaturesOutput("cam", request.detections.frame_offsets, request.detections.source_rows, np.zeros((4, 17, 3), np.float32))
        component.process(module.PersonAppearanceFeaturesInput(request.video, request.detections, poses))


def test_separate_passes_publish_reload_and_cache_without_model_overlap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    from src.tennis_scene.pipeline.components.person_detection import (
        PersonDetectionModule,
    )
    from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource
    from src.tennis_scene.pipeline.execution_plan import (
        ClipExecution,
        ComponentNode,
        ExecutionPlan,
    )
    from src.tennis_scene.pipeline.input_assembly.preprocessing import (
        PersonAppearanceFeaturesInputAssembler,
        PersonDetectionInputAssembler,
        PersonPoseFeaturesInputAssembler,
    )
    from src.tennis_scene.pipeline.model_assets import file_identity
    from src.tennis_scene.pipeline.model_resources import model_identities
    from src.tennis_scene.pipeline.runner import PlanExecutor
    from src.tennis_scene.pipeline.storage.clip_store import ClipStore
    original = inputs(tmp_path)
    people = make_people_config(tmp_path)
    class Pose:
        def __init__(self, *a: Any, **kw: Any) -> None: pass
        def load(self) -> None: pass
        def unload(self) -> None: pass
        def predict(self, request: Any) -> Any:
            list(request.frames_bgr)
            values = torch.zeros((len(request.bbx_xys), 17, 3))
            values[..., :2] = request.bbx_xys[:, None, :2]
            return SimpleNamespace(keypoints=values)
    monkeypatch.setattr(module, 'ViTPosePose2D', Pose)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', reader)
    source = ClipSource('clip', (replace(original.video, camera_id='cam0'), replace(original.video, camera_id='cam1')))
    nodes = []
    for video in source.videos:
        camera = video.camera_id
        context = AssemblyContext(source, camera)
        detection = replace(original.detections, camera_id=camera)
        fixed = SimpleNamespace(process=lambda _, value=detection: value)
        nodes.append(ComponentNode(f'person_detection/{camera}', fixed, PersonDetectionModule.io, PersonDetectionInputAssembler(), {}, context, {}, 'test'))
        pose = materialize_model_weights(module.PersonPoseFeaturesModule(people))
        appearance = materialize_model_weights(module.PersonAppearanceFeaturesModule(fake_encoder_spec(Encoder), batch_size=3))
        nodes.append(ComponentNode(f'person_pose_features/{camera}', pose, pose.io, PersonPoseFeaturesInputAssembler(),
            {'detections': f'person_detection/{camera}'}, context, {'models': model_identities(pose.model_specs(), file_identity)}, 'test'))
        nodes.append(ComponentNode(f'person_appearance_features/{camera}', appearance, appearance.io, PersonAppearanceFeaturesInputAssembler(),
            {'detections': f'person_detection/{camera}', 'poses': f'person_pose_features/{camera}'}, context, {'models': model_identities(appearance.model_specs(), file_identity)}, 'test'))
    active: set[str] = set()
    events: list[tuple[str, str]] = []
    @contextmanager
    def model_event(action: str, model: str, **attributes: Any) -> Any:
        events.append((action, model))
        if action == 'load':
            assert not active, 'ViTPose and CLIP must occupy separate passes'
            active.add(model)
        yield
        if action == 'unload':
            active.remove(model)
    store = ClipStore(tmp_path / 'store', {'clip_id': source.clip_id}, memory_entries=0)
    clip = ClipExecution(source.clip_id, tuple(nodes), store)
    plan = ExecutionPlan([clip])
    executor = PlanExecutor(plan, model_event=model_event)
    executor.run()
    assert events == [('load', 'vitpose'), ('unload', 'vitpose'), ('load', 'appearance'), ('unload', 'appearance')]
    assert not active
    result = executor.output(('clip', 'person_appearance_features/cam1'))
    np.testing.assert_array_equal(result.source_rows, [3, 8, 11, 19])
    events.clear()
    cached = ExecutionPlan([clip])
    PlanExecutor(cached, model_event=model_event).run()
    assert set(cached.statuses.values()) == {'cached'} and not events
    loaded = ExecutionPlan([replace(clip, nodes=tuple(replace(n, source='load') for n in nodes))])
    PlanExecutor(loaded, model_event=model_event).run()
    assert set(loaded.statuses.values()) == {'loaded'} and not events


def test_appearance_invalid_crops_stay_masked_and_unprompted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = inputs(tmp_path)
    boxes = request.detections.boxes_xyxy.copy()
    boxes[1] = [80, 4, 100, 20]  # entirely outside the decoded image
    detections = replace(request.detections, boxes_xyxy=boxes)
    poses = module.PersonPoseFeaturesOutput('cam', detections.frame_offsets, detections.source_rows, np.full((4, 17, 3), 999., np.float32))
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', reader)
    Encoder.calls = []
    component = materialize_model_weights(module.PersonAppearanceFeaturesModule(fake_encoder_spec(Encoder), batch_size=2))
    result = component.process(module.PersonAppearanceFeaturesInput(request.video, detections, poses))
    assert Encoder.calls == [2, 1]
    assert result.appearance_valid.tolist() == [True, False, True, True]
    assert not result.embeddings[1].any()
