"""The split feature stages use the current standard ModelSpec/cache contract."""
import weakref
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components import (
    person_features,
    person_tracking,
    player_selection,
)
from src.tennis_scene.pipeline.components.person_detection import (
    PersonDetectionModule,
    PersonDetectionOutput,
)
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionInput
from src.tennis_scene.pipeline.components.pose_estimation import PoseEstimationInput
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
    standard_model_components,
)
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.model_assets import asset_hashes
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator
from src.tennis_scene.pipeline.runner import PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from tests.unit.tennis_scene.pipeline.components.test_person_features import reader
from tests.unit.tennis_scene.pipeline.components.test_player_selection import _input
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    FixedStage,
    materialize_assets,
    runtime,
)


def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str) -> Any:
    cfg = runtime(tmp_path, overrides=('device=cuda', 'gvhmr.enabled=false', f'people_models.runtime.vitpose.precision={precision}'))
    materialize_assets(enabled_model_assets(cfg).values(), tmp_path, court=cfg.court_kp)
    source = ClipSource('clip', tuple(SourceVideo(f'cam{i}', tmp_path / f'{i}.mp4', 'media', 5, 30., 64, 48) for i in range(2)))
    state = SimpleNamespace(failure=None, events=[], factories=[], references=[], encoder_factories=0)

    class Pose:
        def __init__(self, checkpoint: Path, **options: Any) -> None:
            assert options['precision'] == precision and options['device'] == 'cuda'
            assert options['flip_test'] == cfg.people.runtime.vitpose.flip_test
            state.factories.append(options)
            state.references.append(weakref.ref(self))
        def load(self) -> None:
            state.events.append('load')
            if state.failure == 'initialize':
                raise RuntimeError('feature initialize failed')
        def unload(self) -> None:
            state.events.append('unload')
            if state.failure == 'release':
                raise RuntimeError('feature release failed')
        def predict(self, request: Any) -> Any:
            state.events.append('predict')
            if state.failure == 'inference':
                raise RuntimeError('feature inference failed')
            list(request.frames_bgr)
            values = torch.zeros((len(request.bbx_xys), 17, 3), dtype=torch.float32)
            values[..., :2] = request.bbx_xys[:, None, :2]
            values[..., 2] = 1.4
            return SimpleNamespace(keypoints=values)

    class Encoder:
        name = cfg.tracking.encoder
        input_size = (8, 4)
        def embed(self, pixels: torch.Tensor) -> torch.Tensor:
            values = torch.zeros((len(pixels), 1280), dtype=torch.float32)
            values[:, 0] = 1
            return values

    def encoder(*args: Any, **kwargs: Any) -> Encoder:
        state.encoder_factories += 1
        assert kwargs['checkpoint'] == cfg.tracking_encoder_weights
        return Encoder()

    class Link:
        def __init__(self, path: Path) -> None: pass
        def links(self, boxes: np.ndarray, observed: np.ndarray) -> Any:
            return {i: i for i in range(len(boxes))}, []

    monkeypatch.setattr(person_features, 'ViTPosePose2D', Pose)
    monkeypatch.setattr(person_features, 'OpenCVVideoFrameReader', reader)
    monkeypatch.setattr(person_tracking, 'AFLink', Link)
    monkeypatch.setattr('src.tennis_scene.pipeline.definition.build_encoder', encoder)
    overrides = {}
    for video in source.videos:
        detection = PersonDetectionOutput(video.camera_id, np.arange(6, dtype=np.int64),
            np.tile(np.array([[2, 3, 18, 35]], np.float32), (5, 1)), np.full(5, .9, np.float32),
            np.array([2, 5, 8, 11, 14], np.int64))
        fixed = FixedStage(detection)
        fixed.io = PersonDetectionModule.io
        overrides[f'person_detection/{video.camera_id}'] = fixed
    nodes = standard_definition(cfg, source, code_identity='feature-runtime', overrides=overrides)
    store = ClipStore(tmp_path / 'store', json_value(source), memory_entries=0)
    return SimpleNamespace(cfg=cfg, source=source, nodes=nodes, store=store, state=state)


@pytest.mark.parametrize('precision,dtype', [('fp32', 'float32'), ('fp16', 'float16')])
def test_split_features_keep_precision_identity_reuse_and_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, precision: str, dtype: str) -> None:
    case = harness(tmp_path, monkeypatch, precision)
    targets = [f'person_tracking/{v.camera_id}' for v in case.source.videos]
    clip = ClipExecution('clip', case.nodes, case.store, targets=targets)
    executor = PlanExecutor(ExecutionPlan([clip]))
    executor.run()
    assert case.state.events == ['load', 'predict', 'predict', 'unload']
    assert len(case.state.factories) == case.state.encoder_factories == 1
    assert all(reference() is None for reference in case.state.references)
    marker = TennisSceneOrchestrator(case.cfg).publication_identity()
    for camera in case.source.camera_ids:
        node = next(n for n in case.nodes if n.name == f'person_pose_features/{camera}')
        assert node.settings['models']['pose']['dtype'] == node.component.model_specs()['pose'].dtype == dtype
        assert node.component.people.vitpose_checkpoint in asset_hashes(marker)
        appearance = executor.output(('clip', f'person_appearance_features/{camera}'))
        raw = executor.output(('clip', f'person_tracking/{camera}'))
        appearance_node = next(n for n in case.nodes if n.name == f'person_appearance_features/{camera}')
        assert raw.evidence.encoder_identity == appearance.encoder_identity == document_digest(appearance_node.component.model_specs()['encoder'].key)
        assert raw.evidence.detection_rows.tolist() == [[-1, -1, 8, 11, 14]]
        assert raw.evidence.require_poses().dtype == np.float32
    raw = executor.output(('clip', 'person_tracking/cam0'))
    components = standard_model_components(case.cfg, case.source.camera_ids)
    monkeypatch.setattr(player_selection, 'embed_tracks', lambda *a, **kw: pytest.fail('matching saved identity must be reused'))
    calibration = _input([(0., 5.)], np.ones((1, 5), bool)).calibration
    selected = components['player_selection'].process(PlayerSelectionInput(case.source.videos[0], calibration, raw))
    assert selected.diagnostics['appearance']['source'] == 'tracked_detection_features'
    assert selected.tracks.evidence.encoder_identity == raw.evidence.encoder_identity
    observed = components['pose_estimation'].process(PoseEstimationInput(case.source.videos[0], raw))
    assert observed.confidence.max() == np.float32(1.4)
    assert case.state.encoder_factories == 1
    for mode in ('execute', 'load'):
        plan = ExecutionPlan([replace(clip, nodes=tuple(replace(node, source=mode) for node in case.nodes))])
        PlanExecutor(plan).run()
        assert set(plan.statuses.values()) == {'cached' if mode == 'execute' else 'loaded'}
    assert len(case.state.factories) == case.state.encoder_factories == 1


@pytest.mark.parametrize('failure', ['initialize', 'inference', 'release'])
def test_feature_model_failure_cannot_publish_under_suppressing_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    case = harness(tmp_path, monkeypatch, 'fp16')
    case.state.failure = failure
    target = ('clip', 'person_pose_features/cam0')
    plan = ExecutionPlan([ClipExecution('clip', case.nodes, case.store, targets=[target[1]])])
    @contextmanager
    def hook(*args: Any, **kwargs: Any) -> Iterator[None]:
        with suppress(RuntimeError):
            yield
    completed: list[str] = []
    with pytest.raises(RuntimeError, match=f'feature {failure} failed') as caught:
        PlanExecutor(plan, model_event=hook).run(on_clip_complete=completed.append)
    assert caught.value.__traceback__ is not None
    assert case.state.events.count('load') == case.state.events.count('unload') == 1
    assert all(reference() is None for reference in case.state.references)
    assert target not in plan.references and case.store.active(target[1]) is None and not completed
    assert plan.statuses[target] == 'failed'
