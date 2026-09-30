from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tasks.person_tracking.sequence import TrackingConfig
from src.tennis_scene.pipeline.components import person_tracking as module
from src.tennis_scene.pipeline.components import pose_estimation
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import (
    materialize_assets,
    runtime,
)
from tests.unit.tennis_scene.test_pipeline_configuration import _runtime


def test_default_assets_and_schema_wiring_are_explicit(tmp_path: Path) -> None:
    from src.tennis_scene.pipeline.contracts import ClipSource
    cfg = runtime(tmp_path)
    assert cfg.tracking.method == 'strongsort_pp_pose' and not cfg.merge_duplicate_person_boxes
    assert cfg.tracking.online_config().pose_weight == .15
    assets = enabled_model_assets(cfg)
    assert assets['tracking_encoder'] == cfg.tracking_encoder_weights and assets['aflink'] == cfg.aflink_checkpoint
    materialize_assets(assets.values(), tmp_path)
    source = ClipSource('test', tuple(SourceVideo(f'cam{i}', tmp_path / f'{i}.mp4', 'hash', 5, 30., 1920, 1080) for i in range(3)))
    nodes = {n.name: n for n in standard_definition(cfg, source, code_identity='test')}
    node = nodes['person_tracking/cam0']
    assert node.component.config == cfg.tracking
    assert node.io.version == 5 and node.io.inputs['detections'].version == 2
    assert nodes['player_selection/cam0'].io.inputs['tracks'].version == 5
    cfg.aflink_checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match='AFLink'):
        standard_definition(cfg, source, code_identity='test')
    assert _runtime(['person_tracking.method=all_person_botsort']).tracking.method == 'all_person_botsort'
    with pytest.raises(ValueError, match='Unknown tracking'):
        _runtime(['person_tracking.method=invalid'])


def test_pipeline_consumes_detection_features_once_and_reuses_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    class Pose:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert kwargs['precision'] == 'float32'
        def predict(self, request: Any) -> Any:
            calls.append(1)
            values = torch.zeros((1, 17, 3))
            values[..., 0], values[..., 1], values[..., 2] = 30., 60., 1.4
            return SimpleNamespace(keypoints=values)
        def unload(self) -> None:
            pass
    class Encoder:
        name = 'clipreid_vitb16_market1501'
        input_size = (32, 16)
        def embed(self, crops: torch.Tensor) -> torch.Tensor:
            result = torch.zeros((len(crops), 1280))
            result[:, 0] = 1
            return result
    class Link:
        def __init__(self, path: Path) -> None:
            pass
        def links(self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[Any]]:
            return {i: i for i in range(len(boxes))}, []
    monkeypatch.setattr(module, 'ViTPosePose2D', Pose)
    monkeypatch.setattr(module, 'AFLink', Link)
    monkeypatch.setattr(module, 'OpenCVVideoFrameReader', lambda *a, **k:
        [SimpleNamespace(index=i, frame=np.zeros((180, 180, 3), np.uint8)) for i in range(5)])
    video = SourceVideo('cam0', tmp_path / 'unused.mp4', 'hash', 5, 30., 180, 180)
    detections = PersonDetectionOutput('cam0', np.arange(6, dtype=np.int64),
        np.tile(np.array([[10, 20, 50, 120]], np.float32), (5, 1)), np.full(5, .9, np.float32),
        np.array([0, 2, 4, 6, 8], np.int64))
    people = make_people_config(tmp_path)
    tracker = module.PersonTrackingModule(TrackingConfig(), people=people, encoder=Encoder, aflink_checkpoint=tmp_path / 'af.pth')
    result = tracker.process(module.PersonTrackingInput(video, detections))
    assert len(calls) == 5 and result.evidence is not None
    assert result.evidence.detection_rows.tolist() == [[-1, -1, 4, 6, 8]]
    def forbidden(*a: Any, **k: Any) -> Any:
        raise AssertionError('Pose must not be recomputed')
    monkeypatch.setattr(pose_estimation, 'ViTPosePose2D', forbidden)
    pose = pose_estimation.PoseEstimationModule(people, require_evidence=True)
    observed = pose.process(pose_estimation.PoseEstimationInput(video, result))
    assert observed.confidence[0, 2, 0, 0] == np.float32(1.4)
    with pytest.raises(ValueError, match='lost per-detection'):
        pose.process(pose_estimation.PoseEstimationInput(video, replace(result, evidence=None)))
