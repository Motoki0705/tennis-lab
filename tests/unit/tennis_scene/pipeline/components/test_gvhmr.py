"""GVHMR consumes persisted HMR2 features and selected segment observations."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.body_types import BodyRecoveryRequest
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodySelection,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput, GVHMRModule
from src.tennis_scene.pipeline.components.hmr2_features import (
    HMR2BodyFeatures,
    HMR2FeatureSegment,
    HMR2FeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_people_config,
    materialize_model_weights,
)


def selected(tmp_path: Path) -> GVHMRInput:
    source = ClipSource('clip', (SourceVideo('cam1', tmp_path / 'cam1.mp4', 'video-sha', 8, 30., 640, 480),))
    request = BodyRecoveryRequest(tmp_path / 'cam1.mp4', np.array([0, 2, 4], np.int64),
        np.zeros((3, 17, 3), np.float32), np.ones((3, 3), np.float32) * 100,
        (640, 480), np.eye(3, dtype=np.float64))
    features = np.tile(np.array([3, 8, 17], np.float32)[:, None], (1, 1024))
    return GVHMRInput(source, BodyViewSelectionOutput((BodySelection(7, 'cam1', (request,), (3,)),)),
        HMR2FeaturesOutput('clip', (HMR2BodyFeatures(7, 'cam1', 'video-sha', (HMR2FeatureSegment(0, request.source_frames, features),)),)))


def test_only_mesh_model_runs_on_declared_saved_features(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    class Mesh:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def predict(self, request: Any) -> Any:
            calls.append('mesh')
            assert request.kp2d.shape == (3, 17, 3)
            assert request.K_fullimg.shape == (3, 3)
            assert request.f_imgseq.dtype == torch.float32
            torch.testing.assert_close(request.f_imgseq[:, 0], torch.tensor([3., 8., 17.]))
            return SimpleNamespace(smpl_params_incam={name: request.f_imgseq[:, :1].repeat(1, width) for name, width in
                [('body_pose', 63), ('global_orient', 3), ('betas', 10), ('transl', 3)]})
        def load(self) -> None:
            pass
        def unload(self) -> None:
            calls.append('mesh_unload')
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', Mesh)
    module = materialize_model_weights(GVHMRModule(make_people_config(tmp_path)))
    assert set(module.model_specs()) == {'mesh'}
    result = module.process(selected(tmp_path))
    assert result.bodies[0].person_id == 7 and result.bodies[0].camera_id == 'cam1'
    np.testing.assert_array_equal(result.bodies[0].segments[0].source_frames, [0, 2, 4])
    np.testing.assert_array_equal(result.bodies[0].segments[0].parameters.transl[:, 0], [3, 8, 17])
    assert result.bodies[0].segments[0].observed_samples == 3
    assert calls == ['mesh', 'mesh_unload']


def test_invalid_body_input_fails_before_image_feature_extraction(tmp_path: Path) -> None:
    request = selected(tmp_path).selection.selections[0].requests[0]
    with pytest.raises(ValueError, match='ordered source frames'):
        replace(request, source_frames=np.array([0, 2, 2], np.int64))
    with pytest.raises(ValueError, match='shape mismatch'):
        replace(request, keypoints=np.zeros((3, 16, 3), np.float32))


@pytest.mark.parametrize('wrong', ['clip', 'person', 'camera', 'video', 'segment_count', 'frames', 'missing_person'])
def test_features_from_another_source_fail_before_mesh_load(tmp_path: Path, wrong: str) -> None:
    inputs = selected(tmp_path)
    body = inputs.features.bodies[0]
    if wrong == 'clip':
        features = replace(inputs.features, clip_id='another')
    elif wrong == 'missing_person':
        features = replace(inputs.features, bodies=())
    else:
        if wrong == 'person':
            body = replace(body, person_id=3)
        elif wrong == 'camera':
            body = replace(body, camera_id='cam0')
        elif wrong == 'video':
            body = replace(body, video_sha256='wrong')
        elif wrong == 'segment_count':
            body = replace(body, segments=())
        else:
            body = replace(body, segments=(replace(body.segments[0], source_frames=np.array([1, 2, 4], np.int64)),))
        features = replace(inputs.features, bodies=(body,))
    with pytest.raises(ValueError, match='HMR2 features do not match'):
        GVHMRModule(make_people_config(tmp_path)).process(replace(inputs, features=features))


def test_empty_body_selection_needs_no_weights(tmp_path: Path) -> None:
    inputs = selected(tmp_path)
    empty = replace(inputs, selection=BodyViewSelectionOutput(()), features=HMR2FeaturesOutput('clip', ()))
    result = GVHMRModule(make_people_config(tmp_path)).process(empty)
    assert not result.bodies


def test_gvhmr_batches_equal_lengths_and_preserves_person_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = selected(tmp_path)
    choice = first.selection.selections[0]
    body = first.features.bodies[0]
    segment = body.segments[0]
    last = replace(choice.requests[0], source_frames=np.array([3, 7], np.int64),
                   keypoints=choice.requests[0].keypoints[:2], boxes_xys=choice.requests[0].boxes_xys[:2])
    choices = (choice, replace(choice, person_id=2), replace(choice, person_id=9, requests=(last,), observed_samples=(2,)))
    bodies = (
        body,
        replace(body, person_id=2, segments=(replace(segment, features=segment.features + 100),)),
        replace(body, person_id=9, segments=(replace(segment, source_frames=last.source_frames, features=segment.features[:2] + 200),)),
    )
    calls: list[list[tuple[int, float]]] = []

    class Mesh:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def load(self) -> None:
            pass

        def unload(self) -> None:
            pass

        def predict_batch(self, requests: list[Any]) -> list[Any]:
            calls.append([(len(row.kp2d), float(row.f_imgseq[0, 0])) for row in requests])
            return [
                SimpleNamespace(smpl_params_incam={
                    name: row.f_imgseq[:, :1].repeat(1, width)
                    for name, width in (("body_pose", 63), ("global_orient", 3), ("betas", 10), ("transl", 3))
                })
                for row in requests
            ]

        def predict(self, request: Any) -> Any:
            return self.predict_batch([request])[0]

    monkeypatch.setattr("src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery", Mesh)
    config = make_people_config(tmp_path)
    runtime = replace(config.runtime, gvhmr=replace(config.runtime.gvhmr, batch_size=2))
    module = materialize_model_weights(GVHMRModule(replace(config, runtime=runtime)))
    result = module.process(GVHMRInput(first.source, BodyViewSelectionOutput(choices), replace(first.features, bodies=bodies)))
    assert calls == [[(3, 3), (3, 103)], [(2, 203)]]
    assert [value.person_id for value in result.bodies] == [7, 2, 9]
    for value, expected in zip(result.bodies, [[3, 8, 17], [103, 108, 117], [203, 208]], strict=True):
        np.testing.assert_array_equal(value.segments[0].parameters.transl[:, 0], expected)
