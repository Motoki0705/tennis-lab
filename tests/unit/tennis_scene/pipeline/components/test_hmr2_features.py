"""Source-indexed image features, without temporal inference or real video IO."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.pipeline.body_types import BodyRecoveryRequest
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodySelection,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.hmr2_features import (
    HMR2BodyFeatures,
    HMR2FeatureSegment,
    HMR2FeaturesInput,
    HMR2FeaturesModule,
    HMR2FeaturesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from tests.unit.tennis_scene.pipeline.config_factories import (
    make_people_config,
    materialize_model_weights,
)


def asymmetric_selection(root: Path, clip_id: str = 'clip-a') -> HMR2FeaturesInput:
    source = ClipSource(clip_id, tuple(SourceVideo(c, root / clip_id / f'{c}.mp4',
        f'{clip_id}-{c}-sha', 30, 30., 640, 480) for c in ('cam0', 'cam1')))
    def request(camera: str, frames: list[int], x: float) -> BodyRecoveryRequest:
        boxes = np.tile(np.array([x, 70, 50], np.float32), (len(frames), 1))
        kp: NDArray[np.float32] = np.full((len(frames), 17, 3), x / 10, np.float32)
        return BodyRecoveryRequest(source.video(camera).path, np.array(frames, np.int64),
            kp, boxes, (640, 480), np.eye(3, dtype=np.float64))
    return HMR2FeaturesInput(source, BodyViewSelectionOutput((
        BodySelection(9, 'cam1', (request('cam1', [1, 3, 5], 100), request('cam1', [10, 14], 200)), (3, 2)),
        BodySelection(2, 'cam0', (request('cam0', [2, 7, 11, 19], 300),), (3,)),
        BodySelection(17, 'cam1', (), ()),
    )))


class FakeFeatures:
    """Encode source frame, crop and feature column with independently visible values."""
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def load(self) -> None:
        pass
    def unload(self) -> None:
        pass
    def predict(self, request: Any) -> Any:
        assert request.frame_indices.dtype == torch.int64
        values = request.bbx_xys[:, :1] * 100 + request.frame_indices[:, None]
        return SimpleNamespace(features=values + torch.arange(1024, dtype=torch.float32)[None] / 4)


def test_features_preserve_asymmetric_people_segments_source_frames_and_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('src.tennis_scene.pipeline.components.hmr2_features.Hmr2FeatureExtractor', FakeFeatures)
    inputs = asymmetric_selection(tmp_path)
    result = materialize_model_weights(HMR2FeaturesModule(make_people_config(tmp_path))).process(inputs)
    assert result.clip_id == 'clip-a'
    assert [(b.person_id, b.camera_id, b.video_sha256) for b in result.bodies] == [
        (9, 'cam1', 'clip-a-cam1-sha'), (2, 'cam0', 'clip-a-cam0-sha'), (17, 'cam1', 'clip-a-cam1-sha')]
    assert [len(b.segments) for b in result.bodies] == [2, 1, 0]
    first, second = result.bodies[0].segments
    other = result.bodies[1].segments[0]
    assert [first.segment_index, second.segment_index, other.segment_index] == [0, 1, 0]
    np.testing.assert_array_equal(first.source_frames, [1, 3, 5])
    np.testing.assert_array_equal(second.source_frames, [10, 14])
    np.testing.assert_array_equal(other.source_frames, [2, 7, 11, 19])
    np.testing.assert_array_equal(first.features[:, 0], [10001, 10003, 10005])
    np.testing.assert_array_equal(second.features[:, 0], [20010, 20014])
    np.testing.assert_array_equal(other.features[:, 0], [30002, 30007, 30011, 30019])
    np.testing.assert_array_equal(other.features[:, 1023], [30257.75, 30262.75, 30266.75, 30274.75])


@pytest.mark.parametrize('invalid', ['video', 'size', 'frame_bound', 'overlap', 'observed', 'duplicate', 'camera'])
def test_selection_source_mismatches_fail_before_feature_model_load(tmp_path: Path, invalid: str) -> None:
    inputs = asymmetric_selection(tmp_path)
    first, other, empty = inputs.selection.selections
    request, tail = first.requests
    if invalid == 'video':
        first = replace(first, requests=(replace(request, video_path=tmp_path / 'another.mp4'), tail))
    elif invalid == 'size':
        first = replace(first, requests=(replace(request, size=(641, 480)), tail))
    elif invalid == 'frame_bound':
        first = replace(first, requests=(request, replace(tail, source_frames=np.array([10, 30], np.int64))))
    elif invalid == 'overlap':
        first = replace(first, requests=(request, replace(tail, source_frames=np.array([5, 9], np.int64))))
    elif invalid == 'observed':
        first = replace(first, observed_samples=(3,))
    elif invalid == 'duplicate':
        other = replace(other, person_id=9)
    else:
        first = replace(first, camera_id='unknown')
    bad = replace(inputs, selection=BodyViewSelectionOutput((first, other, empty)))
    # No assets exist: a model borrow would raise FileNotFoundError instead.
    with pytest.raises(ValueError, match='Body selection'):
        HMR2FeaturesModule(make_people_config(tmp_path)).process(bad)


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'nan', 'frames', 'index'])
def test_feature_schema_rejects_invalid_arrays(invalid: str) -> None:
    frames = np.array([1, 4], np.int64)
    features: NDArray[Any] = np.zeros((2, 1024), np.float32)
    index = 0
    if invalid == 'shape':
        features = features[:, :1023]
    elif invalid == 'dtype':
        features = features.astype(np.float64)
    elif invalid == 'nan':
        features[0, 0] = np.nan
    elif invalid == 'frames':
        frames[:] = [4, 1]
    else:
        index = -1
    with pytest.raises(ValueError, match='HMR2'):
        HMR2FeatureSegment(index, frames, features)


def test_feature_schema_rejects_duplicate_person_and_segment_identity() -> None:
    segment = HMR2FeatureSegment(0, np.array([1, 4], np.int64), np.zeros((2, 1024), np.float32))
    body = HMR2BodyFeatures(9, 'cam1', 'sha', (segment,))
    with pytest.raises(ValueError, match='unique person'):
        HMR2FeaturesOutput('clip', (body, body))
    with pytest.raises(ValueError, match='selection order'):
        replace(body, segments=(replace(segment, segment_index=1),))
    with pytest.raises(ValueError, match='disjoint'):
        replace(body, segments=(segment, replace(segment, segment_index=1)))


@pytest.mark.parametrize('disabled', [False, True])
def test_empty_or_disabled_selection_needs_no_weights(tmp_path: Path, disabled: bool) -> None:
    inputs = asymmetric_selection(tmp_path)
    if not disabled:
        inputs = replace(inputs, selection=BodyViewSelectionOutput(()))
    result = HMR2FeaturesModule(make_people_config(tmp_path), enabled=not disabled).process(inputs)
    assert result.clip_id == 'clip-a' and not result.bodies
