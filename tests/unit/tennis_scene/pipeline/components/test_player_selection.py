"""Component boundary preserves full selection evidence and caps linked groups."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.person_tracking.court_linking import LinkingConfig, exclusive_region
from src.tasks.person_tracking.sequence import TrackEvidence
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    TrackAppearance,
)
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import (
    PlayerSelectionInput,
    PlayerSelectionModule,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.inference_runtime import ModelSpec
from tests.unit.tasks.person_tracking.test_court_candidates import camera_tracks


def _input(positions: list[tuple[float, float]], seen: np.ndarray) -> PlayerSelectionInput:
    tracks = camera_tracks(positions, seen)
    raw = PersonTrackingOutput('cam0', tracks.track_ids, tracks.boxes_xyxy.astype(np.float32), seen,
                               tuple((int(i),) for i in tracks.track_ids), ())
    calibration = cast(CourtCalibrationOutput, SimpleNamespace(calibration=SimpleNamespace(views=[SimpleNamespace(camera=tracks.camera)])))
    return PlayerSelectionInput(SourceVideo('cam0', Path('unused.mp4'), 'hash', seen.shape[1], 30., 1920, 1080), calibration, raw)


def _module() -> PlayerSelectionModule:
    return PlayerSelectionModule(LinkingConfig(), footpoints=FootpointConfig(), sampling=CropSamplingConfig(),
                                 encoder=None, encoder_name=None, device='cpu', enabled=True)


@pytest.mark.parametrize('difference', ['same_content', 'weights', 'dtype', 'unknown'])
def test_tracking_features_require_the_same_model_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: str) -> None:
    import src.tennis_scene.pipeline.components.player_selection as selection
    inputs = _input([(0., 5.)], np.ones((1, 80), bool))
    first, second = tmp_path / 'tracking.pth', tmp_path / 'selection.pth'
    first.write_bytes(b'weights A')
    second.write_bytes(b'weights B' if difference == 'weights' else b'weights A')
    created: list[bool] = []
    profile = 'clipreid_vitb16_market1501'
    def create() -> SimpleNamespace:
        created.append(True)
        return SimpleNamespace(name=profile)
    tracked: ModelSpec[Any] = ModelSpec('appearance', {'checkpoint': first}, profile, 'cpu', 'float32', {}, object)
    requested = replace(tracked, weights={'checkpoint': second}, dtype='bfloat16' if difference == 'dtype' else 'float32',
        factory=create)
    evidence = TrackEvidence(np.arange(80, dtype=np.int64)[None], np.zeros((1, 80, 17, 3), np.float32),
        np.tile(np.array([1., 0.], np.float32), (1, 80, 1)), np.ones((1, 80), bool), profile)
    # Bind provenance without relying on the field being implemented yet: this
    # reproducer can execute against the old name-only reuse path.
    object.__setattr__(evidence, 'encoder_identity', None if difference == 'unknown' else document_digest(json_value(tracked.key)))
    inputs = replace(inputs, tracks=replace(inputs.tracks, evidence=evidence))
    recomputed = (TrackAppearance(np.array([0], np.int64), np.array([[0., 1.]], np.float32)),)
    monkeypatch.setattr(selection, 'embed_tracks', lambda *args: (recomputed, []))
    inspected = []
    original = selection.select_linked_candidates
    def inspect(tracks: Any, *args: Any) -> Any:
        inspected.append(tracks.appearance)
        return original(tracks, *args)
    monkeypatch.setattr(selection, 'select_linked_candidates', inspect)
    module = PlayerSelectionModule(LinkingConfig(), footpoints=FootpointConfig(), sampling=CropSamplingConfig(),
        encoder=requested, encoder_name=profile, device='cpu', enabled=True)
    result = module.process(inputs)
    reuse = difference == 'same_content'
    assert len(created) == (0 if reuse else 1)
    assert (result.diagnostics['appearance'].get('source') == 'tracked_detection_features') == reuse
    np.testing.assert_array_equal(inspected[0][0].embeddings[0], [1., 0.] if reuse else [0., 1.])
    assert result.tracks.evidence is not None
    assert result.tracks.evidence.encoder_identity == evidence.encoder_identity
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    codec = ArtifactCodec(type(result))
    payload, arrays = codec.dump(result, tmp_path)
    assert codec.load(payload, tmp_path, arrays).tracks.evidence.encoder_identity == evidence.encoder_identity


def test_fragments_link_before_dwell_and_cap_and_keep_duplicate_observations(tmp_path: Path) -> None:
    seen: np.ndarray = np.zeros((8, 80), bool)
    for row in range(8):
        seen[row, row * 10:min(80, row * 10 + 12)] = True
    result = _module().process(_input([(0., 5.)] * 8, seen))
    assert result.tracks.observed.shape == (1, 80)
    assert result.tracks.observed.all()
    np.testing.assert_array_equal(result.selected, seen)
    assert result.selected.sum() == 94 and result.tracks.observed.sum() == 80
    assert result.diagnostics['linking']['groups'][0]['in_core_frames'] == 80
    assert result.origin_rows[0, 11] == 0 and result.origin_rows[0, 12] == 1
    from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
    codec = ArtifactCodec(type(result))
    payload, arrays = codec.dump(result, tmp_path)
    restored = codec.load(payload, tmp_path, arrays)
    np.testing.assert_array_equal(restored.selected, seen)
    np.testing.assert_array_equal(restored.origin_rows, result.origin_rows)
    assert restored.diagnostics == result.diagnostics


def test_cap_applies_to_eligible_groups_not_all_person_input() -> None:
    inputs = _input([(float(i), 5.) for i in range(-4, 5)] + [(9., 5.)], np.ones((10, 80), bool))
    result = _module().process(inputs)
    assert result.selected[:6].all() and not result.selected[6:].any()
    assert len(result.tracks.track_ids) == 6
    assert result.diagnostics['linking']['groups'][-1]['reason'] == 'insufficient_dwell'


def test_membership_boundary_is_singles_width_and_baseline_plus_five_metres() -> None:
    points = np.array([[4.115, 16.885], [-4.115, -16.885], [4.116, 0.], [0., 16.886]])
    assert exclusive_region(points, LinkingConfig(), core=True).tolist() == [True, True, False, False]


def test_wide_observations_survive_component_and_uncalibrated_camera_is_explicit() -> None:
    inputs = _input([(0., 5.), (6., 5.)], np.ones((2, 120), bool))
    boxes = inputs.tracks.boxes_xyxy.copy()
    for frame in range(30, 90):
        fraction = (frame - 30) / 59
        boxes[0, frame] = (1 - fraction) * boxes[0, frame] + fraction * boxes[1, frame]
    boxes[0, 90:] = boxes[1, 90:]
    result = _module().process(replace(inputs, tracks=replace(inputs.tracks, boxes_xyxy=boxes)))
    assert result.selected[0].all() and not result.selected[1].any()
    np.testing.assert_array_equal(result.tracks.boxes_xyxy[0], boxes[0])
    uncalibrated = cast(CourtCalibrationOutput, SimpleNamespace(calibration=SimpleNamespace(views=[])))
    rejected = _module().process(replace(inputs, calibration=uncalibrated))
    assert not rejected.selected.any() and rejected.diagnostics['reason'] == 'camera_not_calibrated'
    with pytest.raises(ValueError, match='camera/timeline'):
        _module().process(replace(inputs, video=replace(inputs.video, camera_id='other')))
