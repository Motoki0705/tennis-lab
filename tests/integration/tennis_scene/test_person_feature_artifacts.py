"""Frozen pre-split features must reproduce CPU tracks, GSI and court selection."""
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.person_tracking.sequence import TrackingConfig
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components import person_tracking as tracking
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonAppearanceFeaturesOutput,
    PersonPoseFeaturesOutput,
    appearance_feature_settings,
    pose_feature_settings,
)
from src.tennis_scene.pipeline.components.player_selection import (
    PlayerSelectionInput,
    PlayerSelectionModule,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.unit.tennis_scene.pipeline.components.test_player_selection import _input
from tests.unit.tennis_scene.pipeline.config_factories import (
    fake_encoder_spec,
    fake_pose_spec,
)

FIXTURES = Path(__file__).parent / 'fixtures'


def saved_inputs(tmp_path: Path) -> tracking.PersonTrackingInput:
    frames, provenance = load_features(FIXTURES / 'person-features-before-split.npz')
    assert provenance['revision'] == 'f7b016700'
    offsets = np.array([0, *np.cumsum([len(f.rows) for f in frames])], np.int64)
    rows = np.concatenate([f.rows for f in frames])
    detections = PersonDetectionOutput('cam0', offsets, np.concatenate([f.boxes for f in frames]),
                                       np.concatenate([f.scores for f in frames]), rows)
    poses = PersonPoseFeaturesOutput('cam0', offsets, rows, np.concatenate([f.require_poses() for f in frames]),
        document_digest(fake_pose_spec().key), document_digest(pose_feature_settings(TrackingConfig().features)))
    appearance = PersonAppearanceFeaturesOutput('cam0', offsets, rows, np.concatenate([f.embeddings for f in frames]),
        np.concatenate([f.appearance_valid for f in frames]), TrackingConfig().encoder, document_digest(fake_encoder_spec(object).key),
        document_digest(appearance_feature_settings(TrackingConfig().features)))
    # Exercise real schema-directed persistence, with mmap inputs as in ClipStore.
    for name, artifact in [('pose', poses), ('appearance', appearance)]:
        directory = tmp_path / name
        directory.mkdir()
        codec = ArtifactCodec(type(artifact))
        payload, arrays = codec.dump(artifact, directory)
        if name == 'pose':
            poses = codec.load(payload, directory, arrays)
        else:
            appearance = codec.load(payload, directory, arrays)
    video = SourceVideo('cam0', tmp_path / 'must-not-open.mp4', 'frozen', len(frames), 30., 1920, 1080)
    return tracking.PersonTrackingInput(video, detections, poses, appearance)


class NoLinks:
    def __init__(self, _: Path) -> None: pass
    def links(self, boxes: np.ndarray, observed: np.ndarray) -> tuple[dict[int, int], list[Any]]:
        return {i: i for i in range(len(boxes))}, []


def test_saved_features_exactly_match_pre_split_tracking_selection_and_gsi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracking, 'AFLink', NoLinks)
    inputs = saved_inputs(tmp_path)
    component = tracking.PersonTrackingModule(TrackingConfig(), aflink_checkpoint=tmp_path / 'fake-aflink', show_progress=False,
        expected_pose=fake_pose_spec(), expected_encoder=fake_encoder_spec(object))
    actual = component.process(inputs)
    assert actual.evidence is not None and actual.reconstruction is not None
    with np.load(FIXTURES / 'person-tracking-before-split.npz') as expected:
        for name, values in {'track_ids': actual.track_ids, 'boxes': actual.boxes_xyxy, 'observed': actual.observed,
                'detection_rows': actual.evidence.detection_rows, 'poses': actual.evidence.poses,
                'embeddings': actual.evidence.embeddings, 'appearance_valid': actual.evidence.appearance_valid,
                'reconstructed_boxes': actual.reconstruction.boxes, 'interpolated': actual.reconstruction.interpolated,
                'source_track_ids': actual.source_track_ids}.items():
            np.testing.assert_array_equal(values, expected[name], err_msg=name)
        assert actual.reconstruction.interpolated[:, 15:17].all()
        assert not actual.observed[:, 15:17].any()
        calibration = _input([(0., 5.)] * 2, np.ones((2, inputs.video.num_frames), bool)).calibration
        selection = PlayerSelectionModule(LinkingConfig(), footpoints=FootpointConfig(), sampling=CropSamplingConfig(),
            encoder=fake_encoder_spec(lambda: pytest.fail('selection must use saved embeddings')), encoder_name=inputs.appearance.encoder, device='cpu', enabled=True)
        # Supply an explicit declaration, but no inference may be borrowed.
        selected = selection.process(PlayerSelectionInput(inputs.video, calibration, actual))
        for name, values in {'selected': selected.selected, 'group_ids': selected.tracks.track_ids,
                'group_boxes': selected.tracks.boxes_xyxy, 'group_observed': selected.tracks.observed,
                'origin_rows': selected.origin_rows, 'group_detection_rows': selected.tracks.evidence.detection_rows}.items():
            np.testing.assert_array_equal(values, expected[name], err_msg=name)
    frozen = json.loads((FIXTURES / 'person-selection-before-split.json').read_text())
    assert selected.diagnostics == frozen['diagnostics']
    # Running another camera/clip cannot inherit tracker state from this call.
    np.testing.assert_equal(asdict(component.process(inputs)), asdict(actual))


@pytest.mark.parametrize('mismatch', ['camera', 'rows', 'offsets'])
def test_saved_features_reject_wrong_source_before_cpu_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str) -> None:
    inputs = saved_inputs(tmp_path)
    pose = inputs.poses
    changes = {'camera_id': 'other'} if mismatch == 'camera' else ({'source_rows': pose.source_rows + 1} if mismatch == 'rows' else
              {'frame_offsets': np.r_[pose.frame_offsets[:2], pose.frame_offsets[2] - 1, pose.frame_offsets[3:]].astype(np.int64)})
    inputs = replace(inputs, poses=replace(pose, **changes))
    monkeypatch.setattr(tracking, 'AFLink', lambda *_: pytest.fail('must validate the join before tracking'))
    with pytest.raises(ValueError, match='camera/timeline/source row mismatch'):
        tracking.PersonTrackingModule(TrackingConfig(), aflink_checkpoint=tmp_path / 'fake',
            expected_pose=fake_pose_spec(), expected_encoder=fake_encoder_spec(object)).process(inputs)


@pytest.mark.parametrize('feature,field', [('poses', 'model_identity'), ('appearance', 'encoder_identity'),
                                         ('poses', 'preprocessing_identity'), ('appearance', 'preprocessing_identity')])
def test_wrong_feature_identity_cannot_enter_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, feature: str, field: str) -> None:
    inputs = saved_inputs(tmp_path)
    inputs = replace(inputs, **{feature: replace(getattr(inputs, feature), **{field: '0' * 64})})
    monkeypatch.setattr(tracking, 'AFLink', lambda *_: pytest.fail('identity must be checked before tracking'))
    with pytest.raises(ValueError, match='feature model/preprocessing identity'):
        tracking.PersonTrackingModule(TrackingConfig(), aflink_checkpoint=tmp_path / 'fake',
            expected_pose=fake_pose_spec(), expected_encoder=fake_encoder_spec(object)).process(inputs)
