from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from tests.unit.tasks.person_tracking.test_strongsort_pose import two_people


def no_links(monkeypatch: pytest.MonkeyPatch) -> AFLink:
    af = AFLink.__new__(AFLink)
    monkeypatch.setattr(af, 'links', lambda boxes, seen: ({i: i for i in range(len(boxes))}, []))
    return af


def test_shared_default_retains_original_rows_poses_and_synthetic_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames = [two_people(i, reverse=i >= 3) for i in range(7)]
    blank = frames[5]
    frames[5] = replace(blank, rows=blank.rows[:0], boxes=blank.boxes[:0], scores=blank.scores[:0],
                        poses=blank.poses[:0], embeddings=blank.embeddings[:0], appearance_valid=blank.appearance_valid[:0])
    config = TrackingConfig()
    result = track_sequence(iter(frames), fps=30., config=config, aflink=no_links(monkeypatch))
    assert config.method == 'strongsort_pp_pose' and config.online_config().pose_weight == .15
    assert result.evidence.detection_rows[:, 3].tolist() == [7, 6]
    np.testing.assert_array_equal(result.evidence.poses[0, 3], frames[3].poses[1])
    np.testing.assert_array_equal(result.boxes[0, 3], frames[3].boxes[1])
    assert result.reconstruction is not None and result.reconstruction.interpolated[:, 5].all()
    assert not result.observed[:, 5].any() and (result.evidence.detection_rows[:, 5] == -1).all()
    output = PersonTrackingOutput('cam0', result.track_ids, result.boxes, result.observed,
                                  result.source_track_ids, (), result.evidence, result.reconstruction)
    codec = ArtifactCodec(PersonTrackingOutput)
    payload, arrays = codec.dump(output, tmp_path)
    restored = codec.load(payload, tmp_path, arrays)
    assert restored.reconstruction is not None and restored.evidence is not None
    np.testing.assert_array_equal(restored.reconstruction.interpolated, result.reconstruction.interpolated)
    np.testing.assert_array_equal(restored.evidence.detection_rows, result.evidence.detection_rows)
    with pytest.raises(ValueError, match='separate from real'):
        replace(output, reconstruction=replace(result.reconstruction, interpolated=result.observed))


def test_missing_aflink_and_invalid_timeline_do_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match='AFLink must'):
        track_sequence([two_people(0)], fps=30., config=TrackingConfig(), aflink=None)
    with pytest.raises(ValueError, match='Unknown tracking'):
        TrackingConfig(method='typo')
    with pytest.raises(ValueError, match='contiguous'):
        track_sequence([two_people(1)], fps=30., config=TrackingConfig(), aflink=no_links(monkeypatch))


def test_group_evidence_does_not_promote_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    result = track_sequence([two_people(i) for i in range(5)], fps=30.,
                            config=TrackingConfig(), aflink=no_links(monkeypatch))
    mapping = np.array([[-1, -1, 1, 0, 1]], np.int64)
    evidence = result.evidence.regroup(mapping)
    assert evidence.detection_rows.tolist() == [[-1, -1, 5, 6, 9]]
    assert not evidence.poses[0, :2].any()
    mapping[0, 0] = 0
    with pytest.raises(ValueError, match='synthetic/missing'):
        result.evidence.regroup(mapping)
