"""External annotations obey the detector schema without inventing observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.imports.ball_annotations import convert_ball_annotation


def annotation(tmp_path: Path) -> tuple[Path, SourceVideo]:
    video = SourceVideo('cam0', tmp_path / 'video.mp4', 'verified_media_hash', 4, 30., 640, 480)
    kinds = [('observed', 'visible'), ('interpolated', 'not_independently_visible'), ('occlusion_estimated', 'occluded'), ('unresolved', 'unknown')]
    payload = {'schema_version': 'video_ball_annotation.v2',
        'source': {'sha256': video.sha256, 'width': 640, 'height': 480, 'frame_count': 4, 'fps_numerator': 30, 'fps_denominator': 1},
        'coordinate_system': {'origin': 'top_left', 'frame_index': 'zero_based', 'normalization': {'x': 'x_px / width', 'y': 'y_px / height'}},
        'target': {'track_id': 1},
        'frames': [{'frame_index': i, 'status': status, 'visibility': visible, 'label': 'tennis_ball', 'track_id': 1,
            'center_px': {'x': 320., 'y': 240.} if i < 3 else None,
            'center_normalized': {'x': .5, 'y': .5} if i < 3 else None, 'image_score': 127.42 if i == 0 else None}
            for i, (status, visible) in enumerate(kinds)]}
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(payload))
    return path, video


def test_estimates_preserved_but_never_promoted_to_observations(tmp_path: Path) -> None:
    path, video = annotation(tmp_path)
    result, provenance = convert_ball_annotation(path, video)
    np.testing.assert_array_equal(result.observed, [True, False, False, False])
    np.testing.assert_array_equal(result.confidence, [1., 0., 0., 0.])
    np.testing.assert_array_equal(result.point_kind, [1, 2, 3, 0])
    np.testing.assert_array_equal(result.uv_px[1:3], [[320, 240], [320, 240]])
    assert result.score_semantics == 'annotation_acceptance_not_probability'
    assert provenance['origin'] == 'external_annotation'
    assert provenance['frame_offset'] == 0


@pytest.mark.parametrize('mutation', ['media', 'duplicate', 'normalization'])
def test_wrong_video_or_timeline_or_coordinates_fail(tmp_path: Path, mutation: str) -> None:
    path, video = annotation(tmp_path)
    payload = json.loads(path.read_text())
    if mutation == 'media':
        payload['source']['sha256'] = 'another_clip'
    elif mutation == 'duplicate':
        payload['frames'][1]['frame_index'] = 0
    else:
        payload['frames'][0]['center_normalized']['x'] = .8
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        convert_ball_annotation(path, video)
