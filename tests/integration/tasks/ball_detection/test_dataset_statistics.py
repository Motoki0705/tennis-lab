import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_detection.dataset_statistics import (
    StatisticsConfig,
    compute_dataset_statistics,
)
from src.tasks.ball_detection.dataset_statistics.inputs import annotation_details
from src.tasks.ball_detection.visualization.inference.service import DetectionService
from src.tasks.ball_detection.visualization.review.statistics import (
    StatisticsService,
    statistics_router,
)
from src.tasks.base.visualization.detection.web import create_detection_app
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


@pytest.fixture
def dataset(tmp_path: Path):
    root = write_store_clip(tmp_path / 'data/ball_detection/test', 'tracknet/game/clip',
                            [frame(i, ball(xy=(20., 20.))) for i in range(40)])
    return tmp_path, root


def test_statistics_http_lifecycle_scope_assets_and_origin(dataset):
    project, _ = dataset
    backend = DetectionService(project)
    statistics = StatisticsService(project, project / 'data', None)
    app = create_detection_app(backend, task='ball_detection', mode='review', service_config={},
                               review_router=statistics_router(statistics), statistics_ui='/statistics-static/panel.mjs')
    try:
        with TestClient(app) as client:
            assert client.get('/api/catalog').json()['statistics_ui'] == '/statistics-static/panel.mjs'
            settings = client.get('/api/statistics/config').json()
            assert client.get('/statistics-static/panel.mjs').status_code == 200
            for name in ('plot_base.mjs', 'plots.mjs', 'dashboard.mjs'):
                assert client.get(f'/statistics-static/{name}').status_code == 200
            assert client.get('/statistics-static/secret.py').status_code == 404
            body = dict(dataset='store/test', settings=settings)
            assert client.post('/api/statistics/jobs', json=body, headers={'Origin': 'http://elsewhere'}).status_code == 403
            invalid = dict(body, settings=dict(settings, strides=None))
            assert client.post('/api/statistics/jobs', json=invalid).status_code == 422
            response = client.post('/api/statistics/jobs', json=body)
            assert response.status_code == 202
            identifier = response.json()['id']
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                status = client.get(f'/api/statistics/jobs/{identifier}').json()
                if status['state'] not in ('running', 'pending'):
                    break
                time.sleep(.02)
            assert status['state'] == 'complete', status
            overview = client.get(f'/api/statistics/jobs/{identifier}/result').json()
            assert overview['groups']['all']['clip']['pooled']['counts']['frames'] == 40
            assert len(overview['clips']) == 1
            detail = client.get(f'/api/statistics/jobs/{identifier}/clip', params={'scene': 'store/test::tracknet/game/clip'})
            assert detail.status_code == 200 and len(detail.json()['times']) == 40
            assert client.get(f'/api/statistics/jobs/{identifier}/clip', params={'scene': 'store/other::tracknet/game/clip'}).status_code == 422
            assert client.get('/api/statistics/jobs/unknown').status_code == 404
    finally:
        statistics.close()


def test_raw_text_and_interpolation_support_require_matching_snapshot(tmp_path):
    root = write_store_clip(tmp_path / 'ball', 'chat_annotation/example', [
        frame(i, ball('interpolated' if i == 10 else 'observed')) for i in range(40)
    ], source='chat_annotation')
    original = tmp_path / 'original.json'
    rows = [dict(frame_index=i, notes='背景との重なり' if i == 10 else '',
                 balls=[dict(interpolation_frames=[9, 11] if i == 10 else None)]) for i in range(40)]
    original.write_text(json.dumps(dict(schema_version='tennis_chat_ball_annotation.v1', frames=rows)))
    metadata = json.loads((root / 'metadata.json').read_text())
    record = metadata['clips'][0]
    record.update(annotation_path=str(original), annotation_sha256=dual_sha256(original), provenance={'annotation_issues': ['イベント位置が未確定']})
    (root / 'metadata.json').write_text(json.dumps(metadata))
    store = BallFrameStore(root)
    details = annotation_details(record, store.clips[0], tmp_path)
    assert details.notes == {10: '背景との重なり'} and details.endpoints == {10: (9, 11)}
    assert details.issues == ('イベント位置が未確定',)
    original.write_text('{}')
    assert annotation_details(record, store.clips[0], tmp_path).availability == 'hash_mismatch'
    original.unlink()
    assert annotation_details(record, store.clips[0], tmp_path).availability == 'missing_file'


def test_pose_integrity_and_frame_alignment(dataset):
    project, root = dataset
    store = BallFrameStore(root)
    poses = project / 'data/poses'
    poses.mkdir()
    artifact = poses / 'pose.npz'
    points: NDArray[np.float32] = np.full((40, 1, 17, 3), .9, np.float32)
    points[..., :2] = 20
    observed: NDArray[np.bool_] = np.ones((40, 1), bool)
    boxes = np.tile([0, 0, 40, 40], (40, 1, 1)).astype(np.float32)
    np.savez(artifact, keypoints=points, observed=observed, boxes_xyxy=boxes,
             player_ids=np.array(['p1']), frame_index=np.arange(40), pts=store.frames['pts'],
             raw_track_ids=np.ones((40, 1), np.int64), detection_rows=np.arange(40)[:, None])
    review = poses / 'review.json'
    raw_tracks_sha256 = 'fixture-raw-tracks'
    review.write_text(json.dumps(dict(status='approved', clip_id=store.clips[0].clip_id,
                                     raw_tracks_sha256=raw_tracks_sha256)))
    manifest = dict(schema='ball_detection_player_poses.v1', coordinate_system='stored_jpeg_pixels',
                    ball_store=dict(directory=str(root), hashes={name: dual_sha256(root / name) for name in ('metadata.json', 'index.npz')}),
                    clips=[dict(clip_id=store.clips[0].clip_id, pose_status='approved', file=artifact.name,
                                sha256=dual_sha256(artifact), raw_tracks_sha256=raw_tracks_sha256,
                                review_file=review.name, review_sha256=dual_sha256(review))])
    (poses / 'manifest.json').write_text(json.dumps(manifest))
    report = compute_dataset_statistics(store, [store.clips[0].clip_id], StatisticsConfig.shipped(), project_root=project, pose_directory=poses)
    assert report['pose_available_clips'] == 1
    assert report['groups']['all']['clip']['pooled']['distributions']['pose/left_wrist/speed']['max'] == 0
    with artifact.open('ab') as out:
        out.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        compute_dataset_statistics(store, [store.clips[0].clip_id], StatisticsConfig.shipped(), project_root=project, pose_directory=poses)


def test_changed_dataset_during_computation_is_rejected(dataset):
    project, root = dataset
    store = BallFrameStore(root)
    def changed(completed, total):
        with (root / 'metadata.json').open('a') as out:
            out.write(' ')
    with pytest.raises(ValueError, match='changed during'):
        compute_dataset_statistics(store, [store.clips[0].clip_id], StatisticsConfig.shipped(), project_root=project, progress=changed)
