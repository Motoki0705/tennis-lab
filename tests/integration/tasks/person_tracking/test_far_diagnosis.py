import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.far_report import summarize_report
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.utils.checksum import dual_sha256


def test_cpu_report_counts_same_units_extra_nonplayers_scores_and_summed_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = 'video_000/clip_000'
    boxes = np.array([[0, 0, 10, 20], [0, 100, 30, 200], [40, 0, 50, 20]], np.float32)
    labels = ClipLabels(clip, 1, (LabelledPerson('a', 'player', ''), LabelledPerson('b', 'player', ''),
                       LabelledPerson('c', 'non_player', '')), {'cam0': CameraLabels(np.zeros(3, np.int64),
                       np.arange(3, dtype=np.int64), boxes.astype(np.float64))}, {})
    path = tmp_path / 'labels.json'
    labels.save(path)
    video = tmp_path / 'source.mp4'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*'mp4v'), 30., (1920, 1080))
    assert writer.isOpened()
    writer.write(np.zeros((1080, 1920, 3), np.uint8))
    writer.release()
    key = clip + '/cam0'
    plan: dict[str, Any] = {'status': 'ok', 'capacity': {'largest_successful_short_side': 1080},
        'thresholds': [.01, .05, .1, .2, .3, .5], 'union_max_height_at_1080p': 64, 'dedup_iou': .5,
        'scope': 'same_saved_court_roi', 'interpretation': 'not recall', 'near_far': 'bottom rank', 'runtime_scope': 'predictor',
        'inputs': [{'clip': clip, 'camera': 'cam0', 'label_path': str(path), 'label_sha256': dual_sha256(path),
                    'roi': [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
                    'video': {'height': 1080, 'path': str(video), 'sha256': dual_sha256(video)}}], 'archives': {}}
    for name, scores, ms in [('ft_base', [.02, .9, .02], 10.), ('ft_1080', [.8, .9, .1], 20.),
                             ('ft_tiles', [.8, .1, .1], 40.), ('coco_base', [.9, .9, .9], 10.)]:
        archive = DetectionArchive(np.array([0, 3], np.int64), boxes, np.array(scores, np.float32), np.array([ms], np.float64))
        plan['archives'][name] = {key: archive.save(tmp_path / f'{name}.npz')}
    baseline = PersonDetectionOutput('cam0', np.array([0, 1], np.int64), boxes[[1]], np.array([.9], np.float32))
    monkeypatch.setattr('src.tasks.player_detection.evaluation.far_report.load_baseline', lambda _: baseline)
    (tmp_path / 'inference.json').write_text(json.dumps(plan))
    result = summarize_report(tmp_path)
    table = {row['variant']: row for row in result['table'] if row['camera'] == 'cam0' and row['near_far'] == 'far'}
    assert table['ft_s0.30']['old_box_agreement'] == 0
    assert table['ft_s0.01']['old_box_agreement'] == 1
    assert table['ft_s0.01']['added_non_player_units'] == 1
    assert table['ft_1080_s0.30']['added_non_player_units'] == 0
    assert table['ft_union_coco_h64_s0.30']['added_non_player_units'] == 1
    assert table['ft_union_coco_h64_s0.30']['ms_per_frame'] >= 20
    assert table['ft_plus_far_tiles_s0.30']['ms_per_frame'] >= 50
    summary = result['missed_old_scores']['all']
    assert summary['missed_old_player_units'] == 1
    assert summary['best_score_after_roi']['p10_p50_p90'][1] == pytest.approx(.02)
    assert (tmp_path / 'diagnosis.csv').exists() and (tmp_path / 'diagnosis.md').exists()
    assert Path(result['video']).stat().st_size > 0
    capture = cv2.VideoCapture(result['video'])
    ok, image = capture.read()
    capture.release()
    assert ok and image.shape == (680, 1280, 3)
    with pytest.raises(FileExistsError):
        summarize_report(tmp_path, render=False)
    # CPU resummarization must reject a modified raw archive before reporting it.
    record = plan['archives']['ft_base'][key]
    with Path(record['path']).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        DetectionArchive.load(record)
