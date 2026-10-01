"""Review colors, global identity names and full-frame CPU video output."""
import importlib
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
from src.utils.paths import PROJECT_ROOT


@pytest.fixture
def review(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(PROJECT_ROOT/'tests/benchmarks'))
    return importlib.import_module('person_unseen_review_video')


def test_colors_distinguish_identity_mismatch_missing_and_ambiguous(review: Any) -> None:
    mapping = {0:'A',1:'B'}
    assert review.category('A','player',0,mapping) == 'agree'
    assert review.category('A','player',1,mapping) == 'disagree'
    assert review.category('A','player',-1,mapping) == 'label_only'
    assert review.category(None,None,0,mapping) == 'prediction_only'
    assert review.category(None,None,-1,mapping) == 'unknown'
    assert review.category('X','non_player',-1,mapping) == 'agree'
    assert review.category('X','non_player',0,mapping) == 'disagree'
    rows = [{'clip':'test','camera':c,'person':p,'track_id':0} for c,p in [('cam0','A'),('cam1','B')]]
    with pytest.raises(ValueError,match='Conflicting'):
        review.saved_mapping({'tracking':{'associated':{'mapping':rows}}},'test')


def test_all_review_frames_and_raw_reference_contract(review: Any, tmp_path: Path) -> None:
    videos, arrays, cameras = [], {}, {}
    for camera in ['cam0','cam1','cam2']:
        path = tmp_path/f'{camera}.avi'
        writer = cv2.VideoWriter(str(path),cv2.VideoWriter.fourcc(*'MJPG'),12.,(64,36))
        assert writer.isOpened()
        for frame in range(4):
            writer.write(np.full((36,64,3),frame*30,np.uint8))
        writer.release()
        videos.append({'camera_id':camera,'path':str(path),'width':64,'height':36,'fps':12.,'num_frames':4})
        boxes = np.tile([8,4,24,30],(1,4,1)).astype(float)
        arrays.update({f'{camera}_boxes':boxes,f'{camera}_observed':np.ones((1,4),bool),
                       f'{camera}_ids':np.array([[0,1,0,-1]])})
        cameras[camera] = CameraLabels(np.arange(4),np.array([0,0,-1,0]),boxes[0].copy())
    labels = ClipLabels('test',4,(LabelledPerson('A','player','test'),),cameras,{})
    record = {'clip':'test','source':{'videos':videos}}
    result = review.render_clip(record,arrays,labels,{0:'A',1:'B'},tmp_path/'review.mp4','ok')
    assert result['frames_read'] == result['frames_written'] == 4
    assert result['size'] == [2880,620]
    with pytest.raises(FileExistsError):
        review.render_clip(record,arrays,labels,{0:'A'},tmp_path/'review.mp4','ok')
    arrays['cam0_boxes'][0,0] += 5
    with pytest.raises(ValueError,match='boxes differ'):
        review.render_clip(record,arrays,labels,{0:'A'},tmp_path/'invalid.mp4','ok')
