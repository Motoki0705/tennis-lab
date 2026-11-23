"""The 3D route scores on independently projected real source boxes."""

import numpy as np
import pytest

from src.tasks.person_tracking.comparison import pool_scores, score_clip
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.projection import project_3d_prediction
from src.tasks.player_association.evaluation.labels import (
    CameraLabels,
    ClipLabels,
    LabelledPerson,
)
from src.utils.geometry.triangulation import PinholeCamera


def _detections(boxes: list[list[float]], joints: list[list[list[float]]], frame: int = 0) -> DetectionFeatures:
    n = len(boxes)
    pose: np.ndarray = np.zeros((n, 17, 3), np.float32)
    pose[:, :4, :2] = joints
    pose[:, :4, 2] = 1
    return DetectionFeatures(frame, np.arange(n, dtype=np.int64) + 10 * frame, np.asarray(boxes, np.float32),
                             np.ones(n, np.float32), pose, np.zeros((n, 1), np.float32), np.zeros(n, bool))


def test_projection_uses_both_calibrations_and_preserves_box_and_global_id() -> None:
    left = PinholeCamera("left", np.array([[100, 0, 100], [0, 100, 90], [0, 0, 1.]]), np.eye(3), np.zeros(3))
    right = PinholeCamera("right", np.array([[200, 0, 200], [0, 150, 110], [0, 0, 1.]]), np.eye(3),
                          np.array([.5, .2, 0.]))
    xyz = np.full((3, 1, 17, 3), np.nan)
    xyz[0, 0, :4] = [[-.2, -.2, 2.], [.2, -.2, 2.], [-.2, .2, 2.], [.2, .2, 2.]]
    xyz[1, 0, :4] = xyz[0, 0, :4]
    xyz[2, 0, :4] = xyz[0, 0, :4] - [0., 0., 4.]
    valid: np.ndarray = np.zeros((3, 1, 17), dtype=bool)
    valid[:, :, :4] = True
    # Row 0 is the real matching crop. Row 1 has an unrelated 2D pose.
    left_box = [80., 70., 122., 113.]
    right_box = [222., 98., 284., 149.]
    candidates = {
        "left": [_detections([left_box, [150, 130, 210, 170]],
                             [[[90, 80], [110, 80], [90, 100], [110, 100]], [[175, 140]] * 4])],
        "right": [_detections([[380, 310, 440, 360], right_box],
                              [[[410, 330]] * 4, [[230, 110], [270, 110], [230, 140], [270, 140]]])],
    }
    result = project_3d_prediction({"left": left, "right": right}, (1920, 1080), candidates,
                                   np.array([11, 50, 90]), xyz, valid, np.ones((3, 1), bool))
    for name, box in (("left", left_box), ("right", right_box)):
        np.testing.assert_array_equal(result.raw[name].observed[:, 0], [True, False, False])
        np.testing.assert_array_equal(result.raw[name].boxes_xyxy[0, 0], box)
        np.testing.assert_array_equal(result.player_ids[name][:, 0], [11, -1, -1])
    labels = ClipLabels("video/clip", 1, (LabelledPerson("P1", "player", ""),),
                        {"left": CameraLabels(np.array([0]), np.array([0]), np.asarray([left_box])),
                         "right": CameraLabels(np.array([0]), np.array([0]), np.asarray([right_box]))}, {})
    scores = pool_scores([score_clip(labels, result, {"all": [0]})["all"]])
    assert scores["raw"]["idf1"] == scores["group"]["idf1"] == scores["pairs"]["f1"] == scores["coverage_3d"] == 1.


def test_projection_requires_four_common_joints_and_same_frame_grid() -> None:
    camera = PinholeCamera("left", np.eye(3), np.eye(3), np.zeros(3))
    xyz = np.ones((1, 1, 17, 3))
    valid: np.ndarray = np.zeros((1, 1, 17), bool)
    valid[..., :3] = True
    args: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = (np.array([6]), xyz, valid, np.ones((1, 1), bool))
    projection = project_3d_prediction({"left": camera}, (10, 10),
                                       {"left": [_detections([[0, 0, 2, 2]], [[[1, 1]] * 4])]}, *args)
    assert not projection.raw["left"].observed.any()
    with pytest.raises(ValueError, match="source frame in order"):
        project_3d_prediction({"left": camera}, (10, 10),
                              {"left": [_detections([[0, 0, 2, 2]], [[[1, 1]] * 4], frame=1)]}, *args)
