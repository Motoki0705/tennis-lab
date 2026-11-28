"""Asymmetric joint and timeline fixtures for physical units and validity."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.post_training.dataset import Clip
from src.tasks.ball_refiner_3d.post_training.labels import Annotations
from src.tasks.ball_refiner_3d.post_training.motion import (
    PLAYER_FEATURES,
    BodySkeleton,
    MotionFeatures,
    extract_motion,
    wrist_features,
)
from src.tasks.ball_refiner_3d.post_training.runner import FrameInputs
from src.tennis_scene.schema import SceneResult
from src.utils.schema.player import SMPLH_JOINT_IDX, SMPLH_SKELETON


def _skeleton() -> BodySkeleton:
    parents: NDArray[np.int32] = np.full(22, -1, np.int32)
    root: NDArray[np.float32] = np.array([4, -2, 3], np.float32)
    rest = np.repeat(root[None], 22, axis=0)
    for parent, child in SMPLH_SKELETON:
        parents[child] = parent
    for side, sign in (("left", 1), ("right", -1)):
        for joint, distance in (("shoulder", 0.2), ("elbow", 0.5), ("wrist", 0.8)):
            rest[SMPLH_JOINT_IDX[f"{side}_{joint}"]] += (sign * distance, 0.2, 0)
    return BodySkeleton(rest, parents)


def test_fk_rotates_both_wrist_chains_and_decodes_court_placement() -> None:
    pose: NDArray[np.float32] = np.zeros((1, 63), np.float32)
    shoulder = (SMPLH_JOINT_IDX["left_shoulder"] - 1) * 3
    pose[0, shoulder + 2] = np.pi / 2
    features, wrists = wrist_features(
        pose,
        np.array([[np.pi / 2, 0, 0]], np.float32),
        np.array([np.pi / 2], np.float32),
        2,
        _skeleton(),
    )
    np.testing.assert_allclose(wrists[0], [[-1.6, 0.4, 0], [-0.4, -1.6, 0]], atol=1e-6)
    start = shoulder * 2
    np.testing.assert_allclose(
        features[0, start : start + 6], [0, -1, 0, 1, 0, 0], atol=1e-6
    )


def test_stable_slots_and_velocity_masks_do_not_cross_gaps() -> None:
    p, t = 3, 5
    roots: NDArray[np.float32] = np.zeros((p, t, 3), np.float32)
    roots[1] = [[2, 5, 1], [2.2, 4, 1], [2.2, -4, 1], [0, 0, 0], [2, -3, 1]]
    validity = np.array([[True] * t, [True, True, True, False, True], [True] * t])
    headings = validity.copy()
    headings[1, 1] = False
    bodies = np.zeros_like(headings)
    bodies[1] = headings[1]
    ball = np.tile(np.array([3, 6, 2], np.float32), (t, 1))
    ball[2] = 0
    scene = SceneResult(
        t,
        50,
        10,
        10,
        np.zeros((0, t, 14, 2), np.float32),
        np.zeros((0, t, 14), np.float32),
        roots,
        np.zeros((p, t), np.float32),
        smpl_body_pose=np.zeros((p, t, 63), np.float32),
        smpl_global_orient=np.zeros((p, t, 3), np.float32),
        player_track_ids=np.array([9, 3, 15], np.int32),
        ball_3d=ball,
        ball_3d_valid=np.array([True, True, False, True, True]),
        player_valid=validity,
        player_heading_valid=headings,
        player_smpl_valid=bodies,
        metadata={
            "player_sides": [-1, -1, 1],
            "body_placement": {"players": {"3": {"scale": 1}}},
        },
    )
    motion = extract_motion(scene, _skeleton())
    assert motion.track_ids == (3, 9, 15, None)
    assert not motion.players[:, 3].any()
    x = motion.players[:, 0]
    np.testing.assert_allclose(x[1, 126:129], [0.22, 0.2, 0.2])
    np.testing.assert_allclose(x[1, 131:134], [1, -5, 0], atol=1e-6)
    np.testing.assert_allclose(x[1, 146:149], [0.08, 0.1, 0.2])
    assert not x[1, :126].any() and not x[1, 134:146].any()
    assert not x[2, 146:155].any() and not x[3].any() and not x[4, 131:134].any()
    assert x[:, -6:].tolist() == [
        [1, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0],
    ]


@pytest.mark.parametrize(
    ("field", "message"),
    [("players", "player features"), ("root_valid", "boolean player validity")],
)
def test_bad_player_arrays_are_rejected_before_model_execution(
    field: str, message: str
) -> None:
    valid: NDArray[np.bool_] = np.ones((2, 4), dtype=np.bool_)
    motion = MotionFeatures(
        np.zeros((2, 3), np.float32),
        np.ones(2, dtype=np.bool_),
        np.zeros((2, 4, PLAYER_FEATURES), np.float32),
        valid,
        valid,
        (1, 2, 3, 4),
    )
    parts = {
        "players": motion.players[..., :2],
        "root_valid": valid.astype(np.uint8),
    }
    clip = Clip(
        "video_000/clip_001",
        60.0,
        replace(motion, **{field: parts[field]}),
        Annotations((), motion.ball_valid),
        None,
    )
    with pytest.raises(ValueError, match=message):
        FrameInputs.from_clip(clip)
