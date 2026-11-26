"""Verify placed-SMPL projection and explicit evidence masks with small fixtures."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.evaluation.player_pose.reference import (
    BoxQuality,
    Geometry,
    JointQuality,
    SmplRegressors,
    joint_boxes,
    project_placed_smpl,
    qualify_reference,
)
from src.tennis_scene.schema import SceneResult
from src.utils.geometry.triangulation import PinholeCamera


def test_projected_reference_applies_placement_once_and_keeps_saved_frames() -> None:
    vertices = np.tile(
        np.array([[2, 0, -1], [4, 1, 3], [1, 3, 0]], np.float32), (1, 3, 1, 1)
    )
    coco = np.tile([0, 0, 1.0], (17, 1)).astype(np.float32)
    coco[0] = [0, 0.5, 0]
    scene = SceneResult(
        num_frames=3,
        fps=60,
        width=100,
        height=80,
        court_kp=np.empty((1, 3, 14, 2), np.float32),
        court_vis=np.empty((1, 3, 14), np.float32),
        player_position=np.tile([10, -3, 2], (1, 3, 1)).astype(np.float32),
        player_yaw=np.full((1, 3), np.pi / 2, np.float32),
        smpl_vertices_local=vertices,
        smpl_global_orient=np.tile([0, 0, np.pi / 2], (1, 3, 1)).astype(np.float32),
        player_smpl_valid=np.array([[True, True, False]]),
        player_kp_3d=np.full((1, 3, 17, 3), 99.0, np.float32),
    )
    camera = PinholeCamera(
        "cam0",
        np.array([[20, 0, 40], [0, 30, 30], [0, 0, 1.0]]),
        np.eye(3),
        np.array([-10.0, 3.0, 5.0]),
    )
    inferred = np.array([[True, False, True]])
    geometry = project_placed_smpl(
        scene,
        (camera,),
        inferred,
        SmplRegressors(np.array([[1, 0, 0]], np.float32), coco),
    )
    np.testing.assert_allclose(
        geometry.joints_m[0, 0, :2], [[12.0, -2.5, 1.0], [11.0, 0.0, 3.0]], atol=1e-6
    )
    np.testing.assert_allclose(
        geometry.pixels[0, 0, 0, :2], [[140 / 3, 32.5], [42.5, 41.25]], atol=1e-6
    )
    assert geometry.inside[0, 0, 0].all()
    assert np.isnan(geometry.joints_m[0, 1:]).all()
    assert not geometry.front[0, 0, 1:].any()


def test_boxes_clip_expanded_edges_and_require_joint_count() -> None:
    pixels = np.full((2, 17, 2), np.nan)
    pixels[:, :3] = [(0, 5), (20, 25), (8, 10)]
    valid = np.isfinite(pixels).all(-1)
    valid[1, 2] = False
    boxes, keep = joint_boxes(pixels, valid, (40, 30), margin=0.10, min_joints=3)
    assert keep.tolist() == [True, False]
    assert boxes[0] == pytest.approx([0, 3, 22, 27])
    assert np.isnan(boxes[1]).all()
    expanded, _ = joint_boxes(pixels, valid, (40, 30), margin=0.50, min_joints=3)
    assert expanded[0] == pytest.approx([0, 0, 30, 30])


def test_quality_distinguishes_occluded_behind_outside_and_undetermined() -> None:
    pixels = np.tile(
        np.column_stack((np.arange(17) + 40.0, np.arange(17) + 25.0)), (2, 2, 1, 1, 1)
    )
    pixels[1, :, 0, [5, 6, 11, 12], :] = np.array(
        [[10, 10], [20, 10], [10, 20], [20, 20]]
    )[:, None, :]
    pixels[0, 0, 0, :4] = [(12, 12), (30, 15), (np.nan, np.nan), (120, 10)]
    depth = np.broadcast_to(
        np.array([6.0, 4.0])[:, None, None, None], (2, 2, 1, 17)
    ).copy()
    depth[0, 0, 0, 2] = -1
    front = depth > 0
    inside = front & ((pixels >= 0) & (pixels < (100, 80))).all(-1)
    geometry = Geometry(
        np.empty((2, 1, 17, 3), np.float32),
        pixels,
        front,
        inside,
        depth,
        np.empty((2, 2, 1, 4)),
    )
    observed: NDArray[np.bool_] = np.ones((2, 2, 1, 17), bool)
    observed[0, 0, 0, :2] = False
    observed[0, 1, 0, 4] = False
    available: NDArray[np.bool_] = np.ones((2, 1), bool)
    result = qualify_reference(geometry, available, available, observed, (100, 80))
    assert result.joint_reasons[0, 0, 0, :6].tolist() == [
        JointQuality.LIKELY_OCCLUDED,
        JointQuality.UNDETERMINED,
        JointQuality.BEHIND_CAMERA,
        JointQuality.OUTSIDE_IMAGE,
        JointQuality.INSUFFICIENT_VIEW_SUPPORT,
        JointQuality.ELIGIBLE,
    ]
    assert result.box_reasons[0, 0, 0] == BoxQuality.ELIGIBLE
    assert result.joint_view_support[0, 0, :5].tolist() == [1, 1, 2, 2, 1]
    assert np.count_nonzero(result.likely_occluded) == 1

    result = qualify_reference(
        geometry,
        available,
        available,
        observed & np.array([True, False])[None, :, None, None],
        (100, 80),
    )
    assert (result.box_reasons == BoxQuality.INSUFFICIENT_VIEW_SUPPORT).all()
