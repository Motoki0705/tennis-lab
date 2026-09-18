"""CPU diagnostics distinguish image support and root versus limb residuals."""

from copy import deepcopy

import numpy as np
import pytest

from src.tennis_scene.dataset_pipeline.reprojection_audit import (
    audit_reprojection_stages,
    pose_reprojection_diagnostics,
)
from src.tennis_scene.schema import SceneResult


def make_scene() -> SceneResult:
    positions = np.broadcast_to([50.0, 40.0, 1.0], (2, 3, 3)).copy()
    joints = np.repeat(positions[:, :, None], 17, axis=2)
    return SceneResult(
        3,
        30.0,
        100,
        80,
        np.zeros((2, 3, 14, 2)),
        np.ones((2, 3, 14)),
        positions,
        np.zeros((2, 3)),
        human_kp_2d=np.broadcast_to([0.5, 0.5], (2, 2, 3, 17, 2)).copy(),
        human_kp_vis=np.ones((2, 2, 3, 17)),
        player_kp_3d=joints,
        metadata={
            "reference": {
                "camera_ids": ["cam0", "cam1"],
                "camera_fits": [
                    {"K": np.eye(3), "R": np.eye(3), "t": np.zeros(3)} for _ in range(2)
                ],
            }
        },
    )


def test_exact_image_boundaries_partition_without_hiding_outliers() -> None:
    scene = make_scene()
    assert scene.human_kp_2d is not None and scene.player_kp_3d is not None
    # Origin belongs inside; width and height belong outside. Cover each axis
    # and each source independently, including one pair where both are outside.
    scene.human_kp_2d[0, 0, 0, 0] = [0, 0]
    scene.player_kp_3d[0, 0, 0] = [0, 0, 1]
    scene.human_kp_2d[0, 0, 0, 1] = [1, 0.5]
    scene.human_kp_2d[0, 0, 0, 2] = [0.5, 1]
    scene.player_kp_3d[0, 0, 3] = [100, 40, 1]
    scene.player_kp_3d[0, 0, 4] = [50, 80, 1]
    scene.player_kp_3d[0, 0, 5] = [-1000, 40, 1]
    scene.human_kp_2d[0, 0, 0, 5] = [-0.1, 0.5]
    row = pose_reprojection_diagnostics(scene, np.ones((2, 3)))["per_camera"]["cam0"][
        "0"
    ]
    assert row["all_confident"]["count"] == 51
    assert row["both_inside_image"]["count"] == 46
    assert row["either_outside_image"]["count"] == 5
    assert row["either_outside_image"]["max"] == 990
    assert row["root_to_observed_hip_center_px"]["max"] == 0


def test_positive_weight_confidence_depth_and_finiteness_gate_each_view_player() -> (
    None
):
    scene = make_scene()
    assert scene.human_kp_vis is not None and scene.human_kp_2d is not None
    assert scene.player_kp_3d is not None
    scene.human_kp_vis[0, 0, 0, 0] = 0.299
    scene.human_kp_vis[0, 0, 0, 1] = 0.3
    scene.player_kp_3d[0, 0, 2, 2] = -1
    scene.human_kp_2d[0, 0, 0, 3] = np.nan
    mask = np.array([[0.1, 0, 0], [0, 0, 0]])
    rows = pose_reprojection_diagnostics(scene, mask)["per_camera"]
    assert rows["cam0"]["0"]["all_confident"]["count"] == 14
    assert rows["cam1"]["0"]["all_confident"]["count"] == 16
    assert rows["cam0"]["1"]["all_confident"] == dict(
        count=0, mean=None, median=None, p95=None, max=None
    )


def test_root_requires_both_hips_and_front_positive_frame_but_not_limbs() -> None:
    scene = make_scene()
    assert scene.human_kp_vis is not None and scene.player_kp_3d is not None
    scene.human_kp_vis[0, 0, 0, 11] = 0.29
    scene.human_kp_vis[0, 0, 1, 12] = 0.29
    scene.player_position[1, 0, 2] = -1
    scene.player_kp_3d[:, :, 0, 0] = 1050
    mask = np.ones((2, 3))
    mask[1, 1] = 0
    rows = pose_reprojection_diagnostics(scene, mask)["per_camera"]
    assert rows["cam0"]["0"]["root_to_observed_hip_center_px"]["count"] == 1
    assert rows["cam0"]["1"]["root_to_observed_hip_center_px"]["count"] == 1
    assert rows["cam1"]["0"]["root_to_observed_hip_center_px"]["count"] == 3
    assert rows["cam0"]["0"]["root_to_observed_hip_center_px"]["max"] == 0
    assert rows["cam0"]["0"]["all_confident"]["max"] == 1000


def test_raw_refined_share_final_mask_and_preserve_different_residuals() -> None:
    refined = make_scene()
    raw = deepcopy(refined)
    assert raw.player_kp_3d is not None
    raw.player_kp_3d[:, 0, :, 0] += 10
    raw.player_kp_3d[:, 1:, :, 0] += 10000
    result = audit_reprojection_stages(raw, refined, np.array([[1, 0, 0], [0, 0, 0]]))
    for stage in result.values():
        assert stage["per_camera"]["cam0"]["0"]["all_confident"]["count"] == 17
        assert stage["per_camera"]["cam0"]["1"]["all_confident"]["count"] == 0
    assert result["raw"]["per_camera"]["cam0"]["0"]["all_confident"]["max"] == 10
    assert result["refined"]["per_camera"]["cam0"]["0"]["all_confident"]["max"] == 0
    raw.metadata["reference"]["camera_ids"].reverse()
    with pytest.raises(ValueError, match="camera order"):
        audit_reprojection_stages(raw, refined, np.ones((2, 3)))


@pytest.mark.parametrize("kind", ["mask", "confidence", "fits", "duplicates", "matrix"])
def test_malformed_boundaries_fail_explicitly(kind: str) -> None:
    scene = make_scene()
    mask = np.ones((2, 3))
    if kind == "mask":
        mask = np.ones((3, 2))
    elif kind == "confidence":
        scene.human_kp_vis = np.ones((2, 3, 2, 17))
    elif kind == "fits":
        scene.metadata["reference"]["camera_fits"].pop()
    elif kind == "duplicates":
        scene.metadata["reference"]["camera_ids"] = ["cam0", "cam0"]
    else:
        scene.metadata["reference"]["camera_fits"][0]["K"] = np.eye(2)
    with pytest.raises(ValueError):
        pose_reprojection_diagnostics(scene, mask)
