"""Reference-frame Dataset tests for BLCS camera-view CourtKP20."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.tasks.base.generate_dataset import (
    CAMERA_VIEW_V2_SELECTOR,
    apply_court_view_record,
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.data.dataset import BallTrajectoryDataset
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.utils.schema.court import COURT_KP20_HALF_TURN_INDEX
from src.utils.schema.court_normalization import (
    normalize_court_position,
    normalize_court_velocity,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def _camera(
    camera_id: str,
    center: tuple[float, float, float],
    physical_court: np.ndarray,
) -> CameraData:
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    view = build_court_view_record(
        camera_id=camera_id,
        camera_center_court_m=center,
        contract=contract,
    )
    disk = apply_court_view_record(physical_court, view, keypoint_axis=0)
    assert isinstance(disk, np.ndarray)
    return CameraData(
        camera_params={
            "R": np.eye(3).tolist(),
            "C": list(center),
            "f": 100.0,
            "cx": 50.0,
            "cy": 40.0,
            "w": 100,
            "h": 80,
        },
        ball_uv=np.full((2, 2), 0.5, dtype=np.float32),
        ball_vis=np.ones(2, dtype=np.bool_),
        ball_visibility_ratio=1.0,
        court_kp_uv=disk,
        court_kp_vis=np.ones(20, dtype=np.bool_),
        court_visibility_count=20.0,
        court_view=view,
    )


def test_v2_dataset_aligns_reordered_views_targets_velocity_and_extrinsics(
    tmp_path: Path,
) -> None:
    physical_court = np.linspace(0.01, 0.4, 40, dtype=np.float32).reshape(20, 2)
    ball_pos_world = torch.tensor(
        [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=torch.float32
    )
    ball_vel_world = torch.tensor(
        [[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], dtype=torch.float32
    )
    scene = BLCSSceneData(
        scene_id="scene_000000",
        initial_from_cell=0,
        initial_from_side="near",
        rally_length=1,
        end_reason="finished",
        winner_side=None,
        shots=[],
        ball_pos_world=ball_pos_world,
        ball_pos_norm=normalize_court_position(ball_pos_world),
        ball_vel_world=ball_vel_world,
        ball_vel_norm=normalize_court_velocity(ball_vel_world),
        cameras=[
            _camera("cam_0", (0.0, 12.0, 5.0), physical_court),
            _camera("cam_1", (0.0, -12.0, 5.0), physical_court),
        ],
        num_cameras_sampled=2,
        fps_out=30,
        sim_fps=120,
        physics_config_dict={},
        court_config_dict={},
        num_balls=1,
    )
    contract = resolve_court_keypoint_contract(CAMERA_VIEW_V2_SELECTOR)
    writer = BLCSDatasetWriter(tmp_path, court_keypoint_contract=contract)
    writer.save_scene(scene)
    writer.save_meta_json()
    (tmp_path / "test.txt").write_text("scene_000000\n", encoding="utf-8")

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train",
            overrides=["court_keypoints=camera_view_v2"],
        )
    config.data.seq_len_range = [2, 2]
    config.data.num_views_range = [2, 2]
    config.data.camera_mode = 1
    sample = BallTrajectoryDataset(
        scene_dir=tmp_path,
        split_file="test.txt",
        config=config,
        augment=False,
    )[0]

    expected_court = torch.from_numpy(
        physical_court[np.asarray(COURT_KP20_HALF_TURN_INDEX)]
    )
    torch.testing.assert_close(sample["court_kp"][0, 0], expected_court)
    torch.testing.assert_close(sample["court_kp"][1, 0], expected_court)
    provenance = sample["court_reference_provenance"]
    assert provenance.reference_camera_id == "cam_0"
    assert provenance.reference_camera_local_index == 1

    expected_position = normalize_court_position(
        scene.ball_pos_world * torch.tensor([-1.0, -1.0, 1.0])
    )
    expected_velocity = normalize_court_velocity(
        scene.ball_vel_world * torch.tensor([-1.0, -1.0, 1.0])
    )
    torch.testing.assert_close(sample["position_3d"], expected_position)
    torch.testing.assert_close(sample["velocity_3d"], expected_velocity)
    torch.testing.assert_close(
        sample["camera_R"],
        torch.diag(torch.tensor([-1.0, -1.0, 1.0])).expand(2, -1, -1),
    )
    torch.testing.assert_close(
        sample["camera_C"],
        torch.tensor([[0.0, 12.0, 5.0], [0.0, -12.0, 5.0]]),
    )
