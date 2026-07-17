from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.tasks.blcs.data.tracking_dataset import BLCSTrackingDataset
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.utils.projection.camera_projector import CameraConfig, CameraProjector
from src.utils.schema.court import CourtConfig


class _PhysicalSceneStub:
    """Small deterministic stand-in for the separately tested rally simulator."""

    def __init__(self) -> None:
        camera = CameraConfig(
            fixed_position_noise_radius=0.0, fixed_look_at_xy_radius=0.0
        )
        self.config = SimpleNamespace(camera=camera, court=CourtConfig())
        self.calls = 0

    @staticmethod
    def sample_from_cell() -> int:
        return 0

    @staticmethod
    def sample_side() -> str:
        return "near"

    def generate_scene(self, from_cell: int, side: str, scene_id: str) -> BLCSSceneData:
        del from_cell, side
        offset = float(self.calls)
        self.calls += 1
        trajectory = torch.tensor(
            [[-1.0 + offset, -3.0, 1.0], [0.0 + offset, 0.0, 1.5], [1.0 + offset, 3.0, 0.8]]
        )
        projector = CameraProjector(self.config.camera, self.config.court)
        cameras = []
        for camera in projector.cameras():
            view = projector.generate_camera_view(trajectory, camera)
            assert view.points_uv is not None
            assert view.points_visible is not None
            cameras.append(CameraData(
                camera_params=view.camera_params,
                ball_uv=view.points_uv.numpy(),
                ball_visible=view.points_visible.numpy(),
                ball_visibility_ratio=float(view.points_visible.float().mean()),
                court_kp_uv=view.court_kp_uv.numpy(),
                court_kp_visible=view.court_kp_visible.numpy(),
                court_visibility_count=float(view.court_kp_visible.sum()),
            ))
        return BLCSSceneData(
            scene_id=scene_id, initial_from_cell=0, initial_from_side="near",
            rally_length=1, end_reason="test", winner_side=None, shots=[],
            ball_pos_world=trajectory, ball_pos_norm=trajectory / 10.0,
            ball_vel_world=torch.zeros_like(trajectory), cameras=cameras,
            num_cameras_sampled=len(cameras), fps_out=30, sim_fps=120,
            physics_config_dict={}, court_config_dict={},
        )


def test_multi_ball_uses_physical_scenes_and_canonical_writer(tmp_path) -> None:
    scene = MultiBallSceneGenerator(
        _PhysicalSceneStub(), min_balls=2, max_balls=3
    ).generate_scene("scene_000000")
    assert scene.num_balls in {2, 3}
    assert scene.ball_present is not None
    assert scene.ball_pos_world.shape == (3, 3, 3)
    assert scene.ball_present[:, :scene.num_balls].all()
    assert not scene.ball_present[:, scene.num_balls:].any()
    assert scene.cameras[0].ball_uv.shape == (3, 3, 2)

    writer = BLCSDatasetWriter(tmp_path)
    scene_path = writer.save_scene(scene)
    (tmp_path / "train.txt").write_text("scene_000000\n")
    assert (scene_path / "ball_pos_world.npy").exists()
    assert (scene_path / "cam_0_ball_uv.npy").exists()
    sample = BLCSTrackingDataset(scene_dir=tmp_path, split_file="train.txt")[0]
    assert sample["ball_uv"].shape[:3] == (6, 3, 3)
    assert torch.equal(sample["target_ball_mask"], scene.ball_present.any(0))
    assert not torch.equal(
        sample["candidate_gt_index"][0, 0],
        sample["candidate_gt_index"][0, 1],
    )


def test_invalid_ball_cardinality_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        MultiBallSceneGenerator(_PhysicalSceneStub(), min_balls=0, max_balls=3)
