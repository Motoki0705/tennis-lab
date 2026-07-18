from __future__ import annotations

import numpy as np
import torch

from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.plcs.data.tracking_dataset import PLCSTrackingDataset
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.multi_object_scene_generator import (
    MultiPersonSceneGenerator,
)
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.utils.projection.camera_projector import CameraConfig, CameraProjector


class _MotionSceneStub:
    """Small deterministic stand-in for the separately tested AMASS sampler."""

    def __init__(self) -> None:
        self.camera_projector = CameraProjector(CameraConfig(
            fixed_position_noise_radius=0.0, fixed_look_at_xy_radius=0.0
        ))
        self.calls = 0

    def generate_scene(self, scene_id: str) -> SceneData:
        offset = float(self.calls)
        self.calls += 1
        frames = 4
        position: np.ndarray = np.zeros((frames, 3), dtype=np.float32)
        position[:, 0] = (-1.0 + offset) / 5.485
        rotation: np.ndarray = np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32), (frames, 1)
        )
        skeleton: np.ndarray = np.zeros((frames, 17, 3), dtype=np.float32)
        skeleton[..., 2] = np.linspace(0.1, 1.8, 17)
        world = skeleton.copy()
        world[..., 0] += -1.0 + offset
        cameras = []
        for camera in self.camera_projector.cameras():
            uv, visible = self.camera_projector.project_points_to_uv(
                torch.from_numpy(world), camera
            )
            court_uv, court_visible = self.camera_projector.project_court_keypoints(camera)
            cameras.append(CameraData(
                camera_params={"C": camera.C.tolist(), "R": camera.R.tolist(), "f": camera.f,
                               "cx": camera.cx, "cy": camera.cy, "w": camera.w, "h": camera.h,
                               "image_size": self.camera_projector.config.image_size},
                human_kp_uv=uv.numpy(),
                court_kp_uv=np.tile(court_uv.numpy()[None], (frames, 1, 1)),
                human_kp_visible=visible.numpy(),
                court_kp_visible=np.tile(court_visible.numpy()[None], (frames, 1)),
                human_visibility_ratio=float(visible.any(-1).float().mean()),
                court_visibility_count=float(court_visible.sum()),
            ))
        return SceneData(
            meta={"scene_id": scene_id, "motion_source": "stub", "motion_category": "test",
                  "gender": "neutral", "fps": 30.0, "num_frames": frames,
                  "initial_position": (0.0, 0.0), "initial_yaw": 0.0,
                  "num_cameras_sampled": len(cameras)},
            position=position, rotation=rotation, canonical_pose_3d=skeleton,
            cameras=cameras, human_kp_3d=world,
        )


def test_multi_person_uses_motion_scenes_and_canonical_writer(tmp_path) -> None:
    scene = MultiPersonSceneGenerator(
        _MotionSceneStub(),
        timeline=TimelineConfig(
            num_frames=12,
            min_tracks=2,
            max_tracks=2,
            max_concurrent=2,
            start_index_range=(-2, 8),
            min_active_frames=2,
            overlap_probability=0.5,
            min_gap_frames=1,
            max_gap_frames=3,
        ),
    ).generate_scene("scene_000000")
    assert scene.num_persons == 2
    assert scene.person_present is not None
    assert scene.position.shape == (12, 2, 3)
    assert scene.person_present[:, :scene.num_persons].any(0).all()
    assert scene.cameras[0].human_kp_uv.shape == (12, 2, 17, 2)
    assert len(scene.track_instances) == 2
    assert not scene.cameras[0].human_kp_visible[
        ~scene.person_present
    ].any()

    writer = PLCSDatasetWriter(tmp_path)
    scene_path = writer.save_scene(scene)
    (tmp_path / "train.txt").write_text("scene_000000\n")
    assert (scene_path / "position.npy").exists()
    assert (scene_path / "cam_0_human_kp_uv.npy").exists()
    sample = PLCSTrackingDataset(
        scene_dir=tmp_path,
        split_file="train.txt",
        config={
            "data": {
                "seq_len_range": [12, 12],
                "num_views_range": [6, 6],
                "camera_mode": "first",
                "lifecycle": {"min_reuse_gap_frames": 0},
            },
            "model": {"num_queries": 2},
        },
    )[0]
    assert sample["human_kp"].shape[:4] == (6, 12, 2, 17)
    assert "bbox" not in sample
    assert 1 <= int(sample["target_slot_mask"].sum()) <= 2
    assert set(sample["target_instance_id"].unique().tolist()) == {-1, 0, 1}
    assert torch.equal(
        sample["target_rotation"][~sample["target_presence"]],
        torch.tensor([1.0, 0.0]).expand(
            (~sample["target_presence"]).sum(), 2
        ),
    )
    detection_ids = sample["detection_gt_index"][0]
    reused_columns = []
    for track_id in range(2):
        locations = torch.nonzero(detection_ids == track_id, as_tuple=False)
        reused_columns.append(locations[:, 1].unique().numel())
    assert max(reused_columns) > 1


def test_invalid_person_cardinality_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        TimelineConfig(min_tracks=0)
