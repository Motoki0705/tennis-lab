from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.plcs.data.tracking_dataset import PLCSTrackingDataset
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.multi_object_scene_generator import (
    MultiPersonSceneGenerator,
)
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.utils.projection.camera_projector import CameraConfig, CameraProjector
from src.utils.schema.court import NET_POST_OFFSET_X, CourtConfig

_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "src/tasks/plcs/configs/data/_augmentation.yaml"
)


def _camera_config() -> CameraConfig:
    return CameraConfig(
        z_min=3.0,
        z_max=5.0,
        hfov_deg=60.0,
        image_size=(1280, 720),
        fixed_look_at=(0.0, 0.0, 0.0),
        fixed_baseline_clear_extra=0.0,
        fixed_position_noise_radius=0.0,
        fixed_look_at_xy_radius=0.0,
        layout="fixed",
        broadcast_setback=20.0,
        broadcast_height=7.0,
        broadcast_hfov_deg=35.0,
        broadcast_look_at_y=0.0,
        broadcast_look_at_height=0.5,
        broadcast_position_noise_radius=1.0,
        broadcast_look_at_xy_radius=1.0,
        broadcast_hfov_jitter_deg=2.0,
        broadcast_setback_range=None,
        broadcast_height_range=None,
        broadcast_court_width_frac_range=None,
    )


def _timeline(*, min_tracks: int = 2) -> TimelineConfig:
    return TimelineConfig(
        num_frames=12,
        min_tracks=min_tracks,
        max_tracks=2,
        max_concurrent=2,
        min_reuse_gap_frames=4,
        start_index_range=(-2, 8),
        min_active_frames=2,
        overlap_probability=0.5,
        min_gap_frames=1,
        max_gap_frames=3,
    )


def _tracking_config() -> dict[str, object]:
    return {
        "data": {
            "seq_len_range": [12, 12],
            "num_views_range": [6, 6],
            "camera_mode": "first",
            "lifecycle": {
                "pack_to_query_slots": True,
                "min_reuse_gap_frames": 0,
                "randomize_slots_train": False,
            },
            "augmentation": OmegaConf.load(_AUGMENTATION_CONFIG).augmentation,
        },
        "model": {"num_queries": 2},
    }


class _MotionSceneStub:
    """Small deterministic stand-in for the separately tested AMASS sampler."""

    def __init__(self) -> None:
        self.camera_projector = CameraProjector(
            _camera_config(),
            court_config=CourtConfig(
                net_post_offset_x=NET_POST_OFFSET_X,
                net_post_offset_x_range=None,
            ),
        )
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
            court_uv, court_visible = self.camera_projector.project_court_keypoints(
                camera
            )
            cameras.append(
                CameraData(
                    camera_params={
                        "C": camera.C.tolist(),
                        "R": camera.R.tolist(),
                        "f": camera.f,
                        "cx": camera.cx,
                        "cy": camera.cy,
                        "w": camera.w,
                        "h": camera.h,
                        "image_size": self.camera_projector.config.image_size,
                    },
                    human_kp_uv=uv.numpy(),
                    court_kp_uv=np.tile(court_uv.numpy()[None], (frames, 1, 1)),
                    human_kp_vis=visible.numpy(),
                    court_kp_vis=np.tile(court_visible.numpy()[None], (frames, 1)),
                    human_visibility_ratio=float(visible.any(-1).float().mean()),
                    court_visibility_count=float(court_visible.sum()),
                )
            )
        return SceneData(
            meta={
                "scene_id": scene_id,
                "motion_source": "stub",
                "motion_category": "test",
                "gender": "neutral",
                "fps": 30.0,
                "num_frames": frames,
                "initial_position": (0.0, 0.0),
                "initial_yaw": 0.0,
                "num_cameras_sampled": len(cameras),
            },
            position=position,
            rotation=rotation,
            canonical_pose_3d=skeleton,
            cameras=cameras,
            num_persons=1,
            human_kp_3d=world,
        )


def test_multi_person_uses_motion_scenes_and_canonical_writer(tmp_path) -> None:
    scene = MultiPersonSceneGenerator(
        _MotionSceneStub(),
        timeline=_timeline(),
    ).generate_scene("scene_000000")
    assert scene.num_persons == 2
    assert scene.person_present is not None
    assert scene.position.shape == (12, 2, 3)
    assert scene.person_present[:, : scene.num_persons].any(0).all()
    assert scene.cameras[0].human_kp_uv.shape == (12, 2, 17, 2)
    assert len(scene.track_instances) == 2
    assert not scene.cameras[0].human_kp_vis[~scene.person_present].any()

    dataset_root = tmp_path / "dataset"
    writer = PLCSDatasetWriter(dataset_root)
    scene_path = writer.save_scene(scene)
    (dataset_root / "train.txt").write_text("scene_000000\n")
    assert (scene_path / "position.npy").exists()
    assert (scene_path / "cam_0_human_kp_uv.npy").exists()
    assert (scene_path / "cam_0_human_kp_vis.npy").exists()
    assert (scene_path / "cam_0_court_kp_vis.npy").exists()
    assert not (scene_path / "cam_0_human_kp_visible.npy").exists()
    assert not (scene_path / "cam_0_court_kp_visible.npy").exists()
    sample = PLCSTrackingDataset(
        scene_dir=dataset_root,
        split_file="train.txt",
        config=_tracking_config(),
    )[0]
    assert sample["human_kp"].shape[:4] == (6, 12, 2, 17)
    assert "bbox" not in sample
    assert 1 <= int(sample["target_slot_mask"].sum()) <= 2
    assert set(sample["target_instance_id"].unique().tolist()) == {-1, 0, 1}
    missing_count = int((~sample["target_presence"]).sum().item())
    assert torch.equal(
        sample["target_rotation"][~sample["target_presence"]],
        torch.tensor([1.0, 0.0]).expand(missing_count, 2),
    )
    detection_ids = sample["detection_gt_index"][0]
    target_ids = sample["target_instance_id"]
    assert bool(((detection_ids == target_ids) | (detection_ids == -1)).all())
    for object_id in range(2):
        assert bool((detection_ids == object_id).any())


def test_invalid_person_cardinality_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        _timeline(min_tracks=0)
