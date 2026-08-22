from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.tasks.base.generate_dataset.timeline_composer import TimelineConfig
from src.tasks.blcs.data.tracking_dataset import BLCSTrackingDataset
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCSDatasetWriter, load_scene
from src.tasks.blcs.generate_dataset.multi_object_scene_generator import (
    MultiBallSceneGenerator,
)
from src.tasks.blcs.generate_dataset.scene_generator import BLCSSceneData, CameraData
from src.tasks.blcs.generate_dataset.simulation.targeted_velocity_sampler import (
    FULL_PHYSICS_REJECTION_PREFIX,
)
from src.utils.projection.camera_projector import CameraConfig, CameraProjector
from src.utils.schema.court import NET_POST_OFFSET_X, CourtConfig

_AUGMENTATION_CONFIG = (
    Path(__file__).resolve().parents[5]
    / "src/tasks/blcs/configs/data/_augmentation.yaml"
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


class _PhysicalSceneStub:
    """Small deterministic stand-in for the separately tested rally simulator."""

    def __init__(self) -> None:
        camera = _camera_config()
        court = CourtConfig(
            net_post_offset_x=NET_POST_OFFSET_X,
            net_post_offset_x_range=None,
        )
        self.config = SimpleNamespace(camera=camera, court=court)
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
            [
                [-1.0 + offset, -3.0, 1.0],
                [0.0 + offset, 0.0, 1.5],
                [1.0 + offset, 3.0, 0.8],
            ]
        )
        projector = CameraProjector(
            self.config.camera,
            court_config=self.config.court,
        )
        cameras = []
        for camera in projector.cameras():
            view = projector.generate_camera_view(trajectory, camera)
            assert view.points_uv is not None
            assert view.points_vis is not None
            cameras.append(
                CameraData(
                    camera_params=view.camera_params,
                    ball_uv=view.points_uv.numpy(),
                    ball_vis=view.points_vis.numpy(),
                    ball_visibility_ratio=float(view.points_vis.float().mean()),
                    court_kp_uv=view.court_kp_uv.numpy(),
                    court_kp_vis=view.court_kp_vis.numpy(),
                    court_visibility_count=float(view.court_kp_vis.sum()),
                )
            )
        return BLCSSceneData(
            scene_id=scene_id,
            initial_from_cell=0,
            initial_from_side="near",
            rally_length=1,
            end_reason="test",
            winner_side=None,
            shots=[],
            ball_pos_world=trajectory,
            ball_pos_norm=trajectory / 10.0,
            ball_vel_world=torch.zeros_like(trajectory),
            cameras=cameras,
            num_cameras_sampled=len(cameras),
            fps_out=30,
            sim_fps=120,
            physics_config_dict={},
            court_config_dict={},
            num_balls=1,
        )


class _RejectOncePhysicalSceneStub(_PhysicalSceneStub):
    def __init__(self) -> None:
        super().__init__()
        self.attempts: dict[str, int] = {}

    def generate_scene(self, from_cell: int, side: str, scene_id: str) -> BLCSSceneData:
        attempt = self.attempts.get(scene_id, 0) + 1
        self.attempts[scene_id] = attempt
        if attempt == 1:
            raise RuntimeError(
                f"{FULL_PHYSICS_REJECTION_PREFIX} within the requested-side "
                "tolerance."
            )
        return super().generate_scene(from_cell, side, scene_id)


class _AlwaysRejectPhysicalSceneStub(_PhysicalSceneStub):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message
        self.attempts = 0

    def generate_scene(self, from_cell: int, side: str, scene_id: str) -> BLCSSceneData:
        del from_cell, side, scene_id
        self.attempts += 1
        raise RuntimeError(self.message)


def _multi_ball_generator(
    source: _PhysicalSceneStub,
    *,
    maximum_attempts: int = 3,
) -> MultiBallSceneGenerator:
    return MultiBallSceneGenerator(
        source,
        timeline=_timeline(),
        maximum_physics_attempts_per_object=maximum_attempts,
        rng=random.Random(2),
    )


def test_multi_ball_uses_physical_scenes_and_canonical_writer(tmp_path) -> None:
    scene = _multi_ball_generator(_PhysicalSceneStub()).generate_scene("scene_000000")
    assert scene.num_balls == 2
    assert scene.ball_present is not None
    assert scene.ball_pos_world.shape == (12, 2, 3)
    assert scene.ball_present[:, : scene.num_balls].any(0).all()
    assert scene.cameras[0].ball_uv.shape == (12, 2, 2)
    assert len(scene.track_instances) == 2
    assert not scene.cameras[0].ball_vis[~scene.ball_present.numpy()].any()

    dataset_root = tmp_path / "dataset"
    writer = BLCSDatasetWriter(dataset_root)
    scene_path = writer.save_scene(scene)
    (dataset_root / "train.txt").write_text("scene_000000\n")
    assert (scene_path / "ball_pos_world.npy").exists()
    assert (scene_path / "cam_0_ball_uv.npy").exists()
    assert (scene_path / "cam_0_ball_vis.npy").exists()
    assert (scene_path / "cam_0_court_kp_vis.npy").exists()
    assert not (scene_path / "cam_0_ball_visible.npy").exists()
    assert not (scene_path / "cam_0_court_kp_visible.npy").exists()
    loaded = load_scene(scene_path)
    assert "ball_vis" in loaded["cameras"][0]
    assert "court_kp_vis" in loaded["cameras"][0]
    assert "ball_visible" not in loaded["cameras"][0]
    assert "court_kp_visible" not in loaded["cameras"][0]
    sample = BLCSTrackingDataset(
        scene_dir=dataset_root,
        split_file="train.txt",
        config=_tracking_config(),
    )[0]
    assert sample["ball_uv"].shape == (6, 12, 2, 2)
    candidate_ids = sample["candidate_gt_index"]
    assigned = candidate_ids >= 0
    assert candidate_ids.shape == assigned.shape == (6, 12, 2)
    torch.testing.assert_close(
        candidate_ids,
        candidate_ids[:1].expand_as(candidate_ids),
    )
    torch.testing.assert_close(
        assigned,
        assigned[:1].expand_as(assigned),
    )
    assert not bool((sample["ball_vis"] & ~assigned).any())
    assert not bool((sample["clean_ball_vis"] & ~assigned).any())

    physical_presence = scene.ball_present[:, : scene.num_balls]
    view_candidate_ids = candidate_ids[0]
    candidate_physical_presence = torch.stack(
        [
            (view_candidate_ids == object_id).any(dim=1)
            for object_id in range(scene.num_balls)
        ],
        dim=1,
    )
    torch.testing.assert_close(candidate_physical_presence, physical_presence)

    target_ids = sample["target_instance_id"]
    torch.testing.assert_close(sample["target_presence"], target_ids >= 0)
    torch.testing.assert_close(
        sample["target_slot_mask"], sample["target_presence"].any(dim=0)
    )
    target_physical_presence = torch.stack(
        [(target_ids == object_id).any(dim=1) for object_id in range(scene.num_balls)],
        dim=1,
    )
    torch.testing.assert_close(target_physical_presence, physical_presence)
    assert set(target_ids.unique().tolist()) == {-1, 0, 1}

    active_ids_by_slot = [
        set(view_candidate_ids[:, slot][assigned[0, :, slot]].tolist())
        for slot in range(view_candidate_ids.shape[1])
    ]
    assert set(range(scene.num_balls)) in active_ids_by_slot

    persisted_uv = torch.from_numpy(np.load(scene_path / "cam_0_ball_uv.npy"))
    persisted_vis = torch.from_numpy(np.load(scene_path / "cam_0_ball_vis.npy"))
    safe_candidate_ids = view_candidate_ids.clamp_min(0)
    expected_uv = persisted_uv.gather(
        1,
        safe_candidate_ids.unsqueeze(-1).expand(-1, -1, 2),
    ).masked_fill(~assigned[0].unsqueeze(-1), 0.0)
    expected_vis = persisted_vis.gather(1, safe_candidate_ids) & assigned[0]
    torch.testing.assert_close(sample["clean_ball_uv"][0], expected_uv)
    torch.testing.assert_close(sample["clean_ball_vis"][0], expected_vis)

    (scene_path / "cam_0_ball_vis.npy").rename(scene_path / "cam_0_ball_visible.npy")
    with pytest.raises(FileNotFoundError, match="cam_0_ball_vis.npy"):
        load_scene(scene_path)


def test_multi_ball_resamples_only_rejected_physics_proposals(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _RejectOncePhysicalSceneStub()
    caplog.set_level("INFO")

    scene = _multi_ball_generator(source).generate_scene("scene_000000")

    assert scene.num_balls == 2
    assert source.attempts == {
        "scene_000000_ball_00": 2,
        "scene_000000_ball_01": 2,
    }
    assert caplog.text.count("Accepted BLCS physics proposal") == 2


def test_multi_ball_physics_proposal_exhaustion_is_bounded() -> None:
    source = _AlwaysRejectPhysicalSceneStub(
        f"{FULL_PHYSICS_REJECTION_PREFIX}."
    )

    with pytest.raises(RuntimeError, match="exhausted 2 bounded attempts"):
        _multi_ball_generator(source, maximum_attempts=2).generate_scene(
            "scene_000000"
        )

    assert source.attempts == 2


def test_multi_ball_does_not_retry_unexpected_runtime_errors() -> None:
    source = _AlwaysRejectPhysicalSceneStub("unexpected implementation failure")

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        _multi_ball_generator(source).generate_scene("scene_000000")

    assert source.attempts == 1


def test_invalid_ball_cardinality_is_rejected() -> None:
    with np.testing.assert_raises(ValueError):
        _timeline(min_tracks=0)
