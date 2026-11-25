"""Unit tests for monocular PLCS SceneDataset, calibration noise, and batch collation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.base.generate_dataset import (
    apply_court_view_record,
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.datamodule import PLCSDataModule
from src.tasks.plcs.data.dataset import SceneDataset, collate_plcs_batch
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tasks.plcs.generate_dataset.scene_generator import CameraData, SceneData
from src.tasks.plcs.geometry import (
    CAMERA_FEATURE_DIM,
    cliff_and_local_pose_to_world_coco17,
    world_coco17_to_cliff_and_local_pose,
)
from src.utils.projection.camera_projector import (
    MIXED_LAYOUT,
    CameraConfig,
    CameraProjector,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG
from src.utils.schema.court_normalization import normalize_court_position

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def _write_mixed_synthetic_dataset(root: Path) -> Path:
    """Create a valid physical_v1 dataset with mixed fixed + broadcast cameras."""
    dataset_dir = root / "plcs_mono_test"
    contract = resolve_court_keypoint_contract("physical_v1")
    projector = CameraProjector(
        config=CameraConfig(
            z_min=3.2,
            z_max=5.0,
            hfov_deg=60.0,
            image_size=(1280, 720),
            fixed_look_at=(0.0, 0.0, 0.0),
            fixed_baseline_clear_extra=0.0,
            fixed_position_noise_radius=0.0,
            fixed_look_at_xy_radius=0.0,
            layout=MIXED_LAYOUT,
            broadcast_setback=22.0,
            broadcast_height=8.0,
            broadcast_hfov_deg=35.0,
            broadcast_look_at_y=0.0,
            broadcast_look_at_height=0.5,
            broadcast_position_noise_radius=0.0,
            broadcast_look_at_xy_radius=0.0,
            broadcast_hfov_jitter_deg=0.0,
            broadcast_setback_range=(20.0, 26.0),
            broadcast_height_range=(7.0, 10.0),
            broadcast_court_width_frac_range=(0.55, 0.75),
        ),
        court_config=STANDARD_COURT_CONFIG,
    )
    writer = PLCSDatasetWriter(dataset_dir, court_keypoint_contract=contract)
    writer.save_meta_json()

    court_3d = projector.court_kp_3d
    for idx, num_frames in enumerate((20, 28)):
        t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
        root_xyz = np.stack(
            [1.0 * np.sin(2.0 * np.pi * t), -5.0 + 2.0 * t, np.full_like(t, 0.92)],
            axis=-1,
        )
        joint_offsets = np.linspace(-0.25, 0.45, 17 * 3, dtype=np.float32).reshape(
            17, 3
        )
        joint_offsets[11] = [-0.11, 0.0, 0.0]
        joint_offsets[12] = [0.11, 0.0, 0.0]
        world_kp = (root_xyz[:, None, :] + joint_offsets[None, :, :]).astype(np.float32)
        position = np.asarray(normalize_court_position(root_xyz), dtype=np.float32)
        rotation = np.repeat(
            np.array([[1.0, 0.0]], dtype=np.float32), num_frames, axis=0
        )
        canonical = np.repeat(joint_offsets[None, :, :], num_frames, axis=0)

        cameras_data: list[CameraData] = []
        for cam_i, cam in enumerate(projector.cameras()):
            uv_t, vis_t = projector.project_points_to_uv(
                torch.from_numpy(world_kp), cam
            )
            court_uv_t, court_vis_t = projector.project_points_to_uv(court_3d, cam)
            court_view = build_court_view_record(
                camera_id=f"camera_{cam_i}",
                camera_center_court_m=cam.C.numpy(),
                contract=contract,
            )
            court_uv_arr = apply_court_view_record(
                court_uv_t.numpy(), court_view, keypoint_axis=0
            )
            court_vis_arr = apply_court_view_record(
                court_vis_t.numpy(), court_view, keypoint_axis=0
            )
            cameras_data.append(
                CameraData(
                    camera_params={
                        "C": cam.C.tolist(),
                        "R": cam.R.tolist(),
                        "f": cam.f,
                        "cx": cam.cx,
                        "cy": cam.cy,
                        "w": cam.w,
                        "h": cam.h,
                        "image_size": (cam.w, cam.h),
                    },
                    human_kp_uv=uv_t.numpy().astype(np.float32),
                    court_kp_uv=np.tile(court_uv_arr[None], (num_frames, 1, 1)).astype(
                        np.float32
                    ),
                    human_kp_vis=vis_t.numpy().astype(bool),
                    court_kp_vis=np.tile(court_vis_arr[None], (num_frames, 1)).astype(
                        bool
                    ),
                    human_visibility_ratio=float(vis_t.any(dim=-1).float().mean()),
                    court_visibility_count=float(court_vis_t.float().sum()),
                    court_view=court_view,
                )
            )

        scene = SceneData(
            meta={
                "scene_id": f"scene_{idx:06d}",
                "motion_source": f"synthetic_{idx}",
                "motion_category": "tennis",
                "gender": "neutral",
                "fps": 60.0,
                "num_frames": num_frames,
                "initial_position": [0.0, -5.0],
                "initial_yaw": 0.0,
                "num_cameras_sampled": len(cameras_data),
                "motion_source_kind": "amass_smplh_v1",
                "motion_source_id": f"src_{idx}",
            },
            position=position,
            rotation=rotation,
            canonical_pose_3d=canonical,
            cameras=cameras_data,
            num_persons=1,
            human_kp_3d=world_kp,
            court_keypoint_contract=contract,
        )
        writer.save_scene(scene)

    writer.save_meta_json()
    split_content = "scene_000000\nscene_000001\n"
    for split in ("train", "val", "test"):
        (dataset_dir / f"{split}.txt").write_text(split_content, encoding="utf-8")
    return dataset_dir


def _compose_monocular_config(data_root: Path) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["data=_monocular_sequence"])
    with open_dict(cfg):
        cfg.paths.data_root = str(data_root)
        cfg.data.scene_dir = "plcs_mono_test"
        cfg.data.seq_len_range = [12, 20]
        cfg.data.batch_size = 2
        cfg.data.num_workers = 0
        cfg.model = {
            "name": "plcs_monocular_gvhmr",
            "io": {"input_profile": "monocular"},
            "hidden_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
            "max_seq_len": 256,
            "sliding_window_len": 120,
            "enable_pva_heads": True,
            "predict_static_contact": True,
        }
    return cfg


def test_monocular_dataset_clean_and_augmented_calibration_contract(
    tmp_path: Path,
) -> None:
    _write_mixed_synthetic_dataset(tmp_path)
    cfg = _compose_monocular_config(tmp_path)
    runtime = PLCSTrainingConfig.from_config(cfg)
    assert runtime.model.input_profile == "monocular"
    assert runtime.data.values["mode"] == "monocular_sequence"

    clean_ds = SceneDataset(
        scene_dir=runtime.data.scene_dir,
        split_file="val.txt",
        config=runtime.raw,
        seed=7,
        augment=False,
    )
    sample_clean = clean_ds[0]
    assert sample_clean["human_kp"].shape[0] == 1  # single monocular view
    seq_len = int(sample_clean["human_kp"].shape[1])
    assert sample_clean["bbx_xys"].shape == (1, seq_len, 3)
    assert sample_clean["frame_observed"].shape == (1, seq_len)
    assert sample_clean["kp2d_norm"].shape == (1, seq_len, 17, 3)
    assert sample_clean["cliffcam"].shape == (1, seq_len, 3)
    assert sample_clean["cam_feat"].shape == (1, seq_len, CAMERA_FEATURE_DIM)
    # Without augmentation, input camera parameters equal exact ground-truth calibration.
    torch.testing.assert_close(sample_clean["camera_R_in"], sample_clean["camera_R"])
    torch.testing.assert_close(sample_clean["camera_C_in"], sample_clean["camera_C"])
    torch.testing.assert_close(sample_clean["camera_f_in"], sample_clean["camera_f"])

    # Clean sample supports exact CLIFF + camera extrinsics round-trip to human_kp_3d!
    pred_cam, local_pose = world_coco17_to_cliff_and_local_pose(
        sample_clean["human_kp_3d"].unsqueeze(0),
        sample_clean["bbx_xys"],
        R_w2c=sample_clean["camera_R_in"],
        C_w=sample_clean["camera_C_in"],
        fx=sample_clean["camera_f_in"],
        fy=sample_clean["camera_f_in"],
        cx0=sample_clean["camera_cx_in"],
        cy0=sample_clean["camera_cy_in"],
    )
    reconstructed_3d = cliff_and_local_pose_to_world_coco17(
        pred_cam,
        local_pose,
        sample_clean["bbx_xys"],
        R_w2c=sample_clean["camera_R_in"],
        C_w=sample_clean["camera_C_in"],
        fx=sample_clean["camera_f_in"],
        fy=sample_clean["camera_f_in"],
        cx0=sample_clean["camera_cx_in"],
        cy0=sample_clean["camera_cy_in"],
    )
    torch.testing.assert_close(
        reconstructed_3d[0], sample_clean["human_kp_3d"], atol=1e-5, rtol=1e-5
    )

    # With augmentation enabled, camera_*_in is perturbed while GT camera_* stays true,
    # and two datasets with the same seed produce bit-identical augmented samples.
    aug_ds_a = SceneDataset(
        scene_dir=runtime.data.scene_dir,
        split_file="train.txt",
        config=runtime.raw,
        seed=123,
        augment=True,
    )
    aug_ds_b = SceneDataset(
        scene_dir=runtime.data.scene_dir,
        split_file="train.txt",
        config=runtime.raw,
        seed=123,
        augment=True,
    )
    torch.manual_seed(999)
    aug_a = aug_ds_a[0]
    torch.manual_seed(999)
    aug_b = aug_ds_b[0]
    torch.testing.assert_close(aug_a["bbx_xys"], aug_b["bbx_xys"])
    torch.testing.assert_close(aug_a["camera_R_in"], aug_b["camera_R_in"])
    assert not torch.allclose(aug_a["camera_R_in"], aug_a["camera_R"])
    assert not torch.allclose(aug_a["camera_C_in"], aug_a["camera_C"])


def test_monocular_datamodule_collation_and_padding_invariants(tmp_path: Path) -> None:
    _write_mixed_synthetic_dataset(tmp_path)
    cfg = _compose_monocular_config(tmp_path)
    dm = PLCSDataModule(cfg)
    dm.setup("fit")
    batch = next(iter(dm.val_dataloader()))

    # Batch has B=2, N=1, T=max_T
    bsz, n_views, max_t = batch["padding_mask"].shape
    assert (bsz, n_views) == (2, 1)
    assert batch["bbx_xys"].shape == (2, 1, max_t, 3)
    assert batch["kp2d_norm"].shape == (2, 1, max_t, 17, 3)
    assert batch["cliffcam"].shape == (2, 1, max_t, 3)
    assert batch["cam_feat"].shape == (2, 1, max_t, CAMERA_FEATURE_DIM)

    # Create variable-length pair explicitly and verify padded frames are strictly zero/False.
    assert dm.val_dataset is not None
    s0: dict[str, Any] = dm.val_dataset[0]
    s1: dict[str, Any] = dm.val_dataset[1]
    # Truncate s0 to 8 frames to force sequence padding against s1.
    for key in (
        "human_kp",
        "court_kp",
        "human_vis",
        "court_vis",
        "padding_mask",
        "human_kp_target",
        "human_vis_target",
        "bbx_xys",
        "frame_observed",
        "kp2d_norm",
        "cliffcam",
        "cam_feat",
    ):
        s0[key] = s0[key][:, :8]
    for key in ("position", "rotation", "human_kp_3d"):
        s0[key] = s0[key][:8]

    padded = collate_plcs_batch([s0, s1])
    pad_mask_0 = padded["padding_mask"][0, 0]
    assert not pad_mask_0[:8].any()
    assert pad_mask_0[8:].all()
    assert not padded["frame_observed"][0, 0, 8:].any()
    torch.testing.assert_close(
        padded["bbx_xys"][0, 0, 8:], torch.zeros_like(padded["bbx_xys"][0, 0, 8:])
    )
    torch.testing.assert_close(
        padded["kp2d_norm"][0, 0, 8:], torch.zeros_like(padded["kp2d_norm"][0, 0, 8:])
    )
    torch.testing.assert_close(
        padded["cliffcam"][0, 0, 8:], torch.zeros_like(padded["cliffcam"][0, 0, 8:])
    )
    torch.testing.assert_close(
        padded["cam_feat"][0, 0, 8:], torch.zeros_like(padded["cam_feat"][0, 0, 8:])
    )

    # Partial monocular fields must be rejected loudly.
    broken = dict(s0)
    del broken["cliffcam"]
    with pytest.raises(ValueError, match="complete group"):
        collate_plcs_batch([broken])


def test_false_positive_joints_do_not_fabricate_bbox_on_unobserved_frames(
    tmp_path: Path,
) -> None:
    """Synthetic false-positive 2D keypoints must never create bboxes or bridge AR(1) gaps."""
    _write_mixed_synthetic_dataset(tmp_path)
    cfg = _compose_monocular_config(tmp_path)
    with open_dict(cfg):
        cfg.data.augmentation.false_positive.enabled = True
        cfg.data.augmentation.false_positive.prob = 1.0
        cfg.data.augmentation.false_positive.human_prob_absent = 1.0
        cfg.data.augmentation.false_positive.human_prob_after_dropout = 1.0
        cfg.data.augmentation.visibility_dropout.enabled = False
        cfg.data.augmentation.burst_dropout.enabled = False
        cfg.data.augmentation.edge_degradation.enabled = False

    ds = SceneDataset(
        scene_dir=Path(cfg.paths.data_root) / cfg.data.scene_dir,
        split_file="train.txt",
        config=cfg,
        seed=42,
        augment=True,
    )
    raw = ds._load_base_sample(0) if hasattr(ds, "_load_base_sample") else None
    del raw

    # Construct a base sample with a clean missing gap on frames 4..7 and inject FPs via augment_sample.
    clean_ds = SceneDataset(
        scene_dir=Path(cfg.paths.data_root) / cfg.data.scene_dir,
        split_file="train.txt",
        config=cfg,
        seed=42,
        augment=False,
    )
    base_sample = clean_ds[0]
    base_sample["human_vis"][:, 4:8, :] = False
    base_sample["human_kp"][:, 4:8, :, :] = 0.0

    augmented = ds.augment_sample(base_sample)
    # Even though false_positive flips human_vis True on frames 4..7, frame_observed & bbx_xys
    # come from pre-FP visibility, so frames 4..7 stay strictly unobserved!
    assert bool(augmented["human_vis"][:, 4:8, :].any())
    assert not bool(augmented["frame_observed"][:, 4:8].any())
    torch.testing.assert_close(
        augmented["bbx_xys"][:, 4:8], torch.zeros_like(augmented["bbx_xys"][:, 4:8])
    )
    torch.testing.assert_close(
        augmented["cliffcam"][:, 4:8], torch.zeros_like(augmented["cliffcam"][:, 4:8])
    )
    torch.testing.assert_close(
        augmented["kp2d_norm"][:, 4:8],
        torch.zeros_like(augmented["kp2d_norm"][:, 4:8]),
    )


def test_missing_human_kp_3d_rejected_instead_of_reinterpreting_coco17_as_smplh(
    tmp_path: Path,
) -> None:
    """Scenes lacking human_kp_3d must fail loudly rather than indexing 17-joint canonical_pose_3d as SMPL-H."""
    from src.tasks.plcs.data.targets import build_coco17_world_targets

    with pytest.raises(ValueError, match=">=22 joints"):
        build_coco17_world_targets(
            {
                "canonical_pose_3d": np.ones((4, 17, 3), dtype=np.float32),
                "position": np.zeros((4, 3), dtype=np.float32),
                "rotation": np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (4, 1)),
            }
        )

    dataset_dir = _write_mixed_synthetic_dataset(tmp_path)
    (dataset_dir / "scenes" / "scene_000000" / "human_kp_3d.npy").unlink()
    cfg = _compose_monocular_config(tmp_path)
    ds = SceneDataset(
        scene_dir=dataset_dir,
        split_file="val.txt",
        config=cfg,
        seed=0,
        augment=False,
    )
    with pytest.raises(ValueError, match="human_kp_3d"):
        _ = ds[0]
