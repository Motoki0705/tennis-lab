"""Unit tests for PLCSMonocularGVHMRModel, monocular I/O adapter, PVA losses, and metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from torch import Tensor

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.configuration import PLCSModelConfig, PLCSTrainingConfig
from src.tasks.plcs.court_keypoint_contract import court_keypoint_contract_document
from src.tasks.plcs.geometry import (
    CAMERA_FEATURE_DIM,
    SyntheticBBoxConfig,
    compute_camera_features,
    compute_cliffcam_features,
    compute_synthetic_bbx_xys,
    normalize_kp2d_bbox,
    project_world_coco17_to_pixel,
    so3_exp_map,
)
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io import PLCSInputProfile, build_plcs_model_io
from src.tasks.plcs.models.plcs_monocular_gvhmr_model import PLCSMonocularGVHMRModel
from src.tasks.plcs.training.losses import PLCSLoss, PLCSLossConfig
from src.tasks.plcs.training.metrics import PLCSMetrics
from src.utils.configuration import PathResolver
from src.utils.geometry.court_pose import canonical_pose_to_world_pose
from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.projection.camera_projector import (
    MIXED_LAYOUT,
    CameraConfig,
    CameraProjector,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG
from src.utils.schema.court_normalization import (
    add_court_coordinate_normalization,
    normalize_court_position,
)

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


def _make_projector() -> CameraProjector:
    return CameraProjector(
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


def _synthetic_monocular_inputs(
    *,
    batch_size: int = 2,
    seq_len: int = 16,
) -> dict[str, torch.Tensor]:
    projector = _make_projector()
    cams = projector.cameras()
    t = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32)

    base_offsets = torch.tensor(
        [
            [0.00, -0.08, 0.66],
            [0.03, -0.10, 0.69],
            [-0.03, -0.10, 0.69],
            [0.07, -0.06, 0.67],
            [-0.07, -0.06, 0.67],
            [0.19, 0.00, 0.48],
            [-0.19, 0.00, 0.48],
            [0.28, -0.08, 0.24],
            [-0.28, -0.08, 0.24],
            [0.33, -0.20, 0.06],
            [-0.33, -0.20, 0.06],
            [0.12, 0.00, 0.00],
            [-0.12, 0.00, 0.00],
            [0.14, -0.06, -0.43],
            [-0.14, -0.06, -0.43],
            [0.14, 0.03, -0.86],
            [-0.14, 0.03, -0.86],
        ],
        dtype=torch.float32,
    )

    world_list = []
    R_list, C_list, fx_list, cx_list, cy_list, w_list, h_list = [], [], [], [], [], [], []
    for b in range(batch_size):
        cam = cams[b % len(cams)]
        root = torch.stack(
            [
                1.2 * torch.sin(2.0 * math.pi * t + 0.3 * b),
                -5.5 + 1.5 * t,
                torch.full_like(t, 0.90),
            ],
            dim=-1,
        )
        world_list.append(root[:, None, :] + base_offsets[None, :, :])
        R_list.append(cam.R)
        C_list.append(cam.C)
        fx_list.append(cam.f)
        cx_list.append(cam.cx)
        cy_list.append(cam.cy)
        w_list.append(float(cam.w))
        h_list.append(float(cam.h))

    world_kp = torch.stack(world_list, dim=0)  # (B, T, 17, 3)
    R = torch.stack(R_list, dim=0)
    C = torch.stack(C_list, dim=0)
    fx = torch.tensor(fx_list, dtype=torch.float32)
    cx = torch.tensor(cx_list, dtype=torch.float32)
    cy = torch.tensor(cy_list, dtype=torch.float32)
    w = torch.tensor(w_list, dtype=torch.float32)
    h = torch.tensor(h_list, dtype=torch.float32)

    uv_px, in_front = project_world_coco17_to_pixel(
        world_kp, R_w2c=R, C_w=C, fx=fx, fy=fx, cx0=cx, cy0=cy
    )
    vis = in_front.float()
    bbx_xys, frame_observed = compute_synthetic_bbx_xys(
        uv_px, vis, config=SyntheticBBoxConfig(min_visible_joints=2)
    )
    kp2d_norm, _ = normalize_kp2d_bbox(uv_px, vis, bbx_xys, frame_observed)
    cliffcam = compute_cliffcam_features(
        bbx_xys, frame_observed, fx=fx, fy=fx, cx0=cx, cy0=cy
    )
    cam_feat = compute_camera_features(
        R, C, fx=fx, fy=fx, cx0=cx, cy0=cy, image_w=w, image_h=h, seq_len=seq_len
    )
    padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    pelvis = 0.5 * (world_kp[:, :, 11] + world_kp[:, :, 12])
    position = normalize_court_position(pelvis)
    transverse = world_kp[:, :, 11, :2] - world_kp[:, :, 12, :2]
    rotation = torch.nn.functional.normalize(transverse, dim=-1)

    return {
        "kp2d_norm": kp2d_norm,
        "cliffcam": cliffcam,
        "cam_feat": cam_feat,
        "bbx_xys": bbx_xys,
        "frame_observed": frame_observed,
        "padding_mask": padding_mask,
        "camera_R_in": R,
        "camera_C_in": C,
        "camera_f_in": fx,
        "camera_cx_in": cx,
        "camera_cy_in": cy,
        "camera_w": w,
        "camera_h": h,
        "human_kp_3d": world_kp,
        "position": position,
        "rotation": rotation,
        "human_kp_px": uv_px,
        "human_vis": in_front,
    }


class TestPLCSMonocularGVHMRModel:
    def test_gvhmr_architecture_no_hmr2_and_non_degenerate_skeleton_output(self) -> None:
        model = PLCSMonocularGVHMRModel(
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
            max_seq_len=512,
            sliding_window_len=120,
            enable_pva_heads=True,
            predict_static_contact=True,
        )
        model.eval()

        # Verify exact GVHMR hyperparameters and complete absence of HMR2 / imgseq parameters.
        assert len(model.blocks) == 12
        assert model.hidden_dim == 512
        assert not hasattr(model, "imgseq_embedder")
        assert all("imgseq" not in name and "hmr2" not in name for name, _ in model.named_parameters())

        inputs = _synthetic_monocular_inputs(batch_size=2, seq_len=12)
        # Include an unobserved gap at frames 5..6 on sample 0 to test occlusion handling!
        inputs["frame_observed"][0, 5:7] = False
        inputs["bbx_xys"][0, 5:7] = 0.0
        inputs["cliffcam"][0, 5:7] = 0.0
        inputs["kp2d_norm"][0, 5:7] = 0.0

        out = model(
            kp2d_norm=inputs["kp2d_norm"],
            cliffcam=inputs["cliffcam"],
            cam_feat=inputs["cam_feat"],
            bbx_xys=inputs["bbx_xys"],
            frame_observed=inputs["frame_observed"],
            padding_mask=inputs["padding_mask"],
            camera_R_in=inputs["camera_R_in"],
            camera_C_in=inputs["camera_C_in"],
            camera_f_in=inputs["camera_f_in"],
            camera_cx_in=inputs["camera_cx_in"],
            camera_cy_in=inputs["camera_cy_in"],
        )
        assert out["world_kp_3d"].shape == (2, 12, 17, 3)
        assert out["position"].shape == (2, 12, 3)
        assert out["rotation"].shape == (2, 12, 2)
        assert out["canonical_pose"].shape == (2, 12, 17, 3)
        assert out["pred_cam"].shape == (2, 12, 3)
        assert out["pred_vel_3d"].shape == (2, 12, 17, 3)
        assert out["pred_acc_3d"].shape == (2, 12, 17, 3)
        assert out["static_contact_logits"].shape == (2, 12, 4)

        # Canonical pose composed with position + rotation recovers world_kp_3d exactly!
        recomposed = canonical_pose_to_world_pose(
            out["canonical_pose"], out["position"], out["rotation"]
        )
        torch.testing.assert_close(recomposed, out["world_kp_3d"], atol=1e-5, rtol=1e-5)

        # Every non-padded skeleton frame MUST pass all 4 SkeletonValues.require_valid checks!
        for check_fn in (
            compute_bone_lengths,
            compute_joint_angles,
            compute_torsion_angles,
            compute_torso_twist,
        ):
            check_fn(out["world_kp_3d"]).require_valid("world_kp_3d")
            check_fn(out["canonical_pose"]).require_valid("canonical_pose")

    def test_strict_padding_invariance_even_beyond_sliding_window_with_gradients(self) -> None:
        """Valid frames must be invariant to trailing padding even when T_padded >> sliding_window_len."""
        torch.manual_seed(0)
        model = PLCSMonocularGVHMRModel(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            max_seq_len=256,
            sliding_window_len=8,
            enable_pva_heads=True,
            predict_static_contact=True,
        )
        # Perturb zero-init gates so attention, MLP, and Conv1d PVA paths are actively exercised!
        with torch.no_grad():
            for block in model.blocks:
                block.gate_msa.fill_(0.5)
                block.gate_mlp.fill_(0.5)
        model.eval()

        valid_len = 10
        pad_extra = 24  # > 3x sliding_window_len so trailing padded queries have all-padded windows!
        unpadded = _synthetic_monocular_inputs(batch_size=2, seq_len=valid_len)

        padded = {
            "kp2d_norm": torch.cat(
                [unpadded["kp2d_norm"], torch.zeros((2, pad_extra, 17, 3))], dim=1
            ),
            "cliffcam": torch.cat(
                [unpadded["cliffcam"], torch.zeros((2, pad_extra, 3))], dim=1
            ),
            "cam_feat": torch.cat(
                [unpadded["cam_feat"], torch.zeros((2, pad_extra, CAMERA_FEATURE_DIM))],
                dim=1,
            ),
            "bbx_xys": torch.cat(
                [unpadded["bbx_xys"], torch.zeros((2, pad_extra, 3))], dim=1
            ),
            "frame_observed": torch.cat(
                [
                    unpadded["frame_observed"],
                    torch.zeros((2, pad_extra), dtype=torch.bool),
                ],
                dim=1,
            ),
            "padding_mask": torch.cat(
                [
                    unpadded["padding_mask"],
                    torch.ones((2, pad_extra), dtype=torch.bool),
                ],
                dim=1,
            ),
            "camera_R_in": unpadded["camera_R_in"],
            "camera_C_in": unpadded["camera_C_in"],
            "camera_f_in": unpadded["camera_f_in"],
            "camera_cx_in": unpadded["camera_cx_in"],
            "camera_cy_in": unpadded["camera_cy_in"],
        }

        out_unpadded = model(
            kp2d_norm=unpadded["kp2d_norm"],
            cliffcam=unpadded["cliffcam"],
            cam_feat=unpadded["cam_feat"],
            bbx_xys=unpadded["bbx_xys"],
            frame_observed=unpadded["frame_observed"],
            padding_mask=unpadded["padding_mask"],
            camera_R_in=unpadded["camera_R_in"],
            camera_C_in=unpadded["camera_C_in"],
            camera_f_in=unpadded["camera_f_in"],
            camera_cx_in=unpadded["camera_cx_in"],
            camera_cy_in=unpadded["camera_cy_in"],
        )
        out_padded = model(**padded)

        # Ensure no NaNs anywhere (including padded positions) AND backward pass has finite gradients!
        for key, tensor in out_padded.items():
            assert torch.isfinite(tensor).all(), f"{key} contains NaN/Inf on padded batch"
            torch.testing.assert_close(
                tensor[:, :valid_len],
                out_unpadded[key],
                atol=1e-5,
                rtol=1e-5,
            )

        loss = out_padded["world_kp_3d"][:, :valid_len].sum() + out_padded["pred_vel_3d"][:, :valid_len].sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"

    def test_camera_rigid_transform_consistency(self) -> None:
        """Under a rigid world coordinate transform (R_delta, t_delta) with zero-init cam_feat gate,
        predicted world_kp_3d transforms by the exact same rigid transform."""
        model = PLCSMonocularGVHMRModel(
            hidden_dim=128,
            num_layers=2,
            num_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            max_seq_len=128,
            sliding_window_len=60,
            enable_pva_heads=False,
            predict_static_contact=False,
        )
        model.eval()
        inputs = _synthetic_monocular_inputs(batch_size=2, seq_len=2)  # also tests B == T == 2!

        out_orig = model(
            kp2d_norm=inputs["kp2d_norm"],
            cliffcam=inputs["cliffcam"],
            cam_feat=inputs["cam_feat"],
            bbx_xys=inputs["bbx_xys"],
            frame_observed=inputs["frame_observed"],
            padding_mask=inputs["padding_mask"],
            camera_R_in=inputs["camera_R_in"],
            camera_C_in=inputs["camera_C_in"],
            camera_f_in=inputs["camera_f_in"],
            camera_cx_in=inputs["camera_cx_in"],
            camera_cy_in=inputs["camera_cy_in"],
        )

        R_delta = so3_exp_map(torch.tensor([0.0, 0.0, 0.6], dtype=torch.float32))
        t_delta = torch.tensor([1.5, -2.0, 0.4], dtype=torch.float32)

        R_new = inputs["camera_R_in"] @ R_delta.T
        C_new = (inputs["camera_C_in"] @ R_delta.T) + t_delta
        cam_feat_new = compute_camera_features(
            R_new,
            C_new,
            fx=inputs["camera_f_in"],
            fy=inputs["camera_f_in"],
            cx0=inputs["camera_cx_in"],
            cy0=inputs["camera_cy_in"],
            image_w=inputs["camera_w"],
            image_h=inputs["camera_h"],
            seq_len=2,
        )

        out_transformed = model(
            kp2d_norm=inputs["kp2d_norm"],
            cliffcam=inputs["cliffcam"],
            cam_feat=cam_feat_new,
            bbx_xys=inputs["bbx_xys"],
            frame_observed=inputs["frame_observed"],
            padding_mask=inputs["padding_mask"],
            camera_R_in=R_new,
            camera_C_in=C_new,
            camera_f_in=inputs["camera_f_in"],
            camera_cx_in=inputs["camera_cx_in"],
            camera_cy_in=inputs["camera_cy_in"],
        )
        expected_world = (out_orig["world_kp_3d"] @ R_delta.T) + t_delta
        torch.testing.assert_close(
            out_transformed["world_kp_3d"], expected_world, atol=1e-5, rtol=1e-5
        )


class TestMonocularAdapterLossesMetricsAndCheckpoint:
    def test_pva_losses_and_metrics_do_not_cross_padding_or_missing_boundaries(self) -> None:
        base_loss_dict: dict[str, object] = {
            "position_weight": 1.0,
            "position_smooth_l1_beta": 1.0,
            "rotation_weight": 1.0,
            "angle_weight": 0.5,
            "position_smoothness_weight": 0.0,
            "canonical_pose_weight": 1.0,
            "canonical_pose_smooth_l1_beta": 0.1,
            "reprojection_weight": 0.0,
            "reprojection_smooth_l1_beta": 0.01,
            "joint_angle_weight": 0.0,
            "torsion_angle_weight": 0.0,
            "torso_twist_weight": 0.0,
            "bone_length_weight": 0.1,
            "joint_angle_velocity_weight": 0.0,
            "torsion_angle_velocity_weight": 0.0,
            "torso_twist_velocity_weight": 0.0,
            "joint_angle_velocity_angle_weights": None,
            "torsion_angle_velocity_angle_weights": None,
            "world_kp_3d_weight": 2.0,
            "pva_velocity_weight": 1.0,
            "pva_acceleration_weight": 0.5,
            "kinematic_velocity_consistency_weight": 0.5,
            "kinematic_acceleration_consistency_weight": 0.25,
            "static_contact_weight": 0.5,
        }
        loss_fn = PLCSLoss(PLCSLossConfig.from_dict(base_loss_dict))
        inputs = _synthetic_monocular_inputs(batch_size=2, seq_len=10)

        # Make frames 7..9 padded on sample 0, and inject huge garbage on padded frames 7..9!
        pad_mask = torch.zeros((2, 10), dtype=torch.bool)
        pad_mask[0, 7:] = True
        target_kp = inputs["human_kp_3d"].clone()
        pred_kp_clean = target_kp + 0.02
        pred_kp_dirty_pad = pred_kp_clean.clone()
        pred_kp_dirty_pad[0, 7:] = 999.0  # huge jump across padding boundary t=6 -> t=7

        fps = torch.tensor([60.0, 30.0], dtype=torch.float32)
        pred_vel = torch.zeros_like(pred_kp_clean)
        pred_acc = torch.zeros_like(pred_kp_clean)
        pred_contact = torch.zeros((2, 10, 4), dtype=torch.float32)

        terms_a = loss_fn.prepare_inputs(
            pred_position=inputs["position"],
            pred_rotation=inputs["rotation"],
            target_position=inputs["position"],
            target_rotation=inputs["rotation"],
            pred_canonical_pose=pred_kp_clean - pred_kp_clean[:, :, [11]].mean(dim=-2, keepdim=True),
            target_human_kp_3d=target_kp,
            padding_mask=pad_mask,
            pred_world_kp_3d=pred_kp_clean,
            pred_vel_3d=pred_vel,
            pred_acc_3d=pred_acc,
            pred_static_contact_logits=pred_contact,
            frame_rate_hz=fps,
        )
        terms_b = loss_fn.prepare_inputs(
            pred_position=inputs["position"],
            pred_rotation=inputs["rotation"],
            target_position=inputs["position"],
            target_rotation=inputs["rotation"],
            pred_canonical_pose=pred_kp_dirty_pad - pred_kp_dirty_pad[:, :, [11]].mean(dim=-2, keepdim=True),
            target_human_kp_3d=target_kp,
            padding_mask=pad_mask,
            pred_world_kp_3d=pred_kp_dirty_pad,
            pred_vel_3d=pred_vel,
            pred_acc_3d=pred_acc,
            pred_static_contact_logits=pred_contact,
            frame_rate_hz=fps,
        )
        out_a = loss_fn(terms_a)
        out_b = loss_fn(terms_b)
        for key in (
            "world_kp_3d",
            "pva_velocity",
            "pva_acceleration",
            "kinematic_velocity_consistency",
            "kinematic_acceleration_consistency",
            "static_contact",
            "total",
        ):
            assert key in out_a
            torch.testing.assert_close(out_a[key], out_b[key], atol=1e-6, rtol=1e-6)

        metrics = PLCSMetrics(
            position_threshold_m=0.5,
            angle_threshold_deg=15.0,
            predict_canonical_pose=False,
            predict_world_pose=True,
        )
        batch_m = metrics.update(
            inputs["position"],
            inputs["rotation"],
            inputs["position"],
            inputs["rotation"],
            padding_mask=pad_mask,
            pred_world_kp_3d=pred_kp_dirty_pad,
            target_world_kp_3d=target_kp,
            frame_rate_hz=fps,
        )
        epoch_m = metrics.compute()
        assert "world_mpjpe_m" in batch_m and "pa_mpjpe_m" in epoch_m
        assert "mpjve_mps" in epoch_m and "mpjae_mps2" in epoch_m
        assert abs(epoch_m["world_mpjpe_m"] - math.sqrt(3.0 * 0.02**2)) < 1e-4

    def test_hydra_composition_factory_and_predictor_checkpoint_roundtrip(
        self, tmp_path: Path
    ) -> None:
        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "model=monocular_gvhmr_htd",
                    "data=monocular_sequence",
                    "loss=monocular_htd",
                ],
            )
        with open_dict(cfg):
            cfg.model.hidden_dim = 64
            cfg.model.num_layers = 2
            cfg.model.num_heads = 4

        runtime = PLCSTrainingConfig.from_config(cfg)
        bound = build_plcs_model_io(runtime)
        assert bound.adapter.profile is PLCSInputProfile.MONOCULAR

        # Save minimal checkpoint and reload through PLCSPredictor.load_from_checkpoint!
        ckpt_path = tmp_path / "mono_gvhmr.ckpt"
        ckpt_payload: dict[str, object] = {
            "state_dict": {f"model.{k}": v for k, v in bound.model.state_dict().items()},
            "hyper_parameters": {"config": runtime.raw},
        }
        add_court_coordinate_normalization(ckpt_payload, artifact="PLCS checkpoint")
        from src.tasks.plcs.model_io import write_plcs_checkpoint_court_keypoints

        write_plcs_checkpoint_court_keypoints(
            ckpt_payload, resolve_court_keypoint_contract("physical_v1")
        )
        torch.save(ckpt_payload, ckpt_path)

        from src.utils.configuration.paths import RuntimePathRoots

        resolver = PathResolver(
            roots=RuntimePathRoots(
                project_root=tmp_path,
                data_root=tmp_path,
                checkpoint_root=tmp_path,
                output_root=tmp_path,
                artifact_root=tmp_path,
                cache_root=tmp_path,
                external_asset_root=tmp_path,
            )
        )
        predictor = PLCSPredictor.load_from_checkpoint(
            ckpt_path, resolver=resolver, device="cpu"
        )
        assert predictor.input_profile is PLCSInputProfile.MONOCULAR

        raw = _synthetic_monocular_inputs(batch_size=1, seq_len=6)
        pred = predictor.predict_monocular_sequence(
            kp2d_px=raw["human_kp_px"][0].numpy(),
            kp_vis=raw["human_vis"][0].float().numpy(),
            bbx_xys=raw["bbx_xys"][0].numpy(),
            frame_observed=raw["frame_observed"][0].numpy(),
            R_w2c=raw["camera_R_in"][0].numpy(),
            C_w=raw["camera_C_in"][0].numpy(),
            fx=float(raw["camera_f_in"][0].item()),
            fy=float(raw["camera_f_in"][0].item()),
            cx0=float(raw["camera_cx_in"][0].item()),
            cy0=float(raw["camera_cy_in"][0].item()),
            image_w=int(raw["camera_w"][0].item()),
            image_h=int(raw["camera_h"][0].item()),
        )
        assert pred.world_kp_3d_meters is not None
        assert pred.world_kp_3d_meters.shape == (1, 6, 17, 3)
        assert pred.position_meters.shape == (1, 6, 3)

        # Reject invalid model config mismatch loudly.
        with pytest.raises((ValueError, TypeError, KeyError)):
            PLCSModelConfig.from_mapping({"name": "plcs_monocular_gvhmr"})

    def test_bf16_autocast_accepts_128_orthonormal_rotations_and_preserves_skeleton(
        self,
    ) -> None:
        """Finding #1: bf16-mixed must run camera validation & lifting in FP32."""
        from src.tasks.plcs.geometry.monocular_camera_bbox import so3_exp_map
        from src.utils.geometry.skeleton import compute_bone_lengths

        gen = torch.Generator().manual_seed(7)
        omega = torch.randn((128, 3), generator=gen) * 1.5
        R_128 = so3_exp_map(omega)  # exact FP32 orthonormal SO(3)
        C_128 = torch.tensor([0.0, -18.0, 4.0]).expand(128, 3).clone()
        fx_128 = torch.full((128,), 1100.0)
        fy_128 = torch.full((128,), 1100.0)
        cx_128 = torch.full((128,), 640.0)
        cy_128 = torch.full((128,), 360.0)
        raw = _synthetic_monocular_inputs(batch_size=2, seq_len=5)
        model = PLCSMonocularGVHMRModel(
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            sliding_window_len=8,
            enable_pva_heads=True,
        ).eval()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # All 128 valid rotations must pass camera validation under BF16 autocast.
            kp_probe = torch.zeros((128, 1, 17, 3), dtype=torch.bfloat16)
            kp_probe[..., 11, 0] = -0.15
            kp_probe[..., 12, 0] = 0.15
            uv, in_front = project_world_coco17_to_pixel(
                kp_probe,
                R_w2c=R_128,
                C_w=C_128,
                fx=fx_128,
                fy=fy_128,
                cx0=cx_128,
                cy0=cy_128,
            )
            assert uv.dtype == torch.float32
            assert torch.isfinite(uv).all()
            assert in_front.shape == (128, 1, 17)

            out = model(
                kp2d_norm=raw["kp2d_norm"],
                cliffcam=raw["cliffcam"],
                cam_feat=raw["cam_feat"],
                bbx_xys=raw["bbx_xys"],
                frame_observed=raw["frame_observed"],
                padding_mask=raw["padding_mask"],
                camera_R_in=raw["camera_R_in"],
                camera_C_in=raw["camera_C_in"],
                camera_f_in=raw["camera_f_in"],
                camera_cx_in=raw["camera_cx_in"],
                camera_cy_in=raw["camera_cy_in"],
            )
        assert out["world_kp_3d"].dtype == torch.float32
        assert torch.isfinite(out["world_kp_3d"]).all()
        compute_bone_lengths(out["world_kp_3d"]).require_valid("bf16 world_kp_3d")

    def test_zero_init_residual_gates_identity_and_two_step_update_order(self) -> None:
        """Finding #9: gate_msa/gate_mlp start at 0 (identity), update step 1 -> attn/mlp step 2."""
        from src.tasks.plcs.models.plcs_monocular_gvhmr_model import (
            MonocularEncoderRoPEBlock,
            build_monocular_sliding_padding_mask,
        )

        block = MonocularEncoderRoPEBlock(
            hidden_size=32, num_heads=4, mlp_ratio=2.0, dropout=0.0, max_seq_len=16
        )
        assert torch.equal(block.gate_msa, torch.zeros_like(block.gate_msa))
        assert torch.equal(block.gate_mlp, torch.zeros_like(block.gate_mlp))

        pad = torch.tensor([[False, False, False, True, True]])
        obs = torch.tensor([[True, False, True, False, False]])
        active = (~pad) & obs
        attn_mask = build_monocular_sliding_padding_mask(
            pad, sliding_window_len=8, frame_observed=obs
        )
        x = torch.randn((1, 5, 32)) * active.unsqueeze(-1).float()
        y = block(x, attn_mask_4d=attn_mask, valid_3d=active.unsqueeze(-1).float())
        assert torch.equal(y[active], x[active])
        assert torch.equal(y[~active], torch.zeros_like(y[~active]))

        # Step 1 updates gate_msa/gate_mlp while internal attn/mlp stay unchanged;
        # Step 2 updates internal attn/mlp weights once gates are non-zero!
        torch.manual_seed(0)
        model = PLCSMonocularGVHMRModel(
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            dropout=0.0,
            sliding_window_len=8,
            enable_pva_heads=False,
            predict_static_contact=False,
        )
        raw = _synthetic_monocular_inputs(batch_size=1, seq_len=5)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)

        blk = model.blocks[0]
        q_before = blk.attn.query.weight.detach().clone()
        fc1_before = blk.mlp.fc1.weight.detach().clone()

        def _step() -> None:
            opt.zero_grad()
            out = model(
                kp2d_norm=raw["kp2d_norm"],
                cliffcam=raw["cliffcam"],
                cam_feat=raw["cam_feat"],
                bbx_xys=raw["bbx_xys"],
                frame_observed=raw["frame_observed"],
                padding_mask=raw["padding_mask"],
                camera_R_in=raw["camera_R_in"],
                camera_C_in=raw["camera_C_in"],
                camera_f_in=raw["camera_f_in"],
                camera_cx_in=raw["camera_cx_in"],
                camera_cy_in=raw["camera_cy_in"],
            )
            loss = ((out["world_kp_3d"] - raw["human_kp_3d"]) ** 2).mean()
            loss.backward()
            opt.step()

        _step()
        assert float(blk.gate_msa.abs().max().item()) > 0.0
        assert float(blk.gate_mlp.abs().max().item()) > 0.0
        assert torch.equal(blk.attn.query.weight.detach(), q_before)
        assert torch.equal(blk.mlp.fc1.weight.detach(), fc1_before)

        _step()
        assert not torch.equal(blk.attn.query.weight.detach(), q_before)
        assert not torch.equal(blk.mlp.fc1.weight.detach(), fc1_before)

    def test_internal_unobserved_gap_isolates_features_gradients_losses_and_metrics(
        self,
    ) -> None:
        """Finding #2: non-padding unobserved gap blocks attention, conv, loss & metric bleed."""
        from src.tasks.plcs.training.lightning_module import PLCSLightningModule

        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "model=monocular_gvhmr_htd",
                    "data=monocular_sequence",
                    "loss=monocular_htd",
                ],
            )
        with open_dict(cfg):
            cfg.model.hidden_dim = 32
            cfg.model.num_layers = 2
            cfg.model.num_heads = 4
            cfg.model.mlp_ratio = 2.0
            cfg.model.dropout = 0.0
            cfg.model.sliding_window_len = 16  # wider than full sequence!
            cfg.loss.reprojection_weight = 0.0

        module = PLCSLightningModule(cfg)
        # Activate residual gates so attention & MLP actually exchange information inside segments!
        with torch.no_grad():
            for blk in module.model.blocks:
                blk.gate_msa.fill_(0.5)
                blk.gate_mlp.fill_(0.5)
            module.model.pva_block.gate_msa.fill_(0.5)
            module.model.pva_block.gate_mlp.fill_(0.5)

        raw = _synthetic_monocular_inputs(batch_size=1, seq_len=8)
        obs = torch.tensor([[True, True, True, False, False, True, True, True]])
        pad = torch.zeros((1, 8), dtype=torch.bool)  # non-padding internal gap!

        def _make_batch(kp2d_override: Tensor) -> dict[str, object]:
            return {
                "human_kp": (raw["human_kp_px"] / 1280.0).unsqueeze(1),
                "human_vis": raw["human_vis"].unsqueeze(1),
                "court_kp": torch.zeros((1, 1, 8, 14, 2), dtype=torch.float32),
                "court_vis": torch.ones((1, 1, 8, 14), dtype=torch.bool),
                "kp2d_norm": kp2d_override.unsqueeze(1),
                "cliffcam": raw["cliffcam"].unsqueeze(1),
                "cam_feat": raw["cam_feat"].unsqueeze(1),
                "bbx_xys": raw["bbx_xys"].unsqueeze(1),
                "frame_observed": obs.unsqueeze(1),
                "padding_mask": pad.unsqueeze(1),
                "camera_R_in": raw["camera_R_in"].unsqueeze(1),
                "camera_C_in": raw["camera_C_in"].unsqueeze(1),
                "camera_f_in": raw["camera_f_in"].unsqueeze(1),
                "camera_cx_in": raw["camera_cx_in"].unsqueeze(1),
                "camera_cy_in": raw["camera_cy_in"].unsqueeze(1),
                "position": raw["position"],
                "rotation": raw["rotation"],
                "human_kp_3d": raw["human_kp_3d"],
                "frame_rate_hz": torch.tensor([60.0]),
                "court_keypoint_metadata": court_keypoint_contract_document(
                    resolve_court_keypoint_contract("physical_v1")
                ),
            }

        kp_a = raw["kp2d_norm"].clone().detach().requires_grad_(True)
        kp_b = raw["kp2d_norm"].clone().detach()
        kp_b[:, :3, :, :2] += 0.45  # corrupt left segment (t=0..2)
        kp_b[:, 3:5, :, :2] += 5.0  # corrupt gap frames (t=3..4)
        kp_b.requires_grad_(True)

        res_a = module._compute_supervised_result(_make_batch(kp_a), "val")
        res_b = module._compute_supervised_result(_make_batch(kp_b), "val")

        # 1. Right segment outputs (t=5..7) must be IDENTICAL despite left/gap corruption.
        torch.testing.assert_close(
            res_a["outputs"].world_kp_3d[:, 5:],
            res_b["outputs"].world_kp_3d[:, 5:],
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            res_a["outputs"].pred_vel_3d[:, 5:],
            res_b["outputs"].pred_vel_3d[:, 5:],
            atol=1e-5,
            rtol=1e-5,
        )

        # 2. Loss gradients on gap frames (t=3..4) must be strictly ZERO, and left
        #    target/prediction perturbations must NOT leak gradients across the gap.
        res_a["loss"].backward()
        assert kp_a.grad is not None
        assert float(kp_a.grad[:, 3:5].abs().max().item()) == 0.0

        # 3. Direct check that modifying unobserved gap predictions (t=3..4) has ZERO
        #    effect on losses AND metrics (including mpjve_mps / mpjae_mps2 windows).
        pred_kp = raw["human_kp_3d"] + 0.02
        pred_kp_gap = pred_kp.clone()
        pred_kp_gap[:, 3:5] += 50.0  # huge error inside missing gap
        pred_v = torch.zeros_like(pred_kp)
        pred_v_gap = pred_v.clone()
        pred_v_gap[:, 2:5] += 100.0  # t=2 crosses into gap (2->3), t=3,4 are inside gap
        pred_a = torch.zeros_like(pred_kp)
        pred_a_gap = pred_a.clone()
        pred_a_gap[:, 2:6] += 100.0  # only t=1 (0,1,2) and t=6 (5,6,7) are valid triples!
        pred_c = torch.zeros((1, 8, 4))
        pred_c_gap = pred_c.clone()
        pred_c_gap[:, 3:5] += 100.0

        loss_fn = module.loss_fn
        t1 = loss_fn.prepare_inputs(
            pred_position=raw["position"],
            pred_rotation=raw["rotation"],
            target_position=raw["position"],
            target_rotation=raw["rotation"],
            pred_canonical_pose=pred_kp - pred_kp[:, :, [11]].mean(dim=-2, keepdim=True),
            target_human_kp_3d=raw["human_kp_3d"],
            padding_mask=pad,
            frame_observed=obs,
            pred_world_kp_3d=pred_kp,
            pred_vel_3d=pred_v,
            pred_acc_3d=pred_a,
            pred_static_contact_logits=pred_c,
            frame_rate_hz=torch.tensor([60.0]),
        )
        t2 = loss_fn.prepare_inputs(
            pred_position=raw["position"],
            pred_rotation=raw["rotation"],
            target_position=raw["position"],
            target_rotation=raw["rotation"],
            pred_canonical_pose=pred_kp_gap - pred_kp_gap[:, :, [11]].mean(dim=-2, keepdim=True),
            target_human_kp_3d=raw["human_kp_3d"],
            padding_mask=pad,
            frame_observed=obs,
            pred_world_kp_3d=pred_kp_gap,
            pred_vel_3d=pred_v_gap,
            pred_acc_3d=pred_a_gap,
            pred_static_contact_logits=pred_c_gap,
            frame_rate_hz=torch.tensor([60.0]),
        )
        l1 = loss_fn(t1)
        l2 = loss_fn(t2)
        for key in (
            "world_kp_3d",
            "pva_velocity",
            "pva_acceleration",
            "kinematic_velocity_consistency",
            "kinematic_acceleration_consistency",
            "static_contact",
            "total",
        ):
            torch.testing.assert_close(l1[key], l2[key], atol=1e-6, rtol=1e-6)

        m1 = PLCSMetrics(
            position_threshold_m=0.5,
            angle_threshold_deg=15.0,
            predict_canonical_pose=False,
            predict_world_pose=True,
        )
        m2 = PLCSMetrics(
            position_threshold_m=0.5,
            angle_threshold_deg=15.0,
            predict_canonical_pose=False,
            predict_world_pose=True,
        )
        r1 = m1.update(
            raw["position"],
            raw["rotation"],
            raw["position"],
            raw["rotation"],
            padding_mask=pad,
            frame_observed=obs,
            pred_world_kp_3d=pred_kp,
            target_world_kp_3d=raw["human_kp_3d"],
            frame_rate_hz=torch.tensor([60.0]),
        )
        r2 = m2.update(
            raw["position"],
            raw["rotation"],
            raw["position"],
            raw["rotation"],
            padding_mask=pad,
            frame_observed=obs,
            pred_world_kp_3d=pred_kp_gap,
            target_world_kp_3d=raw["human_kp_3d"],
            frame_rate_hz=torch.tensor([60.0]),
        )
        for key in ("world_mpjpe_m", "pa_mpjpe_m", "mpjve_mps", "mpjae_mps2"):
            assert abs(r1[key] - r2[key]) < 1e-6

    def test_public_predictor_stage_b_gap_isolation_batched_and_unobserved_validity(
        self,
    ) -> None:
        """Finding #3: predict_monocular_sequence isolates gaps, handles B>1, and flags unobserved."""
        from src.tasks.plcs.inference.htd_refiner import (
            HTDRefinerConfig,
            PLCSHTDRefiner,
        )

        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "model=monocular_gvhmr_htd",
                    "data=monocular_sequence",
                    "loss=monocular_htd",
                ],
            )
        with open_dict(cfg):
            cfg.model.hidden_dim = 32
            cfg.model.num_layers = 1
            cfg.model.num_heads = 4
            cfg.model.mlp_ratio = 2.0
            cfg.model.dropout = 0.0
        runtime = PLCSTrainingConfig.from_config(cfg)
        bound = build_plcs_model_io(runtime)
        predictor = PLCSPredictor(
            model=bound.model, adapter=bound.adapter, device=torch.device("cpu")
        )
        refiner = PLCSHTDRefiner(HTDRefinerConfig(num_steps=8, lr=0.05))

        raw = _synthetic_monocular_inputs(batch_size=2, seq_len=8)
        obs_gap = np.array([True, True, True, False, False, True, True, True], dtype=bool)
        kp_a = raw["human_kp_px"][0].numpy().copy()
        kp_b = kp_a.copy()
        kp_b[:3] += 120.0  # large shift in left segment only
        vis = raw["human_vis"][0].float().numpy()

        pred_a = predictor.predict_monocular_sequence(
            kp2d_px=kp_a,
            kp_vis=vis,
            bbx_xys=None,
            frame_observed=obs_gap,
            R_w2c=raw["camera_R_in"][0].numpy(),
            C_w=raw["camera_C_in"][0].numpy(),
            fx=float(raw["camera_f_in"][0].item()),
            fy=float(raw["camera_f_in"][0].item()),
            cx0=float(raw["camera_cx_in"][0].item()),
            cy0=float(raw["camera_cy_in"][0].item()),
            image_w=int(raw["camera_w"][0].item()),
            image_h=int(raw["camera_h"][0].item()),
            htd_refiner=refiner,
        )
        pred_b = predictor.predict_monocular_sequence(
            kp2d_px=kp_b,
            kp_vis=vis,
            bbx_xys=None,
            frame_observed=obs_gap,
            R_w2c=raw["camera_R_in"][0].numpy(),
            C_w=raw["camera_C_in"][0].numpy(),
            fx=float(raw["camera_f_in"][0].item()),
            fy=float(raw["camera_f_in"][0].item()),
            cx0=float(raw["camera_cx_in"][0].item()),
            cy0=float(raw["camera_cy_in"][0].item()),
            image_w=int(raw["camera_w"][0].item()),
            image_h=int(raw["camera_h"][0].item()),
            htd_refiner=refiner,
        )
        assert pred_a.valid_frame_mask is not None
        assert np.array_equal(pred_a.valid_frame_mask[0], obs_gap)
        assert pred_a.world_kp_3d_meters is not None and pred_b.world_kp_3d_meters is not None
        # Gap frames (3..4) must be NaN; right segment (5..7) must be bit-for-bit isolated!
        assert np.isnan(pred_a.world_kp_3d_meters[0, 3:5]).all()
        np.testing.assert_allclose(
            pred_a.world_kp_3d_meters[0, 5:],
            pred_b.world_kp_3d_meters[0, 5:],
            atol=1e-4,
            rtol=1e-4,
        )

        # B=2 batched refinement must actually run Stage B on BOTH sequences!
        pred_batch_raw = predictor.predict_monocular_sequence(
            kp2d_px=raw["human_kp_px"].numpy(),
            kp_vis=raw["human_vis"].float().numpy(),
            bbx_xys=raw["bbx_xys"].numpy(),
            frame_observed=raw["frame_observed"].numpy(),
            R_w2c=raw["camera_R_in"].numpy(),
            C_w=raw["camera_C_in"].numpy(),
            fx=raw["camera_f_in"].numpy(),
            fy=raw["camera_f_in"].numpy(),
            cx0=raw["camera_cx_in"].numpy(),
            cy0=raw["camera_cy_in"].numpy(),
            image_w=int(raw["camera_w"][0].item()),
            image_h=int(raw["camera_h"][0].item()),
            htd_refiner=None,
        )
        pred_batch_ref = predictor.predict_monocular_sequence(
            kp2d_px=raw["human_kp_px"].numpy(),
            kp_vis=raw["human_vis"].float().numpy(),
            bbx_xys=raw["bbx_xys"].numpy(),
            frame_observed=raw["frame_observed"].numpy(),
            R_w2c=raw["camera_R_in"].numpy(),
            C_w=raw["camera_C_in"].numpy(),
            fx=raw["camera_f_in"].numpy(),
            fy=raw["camera_f_in"].numpy(),
            cx0=raw["camera_cx_in"].numpy(),
            cy0=raw["camera_cy_in"].numpy(),
            image_w=int(raw["camera_w"][0].item()),
            image_h=int(raw["camera_h"][0].item()),
            htd_refiner=refiner,
        )
        assert pred_batch_raw.world_kp_3d_meters is not None
        assert pred_batch_ref.world_kp_3d_meters is not None
        for b in range(2):
            assert not np.allclose(
                pred_batch_raw.world_kp_3d_meters[b],
                pred_batch_ref.world_kp_3d_meters[b],
                atol=1e-5,
            )

        # All-unobserved sequence must return valid_frame_mask=all-False and NaN skeletons.
        all_unobs: np.ndarray = np.zeros((8,), dtype=bool)
        pred_unobs = predictor.predict_monocular_sequence(
            kp2d_px=kp_a,
            kp_vis=np.zeros_like(vis),
            bbx_xys=None,
            frame_observed=all_unobs,
            R_w2c=raw["camera_R_in"][0].numpy(),
            C_w=raw["camera_C_in"][0].numpy(),
            fx=float(raw["camera_f_in"][0].item()),
            fy=float(raw["camera_f_in"][0].item()),
            cx0=float(raw["camera_cx_in"][0].item()),
            cy0=float(raw["camera_cy_in"][0].item()),
            image_w=int(raw["camera_w"][0].item()),
            image_h=int(raw["camera_h"][0].item()),
            htd_refiner=refiner,
        )
        assert pred_unobs.valid_frame_mask is not None
        assert not pred_unobs.valid_frame_mask.any()
        assert pred_unobs.world_kp_3d_meters is not None
        assert np.isnan(pred_unobs.world_kp_3d_meters).all()

    def test_legacy_main_axial_loss_config_backward_compatibility(
        self, tmp_path: Path
    ) -> None:
        """Finding #7: axial checkpoints saved on main without monocular loss keys still load."""
        from src.tasks.plcs.training.lightning_module import PLCSLightningModule

        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
            cfg = compose(config_name="train", overrides=["loss=canonical"])
        with open_dict(cfg):
            cfg.model.hidden_dim = 32
            cfg.model.num_layers = 1
            cfg.model.num_heads = 4
            cfg.model.ffn_dim = 64
            cfg.model.rope_dim = 8
            # Strip the 6 new monocular/HTD loss keys to mimic a main-branch checkpoint!
            for new_key in (
                "world_kp_3d_weight",
                "pva_velocity_weight",
                "pva_acceleration_weight",
                "kinematic_velocity_consistency_weight",
                "kinematic_acceleration_consistency_weight",
                "static_contact_weight",
            ):
                del cfg.loss[new_key]

        legacy_loss_cfg = PLCSLossConfig.from_dict(dict(cfg.loss))
        assert legacy_loss_cfg.world_kp_3d_weight == 0.0
        assert legacy_loss_cfg.pva_velocity_weight == 0.0

        import pytorch_lightning as pl

        module = PLCSLightningModule(cfg)
        ckpt_path = tmp_path / "legacy_axial.ckpt"
        payload: dict[str, object] = {
            "pytorch-lightning_version": pl.__version__,
            "state_dict": module.state_dict(),
            "hyper_parameters": {"config": cfg},
        }
        add_court_coordinate_normalization(payload, artifact="PLCS checkpoint")
        from src.tasks.plcs.model_io import write_plcs_checkpoint_court_keypoints

        write_plcs_checkpoint_court_keypoints(
            payload, resolve_court_keypoint_contract("physical_v1")
        )
        torch.save(payload, ckpt_path)
        restored = PLCSLightningModule.load_from_checkpoint(
            ckpt_path, map_location="cpu", weights_only=False
        )
        assert restored.loss_fn.config == legacy_loss_cfg

    def test_load_scene_roundtrip_into_monocular_predict_scene(
        self, tmp_path: Path
    ) -> None:
        """Finding #8: canonical load_scene() with camera.params works in monocular predict_scene()."""
        from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
        from tests.integration.tasks.plcs.model_contracts import _write_dataset

        scene_root = tmp_path / "sceneset"
        _write_dataset(scene_root)
        scene = load_scene(
            scene_root / "scenes" / "scene_0",
            court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
        )
        assert hasattr(scene.cameras[0], "params")

        with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
            cfg = compose(
                config_name="train",
                overrides=[
                    "model=monocular_gvhmr_htd",
                    "data=monocular_sequence",
                    "loss=monocular_htd",
                ],
            )
        with open_dict(cfg):
            cfg.model.hidden_dim = 32
            cfg.model.num_layers = 1
            cfg.model.num_heads = 4
            cfg.model.mlp_ratio = 2.0
        runtime = PLCSTrainingConfig.from_config(cfg)
        bound = build_plcs_model_io(runtime)
        predictor = PLCSPredictor(
            model=bound.model, adapter=bound.adapter, device=torch.device("cpu")
        )
        pred = predictor.predict_scene(scene, cameras=[0])
        assert pred.world_kp_3d is not None
        assert pred.pred_vel_3d is not None
        assert pred.pred_acc_3d is not None
        assert pred.world_kp_3d.shape[0] == 1
        assert pred.world_kp_3d.shape[2:] == (17, 3)
        assert torch.isfinite(pred.world_kp_3d).all()
