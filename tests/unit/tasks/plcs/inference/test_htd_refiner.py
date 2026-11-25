"""Unit tests for PLCSHTDRefiner (HTD-Refine Stage B trajectory optimization)."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.plcs.geometry import project_world_coco17_to_pixel
from src.tasks.plcs.inference.htd_refiner import (
    HTDRefineCameraView,
    HTDRefinerConfig,
    PLCSHTDRefiner,
)
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


def _split_step_trajectory(seq_len: int = 40, fps: float = 60.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a tennis split-step + explosive lateral push trajectory with sharp true acceleration peak."""
    dt = 1.0 / fps
    t = torch.arange(seq_len, dtype=torch.float32) * dt
    # Sharp Gaussian velocity impulse around t = 0.33s (split-step direction change!).
    t0 = 0.33
    sigma = 0.045
    x_root = 1.2 * torch.erf((t - t0) / (math.sqrt(2.0) * sigma))
    y_root = -6.0 + 0.8 * t
    z_root = 0.90 - 0.10 * torch.exp(-0.5 * ((t - t0) / sigma) ** 2)
    root = torch.stack([x_root, y_root, z_root], dim=-1)

    offsets = torch.tensor(
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
            [0.14, 0.03, -0.88],
            [-0.14, 0.03, -0.88],
        ],
        dtype=torch.float32,
    )
    world_kp = root[:, None, :] + offsets[None, :, :]
    # Ensure ankles never go below z=0.02 in GT, and plant ankles at t=0..6 (stance phase).
    world_kp[..., 2] = world_kp[..., 2].clamp(min=0.02)
    world_kp[:7, 15, :] = world_kp[0:1, 15, :]
    world_kp[:7, 16, :] = world_kp[0:1, 16, :]

    vel_gt = torch.zeros_like(world_kp)
    vel_gt[:-1] = (world_kp[1:] - world_kp[:-1]) * fps
    vel_gt[-1] = vel_gt[-2]

    acc_gt = torch.zeros_like(world_kp)
    acc_gt[1:-1] = (world_kp[2:] - 2.0 * world_kp[1:-1] + world_kp[:-2]) * (fps**2)
    return world_kp, vel_gt, acc_gt


def _build_cameras() -> list[HTDRefineCameraView]:
    proj = CameraProjector(
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
    views: list[HTDRefineCameraView] = []
    for cam in proj.cameras()[:2]:
        views.append(
            HTDRefineCameraView(
                R_w2c=cam.R,
                C_w=cam.C,
                fx=float(cam.f),
                fy=float(cam.f),
                cx0=float(cam.cx),
                cy0=float(cam.cy),
                image_w=float(cam.w),
                image_h=float(cam.h),
                kp2d_px=torch.empty(0),
                kp_conf=torch.empty(0),
                frame_observed=torch.empty(0, dtype=torch.bool),
            )
        )
    return views


def _attach_observations(
    cam: HTDRefineCameraView, world_kp: torch.Tensor
) -> HTDRefineCameraView:
    uv_px, in_front = project_world_coco17_to_pixel(
        world_kp.unsqueeze(0),
        R_w2c=cam.R_w2c.unsqueeze(0),
        C_w=cam.C_w.unsqueeze(0),
        fx=torch.tensor([cam.fx]),
        fy=torch.tensor([cam.fy]),
        cx0=torch.tensor([cam.cx0]),
        cy0=torch.tensor([cam.cy0]),
    )
    seq_len = world_kp.shape[0]
    return HTDRefineCameraView(
        R_w2c=cam.R_w2c,
        C_w=cam.C_w,
        fx=cam.fx,
        fy=cam.fy,
        cx0=cam.cx0,
        cy0=cam.cy0,
        image_w=cam.image_w,
        image_h=cam.image_h,
        kp2d_px=uv_px[0],
        kp_conf=in_front[0].float() * 0.95,
        frame_observed=torch.ones((seq_len,), dtype=torch.bool),
    )


class TestPLCSHTDRefiner:
    def test_suppresses_jitter_while_retaining_split_step_peak_acceleration_and_bone_invariance(
        self,
    ) -> None:
        fps = 60.0
        gt_kp, gt_vel, gt_acc = _split_step_trajectory(seq_len=40, fps=fps)
        cams = _build_cameras()
        mono_view = _attach_observations(cams[0], gt_kp)

        # Inject per-frame 3D depth/pose jitter + ground penetration (-8 cm) on stance ankles.
        gen = torch.Generator().manual_seed(77)
        jitter = 0.045 * torch.randn(gt_kp.shape, generator=gen)
        noisy_init = gt_kp + jitter
        noisy_init[:7, 15:17, 2] = -0.08  # underground penetration on planted feet!

        contact_logits = torch.full((40, 4), -6.0, dtype=torch.float32)
        contact_logits[:7, 0:2] = 6.0  # left & right ankles in static contact during stance!

        refiner = PLCSHTDRefiner(
            HTDRefinerConfig(
                num_steps=80,
                lr=0.04,
                vel_weight=1.0,
                acc_weight=0.15,
                reproj_weight=1.5,
                bone_var_weight=8.0,
                bone_sym_weight=4.0,
                contact_weight=4.0,
                ground_penetration_weight=10.0,
                jerk_weight=1e-4,
                init_anchor_weight=0.02,
            )
        )
        result = refiner.refine(
            world_kp_init=noisy_init,
            pred_vel_3d=gt_vel,
            pred_acc_3d=gt_acc,
            static_contact_logits=contact_logits,
            views=[mono_view],
            fps=fps,
        )
        refined = result.world_kp_3d

        # 1. W-MPJPE and velocity/acceleration error drop substantially vs noisy init!
        init_mpjpe = (noisy_init - gt_kp).norm(dim=-1).mean().item()
        ref_mpjpe = (refined - gt_kp).norm(dim=-1).mean().item()
        assert ref_mpjpe < 0.55 * init_mpjpe

        init_vel = (noisy_init[1:] - noisy_init[:-1]) * fps
        ref_vel = (refined[1:] - refined[:-1]) * fps
        init_mpjve = (init_vel - gt_vel[:-1]).norm(dim=-1).mean().item()
        ref_mpjve = (ref_vel - gt_vel[:-1]).norm(dim=-1).mean().item()
        assert ref_mpjve < 0.35 * init_mpjve

        # 2. Split-step peak pelvis acceleration is preserved (not over-smoothed away!).
        gt_pelvis_acc = (
            (gt_kp[2:, 11:13].mean(-2) - 2 * gt_kp[1:-1, 11:13].mean(-2) + gt_kp[:-2, 11:13].mean(-2))
            * (fps**2)
        ).norm(dim=-1)
        ref_pelvis_acc = (
            (refined[2:, 11:13].mean(-2) - 2 * refined[1:-1, 11:13].mean(-2) + refined[:-2, 11:13].mean(-2))
            * (fps**2)
        ).norm(dim=-1)
        peak_idx = int(gt_pelvis_acc.argmax().item())
        retention = float((ref_pelvis_acc[peak_idx] / gt_pelvis_acc[peak_idx]).item())
        assert 0.75 <= retention <= 1.25

        # 3. Bone-length temporal std decreases and ground penetration is eliminated!
        init_bone_std = compute_bone_lengths(noisy_init).values.std(dim=0).mean().item()
        ref_bone_std = compute_bone_lengths(refined).values.std(dim=0).mean().item()
        assert ref_bone_std < 0.5 * init_bone_std
        assert float(refined[:7, 15:17, 2].min().item()) >= -0.005

        # 4. Refined skeleton passes all 4 SkeletonValues.require_valid checks!
        for check_fn in (
            compute_bone_lengths,
            compute_joint_angles,
            compute_torsion_angles,
            compute_torso_twist,
        ):
            check_fn(refined).require_valid("refined world_kp_3d")

    def test_missing_gap_and_padding_boundaries_are_strictly_decoupled(self) -> None:
        """Perturbing segment 1 before an unobserved/invalid gap must have zero effect on segment 2!"""
        fps = 60.0
        gt_kp, gt_vel, gt_acc = _split_step_trajectory(seq_len=24, fps=fps)
        cams = _build_cameras()
        view = _attach_observations(cams[0], gt_kp)

        # Mark frames 10..13 as unobserved/invalid track gap, and frames 20..23 as padded!
        valid_mask = torch.ones((24,), dtype=torch.bool)
        valid_mask[10:14] = False
        valid_mask[20:] = False
        obs_mask = view.frame_observed.clone()
        obs_mask[10:14] = False
        obs_mask[20:] = False
        view_gapped = HTDRefineCameraView(
            R_w2c=view.R_w2c,
            C_w=view.C_w,
            fx=view.fx,
            fy=view.fy,
            cx0=view.cx0,
            cy0=view.cy0,
            image_w=view.image_w,
            image_h=view.image_h,
            kp2d_px=view.kp2d_px,
            kp_conf=view.kp_conf,
            frame_observed=obs_mask,
        )

        init_a = gt_kp.clone()
        init_b = gt_kp.clone()
        # Huge corruption on segment 0 (t=0..9) and inside invalid/padded regions (t=10..13, 20..23):
        init_b[:10] += 3.0
        init_b[10:14] = -50.0
        init_b[20:] = 99.0

        refiner = PLCSHTDRefiner(HTDRefinerConfig(num_steps=30, lr=0.03, bone_var_per_segment=True))
        res_a = refiner.refine(
            world_kp_init=init_a,
            pred_vel_3d=gt_vel,
            pred_acc_3d=gt_acc,
            static_contact_logits=None,
            views=[view_gapped],
            fps=fps,
            valid_frame_mask=valid_mask,
        )
        res_b = refiner.refine(
            world_kp_init=init_b,
            pred_vel_3d=gt_vel,
            pred_acc_3d=gt_acc,
            static_contact_logits=None,
            views=[view_gapped],
            fps=fps,
            valid_frame_mask=valid_mask,
        )
        # Segment 2 (frames 14..19) must be IDENTICAL between res_a and res_b!
        torch.testing.assert_close(
            res_a.world_kp_3d[14:20], res_b.world_kp_3d[14:20], atol=1e-5, rtol=1e-5
        )
        # Invalid/padded frames are untouched / explicitly marked False in valid_frame_mask.
        assert torch.equal(res_b.valid_frame_mask, valid_mask)

    def test_multiview_reprojection_improves_depth_further_without_retraining(self) -> None:
        fps = 60.0
        gt_kp, gt_vel, gt_acc = _split_step_trajectory(seq_len=20, fps=fps)
        cams = _build_cameras()
        view0 = _attach_observations(cams[0], gt_kp)
        view1 = _attach_observations(cams[1], gt_kp)

        # Corrupt initial trajectory along view 0's optical depth direction so monocular
        # reprojection alone is weakly sensitive, whereas adding view 1 resolves depth!
        ray_dir = gt_kp - cams[0].C_w.view(1, 1, 3)
        ray_unit = ray_dir / ray_dir.norm(dim=-1, keepdim=True)
        depth_biased_init = gt_kp + 0.35 * ray_unit

        refiner = PLCSHTDRefiner(HTDRefinerConfig(num_steps=60, lr=0.05, init_anchor_weight=0.0))
        mono_res = refiner.refine(
            world_kp_init=depth_biased_init,
            pred_vel_3d=gt_vel,
            pred_acc_3d=gt_acc,
            static_contact_logits=None,
            views=[view0],
            fps=fps,
        )
        multi_res = refiner.refine(
            world_kp_init=depth_biased_init,
            pred_vel_3d=gt_vel,
            pred_acc_3d=gt_acc,
            static_contact_logits=None,
            views=[view0, view1],
            fps=fps,
        )
        mono_err = (mono_res.world_kp_3d - gt_kp).norm(dim=-1).mean().item()
        multi_err = (multi_res.world_kp_3d - gt_kp).norm(dim=-1).mean().item()
        assert multi_err < 0.5 * mono_err

        # Uncalibrated camera in HTDRefineCameraView must fail immediately!
        with pytest.raises(ValueError, match="orthonormal"):
            HTDRefineCameraView(
                R_w2c=torch.zeros((3, 3)),
                C_w=cams[0].C_w,
                fx=cams[0].fx,
                fy=cams[0].fy,
                cx0=cams[0].cx0,
                cy0=cams[0].cy0,
                image_w=cams[0].image_w,
                image_h=cams[0].image_h,
                kp2d_px=view0.kp2d_px,
                kp_conf=view0.kp_conf,
                frame_observed=view0.frame_observed,
            )
