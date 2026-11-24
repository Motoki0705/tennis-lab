"""Unit tests for monocular camera, synthetic bbox, and CLIFF coordinate transforms."""

from __future__ import annotations

import math

import pytest
import torch

from src.tasks.plcs.geometry import (
    CAMERA_FEATURE_DIM,
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    BBoxAugmentationConfig,
    CalibrationNoiseConfig,
    SyntheticBBoxConfig,
    augment_bbx_xys,
    cliff_and_local_pose_to_world_coco17,
    compute_camera_features,
    compute_cliffcam_features,
    compute_synthetic_bbx_xys,
    normalize_kp2d_bbox,
    perturb_camera_calibration,
    project_world_coco17_to_pixel,
    so3_exp_map,
    validate_pinhole_camera_tensors,
    world_coco17_to_cliff_and_local_pose,
)
from src.utils.projection.camera_projector import (
    MIXED_LAYOUT,
    CameraConfig,
    CameraProjector,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG


def _build_mixed_projector() -> CameraProjector:
    cfg = CameraConfig(
        z_min=3.2,
        z_max=5.0,
        hfov_deg=60.0,
        image_size=(1280, 720),
        fixed_look_at=(0.0, 0.0, 0.0),
        fixed_baseline_clear_extra=0.0,
        fixed_position_noise_radius=0.0,
        fixed_look_at_xy_radius=0.0,
        layout=MIXED_LAYOUT,
        broadcast_setback=24.0,
        broadcast_height=8.5,
        broadcast_hfov_deg=35.0,
        broadcast_look_at_y=0.0,
        broadcast_look_at_height=0.5,
        broadcast_position_noise_radius=0.0,
        broadcast_look_at_xy_radius=0.0,
        broadcast_hfov_jitter_deg=0.0,
        broadcast_setback_range=(22.0, 28.0),
        broadcast_height_range=(7.0, 10.0),
        broadcast_court_width_frac_range=(0.55, 0.75),
    )
    return CameraProjector(config=cfg, court_config=STANDARD_COURT_CONFIG)


def _sample_court_coco17_sequence(seq_len: int = 16) -> torch.Tensor:
    """Create a realistic moving 17-joint court-space sequence (T, 17, 3) [m]."""
    t = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32)
    root = torch.stack(
        [
            1.5 * torch.sin(2.0 * math.pi * t),
            -6.0 + 2.0 * t,
            torch.full_like(t, 0.92),
        ],
        dim=-1,
    )
    offsets = torch.tensor(
        [
            [0.00, 0.05, 0.68],   # 0 nose
            [-0.03, 0.06, 0.71],  # 1 left_eye
            [0.03, 0.06, 0.71],   # 2 right_eye
            [-0.07, 0.02, 0.69],  # 3 left_ear
            [0.07, 0.02, 0.69],   # 4 right_ear
            [-0.19, 0.00, 0.48],  # 5 left_shoulder
            [0.19, 0.00, 0.48],   # 6 right_shoulder
            [-0.28, 0.08, 0.24],  # 7 left_elbow
            [0.28, 0.08, 0.24],   # 8 right_elbow
            [-0.34, 0.18, 0.05],  # 9 left_wrist
            [0.34, 0.18, 0.05],   # 10 right_wrist
            [-0.11, 0.00, 0.00],  # 11 left_hip
            [0.11, 0.00, 0.00],   # 12 right_hip
            [-0.13, 0.04, -0.44], # 13 left_knee
            [0.13, 0.04, -0.44],  # 14 right_knee
            [-0.13, -0.02, -0.88],# 15 left_ankle
            [0.13, -0.02, -0.88], # 16 right_ankle
        ],
        dtype=torch.float32,
    )
    swing = 0.08 * torch.sin(4.0 * math.pi * t)[:, None, None]
    return root[:, None, :] + offsets[None, :, :] + swing


class TestSyntheticBBoxAndNormalization:
    def test_invisible_zeroed_joints_do_not_pull_bbox_to_origin(self) -> None:
        kp_px = torch.zeros((1, 4, 17, 2), dtype=torch.float32)
        kp_vis = torch.zeros((1, 4, 17), dtype=torch.float32)

        # Frame 0: visible cluster near (600, 350), other 13 joints zeroed at (0, 0).
        kp_px[0, 0, [5, 6, 11, 12]] = torch.tensor(
            [[560.0, 280.0], [640.0, 280.0], [575.0, 420.0], [625.0, 420.0]]
        )
        kp_vis[0, 0, [5, 6, 11, 12]] = 1.0

        # Frame 1: only 1 visible joint (< min_visible_joints=2) -> unobserved.
        kp_px[0, 1, 0] = torch.tensor([500.0, 300.0])
        kp_vis[0, 1, 0] = 1.0

        # Frame 2: completely missing.
        # Frame 3: wide horizontal span (240 px wide, 80 px tall) -> 3:4 aspect fit.
        kp_px[0, 3, [9, 10, 11]] = torch.tensor(
            [[480.0, 320.0], [720.0, 320.0], [600.0, 400.0]]
        )
        kp_vis[0, 3, [9, 10, 11]] = 0.9

        bbx_xys, observed = compute_synthetic_bbx_xys(kp_px, kp_vis)
        assert observed.tolist() == [[True, False, False, True]]

        # Frame 0 center is exact midpoint of visible joints (600, 350); height=140 -> 140*1.2=168
        torch.testing.assert_close(
            bbx_xys[0, 0], torch.tensor([600.0, 350.0, 168.0]), atol=1e-4, rtol=0.0
        )
        # Unobserved frames are strictly zero.
        torch.testing.assert_close(bbx_xys[0, 1], torch.zeros(3))
        torch.testing.assert_close(bbx_xys[0, 2], torch.zeros(3))
        # Frame 3: width=240 > 0.75*80 -> fitted_h = 240 / 0.75 = 320 -> size = 320 * 1.2 = 384
        torch.testing.assert_close(
            bbx_xys[0, 3], torch.tensor([600.0, 360.0, 384.0]), atol=1e-4, rtol=0.0
        )

        kp_norm, joint_vis = normalize_kp2d_bbox(kp_px, kp_vis, bbx_xys, observed)
        assert not joint_vis[0, 1].any()
        assert not joint_vis[0, 2].any()
        torch.testing.assert_close(kp_norm[0, 1], torch.zeros((17, 3)))
        # Visible joints lie strictly inside [-1, 1], invisible joints are [0, 0, 0].
        vis0 = joint_vis[0, 0]
        assert vis0.sum().item() == 4
        assert bool((kp_norm[0, 0, vis0, :2].abs() <= 1.0 + 1e-6).all())
        torch.testing.assert_close(kp_norm[0, 0, ~vis0], torch.zeros((13, 3)))

    def test_partial_crop_masks_out_of_box_joints_cleanly(self) -> None:
        kp_px = torch.tensor(
            [[[[500.0, 200.0], [500.0, 400.0], [500.0, 520.0]]]], dtype=torch.float32
        ).repeat(1, 1, 6, 1)[..., :17, :]
        kp_vis = torch.ones((1, 1, 17), dtype=torch.float32)
        # Tight box centered at (500, 300) with size 220 excludes y=520.
        bbx = torch.tensor([[[500.0, 300.0, 220.0]]], dtype=torch.float32)
        obs = torch.tensor([[True]])

        kp_norm, joint_vis = normalize_kp2d_bbox(kp_px, kp_vis, bbx, obs)
        # Joint index 2 is at y=520 > 300 + 110 = 410 -> masked out and zeroed.
        assert not bool(joint_vis[0, 0, 2])
        torch.testing.assert_close(kp_norm[0, 0, 2], torch.zeros(3))
        assert bool(joint_vis[0, 0, 0]) and bool(joint_vis[0, 0, 1])

    def test_bbox_augmentation_drift_resets_across_missing_gaps(self) -> None:
        bbx_a = torch.tensor(
            [[[600.0, 360.0, 200.0]] * 8], dtype=torch.float32
        )
        bbx_b = bbx_a.clone()
        # Different values before the unobserved gap at t=3..4 must not change drift state
        # accumulation semantics, and unobserved frames must remain strictly zero.
        obs = torch.tensor([[True, True, True, False, False, True, True, True]])
        bbx_a[0, ~obs[0]] = 0.0
        bbx_b[0, ~obs[0]] = 0.0
        bbx_b[0, :3, 0] += 150.0

        cfg = BBoxAugmentationConfig(
            enabled=True,
            prob=1.0,
            scale_range=(1.1, 1.25),
            shift_xy_frac=0.05,
            scale_drift_std=0.04,
            center_drift_frac_std=0.04,
            drift_decay=0.9,
            partial_crop_prob=0.0,
        )
        gen_a = torch.Generator().manual_seed(1234)
        gen_b = torch.Generator().manual_seed(1234)
        aug_a = augment_bbx_xys(bbx_a, obs, config=cfg, generator=gen_a)
        aug_b = augment_bbx_xys(bbx_b, obs, config=cfg, generator=gen_b)

        torch.testing.assert_close(aug_a[0, 3:5], torch.zeros((2, 3)))
        # Post-gap frames (t=5..7) have identical inputs and reset drift -> identical outputs!
        torch.testing.assert_close(aug_a[0, 5:], aug_b[0, 5:], atol=1e-6, rtol=0.0)


class TestCLIFFWorldRoundTripAndProjection:
    def test_cliff_world_roundtrip_across_fixed_and_broadcast_cameras(self) -> None:
        projector = _build_mixed_projector()
        cameras = projector.cameras()
        assert len(cameras) == 8  # 6 fixed + 2 broadcast

        world_kp = _sample_court_coco17_sequence(seq_len=12)
        for cam in cameras:
            uv_norm, vis = projector.project_points_to_uv(world_kp, cam)
            wh = torch.tensor([float(cam.w), float(cam.h)], dtype=torch.float32)
            kp_px = uv_norm * wh
            bbx_xys, obs = compute_synthetic_bbx_xys(
                kp_px.unsqueeze(0),
                vis.float().unsqueeze(0),
                config=SyntheticBBoxConfig(min_visible_joints=1),
            )
            assert bool(obs.all())

            # Test both isotropic (fx == fy) and anisotropic (fx != fy) intrinsics.
            fx = torch.tensor([cam.f], dtype=torch.float32)
            fy = torch.tensor([cam.f * 1.04], dtype=torch.float32)
            cx0 = torch.tensor([cam.cx + 5.0], dtype=torch.float32)
            cy0 = torch.tensor([cam.cy - 3.0], dtype=torch.float32)
            R = cam.R.unsqueeze(0)
            C = cam.C.unsqueeze(0)

            pred_cam, local_pose = world_coco17_to_cliff_and_local_pose(
                world_kp.unsqueeze(0),
                bbx_xys,
                R_w2c=R,
                C_w=C,
                fx=fx,
                fy=fy,
                cx0=cx0,
                cy0=cy0,
            )
            # Mid-hip of local_pose is strictly centered at origin.
            pelvis_local = 0.5 * (
                local_pose[..., LEFT_HIP_IDX, :] + local_pose[..., RIGHT_HIP_IDX, :]
            )
            torch.testing.assert_close(
                pelvis_local, torch.zeros_like(pelvis_local), atol=1e-6, rtol=0.0
            )
            assert bool((pred_cam[..., 0] > 0.0).all())

            recovered = cliff_and_local_pose_to_world_coco17(
                pred_cam,
                local_pose,
                bbx_xys,
                R_w2c=R,
                C_w=C,
                fx=fx,
                fy=fy,
                cx0=cx0,
                cy0=cy0,
            )
            torch.testing.assert_close(
                recovered[0], world_kp, atol=1e-5, rtol=1e-5
            )

            # Verify pixel projection matches CameraProjector when fx == fy == cam.f.
            proj_px, in_front = project_world_coco17_to_pixel(
                world_kp.unsqueeze(0),
                R_w2c=R,
                C_w=C,
                fx=torch.tensor([cam.f]),
                fy=torch.tensor([cam.f]),
                cx0=torch.tensor([cam.cx]),
                cy0=torch.tensor([cam.cy]),
            )
            assert bool(in_front.all())
            torch.testing.assert_close(proj_px[0], kp_px, atol=1e-4, rtol=1e-4)


class TestCameraFeaturesAndCalibrationNoise:
    def test_camera_features_static_and_panning_shapes_and_rel_rotation(self) -> None:
        projector = _build_mixed_projector()
        cam = projector.cameras()[6]  # broadcast camera
        R_static = cam.R.unsqueeze(0)
        C_static = cam.C.unsqueeze(0)
        fx = torch.tensor([cam.f])
        fy = torch.tensor([cam.f])
        cx0 = torch.tensor([cam.cx])
        cy0 = torch.tensor([cam.cy])
        w = torch.tensor([float(cam.w)])
        h = torch.tensor([float(cam.h)])

        feat_static = compute_camera_features(
            R_static,
            C_static,
            fx=fx,
            fy=fy,
            cx0=cx0,
            cy0=cy0,
            image_w=w,
            image_h=h,
            seq_len=10,
        )
        assert feat_static.shape == (1, 10, CAMERA_FEATURE_DIM)
        assert torch.isfinite(feat_static).all()
        # Static camera has identity inter-frame rotation [1, 0, 0, 0, 1, 0] for all t.
        expected_rel = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(1, 10, 6)
        torch.testing.assert_close(feat_static[..., 15:21], expected_rel, atol=1e-5, rtol=0.0)

        # Panning camera sequence: R_t varies smoothly over time.
        pan_angles = torch.linspace(0.0, 0.1, 10).unsqueeze(-1) * torch.tensor(
            [0.0, 1.0, 0.0]
        )
        R_pan = (so3_exp_map(pan_angles) @ cam.R).unsqueeze(0)  # (1, 10, 3, 3)
        feat_pan = compute_camera_features(
            R_pan,
            C_static,
            fx=fx,
            fy=fy,
            cx0=cx0,
            cy0=cy0,
            image_w=w,
            image_h=h,
            seq_len=10,
        )
        assert feat_pan.shape == (1, 10, CAMERA_FEATURE_DIM)
        torch.testing.assert_close(
            feat_pan[:, 0, 15:21],
            torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )
        assert not torch.allclose(feat_pan[:, 1:, 15:21], expected_rel[:, 1:])

    def test_calibration_noise_preserves_so3_and_rejects_invalid_cameras(self) -> None:
        projector = _build_mixed_projector()
        cams = projector.cameras()
        R = torch.stack([c.R for c in cams], dim=0)
        C = torch.stack([c.C for c in cams], dim=0)
        fx = torch.tensor([c.f for c in cams], dtype=torch.float32)
        cx0 = torch.tensor([c.cx for c in cams], dtype=torch.float32)
        cy0 = torch.tensor([c.cy for c in cams], dtype=torch.float32)
        w = torch.tensor([float(c.w) for c in cams], dtype=torch.float32)
        h = torch.tensor([float(c.h) for c in cams], dtype=torch.float32)

        gen = torch.Generator().manual_seed(42)
        pert = perturb_camera_calibration(
            R,
            C,
            fx=fx,
            fy=fx,
            cx0=cx0,
            cy0=cy0,
            image_w=w,
            image_h=h,
            config=CalibrationNoiseConfig(enabled=True, prob=1.0),
            generator=gen,
        )
        validate_pinhole_camera_tensors(
            pert.R_w2c,
            pert.C_w,
            fx=pert.fx,
            fy=pert.fy,
            cx0=pert.cx0,
            cy0=pert.cy0,
            image_w=w,
            image_h=h,
        )
        assert not torch.allclose(pert.R_w2c, R)
        assert not torch.allclose(pert.C_w, C)
        assert not torch.allclose(pert.fx, fx)

        # CLIFF features stay zero on unobserved frames and finite on observed frames.
        bbx = torch.tensor([[[640.0, 360.0, 200.0], [0.0, 0.0, 0.0]]]).expand(8, 2, 3)
        obs = torch.tensor([[True, False]]).expand(8, 2)
        cliff = compute_cliffcam_features(
            bbx, obs, fx=pert.fx, fy=pert.fy, cx0=pert.cx0, cy0=pert.cy0
        )
        assert cliff.shape == (8, 2, 3)
        torch.testing.assert_close(cliff[:, 1], torch.zeros((8, 3)))

        # Reject improper reflection R (det == -1) and non-positive focal length.
        bad_R = R.clone()
        bad_R[0, 0, :] *= -1.0
        with pytest.raises(ValueError, match="det"):
            validate_pinhole_camera_tensors(
                bad_R, C, fx=fx, fy=fx, cx0=cx0, cy0=cy0, image_w=w, image_h=h
            )
        with pytest.raises(ValueError, match="strictly positive"):
            validate_pinhole_camera_tensors(
                R,
                C,
                fx=torch.zeros_like(fx),
                fy=fx,
                cx0=cx0,
                cy0=cy0,
                image_w=w,
                image_h=h,
            )
