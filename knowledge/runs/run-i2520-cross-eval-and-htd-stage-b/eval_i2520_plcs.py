#!/usr/bin/env python3
"""Controlled synthetic + real-clip (meiji_4cam) evaluation suite for #2475 / #2520.

Evaluates monocular GVHMR/HTD PLCS checkpoints (M0, M1, M2, A1) and multiview
axial PLCS baseline checkpoints (B0) under identical metric definitions on:
  1. Synthetic test splits (fixed, broadcast, mixed)
  2. Pinned real 4-camera benchmark `meiji_4cam/benchmarks/multiview-player-evaluation-v1`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.generate_dataset.io.scene_loader import load_scene
from src.tasks.plcs.geometry.monocular_camera_bbox import (
    LEFT_HIP_IDX,
    RIGHT_HIP_IDX,
    SyntheticBBoxConfig,
    compute_synthetic_bbx_xys,
    project_world_coco17_to_pixel,
)
from src.tasks.plcs.inference.htd_refiner import (
    HTDRefineCameraView,
    HTDRefinerConfig,
    PLCSHTDRefiner,
)
from src.tasks.plcs.inference.predictor import PLCSPredictor
from src.tasks.plcs.model_io.contracts import PLCSInputProfile
from src.utils.configuration import PathResolver
from src.utils.configuration.paths import RuntimePathRoots
from src.utils.geometry.court_pose import canonical_pose_to_world_pose
from src.utils.geometry.skeleton import (
    compute_bone_lengths,
    compute_joint_angles,
    compute_torsion_angles,
    compute_torso_twist,
)
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d
from src.utils.schema.court_normalization import denormalize_court_position
from src.utils.schema.player import COCO17_BONE_LENGTH_EDGES


def _procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Batch Procrustes (s, R, t) alignment of pred (N, J, 3) onto target (N, J, 3)."""
    mu_p = pred.mean(axis=1, keepdims=True)
    mu_t = target.mean(axis=1, keepdims=True)
    x = pred - mu_p
    y = target - mu_t
    var_x = np.sum(x * x, axis=(1, 2))
    cov = np.einsum("nji,njk->nik", x, y)
    u, s, vt = np.linalg.svd(cov)
    r = np.einsum("nij,nkj->nik", vt.transpose(0, 2, 1), u)
    det = np.linalg.det(r)
    neg = det < 0
    if np.any(neg):
        vt_fix = vt[neg].copy()
        vt_fix[:, -1, :] *= -1.0
        s_fix = s[neg].copy()
        s_fix[:, -1] *= -1.0
        r[neg] = np.einsum("nij,nkj->nik", vt_fix.transpose(0, 2, 1), u[neg])
        s[neg] = s_fix
    scale = np.sum(s, axis=-1) / np.maximum(var_x, 1e-8)
    aligned = scale[:, None, None] * np.einsum("njk,nik->nji", x, r) + mu_t
    return aligned


def _temporal_kinematics(
    pred: np.ndarray,
    target: np.ndarray | None,
    valid: np.ndarray,
    fps: float,
) -> dict[str, list[float]]:
    """Compute segment-isolated velocity, acceleration, jerk, and bone-length std."""
    t_len = pred.shape[0]
    vel_errs: list[float] = []
    acc_errs: list[float] = []
    jerks: list[float] = []
    bone_stds: list[float] = []

    if t_len >= 2:
        v_mask = valid[:-1] & valid[1:]
        if np.any(v_mask) and target is not None:
            pv = (pred[1:] - pred[:-1]) * fps
            tv = (target[1:] - target[:-1]) * fps
            ve = np.linalg.norm(pv - tv, axis=-1)[v_mask]
            vel_errs.extend(ve.reshape(-1).tolist())

    if t_len >= 3:
        a_mask = valid[:-2] & valid[1:-1] & valid[2:]
        if np.any(a_mask) and target is not None:
            pa = (pred[2:] - 2.0 * pred[1:-1] + pred[:-2]) * (fps * fps)
            ta = (target[2:] - 2.0 * target[1:-1] + target[:-2]) * (fps * fps)
            ae = np.linalg.norm(pa - ta, axis=-1)[a_mask]
            acc_errs.extend(ae.reshape(-1).tolist())

    if t_len >= 4:
        j_mask = valid[:-3] & valid[1:-2] & valid[2:-1] & valid[3:]
        if np.any(j_mask):
            pj = (pred[3:] - 3.0 * pred[2:-1] + 3.0 * pred[1:-2] - pred[:-3]) * (fps**3)
            jn = np.linalg.norm(pj, axis=-1)[j_mask]
            jerks.extend(jn.reshape(-1).tolist())

    if int(valid.sum()) >= 2:
        vp = pred[valid]  # (Nv, 17, 3)
        for i, j in COCO17_BONE_LENGTH_EDGES:
            bl = np.linalg.norm(vp[:, i] - vp[:, j], axis=-1)
            bone_stds.append(float(np.std(bl)))

    return {
        "vel_errs": vel_errs,
        "acc_errs": acc_errs,
        "jerks": jerks,
        "bone_stds": bone_stds,
    }


def _count_valid_skeleton_frames(world_kp: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Check how many masked frames pass all 4 SkeletonValues.require_valid criteria."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0, 0
    kp_t = torch.from_numpy(world_kp[idx].astype(np.float32))
    finite_ok = torch.isfinite(kp_t).all(dim=(1, 2))
    valid_all = finite_ok.clone()
    if bool(finite_ok.any()):
        sub = kp_t[finite_ok]
        ok_sub = torch.ones((sub.shape[0],), dtype=torch.bool)
        for fn in (
            compute_bone_lengths,
            compute_joint_angles,
            compute_torsion_angles,
            compute_torso_twist,
        ):
            sv = fn(sub)
            ok_sub &= sv.valid.all(dim=-1) if sv.valid.ndim > 1 else sv.valid
        valid_all[finite_ok] = ok_sub
    return int(valid_all.sum().item()), int(len(idx))


def _boxes_xyxy_to_xys(
    boxes_xyxy: np.ndarray,
    aspect_wh: tuple[float, float] = (192.0, 256.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (..., 4) [x1, y1, x2, y2] to GVHMR square bbx_xys [cx, cy, s] + valid mask."""
    finite = np.isfinite(boxes_xyxy).all(axis=-1)
    safe = np.where(finite[..., None], boxes_xyxy, 0.0)
    w = np.maximum(safe[..., 2] - safe[..., 0], 0.0)
    h = np.maximum(safe[..., 3] - safe[..., 1], 0.0)
    cx = 0.5 * (safe[..., 0] + safe[..., 2])
    cy = 0.5 * (safe[..., 1] + safe[..., 3])
    aspect = aspect_wh[0] / aspect_wh[1]
    fitted_h = np.where(w > aspect * h, w / aspect, h)
    fitted_w = np.where(w < aspect * h, h * aspect, w)
    s = np.maximum(fitted_w, fitted_h)
    valid = finite & (s > 1.0)
    xys = np.stack([cx, cy, s], axis=-1).astype(np.float32)
    xys = np.where(valid[..., None], xys, 0.0)
    return xys, valid


def evaluate_synthetic(
    predictor: PLCSPredictor,
    dataset_dir: Path,
    *,
    split: str,
    max_scenes: int | None,
    htd_refine: bool,
    htd_steps: int,
    multiview_cameras: int,
) -> dict[str, Any]:
    split_file = dataset_dir / f"{split}.txt"
    scene_names = [
        line.strip()
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_scenes is not None:
        scene_names = scene_names[:max_scenes]

    contract = predictor.court_keypoint_contract
    refiner = (
        PLCSHTDRefiner(HTDRefinerConfig(num_steps=htd_steps))
        if htd_refine and predictor.input_profile is PLCSInputProfile.MONOCULAR
        else None
    )

    mpjpe_all: list[float] = []
    pa_mpjpe_all: list[float] = []
    pos_err_all: list[float] = []
    ang_err_all: list[float] = []
    vel_all: list[float] = []
    acc_all: list[float] = []
    jerk_all: list[float] = []
    bone_std_all: list[float] = []
    skel_valid_num = 0
    skel_valid_den = 0
    latencies_ms: list[float] = []
    total_frames = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for scene_name in scene_names:
        scene_path = dataset_dir / "scenes" / scene_name
        scene = load_scene(scene_path, court_keypoint_contract=contract)
        fps = float(scene.meta.get("fps", 60.0))
        gt_world = np.asarray(scene.human_kp_3d, dtype=np.float32)  # (T, 17, 3)
        gt_pos = np.asarray(
            denormalize_court_position(torch.as_tensor(scene.position, dtype=torch.float32)),
            dtype=np.float32,
        )  # (T, 3)
        gt_rot = np.asarray(scene.rotation, dtype=np.float32)  # (T, 2)
        gt_yaw = np.arctan2(gt_rot[:, 1], gt_rot[:, 0])
        seq_len = gt_world.shape[0]

        if predictor.input_profile is PLCSInputProfile.MONOCULAR:
            # Evaluate view 0 deterministically across all scenes (or all views? Use view 0 for speed & parity, plus all-view option).
            cam = scene.cameras[0]
            p = cam.params
            R_np = np.asarray(p["R"], dtype=np.float32)
            C_np = np.asarray(p["C"], dtype=np.float32)
            fx = float(p.get("fx", p["f"]))
            fy = float(p.get("fy", p["f"]))
            cx0 = float(p["cx"])
            cy0 = float(p["cy"])
            w = float(p["w"])
            h = float(p["h"])
            kp_uv = np.asarray(cam.human_kp_uv, dtype=np.float32)
            kp_px = np.stack([kp_uv[..., 0] * w, kp_uv[..., 1] * h], axis=-1)
            kp_vis = np.asarray(cam.human_kp_vis, dtype=np.float32)

            t0 = time.perf_counter()
            max_chunk = 2048
            overlap = 128
            if seq_len <= 4096:
                pred = predictor.predict_monocular_sequence(
                    kp2d_px=kp_px,
                    kp_vis=kp_vis,
                    bbx_xys=None,
                    frame_observed=None,
                    R_w2c=R_np,
                    C_w=C_np,
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                    image_w=w,
                    image_h=h,
                    fps=fps,
                    htd_refiner=refiner,
                )
                assert pred.world_kp_3d_meters is not None
                pred_world = pred.world_kp_3d_meters[0]  # (T, 17, 3)
                pred_pos = pred.position_meters[0]  # (T, 3)
                pred_yaw = pred.yaw_radians[0]  # (T,)
                valid_mask = (
                    pred.valid_frame_mask[0]
                    if pred.valid_frame_mask is not None
                    else np.isfinite(pred_world).all(axis=(1, 2))
                )
            else:
                pred_world = np.full((seq_len, 17, 3), np.nan, dtype=np.float32)
                pred_pos = np.full((seq_len, 3), np.nan, dtype=np.float32)
                pred_yaw = np.full((seq_len,), np.nan, dtype=np.float32)
                valid_mask = np.zeros((seq_len,), dtype=np.bool_)
                stride = max_chunk - overlap
                starts = list(range(0, seq_len, stride))
                for idx_c, s0 in enumerate(starts):
                    s1 = min(s0 + max_chunk, seq_len)
                    p_c = predictor.predict_monocular_sequence(
                        kp2d_px=kp_px[s0:s1],
                        kp_vis=kp_vis[s0:s1],
                        bbx_xys=None,
                        frame_observed=None,
                        R_w2c=R_np,
                        C_w=C_np,
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                        image_w=w,
                        image_h=h,
                        fps=fps,
                        htd_refiner=refiner,
                    )
                    assert p_c.world_kp_3d_meters is not None
                    keep_l = 0 if idx_c == 0 else overlap // 2
                    keep_r = (s1 - s0) if (s1 == seq_len) else (s1 - s0 - overlap // 2)
                    pred_world[s0 + keep_l : s0 + keep_r] = p_c.world_kp_3d_meters[0, keep_l:keep_r]
                    pred_pos[s0 + keep_l : s0 + keep_r] = p_c.position_meters[0, keep_l:keep_r]
                    pred_yaw[s0 + keep_l : s0 + keep_r] = p_c.yaw_radians[0, keep_l:keep_r]
                    vm_c = (
                        p_c.valid_frame_mask[0]
                        if p_c.valid_frame_mask is not None
                        else np.isfinite(p_c.world_kp_3d_meters[0]).all(axis=(1, 2))
                    )
                    valid_mask[s0 + keep_l : s0 + keep_r] = vm_c[keep_l:keep_r]
                    if s1 == seq_len:
                        break
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            total_frames += seq_len
        else:
            cam_indices = list(range(min(multiview_cameras, len(scene.cameras))))
            ref_cam_id = (
                scene.cameras[cam_indices[0]].court_view.camera_id
                if contract.selector == "camera_view_v2"
                else None
            )
            t0 = time.perf_counter()
            decoded = predictor.predict_scene(
                scene, cameras=cam_indices, reference_camera_id=ref_cam_id
            )
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            total_frames += seq_len

            pred_pos_t = denormalize_court_position(decoded.position[0])
            pred_pos = pred_pos_t.numpy().astype(np.float32)
            pred_rot = decoded.rotation[0].numpy().astype(np.float32)
            pred_yaw = np.arctan2(pred_rot[:, 1], pred_rot[:, 0])
            if decoded.canonical_pose is not None:
                pred_world = (
                    canonical_pose_to_world_pose(
                        decoded.canonical_pose[0],
                        decoded.position[0],
                        decoded.rotation[0],
                    )
                    .numpy()
                    .astype(np.float32)
                )
            else:
                pred_world = np.full_like(gt_world, np.nan)
            valid_mask = np.ones((seq_len,), dtype=np.bool_)

        if not np.any(valid_mask):
            continue

        # Position & angular errors
        pe = np.linalg.norm(pred_pos[valid_mask] - gt_pos[valid_mask], axis=-1)
        pos_err_all.extend(pe.tolist())
        dyaw = np.abs(np.arctan2(np.sin(pred_yaw - gt_yaw), np.cos(pred_yaw - gt_yaw)))
        ang_err_all.extend(np.degrees(dyaw[valid_mask]).tolist())

        # 3D skeleton metrics
        if np.isfinite(pred_world[valid_mask]).all():
            pw = pred_world[valid_mask]
            gw = gt_world[valid_mask]
            mp = np.linalg.norm(pw - gw, axis=-1).mean(axis=-1)
            mpjpe_all.extend(mp.tolist())
            aligned = _procrustes_align(pw, gw)
            pamp = np.linalg.norm(aligned - gw, axis=-1).mean(axis=-1)
            pa_mpjpe_all.extend(pamp.tolist())

            kin = _temporal_kinematics(pred_world, gt_world, valid_mask, fps)
            vel_all.extend(kin["vel_errs"])
            acc_all.extend(kin["acc_errs"])
            jerk_all.extend(kin["jerks"])
            bone_std_all.extend(kin["bone_stds"])

            vn, vd = _count_valid_skeleton_frames(pred_world, valid_mask)
            skel_valid_num += vn
            skel_valid_den += vd

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else 0.0
    )
    pos_arr = np.asarray(pos_err_all, dtype=np.float64)
    ang_arr = np.asarray(ang_err_all, dtype=np.float64)

    return {
        "dataset": str(dataset_dir.name),
        "split": split,
        "num_scenes": len(scene_names),
        "total_frames": total_frames,
        "htd_refine": bool(refiner is not None),
        "position_error_m": float(np.mean(pos_arr)) if len(pos_arr) else float("nan"),
        "position_accuracy_0.5m": float(np.mean(pos_arr <= 0.5)) if len(pos_arr) else float("nan"),
        "angular_error_deg": float(np.mean(ang_arr)) if len(ang_arr) else float("nan"),
        "angle_accuracy_15deg": float(np.mean(ang_arr <= 15.0)) if len(ang_arr) else float("nan"),
        "world_mpjpe_m": float(np.mean(mpjpe_all)) if mpjpe_all else float("nan"),
        "pa_mpjpe_m": float(np.mean(pa_mpjpe_all)) if pa_mpjpe_all else float("nan"),
        "mpjve_mps": float(np.mean(vel_all)) if vel_all else float("nan"),
        "mpjae_mps2": float(np.mean(acc_all)) if acc_all else float("nan"),
        "jerk_mps3": float(np.mean(jerk_all)) if jerk_all else float("nan"),
        "bone_length_std_m": float(np.mean(bone_std_all)) if bone_std_all else float("nan"),
        "skeleton_valid_rate": float(skel_valid_num / max(skel_valid_den, 1)),
        "skeleton_valid_frames": skel_valid_num,
        "skeleton_evaluated_frames": skel_valid_den,
        "latency_ms_per_scene": float(np.mean(latencies_ms)) if latencies_ms else float("nan"),
        "latency_ms_per_frame": float(np.sum(latencies_ms) / max(total_frames, 1)),
        "peak_vram_mb": peak_vram_mb,
    }


def evaluate_meiji(
    predictor: PLCSPredictor,
    benchmark_dir: Path,
    *,
    htd_refine_mode: str,
    htd_steps: int,
    conf_thresh: float = 0.3,
    image_w: int = 1920,
    image_h: int = 1080,
) -> dict[str, Any]:
    """Evaluate monocular PLCS on pinned meiji_4cam/multiview-player-evaluation-v1.

    htd_refine_mode: 'none' | 'single_view' | 'multiview'
    """
    manifest_path = benchmark_dir / "manifest.json"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clips_dir = benchmark_dir / "clips"

    refiner = (
        PLCSHTDRefiner(HTDRefinerConfig(num_steps=htd_steps))
        if htd_refine_mode in {"single_view", "multiview"}
        else None
    )

    lovo_reproj_all: list[float] = []
    same_reproj_all: list[float] = []
    cross_cam_3d_all: list[float] = []
    vs_dlt_3d_all: list[float] = []
    vs_dlt_pa_all: list[float] = []
    vs_gvhmr_3d_all: list[float] = []
    mv_vs_dlt_3d_all: list[float] = []
    mv_vs_dlt_pa_all: list[float] = []
    mv_vs_gvhmr_3d_all: list[float] = []
    mv_all_cam_reproj_all: list[float] = []
    mv_jerk_all: list[float] = []
    mv_bone_std_all: list[float] = []
    mv_latencies_ms: list[float] = []
    mv_total_frames = 0
    single_view_frames_total = 0
    single_view_frames_reconstructed = 0
    single_view_reproj_all: list[float] = []
    dlt_incomplete_frames_total = 0
    dlt_incomplete_frames_reconstructed = 0
    jerk_all: list[float] = []
    bone_std_all: list[float] = []
    skel_valid_num = 0
    skel_valid_den = 0
    latencies_ms: list[float] = []
    total_track_frames = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for clip_entry in manifest["clips"]:
        clip_name = Path(clip_entry["clip_id"]).name
        clip_path = clips_dir / clip_name
        bri = np.load(clip_path / "body_recovery_inputs.npz")
        pgt = np.load(clip_path / "pseudo_gt.npz")

        K = bri["K"].astype(np.float64)  # (C, 3, 3)
        R = bri["R"].astype(np.float64)  # (C, 3, 3)
        t_vec = bri["t"].astype(np.float64)  # (C, 3)
        C_w = -np.einsum("cji,cj->ci", R, t_vec)  # (C, 3)
        clip_time = bri["clip_time_sec"]
        fps = float(1.0 / np.median(np.diff(clip_time))) if len(clip_time) > 1 else 60.0

        # Infer image resolution from manifest or principal point (approx 2*cx, 2*cy -> standard 1920x1080 or 1280x720)
        w = int(clip_entry.get("width", image_w))
        h = int(clip_entry.get("height", image_h))

        vitpose = pgt["vitpose_keypoints_xyc"].astype(np.float32)  # (P, C, T, 17, 3)
        same_obs = pgt["same_view_observed"].astype(np.bool_)  # (P, C, T, 17)
        boxes_xyxy = pgt["boxes_xyxy"].astype(np.float64)  # (P, C, T, 4)
        dlt_joints = bri["triangulated_joints_m"].astype(np.float32)  # (P, T, 17, 3)
        dlt_valid = bri["triangulated_valid"].astype(np.bool_)  # (P, T, 17)
        gvhmr_joints = pgt["joints_court_m"].astype(np.float32)  # (P, T, 17, 3)
        gvhmr_valid = pgt["gvhmr_inferred"].astype(np.bool_)  # (P, T)
        support_views = pgt["person_support_views"]  # (P, T)

        num_players, num_cams, num_frames = vitpose.shape[:3]

        for p in range(num_players):
            per_cam_world: dict[int, np.ndarray] = {}
            per_cam_valid: dict[int, np.ndarray] = {}
            per_cam_vel: dict[int, np.ndarray] = {}
            per_cam_acc: dict[int, np.ndarray] = {}
            per_cam_contact: dict[int, np.ndarray] = {}
            refine_views_all: list[HTDRefineCameraView] = []

            for c in range(num_cams):
                kp_px = np.nan_to_num(vitpose[p, c, :, :, :2], nan=0.0).astype(np.float32)
                kp_conf = np.where(
                    same_obs[p, c] & np.isfinite(vitpose[p, c, :, :, 2]),
                    np.nan_to_num(vitpose[p, c, :, :, 2], nan=0.0),
                    0.0,
                ).astype(np.float32)
                bbx_xys, bbx_ok = _boxes_xyxy_to_xys(boxes_xyxy[p, c])
                # Fallback to synthetic box from real observed keypoints if pseudo-box excluded while >=2 joints observed
                syn_bbx, syn_ok = compute_synthetic_bbx_xys(
                    torch.from_numpy(kp_px),
                    torch.from_numpy(kp_conf),
                    config=SyntheticBBoxConfig(vis_threshold=conf_thresh),
                )
                use_bbx = np.where(bbx_ok[:, None], bbx_xys, syn_bbx.numpy())
                frame_obs = (same_obs[p, c].sum(axis=-1) >= 2) & (use_bbx[:, 2] > 1.0)
                if not np.any(frame_obs):
                    continue

                fx = float(K[c, 0, 0])
                fy = float(K[c, 1, 1])
                cx0 = float(K[c, 0, 2])
                cy0 = float(K[c, 1, 2])
                R_c = R[c].astype(np.float32)
                C_c = C_w[c].astype(np.float32)

                sv_refiner = refiner if htd_refine_mode == "single_view" else None
                t0 = time.perf_counter()
                pred = predictor.predict_monocular_sequence(
                    kp2d_px=kp_px,
                    kp_vis=kp_conf,
                    bbx_xys=use_bbx,
                    frame_observed=frame_obs,
                    R_w2c=R_c,
                    C_w=C_c,
                    fx=fx,
                    fy=fy,
                    cx0=cx0,
                    cy0=cy0,
                    image_w=w,
                    image_h=h,
                    fps=fps,
                    vis_threshold=conf_thresh,
                    htd_refiner=sv_refiner,
                )
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                total_track_frames += int(frame_obs.sum())

                assert pred.world_kp_3d_meters is not None
                w_kp = pred.world_kp_3d_meters[0]
                v_mask = (
                    pred.valid_frame_mask[0]
                    if pred.valid_frame_mask is not None
                    else frame_obs
                )
                per_cam_world[c] = w_kp
                per_cam_valid[c] = v_mask & np.isfinite(w_kp).all(axis=(1, 2))
                if pred.pred_vel_3d_mps is not None:
                    per_cam_vel[c] = pred.pred_vel_3d_mps[0]
                if pred.pred_acc_3d_mps2 is not None:
                    per_cam_acc[c] = pred.pred_acc_3d_mps2[0]
                if pred.static_contact_logits is not None:
                    per_cam_contact[c] = pred.static_contact_logits[0]

                refine_views_all.append(
                    HTDRefineCameraView(
                        R_w2c=torch.from_numpy(R_c),
                        C_w=torch.from_numpy(C_c),
                        fx=fx,
                        fy=fy,
                        cx0=cx0,
                        cy0=cy0,
                        image_w=float(w),
                        image_h=float(h),
                        kp2d_px=torch.from_numpy(kp_px),
                        kp_conf=torch.from_numpy(kp_conf),
                        frame_observed=torch.from_numpy(frame_obs),
                    )
                )

                # Monocular LOVO 2D reprojection error on held-out cameras + same-view reprojection
                if np.any(per_cam_valid[c]):
                    vn, vd = _count_valid_skeleton_frames(w_kp, per_cam_valid[c])
                    skel_valid_num += vn
                    skel_valid_den += vd
                    kin = _temporal_kinematics(w_kp, None, per_cam_valid[c], fps)
                    jerk_all.extend(kin["jerks"])
                    bone_std_all.extend(kin["bone_stds"])

                    valid_idx = np.flatnonzero(per_cam_valid[c])
                    w_sub = torch.from_numpy(w_kp[valid_idx].astype(np.float32)).unsqueeze(0)
                    for c_eval in range(num_cams):
                        uv_p, in_front = project_world_coco17_to_pixel(
                            w_sub,
                            R_w2c=torch.from_numpy(R[c_eval].astype(np.float32)).unsqueeze(0),
                            C_w=torch.from_numpy(C_w[c_eval].astype(np.float32)).unsqueeze(0),
                            fx=torch.tensor([float(K[c_eval, 0, 0])]),
                            fy=torch.tensor([float(K[c_eval, 1, 1])]),
                            cx0=torch.tensor([float(K[c_eval, 0, 2])]),
                            cy0=torch.tensor([float(K[c_eval, 1, 2])]),
                        )
                        uv_pred = uv_p[0].numpy()
                        front_np = in_front[0].numpy()
                        obs_ev = (
                            same_obs[p, c_eval, valid_idx]
                            & (np.nan_to_num(vitpose[p, c_eval, valid_idx, :, 2], nan=0.0) >= conf_thresh)
                            & np.isfinite(vitpose[p, c_eval, valid_idx, :, :2]).all(axis=-1)
                            & front_np
                        )
                        if np.any(obs_ev):
                            err_px = np.linalg.norm(
                                uv_pred - vitpose[p, c_eval, valid_idx, :, :2], axis=-1
                            )[obs_ev]
                            if c_eval == c:
                                same_reproj_all.extend(err_px.tolist())
                            else:
                                lovo_reproj_all.extend(err_px.tolist())

                    # Agreement vs 4-cam DLT & GVHMR+DLT
                    dlt_m = per_cam_valid[c] & dlt_valid[p].all(axis=-1) & np.isfinite(dlt_joints[p]).all(axis=(1, 2))
                    if np.any(dlt_m):
                        pw = w_kp[dlt_m]
                        dw = dlt_joints[p, dlt_m]
                        vs_dlt_3d_all.extend(np.linalg.norm(pw - dw, axis=-1).mean(axis=-1).tolist())
                        al = _procrustes_align(pw, dw)
                        vs_dlt_pa_all.extend(np.linalg.norm(al - dw, axis=-1).mean(axis=-1).tolist())

                    gv_m = per_cam_valid[c] & gvhmr_valid[p] & np.isfinite(gvhmr_joints[p]).all(axis=(1, 2))
                    if np.any(gv_m):
                        vs_gvhmr_3d_all.extend(
                            np.linalg.norm(w_kp[gv_m] - gvhmr_joints[p, gv_m], axis=-1)
                            .mean(axis=-1)
                            .tolist()
                        )

            # Single-view occlusion recovery stats (support_views == 1)
            sv_frames = support_views[p] == 1
            single_view_frames_total += int(sv_frames.sum())
            if np.any(sv_frames) and per_cam_valid:
                any_rec = np.zeros((num_frames,), dtype=np.bool_)
                for c_idx, vm in per_cam_valid.items():
                    hit = sv_frames & vm
                    any_rec |= hit
                    if np.any(hit):
                        idx_h = np.flatnonzero(hit)
                        uv_p, in_front = project_world_coco17_to_pixel(
                            torch.from_numpy(per_cam_world[c_idx][idx_h].astype(np.float32)).unsqueeze(0),
                            R_w2c=torch.from_numpy(R[c_idx].astype(np.float32)).unsqueeze(0),
                            C_w=torch.from_numpy(C_w[c_idx].astype(np.float32)).unsqueeze(0),
                            fx=torch.tensor([float(K[c_idx, 0, 0])]),
                            fy=torch.tensor([float(K[c_idx, 1, 1])]),
                            cx0=torch.tensor([float(K[c_idx, 0, 2])]),
                            cy0=torch.tensor([float(K[c_idx, 1, 2])]),
                        )
                        obs_h = same_obs[p, c_idx, idx_h] & in_front[0].numpy() & (
                            np.nan_to_num(vitpose[p, c_idx, idx_h, :, 2], nan=0.0) >= conf_thresh
                        )
                        if np.any(obs_h):
                            ep = np.linalg.norm(
                                uv_p[0].numpy() - vitpose[p, c_idx, idx_h, :, :2], axis=-1
                            )[obs_h]
                            single_view_reproj_all.extend(ep.tolist())
                single_view_frames_reconstructed += int(any_rec.sum())

            # Frames where 4-cam DLT missed >=1 joint yet >=1 camera observed the player
            any_cam_obs_p = (same_obs[p].sum(axis=-1) >= 2).any(axis=0)
            dlt_incomplete = any_cam_obs_p & (~dlt_valid[p].all(axis=-1))
            dlt_incomplete_frames_total += int(dlt_incomplete.sum())
            if np.any(dlt_incomplete) and per_cam_valid:
                rec_inc = np.zeros((num_frames,), dtype=np.bool_)
                for vm in per_cam_valid.values():
                    rec_inc |= dlt_incomplete & vm
                dlt_incomplete_frames_reconstructed += int(rec_inc.sum())

            # Pairwise cross-camera 3D consistency
            cam_keys = sorted(per_cam_world.keys())
            for i_idx in range(len(cam_keys)):
                for j_idx in range(i_idx + 1, len(cam_keys)):
                    ci, cj = cam_keys[i_idx], cam_keys[j_idx]
                    both = per_cam_valid[ci] & per_cam_valid[cj]
                    if np.any(both):
                        diff = np.linalg.norm(
                            per_cam_world[ci][both] - per_cam_world[cj][both], axis=-1
                        ).mean(axis=-1)
                        cross_cam_3d_all.extend(diff.tolist())

            # Optional multi-view Stage B refiner evaluation (M2-MV)
            if htd_refine_mode == "multiview" and refiner is not None and cam_keys:
                fused_sum = np.zeros((num_frames, 17, 3), dtype=np.float64)
                w_sum = np.zeros((num_frames,), dtype=np.float64)
                vel_sum = np.zeros((num_frames, 17, 3), dtype=np.float64)
                acc_sum = np.zeros((num_frames, 17, 3), dtype=np.float64)
                cnt_sum = np.zeros((num_frames, 4), dtype=np.float64)
                v_w = np.zeros((num_frames,), dtype=np.float64)
                for c_idx in cam_keys:
                    vm = per_cam_valid[c_idx]
                    fused_sum += np.where(vm[:, None, None], per_cam_world[c_idx], 0.0)
                    w_sum += vm.astype(np.float64)
                    if c_idx in per_cam_vel:
                        vel_sum += np.where(vm[:, None, None], per_cam_vel[c_idx], 0.0)
                        v_w += vm.astype(np.float64)
                    if c_idx in per_cam_acc:
                        acc_sum += np.where(vm[:, None, None], per_cam_acc[c_idx], 0.0)
                    if c_idx in per_cam_contact:
                        cnt_sum += np.where(vm[:, None], per_cam_contact[c_idx], 0.0)
                any_v = w_sum > 0
                if np.any(any_v):
                    init_kp = np.where(
                        any_v[:, None, None], fused_sum / np.maximum(w_sum, 1.0)[:, None, None], 0.0
                    ).astype(np.float32)
                    pv_t = (
                        torch.from_numpy(
                            np.where((v_w > 0)[:, None, None], vel_sum / np.maximum(v_w, 1.0)[:, None, None], 0.0).astype(np.float32)
                        )
                        if np.any(v_w > 0)
                        else None
                    )
                    pa_t = (
                        torch.from_numpy(
                            np.where((v_w > 0)[:, None, None], acc_sum / np.maximum(v_w, 1.0)[:, None, None], 0.0).astype(np.float32)
                        )
                        if np.any(v_w > 0)
                        else None
                    )
                    pc_t = (
                        torch.from_numpy(
                            np.where((v_w > 0)[:, None], cnt_sum / np.maximum(v_w, 1.0)[:, None], 0.0).astype(np.float32)
                        )
                        if np.any(v_w > 0)
                        else None
                    )
                    t_mv0 = time.perf_counter()
                    ref_res = refiner.refine(
                        world_kp_init=torch.from_numpy(init_kp),
                        pred_vel_3d=pv_t,
                        pred_acc_3d=pa_t,
                        static_contact_logits=pc_t,
                        views=refine_views_all,
                        fps=fps,
                        valid_frame_mask=torch.from_numpy(any_v),
                    )
                    mv_latencies_ms.append((time.perf_counter() - t_mv0) * 1000.0)
                    mv_total_frames += int(any_v.sum())
                    ref_kp = ref_res.world_kp_3d.detach().cpu().numpy().astype(np.float32)
                    kin_mv = _temporal_kinematics(ref_kp, None, any_v, fps)
                    mv_jerk_all.extend(kin_mv["jerks"])
                    mv_bone_std_all.extend(kin_mv["bone_stds"])
                    # Multi-view all-camera 2D reprojection error
                    valid_mv_idx = np.flatnonzero(any_v)
                    w_mv_sub = torch.from_numpy(ref_kp[valid_mv_idx].astype(np.float32)).unsqueeze(0)
                    for c_eval in range(num_cams):
                        uv_p, in_front = project_world_coco17_to_pixel(
                            w_mv_sub,
                            R_w2c=torch.from_numpy(R[c_eval].astype(np.float32)).unsqueeze(0),
                            C_w=torch.from_numpy(C_w[c_eval].astype(np.float32)).unsqueeze(0),
                            fx=torch.tensor([float(K[c_eval, 0, 0])]),
                            fy=torch.tensor([float(K[c_eval, 1, 1])]),
                            cx0=torch.tensor([float(K[c_eval, 0, 2])]),
                            cy0=torch.tensor([float(K[c_eval, 1, 2])]),
                        )
                        obs_ev = (
                            same_obs[p, c_eval, valid_mv_idx]
                            & (np.nan_to_num(vitpose[p, c_eval, valid_mv_idx, :, 2], nan=0.0) >= conf_thresh)
                            & np.isfinite(vitpose[p, c_eval, valid_mv_idx, :, :2]).all(axis=-1)
                            & in_front[0].numpy()
                        )
                        if np.any(obs_ev):
                            err_px = np.linalg.norm(
                                uv_p[0].numpy() - vitpose[p, c_eval, valid_mv_idx, :, :2], axis=-1
                            )[obs_ev]
                            mv_all_cam_reproj_all.extend(err_px.tolist())

                    dlt_m = any_v & dlt_valid[p].all(axis=-1) & np.isfinite(dlt_joints[p]).all(axis=(1, 2))
                    if np.any(dlt_m):
                        pw_mv = ref_kp[dlt_m]
                        dw_mv = dlt_joints[p, dlt_m]
                        mv_vs_dlt_3d_all.extend(np.linalg.norm(pw_mv - dw_mv, axis=-1).mean(axis=-1).tolist())
                        al_mv = _procrustes_align(pw_mv, dw_mv)
                        mv_vs_dlt_pa_all.extend(np.linalg.norm(al_mv - dw_mv, axis=-1).mean(axis=-1).tolist())
                    gv_mv_m = any_v & gvhmr_valid[p] & np.isfinite(gvhmr_joints[p]).all(axis=(1, 2))
                    if np.any(gv_mv_m):
                        mv_vs_gvhmr_3d_all.extend(
                            np.linalg.norm(ref_kp[gv_mv_m] - gvhmr_joints[p, gv_mv_m], axis=-1)
                            .mean(axis=-1)
                            .tolist()
                        )

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if torch.cuda.is_available()
        else 0.0
    )

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "count": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "count": int(len(arr)),
        }

    return {
        "benchmark_id": manifest["benchmark_id"],
        "manifest_sha256": manifest_sha256,
        "htd_refine_mode": htd_refine_mode,
        "lovo_reproj_px": _stats(lovo_reproj_all),
        "same_view_reproj_px": _stats(same_reproj_all),
        "cross_camera_3d_consistency_m": _stats(cross_cam_3d_all),
        "vs_dlt_mpjpe_m": _stats(vs_dlt_3d_all),
        "vs_dlt_pa_mpjpe_m": _stats(vs_dlt_pa_all),
        "vs_gvhmr_mpjpe_m": _stats(vs_gvhmr_3d_all),
        "single_view_frames_total": single_view_frames_total,
        "single_view_frames_reconstructed": single_view_frames_reconstructed,
        "single_view_recovery_rate": float(
            single_view_frames_reconstructed / max(single_view_frames_total, 1)
        ),
        "dlt_incomplete_frames_total": dlt_incomplete_frames_total,
        "dlt_incomplete_frames_reconstructed": dlt_incomplete_frames_reconstructed,
        "dlt_incomplete_recovery_rate": float(
            dlt_incomplete_frames_reconstructed / max(dlt_incomplete_frames_total, 1)
        ),
        "single_view_reproj_px": _stats(single_view_reproj_all),
        "jerk_mps3": float(np.mean(jerk_all)) if jerk_all else float("nan"),
        "bone_length_std_m": float(np.mean(bone_std_all)) if bone_std_all else float("nan"),
        "skeleton_valid_rate": float(skel_valid_num / max(skel_valid_den, 1)),
        "latency_ms_per_track": float(np.mean(latencies_ms)) if latencies_ms else float("nan"),
        "latency_ms_per_frame": float(np.sum(latencies_ms) / max(total_track_frames, 1)),
        "peak_vram_mb": peak_vram_mb,
        **(
            {
                "multiview_fused": {
                    "vs_dlt_mpjpe_m": _stats(mv_vs_dlt_3d_all),
                    "vs_dlt_pa_mpjpe_m": _stats(mv_vs_dlt_pa_all),
                    "vs_gvhmr_mpjpe_m": _stats(mv_vs_gvhmr_3d_all),
                    "all_cam_reproj_px": _stats(mv_all_cam_reproj_all),
                    "jerk_mps3": float(np.mean(mv_jerk_all)) if mv_jerk_all else float("nan"),
                    "bone_length_std_m": float(np.mean(mv_bone_std_all)) if mv_bone_std_all else float("nan"),
                    "refine_latency_ms_per_frame": float(np.sum(mv_latencies_ms) / max(mv_total_frames, 1)),
                }
            }
            if htd_refine_mode == "multiview"
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PLCS checkpoints for #2475/#2520.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["synthetic", "meiji", "all"], default="all")
    parser.add_argument(
        "--synthetic-datasets",
        type=Path,
        nargs="*",
        default=[
            Path("data/plcs/single_object"),
            Path("data/plcs/single_object_broadcast_eval"),
            Path("data/plcs/single_object_mixed"),
        ],
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument(
        "--meiji-dir",
        type=Path,
        default=Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-player-evaluation-v1"),
    )
    parser.add_argument("--htd-refine", action="store_true")
    parser.add_argument("--htd-refine-mode", choices=["none", "single_view", "multiview"], default="none")
    parser.add_argument("--htd-steps", type=int, default=25)
    parser.add_argument("--multiview-cameras", type=int, default=4)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    ckpt_resolved = args.checkpoint.resolve()
    ckpt_root = ckpt_resolved.parent
    repo_root = Path.cwd().resolve()
    resolver = PathResolver(
        roots=RuntimePathRoots(
            project_root=repo_root,
            data_root=(repo_root / "data").resolve(strict=False),
            checkpoint_root=ckpt_root.resolve(strict=False),
            output_root=(repo_root / "outputs").resolve(strict=False),
            artifact_root=(repo_root / "artifacts").resolve(strict=False),
            cache_root=(repo_root / ".cache").resolve(strict=False),
            external_asset_root=(repo_root / "data").resolve(strict=False),
        )
    )
    predictor = PLCSPredictor.load_from_checkpoint(
        ckpt_resolved,
        resolver=resolver,
        device=args.device,
    )

    results: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "input_profile": predictor.input_profile.value,
        "device": args.device,
    }
    if args.mode in {"synthetic", "all"}:
        results["synthetic"] = {}
        for ds in args.synthetic_datasets:
            if not ds.exists():
                continue
            results["synthetic"][ds.name] = evaluate_synthetic(
                predictor,
                ds,
                split=args.split,
                max_scenes=args.max_scenes,
                htd_refine=args.htd_refine,
                htd_steps=args.htd_steps,
                multiview_cameras=args.multiview_cameras,
            )
    if args.mode in {"meiji", "all"} and predictor.input_profile is PLCSInputProfile.MONOCULAR:
        results["meiji"] = evaluate_meiji(
            predictor,
            args.meiji_dir,
            htd_refine_mode=args.htd_refine_mode,
            htd_steps=args.htd_steps,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
