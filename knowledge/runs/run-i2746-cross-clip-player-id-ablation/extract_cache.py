"""Deterministic 16-crop sampling and CLIP-ReID feature/metadata cache extraction.

Fixed input: meiji_4cam/video_000 (clip_000..068) multiview-v4.
Reuses the repository's ClipReIDEncoder (clipreid_vitb16_market1501) on GPU via
training-queue without modifying any upstream SceneResult artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from src.tasks.player_association.appearance.encoders import (
    CLIP_REID_PROFILE,
    ClipReIDEncoder,
)
from src.tasks.player_association.appearance.sampling import crop as reid_crop
from src.utils.geometry.bbox import pairwise_iou

DEFAULT_BENCH_ROOT = Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4")
DEFAULT_PROJECTS_JSON = Path("data/tennis_multiview/processed/meiji_4cam/projects.json")
DEFAULT_CKPT = Path("ckpt/player_association/person_vit_clip_reid.pth")
CROPS_PER_CAMERA = 4
MIN_HEIGHT_PX = 16.0
BORDER_PX = 4.0


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_reid_box(
    kp_uv: np.ndarray,
    kp_vis: np.ndarray,
    box_xys: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Deterministic xyxy person crop box from observed 2D keypoints and boxes_xys."""
    cx, cy, s = float(box_xys[0]), float(box_xys[1]), float(box_xys[2])
    valid_kp = kp_vis & (kp_uv[:, 0] > 0) & (kp_uv[:, 1] > 0)
    if valid_kp.sum() >= 4:
        pts = kp_uv[valid_kp]
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        bw = max(x2 - x1, s * 0.22)
        bh = max(y2 - y1, s * 0.50)
        pad_x = max(bw * 0.25, 8.0)
        pad_y_top = max(bh * 0.22, 10.0)
        pad_y_bot = max(bh * 0.14, 8.0)
        rx1, ry1 = x1 - pad_x, y1 - pad_y_top
        rx2, ry2 = x2 + pad_x, y2 + pad_y_bot
    else:
        rx1, ry1 = cx - s * 0.30, cy - s * 0.50
        rx2, ry2 = cx + s * 0.30, cy + s * 0.50

    w = max(rx2 - rx1, 8.0)
    h = max(ry2 - ry1, 16.0)
    # Keep sensible Re-ID aspect ratio (approx 1:2, never thinner than 1:3 or wider than 1:1)
    if w < h * 0.38:
        w = h * 0.45
    elif w > h * 0.95:
        h = w / 0.85
    mcx = 0.5 * (rx1 + rx2)
    mcy = 0.5 * (ry1 + ry2)
    ix1 = max(0.0, mcx - 0.5 * w)
    iy1 = max(0.0, mcy - 0.5 * h)
    ix2 = min(float(img_w), mcx + 0.5 * w)
    iy2 = min(float(img_h), mcy + 0.5 * h)
    return np.array([ix1, iy1, ix2, iy2], dtype=np.float32)


def compute_facing_score(kp_uv: np.ndarray, kp_vis: np.ndarray, kp_conf: np.ndarray) -> float:
    """Estimate front (+1) vs back (-1) facing proxy from face keypoints visibility/confidence.

    COCO-17: 0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear, 5=l_sho, 6=r_sho.
    """
    face_vis = kp_vis[0:3]
    face_conf = kp_conf[0:3]
    ear_conf = kp_conf[3:5]
    front_evidence = float(face_conf[face_vis].mean()) if face_vis.any() else 0.0
    if not face_vis.any() and (ear_conf > 0.3).any():
        return -1.0
    if front_evidence >= 0.45 and int(face_vis.sum()) >= 2:
        return 1.0
    if front_evidence <= 0.20:
        return -1.0
    return 0.0


def extract_all(
    bench_root: Path,
    projects_json: Path,
    ckpt_path: Path,
    out_dir: Path,
    device: str,
    batch_size: int = 64,
    crops_per_camera: int = CROPS_PER_CAMERA,
) -> dict[str, Any]:
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    out_dir.mkdir(parents=True, exist_ok=True)
    proj = json.loads(projects_json.read_text())
    proj_clips = {c["name"]: c.get("score") or {} for c in proj["projects"]["video_000"]["clips"]}

    clips_dir = bench_root / "videos/video_000/clips"
    clip_dirs = sorted(clips_dir.glob("clip_*"))
    n_clips = len(clip_dirs)
    assert n_clips == 69, f"Expected 69 clips, got {n_clips}"

    encoder = ClipReIDEncoder(CLIP_REID_PROFILE, ckpt_path, device=device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    clip_names: list[str] = []
    # Fixed per-node arrays (N_clips, P=4, V=4, K=4)
    P_MAX, V_MAX, K_MAX = 4, 4, crops_per_camera
    embeddings = np.zeros((n_clips, P_MAX, V_MAX, K_MAX, 1280), dtype=np.float32)
    crop_valid = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=bool)
    crop_frames = np.full((n_clips, P_MAX, V_MAX, K_MAX), -1, dtype=np.int32)
    crop_source_rows = np.full((n_clips, P_MAX, V_MAX, K_MAX), -1, dtype=np.int64)
    crop_boxes = np.zeros((n_clips, P_MAX, V_MAX, K_MAX, 4), dtype=np.float32)
    crop_heights = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_areas = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_border_dist = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_truncated = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=bool)
    crop_max_iou = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_kp_vis_ratio = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_kp_conf_mean = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_blur_var = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_facing = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.float32)
    crop_kp_norm = np.zeros((n_clips, P_MAX, V_MAX, K_MAX, 17, 3), dtype=np.float32)
    crop_neg_kp_norm = np.zeros((n_clips, P_MAX, V_MAX, K_MAX, 3, 17, 3), dtype=np.float32)
    crop_neg_count = np.zeros((n_clips, P_MAX, V_MAX, K_MAX), dtype=np.int32)

    # Node-level metadata (N_clips, P=4)
    sides = np.zeros((n_clips, P_MAX), dtype=np.int8)
    obs_frames_pv = np.zeros((n_clips, P_MAX, V_MAX), dtype=np.int32)
    id_amb_counts = np.zeros((n_clips, P_MAX), dtype=np.int32)
    frame_amb_counts = np.zeros((n_clips, P_MAX), dtype=np.int32)
    smpl_betas = np.zeros((n_clips, P_MAX, 10), dtype=np.float32)
    smpl_valid_frames = np.zeros((n_clips, P_MAX), dtype=np.int32)
    gvhmr_seg_count = np.zeros((n_clips, P_MAX), dtype=np.int32)
    gvhmr_obs_samples = np.zeros((n_clips, P_MAX), dtype=np.int32)
    gvhmr_beta_std = np.zeros((n_clips, P_MAX, 10), dtype=np.float32)
    gvhmr_rep_cam = np.full((n_clips, P_MAX), -1, dtype=np.int8)
    pos_start_3d = np.zeros((n_clips, P_MAX, 3), dtype=np.float32)
    pos_end_3d = np.zeros((n_clips, P_MAX, 3), dtype=np.float32)
    pos_mean_3d = np.zeros((n_clips, P_MAX, 3), dtype=np.float32)
    vel_start_3d = np.zeros((n_clips, P_MAX, 3), dtype=np.float32)
    vel_end_3d = np.zeros((n_clips, P_MAX, 3), dtype=np.float32)

    clip_meta_records: list[dict[str, Any]] = []
    gpu_sec = 0.0
    total_crops_extracted = 0

    for ci, clip_dir in enumerate(clip_dirs):
        cname = clip_dir.name
        clip_names.append(cname)
        cmeta = json.loads((clip_dir / "clip.json").read_text())
        img_w, img_h = int(cmeta["width"]), int(cmeta["height"])
        fps = float(cmeta["fps"])

        sj = json.loads((clip_dir / "scene/scene.json").read_text())
        msi_path = clip_dir / "scene" / sj["artifacts"]["multiview_scene_inputs"]["path"]
        msi = json.loads(msi_path.read_text())
        ppl = msi["payload"]["people"]

        raw_idx = np.load(msi_path.parent / ppl["grouped"]["raw_indices"]["array"])  # (P, V, T)
        source_rows = np.load(msi_path.parent / ppl["source_rows"]["array"])  # (P, V, T)
        raw_obs = np.load(msi_path.parent / ppl["raw"]["observed"]["array"])  # (V, T, D)
        boxes_xys = np.load(msi_path.parent / ppl["raw"]["boxes_xys"]["array"])  # (V, T, D, 3)
        raw_uv = np.load(msi_path.parent / ppl["raw"]["uv_px"]["array"])  # (V, T, D, 17, 2)
        raw_conf = np.load(msi_path.parent / ppl["raw"]["confidence"]["array"])  # (V, T, D, 17)
        uv_px = np.load(msi_path.parent / ppl["grouped"]["uv_px"]["array"])  # (P, V, T, 17, 2)
        kp_vis = np.load(msi_path.parent / ppl["grouped"]["visibility"]["array"])  # (P, V, T, 17)
        kp_conf = np.load(msi_path.parent / ppl["grouped"]["confidence"]["array"])  # (P, V, T, 17)
        id_amb = np.load(msi_path.parent / ppl["id_ambiguous"]["array"])  # (P, T)
        frame_amb = np.load(msi_path.parent / ppl["frame_ambiguous"]["array"])  # (P, T)
        sides[ci] = np.asarray(ppl["sides"], dtype=np.int8)

        # Scene npz
        exp_npz = clip_dir / "scene" / sj["exports"]["scene"]["files"]["scene"]["path"]
        scene = np.load(exp_npz, allow_pickle=False)
        identities = np.load(msi_path.parent / ppl["grouped"]["identities"]["array"]).astype(np.int64)
        if "player_track_ids" in scene.files:
            scene_tids = scene["player_track_ids"].astype(np.int64)
            if not np.array_equal(identities, scene_tids):
                raise RuntimeError(
                    f"{cname}: msi identities {identities.tolist()} != scene player_track_ids {scene_tids.tolist()}"
                )
        id_to_p = {int(pid): idx for idx, pid in enumerate(identities.tolist())}
        smpl_betas[ci] = scene["smpl_betas"].astype(np.float32)
        for p in range(P_MAX):
            smpl_valid_frames[ci, p] = int(scene["player_smpl_valid"][p].sum())
            id_amb_counts[ci, p] = int(id_amb[p].sum())
            frame_amb_counts[ci, p] = int(frame_amb[p].sum())
            pv = scene["player_valid"][p]
            ppos = scene["player_position"][p]
            vt = np.flatnonzero(pv)
            if len(vt) > 0:
                pos_start_3d[ci, p] = ppos[vt[0]]
                pos_end_3d[ci, p] = ppos[vt[-1]]
                pos_mean_3d[ci, p] = ppos[vt].mean(axis=0)
                if len(vt) >= 2:
                    k_v = min(15, len(vt) - 1)
                    dt_s = float(vt[k_v] - vt[0]) / fps
                    if dt_s > 0:
                        vel_start_3d[ci, p] = (ppos[vt[k_v]] - ppos[vt[0]]) / dt_s
                    dt_e = float(vt[-1] - vt[-1 - k_v]) / fps
                    if dt_e > 0:
                        vel_end_3d[ci, p] = (ppos[vt[-1]] - ppos[vt[-1 - k_v]]) / dt_e

        # GVHMR per-segment betas
        gvhmr_path = clip_dir / "scene" / sj["artifacts"]["gvhmr"]["path"]
        gvhmr = json.loads(gvhmr_path.read_text())
        for body in gvhmr["payload"]["bodies"]:
            raw_pid = int(body["person_id"])
            if raw_pid not in id_to_p:
                raise RuntimeError(f"{cname}: GVHMR person_id {raw_pid} not in msi identities {identities.tolist()}")
            p = id_to_p[raw_pid]
            cam_str = str(body["camera_id"])
            if cam_str.startswith("cam"):
                gvhmr_rep_cam[ci, p] = int(cam_str[3:])
            segs = body.get("segments", [])
            gvhmr_seg_count[ci, p] = len(segs)
            all_b = []
            obs_sum = 0
            for seg in segs:
                obs_sum += int(seg.get("observed_samples", 0))
                b_ref = seg.get("parameters", {}).get("betas", {}).get("array")
                if b_ref:
                    arr = np.load(gvhmr_path.parent / b_ref)
                    if arr.ndim == 2 and arr.shape[1] == 10:
                        all_b.append(arr.astype(np.float32))
            gvhmr_obs_samples[ci, p] = obs_sum
            if all_b:
                cat = np.concatenate(all_b, axis=0)
                gvhmr_beta_std[ci, p] = cat.std(axis=0)

        # Deterministic equal-spaced frame selection per (p, v) from real observed frames
        needed_by_cam: dict[int, dict[int, list[tuple[int, int, int, np.ndarray]]]] = {v: {} for v in range(V_MAX)}
        for p in range(P_MAX):
            for v in range(V_MAX):
                valid_ts = []
                for t in np.flatnonzero(raw_idx[p, v] >= 0):
                    d = int(raw_idx[p, v, t])
                    if not raw_obs[v, t, d]:
                        continue
                    box = derive_reid_box(uv_px[p, v, t], kp_vis[p, v, t], boxes_xys[v, t, d], img_w, img_h)
                    if (box[3] - box[1]) >= MIN_HEIGHT_PX and (box[2] - box[0]) >= 6.0:
                        valid_ts.append((int(t), d, box))
                obs_frames_pv[ci, p, v] = len(valid_ts)
                if not valid_ts:
                    continue
                if len(valid_ts) <= K_MAX:
                    picked = list(range(len(valid_ts)))
                else:
                    picked = np.unique(np.linspace(0, len(valid_ts) - 1, K_MAX).round().astype(int)).tolist()
                for ki, idx_in_list in enumerate(picked[:K_MAX]):
                    t, d, box = valid_ts[idx_in_list]
                    needed_by_cam[v].setdefault(t, []).append((p, ki, d, box))

        # Decode selected frames per camera and run CLIP-ReID in batches
        crop_tensors: list[np.ndarray] = []
        crop_coords: list[tuple[int, int, int]] = []
        for v in range(V_MAX):
            wanted = needed_by_cam[v]
            if not wanted:
                continue
            cap = cv2.VideoCapture(str(clip_dir / f"media/cam{v}.mp4"))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video for {cname} cam{v}")
            max_f = max(wanted)
            fidx = 0
            while fidx <= max_f:
                ok, frame = cap.read()
                if not ok:
                    break
                items = wanted.get(fidx)
                if items:
                    # Compute all observed person boxes in this camera & frame for IoU overlap
                    obs_d = [d for d in range(raw_obs.shape[2]) if raw_obs[v, fidx, d]]
                    obs_boxes_d: dict[int, np.ndarray] = {}
                    for d_other in obs_d:
                        ov_kp = raw_uv[v, fidx, d_other]
                        ov_vis = (raw_conf[v, fidx, d_other] > 0.2) & (ov_kp[:, 0] > 0) & (ov_kp[:, 1] > 0)
                        obs_boxes_d[d_other] = derive_reid_box(ov_kp, ov_vis, boxes_xys[v, fidx, d_other], img_w, img_h)

                    for p, ki, d, box in items:
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                        bw, bh = x2 - x1, y2 - y1
                        # Overlap IoU with other observed persons
                        max_iou = 0.0
                        other_list = [obs_boxes_d[do] for do in obs_d if do != d]
                        if other_list:
                            ious = pairwise_iou(box[None, :], np.stack(other_list, axis=0))
                            max_iou = float(ious.max())
                        bdist = min(x1, y1, img_w - x2, img_h - y2)
                        trunc = bdist < BORDER_PX

                        # Blur metric on crop
                        ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
                        ix2, iy2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))
                        patch = frame[iy1:iy2, ix1:ix2]
                        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                        blur_v = float(cv2.Laplacian(gray, cv2.CV_64F).var())

                        # Crop normalized keypoints for potential KPR follow-up
                        kp_n = np.zeros((17, 3), dtype=np.float32)
                        kp_n[:, 0] = (uv_px[p, v, fidx, :, 0] - ix1) / max(ix2 - ix1, 1)
                        kp_n[:, 1] = (uv_px[p, v, fidx, :, 1] - iy1) / max(iy2 - iy1, 1)
                        kp_n[:, 2] = np.where(kp_vis[p, v, fidx], kp_conf[p, v, fidx], 0.0)

                        neg_cnt = 0
                        for do in obs_d:
                            if do == d or neg_cnt >= 3:
                                continue
                            # Check if other person has any joint inside/near crop
                            okp = raw_uv[v, fidx, do]
                            oconf = raw_conf[v, fidx, do]
                            onx = (okp[:, 0] - ix1) / max(ix2 - ix1, 1)
                            ony = (okp[:, 1] - iy1) / max(iy2 - iy1, 1)
                            inside = (oconf > 0.3) & (onx >= 0.0) & (onx < 1.0) & (ony >= 0.0) & (ony < 1.0)
                            if inside.any():
                                crop_neg_kp_norm[ci, p, v, ki, neg_cnt, :, 0] = onx
                                crop_neg_kp_norm[ci, p, v, ki, neg_cnt, :, 1] = ony
                                crop_neg_kp_norm[ci, p, v, ki, neg_cnt, :, 2] = oconf
                                neg_cnt += 1

                        crop_valid[ci, p, v, ki] = True
                        crop_frames[ci, p, v, ki] = fidx
                        crop_source_rows[ci, p, v, ki] = int(source_rows[p, v, fidx])
                        crop_boxes[ci, p, v, ki] = box
                        crop_heights[ci, p, v, ki] = bh
                        crop_areas[ci, p, v, ki] = bw * bh
                        crop_border_dist[ci, p, v, ki] = bdist
                        crop_truncated[ci, p, v, ki] = trunc
                        crop_max_iou[ci, p, v, ki] = max_iou
                        crop_kp_vis_ratio[ci, p, v, ki] = float(kp_vis[p, v, fidx].mean())
                        crop_kp_conf_mean[ci, p, v, ki] = (
                            float(kp_conf[p, v, fidx][kp_vis[p, v, fidx]].mean())
                            if kp_vis[p, v, fidx].any()
                            else 0.0
                        )
                        crop_blur_var[ci, p, v, ki] = blur_v
                        crop_facing[ci, p, v, ki] = compute_facing_score(
                            uv_px[p, v, fidx], kp_vis[p, v, fidx], kp_conf[p, v, fidx]
                        )
                        crop_kp_norm[ci, p, v, ki] = kp_n
                        crop_neg_count[ci, p, v, ki] = neg_cnt

                        tensor_chw = reid_crop(frame, box, encoder.input_size)
                        crop_tensors.append(tensor_chw)
                        crop_coords.append((p, v, ki))
                fidx += 1
            cap.release()

        if crop_tensors:
            if use_cuda:
                torch.cuda.synchronize()
            tg0 = time.perf_counter()
            embs_list = []
            for start in range(0, len(crop_tensors), batch_size):
                batch_t = torch.from_numpy(np.stack(crop_tensors[start : start + batch_size], axis=0))
                embs_list.append(encoder.embed(batch_t).numpy())
            if use_cuda:
                torch.cuda.synchronize()
            gpu_sec += time.perf_counter() - tg0
            all_embs = np.concatenate(embs_list, axis=0)
            for (p, v, ki), emb in zip(crop_coords, all_embs, strict=True):
                embeddings[ci, p, v, ki] = emb
            total_crops_extracted += len(crop_tensors)

        clip_meta_records.append(
            {
                "clip_idx": ci,
                "clip_name": cname,
                "clip_id": cmeta["clip_id"],
                "num_frames": int(cmeta["num_frames"]),
                "fps": fps,
                "global_start_sec": float(cmeta["global_start_sec"]),
                "global_end_sec": float(cmeta["global_end_sec"]),
                "cameras": [
                    {
                        "camera_id": cam["camera_id"],
                        "offset_sec": float(cam["offset_sec"]),
                        "source_frame_start": int(cam["source_frame_start"]),
                        "source_frame_end": int(cam["source_frame_end"]),
                    }
                    for cam in cmeta["cameras"]
                ],
                "score": proj_clips.get(cname, {}),
                "scene_export_sha256": sj["exports"]["scene"]["files"]["scene"]["sha256"],
                "msi_artifact_id": sj["artifacts"]["multiview_scene_inputs"]["artifact_id"],
                "gvhmr_artifact_id": sj["artifacts"]["gvhmr"]["artifact_id"],
            }
        )

    npz_path = out_dir / "clip_reid_cache.npz"
    np.savez_compressed(
        npz_path,
        clip_names=np.asarray(clip_names),
        embeddings=embeddings,
        crop_valid=crop_valid,
        crop_frames=crop_frames,
        crop_source_rows=crop_source_rows,
        crop_boxes=crop_boxes,
        crop_heights=crop_heights,
        crop_areas=crop_areas,
        crop_border_dist=crop_border_dist,
        crop_truncated=crop_truncated,
        crop_max_iou=crop_max_iou,
        crop_kp_vis_ratio=crop_kp_vis_ratio,
        crop_kp_conf_mean=crop_kp_conf_mean,
        crop_blur_var=crop_blur_var,
        crop_facing=crop_facing,
        crop_kp_norm=crop_kp_norm,
        crop_neg_kp_norm=crop_neg_kp_norm,
        crop_neg_count=crop_neg_count,
        sides=sides,
        obs_frames_pv=obs_frames_pv,
        id_amb_counts=id_amb_counts,
        frame_amb_counts=frame_amb_counts,
        smpl_betas=smpl_betas,
        smpl_valid_frames=smpl_valid_frames,
        gvhmr_seg_count=gvhmr_seg_count,
        gvhmr_obs_samples=gvhmr_obs_samples,
        gvhmr_beta_std=gvhmr_beta_std,
        gvhmr_rep_cam=gvhmr_rep_cam,
        pos_start_3d=pos_start_3d,
        pos_end_3d=pos_end_3d,
        pos_mean_3d=pos_mean_3d,
        vel_start_3d=vel_start_3d,
        vel_end_3d=vel_end_3d,
    )

    wall_sec = time.perf_counter() - t0_wall
    cpu_sec = time.process_time() - t0_cpu
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024.0**2)) if use_cuda else 0.0

    manifest = {
        "schema_version": 1,
        "dataset_id": "meiji_4cam",
        "video_id": "video_000",
        "benchmark_version": "multiview-v4",
        "benchmark_manifest_sha256": file_sha256(bench_root / "manifest.json"),
        "projects_json_sha256": file_sha256(projects_json),
        "encoder": {
            "profile": CLIP_REID_PROFILE,
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": file_sha256(ckpt_path),
            "input_size_hw": list(encoder.input_size),
            "feature_dim": 1280,
            "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
            "l2_normalized": True,
        },
        "sampling": {
            "crops_per_camera": crops_per_camera,
            "max_crops_per_person_clip": V_MAX * crops_per_camera,
            "min_height_px": MIN_HEIGHT_PX,
            "border_px": BORDER_PX,
            "strategy": "deterministic_uniform_linspace_over_real_observed_frames",
        },
        "stats": {
            "num_clips": n_clips,
            "num_nodes": n_clips * P_MAX,
            "total_crops_extracted": total_crops_extracted,
            "expected_max_crops": n_clips * P_MAX * V_MAX * K_MAX,
            "missing_camera_node_pairs": int((crop_valid.sum(axis=-1) == 0).sum()),
            "cache_npz_path": str(npz_path),
            "cache_npz_sha256": file_sha256(npz_path),
            "cache_npz_bytes": npz_path.stat().st_size,
            "wall_sec": round(wall_sec, 3),
            "cpu_sec": round(cpu_sec, 3),
            "gpu_embed_sec": round(gpu_sec, 3),
            "peak_rss_mb": round(peak_rss_mb, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "device": device,
        },
        "clips": clip_meta_records,
    }
    manifest_path = out_dir / "clip_reid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(
        f"Extracted {total_crops_extracted} crops across {n_clips} clips in {wall_sec:.2f}s "
        f"(gpu={gpu_sec:.2f}s, peak_vram={peak_vram_mb:.1f}MB, cache={npz_path.stat().st_size/1e6:.2f}MB)"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--projects-json", type=Path, default=DEFAULT_PROJECTS_JSON)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/i2746_cross_clip_player_id/cache"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    extract_all(args.bench_root, args.projects_json, args.ckpt, args.out_dir, args.device, args.batch_size)


if __name__ == "__main__":
    main()
