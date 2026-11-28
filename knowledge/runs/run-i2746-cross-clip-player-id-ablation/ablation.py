"""Core E0-E7 cross-clip player identity linking ablation engine (Issue #2746).

Implements:
- Deterministic warm-up anchor selection (input-quality only, GT-independent)
- Fixed chronological game-block dev/test split (no source-frame overlap)
- E0 (clip-local sanity), E1 (all-cam mean NN), E2 (per-cam mean NN),
  E3 (quality-weighted per-cam NN), E4 (E3 + Hungarian 1-to-1 with dummy),
  E5 (E4 + quality-gated SMPL beta shape), E6 (E4 + short-gap 3D continuity),
  E7 (E4 + high-confidence causal gallery update)
- Stratified analysis, 1-camera dropout robustness, leave-one-out open-set test,
  precision-coverage curves, and SMPL beta separability diagnostics.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from contract import (  # noqa: E402
    SCHEMA_VERSION,
    IdentityRegistry,
    LinkEvidence,
    PlayerIdentityLink,
    PlayerIdentityLinksDocument,
)
from metrics import (  # noqa: E402
    EvalNode,
    compute_scope_metrics,
    precision_coverage_curve,
)


@dataclass(frozen=True)
class CacheData:
    clip_names: list[str]
    embeddings: np.ndarray  # (N, 4, 4, K, 1280)
    crop_valid: np.ndarray  # (N, 4, 4, K)
    crop_frames: np.ndarray  # (N, 4, 4, K)
    crop_source_rows: np.ndarray  # (N, 4, 4, K)
    crop_heights: np.ndarray  # (N, 4, 4, K)
    crop_areas: np.ndarray  # (N, 4, 4, K)
    crop_truncated: np.ndarray  # (N, 4, 4, K)
    crop_max_iou: np.ndarray  # (N, 4, 4, K)
    crop_kp_vis_ratio: np.ndarray  # (N, 4, 4, K)
    crop_kp_conf_mean: np.ndarray  # (N, 4, 4, K)
    crop_blur_var: np.ndarray  # (N, 4, 4, K)
    crop_facing: np.ndarray  # (N, 4, 4, K)
    sides: np.ndarray  # (N, 4)
    obs_frames_pv: np.ndarray  # (N, 4, 4)
    id_amb_counts: np.ndarray  # (N, 4)
    smpl_betas: np.ndarray  # (N, 4, 10)
    smpl_valid_frames: np.ndarray  # (N, 4)
    gvhmr_seg_count: np.ndarray  # (N, 4)
    gvhmr_obs_samples: np.ndarray  # (N, 4)
    gvhmr_beta_std: np.ndarray  # (N, 4, 10)
    gvhmr_rep_cam: np.ndarray  # (N, 4)
    pos_start_3d: np.ndarray  # (N, 4, 3)
    pos_end_3d: np.ndarray  # (N, 4, 3)
    pos_mean_3d: np.ndarray  # (N, 4, 3)
    vel_start_3d: np.ndarray  # (N, 4, 3)
    vel_end_3d: np.ndarray  # (N, 4, 3)
    manifest: dict[str, Any]


def load_cache(cache_dir: Path) -> CacheData:
    z = np.load(cache_dir / "clip_reid_cache.npz", allow_pickle=False)
    manifest = json.loads((cache_dir / "clip_reid_manifest.json").read_text())
    return CacheData(
        clip_names=[str(x) for x in z["clip_names"].tolist()],
        embeddings=z["embeddings"],
        crop_valid=z["crop_valid"],
        crop_frames=z["crop_frames"],
        crop_source_rows=z["crop_source_rows"],
        crop_heights=z["crop_heights"],
        crop_areas=z["crop_areas"],
        crop_truncated=z["crop_truncated"],
        crop_max_iou=z["crop_max_iou"],
        crop_kp_vis_ratio=z["crop_kp_vis_ratio"],
        crop_kp_conf_mean=z["crop_kp_conf_mean"],
        crop_blur_var=z["crop_blur_var"],
        crop_facing=z["crop_facing"],
        sides=z["sides"],
        obs_frames_pv=z["obs_frames_pv"],
        id_amb_counts=z["id_amb_counts"],
        smpl_betas=z["smpl_betas"],
        smpl_valid_frames=z["smpl_valid_frames"],
        gvhmr_seg_count=z["gvhmr_seg_count"],
        gvhmr_obs_samples=z["gvhmr_obs_samples"],
        gvhmr_beta_std=z["gvhmr_beta_std"],
        gvhmr_rep_cam=z["gvhmr_rep_cam"],
        pos_start_3d=z["pos_start_3d"],
        pos_end_3d=z["pos_end_3d"],
        pos_mean_3d=z["pos_mean_3d"],
        vel_start_3d=z["vel_start_3d"],
        vel_end_3d=z["vel_end_3d"],
        manifest=manifest,
    )


def compute_crop_quality_weights(cache: CacheData) -> np.ndarray:
    """Pre-fixed input-only crop quality weight in [0.05, 1.0] for E3+ (no test/dev fitting)."""
    h_fac = np.clip(cache.crop_heights / 120.0, 0.15, 1.0)
    trunc_fac = np.where(cache.crop_truncated, 0.35, 1.0)
    iou_fac = np.exp(-3.0 * np.clip(cache.crop_max_iou, 0.0, 1.0))
    vis_fac = 0.30 + 0.70 * np.clip(cache.crop_kp_vis_ratio, 0.0, 1.0)
    blur_fac = np.clip(np.sqrt(np.maximum(cache.crop_blur_var, 1.0) / 150.0), 0.25, 1.0)
    w = h_fac * trunc_fac * iou_fac * vis_fac * blur_fac
    w = np.clip(w, 0.05, 1.0).astype(np.float32)
    return np.where(cache.crop_valid, w, 0.0).astype(np.float32)


def select_warmup_clip_from_input_quality(cache: CacheData, candidate_indices: list[int] | None = None) -> dict[str, Any]:
    """Deterministically pick 1 warm-up registration clip strictly from input quality (no GT)."""
    weights = compute_crop_quality_weights(cache)
    cands = candidate_indices if candidate_indices is not None else list(range(5))  # Game 1 clips
    best_ci = cands[0]
    best_score = -1.0
    details = []
    for ci in cands:
        all_cams_ok = bool((cache.crop_valid[ci].sum(axis=-1) > 0).all())
        no_id_amb = bool((cache.id_amb_counts[ci] == 0).all())
        mean_q = float(weights[ci][cache.crop_valid[ci]].mean()) if cache.crop_valid[ci].any() else 0.0
        min_node_q = float(
            min(
                weights[ci, p][cache.crop_valid[ci, p]].mean()
                if cache.crop_valid[ci, p].any()
                else 0.0
                for p in range(4)
            )
        )
        # Prefer complete 4-cam, zero id_ambiguous, high minimum player quality
        score = (10.0 if all_cams_ok else 0.0) + (2.0 if no_id_amb else 0.0) + 0.5 * mean_q + 0.5 * min_node_q
        details.append(
            {
                "clip_idx": ci,
                "clip_name": cache.clip_names[ci],
                "all_cams_ok": all_cams_ok,
                "no_id_amb": no_id_amb,
                "mean_quality": round(mean_q, 4),
                "min_node_quality": round(min_node_q, 4),
                "selection_score": round(score, 4),
            }
        )
        if score > best_score:
            best_score = score
            best_ci = ci
    return {"warmup_clip_idx": best_ci, "warmup_clip_name": cache.clip_names[best_ci], "candidates": details}


def build_split_manifest(cache: CacheData, warmup_clip_idx: int) -> dict[str, Any]:
    """Fixed chronological game-block dev/test split:
    - warmup: 1 clip (excluded from all query scoring)
    - dev: Games 1-4 (clip_000..clip_034 excluding warmup) -> 34 clips
    - test: Games 5-9 (clip_035..clip_068) -> 34 clips
    """
    dev_indices = [i for i in range(0, 35) if i != warmup_clip_idx]
    test_indices = list(range(35, 69))

    def game_block(ci: int) -> str:
        sc = cache.manifest["clips"][ci].get("score") or {}
        games = (sc.get("before") or {}).get("games")
        if games == [4, 4]:
            # Final game has end-changes every 2 points; split into 2 blocks
            pts = (sc.get("before") or {}).get("points") or [0, 0]
            return f"G4-4_half{0 if sum(pts) < 6 else 1}"
        return f"G{games[0]}-{games[1]}" if games else f"clip_{ci // 5}"

    # Verify zero source-frame overlap across splits
    split_intervals: dict[str, list[tuple[int, int, str, str]]] = {"warmup": [], "dev": [], "test": []}
    for ci in range(len(cache.clip_names)):
        sp = "warmup" if ci == warmup_clip_idx else ("dev" if ci in dev_indices else "test")
        for cam in cache.manifest["clips"][ci]["cameras"]:
            split_intervals[sp].append(
                (int(cam["source_frame_start"]), int(cam["source_frame_end"]), str(cam["camera_id"]), cache.clip_names[ci])
            )
    overlaps = 0
    for cam_id in ("cam0", "cam1", "cam2", "cam3"):
        all_iv = []
        for sp, rows in split_intervals.items():
            for s, e, cid, cn in rows:
                if cid == cam_id:
                    all_iv.append((s, e, sp, cn))
        all_iv.sort()
        for (_s1, e1, _sp1, _n1), (s2, _e2, _sp2, _n2) in zip(all_iv, all_iv[1:], strict=False):
            if s2 <= e1:
                overlaps += 1

    return {
        "warmup_clip_idx": warmup_clip_idx,
        "warmup_clip_name": cache.clip_names[warmup_clip_idx],
        "dev_indices": dev_indices,
        "test_indices": test_indices,
        "dev_clip_names": [cache.clip_names[i] for i in dev_indices],
        "test_clip_names": [cache.clip_names[i] for i in test_indices],
        "clip_blocks": {cache.clip_names[i]: game_block(i) for i in range(len(cache.clip_names))},
        "source_frame_overlaps_verified": overlaps,
    }


def _l2_norm(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n <= 1e-12:
        return None
    return (v / n).astype(np.float64)


def compute_node_descriptors(
    cache: CacheData,
    *,
    quality_weights: np.ndarray | None,
    camera_mask: tuple[bool, bool, bool, bool] = (True, True, True, True),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-node pooled (N,4,1280) and per-camera (N,4,4,1280) unit descriptors + valid masks."""
    N, P, V, K, D = cache.embeddings.shape
    pooled = np.zeros((N, P, D), dtype=np.float64)
    pooled_valid = np.zeros((N, P), dtype=bool)
    per_cam = np.zeros((N, P, V, D), dtype=np.float64)
    per_cam_valid = np.zeros((N, P, V), dtype=bool)

    for ci in range(N):
        for p in range(P):
            all_vecs = []
            all_w = []
            for v in range(V):
                if not camera_mask[v]:
                    continue
                valid_k = cache.crop_valid[ci, p, v]
                if not valid_k.any():
                    continue
                embs = cache.embeddings[ci, p, v, valid_k].astype(np.float64)
                w = (
                    quality_weights[ci, p, v, valid_k].astype(np.float64)
                    if quality_weights is not None
                    else np.ones(len(embs), dtype=np.float64)
                )
                agg = _l2_norm((embs * w[:, None]).sum(axis=0))
                if agg is not None:
                    per_cam[ci, p, v] = agg
                    per_cam_valid[ci, p, v] = True
                all_vecs.append(embs)
                all_w.append(w)
            if all_vecs:
                cat_e = np.concatenate(all_vecs, axis=0)
                cat_w = np.concatenate(all_w, axis=0)
                p_agg = _l2_norm((cat_e * cat_w[:, None]).sum(axis=0))
                if p_agg is not None:
                    pooled[ci, p] = p_agg
                    pooled_valid[ci, p] = True

    return pooled, pooled_valid, per_cam, per_cam_valid


def analyze_smpl_beta_separability(
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    clip_indices: list[int],
    min_valid_frames: int = 30,
    min_obs_samples: int = 15,
    max_beta_std: float = 0.60,
) -> dict[str, Any]:
    """Dev-only diagnostic of SMPL beta intra-person vs inter-person distance."""
    valid_nodes = []
    unavailable = 0
    for ci in clip_indices:
        cname = cache.clip_names[ci]
        for p in range(4):
            gid = gt_map.get((cname, p))
            if gid is None:
                continue
            b = cache.smpl_betas[ci, p].astype(np.float64)
            is_ok = (
                int(cache.smpl_valid_frames[ci, p]) >= min_valid_frames
                and int(cache.gvhmr_obs_samples[ci, p]) >= min_obs_samples
                and float(np.linalg.norm(b)) > 1e-6
                and float(cache.gvhmr_beta_std[ci, p].mean()) <= max_beta_std
            )
            if not is_ok:
                unavailable += 1
                continue
            valid_nodes.append((ci, p, gid, b, int(cache.gvhmr_rep_cam[ci, p])))

    same_dists = []
    diff_dists = []
    same_cam_same = []
    diff_cam_same = []
    for i in range(len(valid_nodes)):
        ci1, _, g1, b1, rc1 = valid_nodes[i]
        for j in range(i + 1, len(valid_nodes)):
            ci2, _, g2, b2, rc2 = valid_nodes[j]
            if ci1 == ci2:
                continue
            d = float(np.linalg.norm(b1 - b2))
            if g1 == g2:
                same_dists.append(d)
                if rc1 == rc2:
                    same_cam_same.append(d)
                else:
                    diff_cam_same.append(d)
            else:
                diff_dists.append(d)

    same_arr = np.asarray(same_dists, dtype=np.float64)
    diff_arr = np.asarray(diff_dists, dtype=np.float64)
    # AUROC for smaller distance = same person
    auroc = None
    if len(same_arr) > 0 and len(diff_arr) > 0:
        wins = float((same_arr[:, None] < diff_arr[None, :]).mean()) + 0.5 * float(
            (same_arr[:, None] == diff_arr[None, :]).mean()
        )
        auroc = round(wins, 4)

    return {
        "valid_nodes": len(valid_nodes),
        "unavailable_nodes": unavailable,
        "valid_rate": round(len(valid_nodes) / max(1, len(valid_nodes) + unavailable), 4),
        "same_person_l2_mean": round(float(same_arr.mean()), 4) if len(same_arr) else None,
        "same_person_l2_std": round(float(same_arr.std()), 4) if len(same_arr) else None,
        "diff_person_l2_mean": round(float(diff_arr.mean()), 4) if len(diff_arr) else None,
        "diff_person_l2_std": round(float(diff_arr.std()), 4) if len(diff_arr) else None,
        "same_rep_cam_same_person_l2_mean": round(float(np.mean(same_cam_same)), 4) if same_cam_same else None,
        "diff_rep_cam_same_person_l2_mean": round(float(np.mean(diff_cam_same)), 4) if diff_cam_same else None,
        "pairwise_auroc": auroc,
        "gate_params": {
            "min_valid_frames": min_valid_frames,
            "min_obs_samples": min_obs_samples,
            "max_beta_std": max_beta_std,
        },
    }


def analyze_temporal_gaps(cache: CacheData, short_gap_max_sec: float = 3.0) -> dict[str, Any]:
    """Audit inter-clip gaps for E6 short-gap 3D continuity applicability."""
    clips = cache.manifest["clips"]
    gaps = []
    for i in range(1, len(clips)):
        dt = float(clips[i]["global_start_sec"]) - float(clips[i - 1]["global_end_sec"])
        gaps.append({"prev": clips[i - 1]["clip_name"], "curr": clips[i]["clip_name"], "gap_sec": round(dt, 3)})
    vals = [g["gap_sec"] for g in gaps]
    eligible = [g for g in gaps if 0.0 <= g["gap_sec"] <= short_gap_max_sec]
    return {
        "short_gap_max_sec": short_gap_max_sec,
        "total_consecutive_pairs": len(gaps),
        "eligible_pairs": len(eligible),
        "eligible_rate": round(len(eligible) / max(1, len(gaps)), 6),
        "min_gap_sec": min(vals) if vals else None,
        "median_gap_sec": round(float(np.median(vals)), 3) if vals else None,
        "max_gap_sec": max(vals) if vals else None,
        "gaps_under_10s": sum(1 for v in vals if v < 10.0),
    }


def _hungarian_with_dummy(
    score_mat: np.ndarray,
    min_score: float,
    *,
    use_swap_margin: bool = False,
) -> tuple[list[int | None], list[float | None], list[float | None], list[float | None]]:
    """Solve 1-to-1 assignment of P queries to K gallery slots, each query also allowed an
    unassigned dummy slot with score `min_score`.
    Returns assigned gallery index (or None), best score, selected margin, and assignment swap margin.
    """
    P, K = score_mat.shape
    aug = np.full((P, K + P), -1e6, dtype=np.float64)
    aug[:, :K] = np.where(np.isfinite(score_mat), score_mat, -1e6)
    for p in range(P):
        aug[p, K + p] = min_score

    row_ind, col_ind = linear_sum_assignment(-aug)
    opt_total = float(aug[row_ind, col_ind].sum())
    assigned: list[int | None] = [None] * P
    best_scores: list[float | None] = [None] * P
    row_margins: list[float | None] = [None] * P
    swap_margins: list[float | None] = [None] * P

    for r, c in zip(row_ind, col_ind, strict=True):
        if c < K and np.isfinite(score_mat[r, c]) and score_mat[r, c] >= min_score:
            assigned[r] = int(c)
            s_best = float(score_mat[r, c])
            best_scores[r] = s_best
            others = [float(score_mat[r, k]) for k in range(K) if k != c and np.isfinite(score_mat[r, k])]
            runner = max(others) if others else -1.0
            row_margins[r] = s_best - runner
            # Assignment swap margin: objective drop if edge (r, c) is forbidden
            aug_forbid = aug.copy()
            aug_forbid[r, c] = -1e6
            r2, c2 = linear_sum_assignment(-aug_forbid)
            swap_margins[r] = opt_total - float(aug_forbid[r2, c2].sum())
        else:
            valid_row = [float(score_mat[r, k]) for k in range(K) if np.isfinite(score_mat[r, k])]
            if valid_row:
                s_sorted = sorted(valid_row, reverse=True)
                best_scores[r] = s_sorted[0]
                row_margins[r] = s_sorted[0] - (s_sorted[1] if len(s_sorted) > 1 else -1.0)
                swap_margins[r] = 0.0
    chosen_margins = swap_margins if use_swap_margin else row_margins
    return assigned, best_scores, chosen_margins, swap_margins


def run_method(
    method: str,
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    clip_indices: list[int],
    *,
    min_score: float = 0.40,
    min_margin: float = 0.03,
    shape_weight: float = 0.08,
    shape_gate_max_std: float = 0.60,
    update_min_score: float = 0.72,
    update_min_margin: float = 0.08,
    update_min_quality: float = 0.35,
    max_gallery_per_cam: int = 8,
    short_gap_max_sec: float = 3.0,
    camera_mask: tuple[bool, bool, bool, bool] = (True, True, True, True),
    excluded_anchor_gt_id: str | None = None,
    cross_cam_fallback: bool = False,
    use_swap_margin: bool = False,
    force_hungarian: bool = False,
    n_bootstrap: int = 400,
) -> dict[str, Any]:
    """Run one method (E0..E7) on `clip_indices` using the warm-up clip as fixed initial gallery."""
    t0 = time.perf_counter()
    t0_cpu = time.process_time()

    warmup_ci = int(split["warmup_clip_idx"])
    warmup_cname = cache.clip_names[warmup_ci]
    use_quality = method in {"E3", "E4", "E5", "E6", "E7"}
    q_weights = compute_crop_quality_weights(cache) if use_quality else None
    all_q_weights = compute_crop_quality_weights(cache)

    pooled, pooled_valid, per_cam, per_cam_valid = compute_node_descriptors(
        cache, quality_weights=q_weights, camera_mask=camera_mask
    )

    # Initialize persistent registry from warm-up clip (optionally omitting 1 player for open-set test)
    registry = IdentityRegistry(identity_scope="meiji_4cam/video_000")
    gallery_ids: list[str] = []
    gallery_pooled: list[np.ndarray] = []
    # Per-camera gallery lists (allow E7 updates up to max_gallery_per_cam)
    gallery_per_cam: list[dict[int, list[np.ndarray]]] = []
    gallery_betas: list[np.ndarray | None] = []
    gallery_gt_id: dict[str, str | None] = {}

    for p in range(4):
        gt_p = gt_map.get((warmup_cname, p))
        if excluded_anchor_gt_id is not None and gt_p == excluded_anchor_gt_id:
            continue
        gid = registry.register_anchor(
            f"{warmup_cname}:p{p}",
            clip_id=f"video_000/{warmup_cname}",
            local_player_id=p,
            metadata={"side": int(cache.sides[warmup_ci, p])},
        )
        gallery_ids.append(gid)
        gallery_gt_id[gid] = gt_p
        gallery_pooled.append(pooled[warmup_ci, p].copy())
        cam_dict: dict[int, list[np.ndarray]] = {}
        for v in range(4):
            if per_cam_valid[warmup_ci, p, v]:
                cam_dict[v] = [per_cam[warmup_ci, p, v].copy()]
        gallery_per_cam.append(cam_dict)
        # Warm-up beta validity
        b = cache.smpl_betas[warmup_ci, p].astype(np.float64)
        b_ok = (
            int(cache.smpl_valid_frames[warmup_ci, p]) >= 30
            and int(cache.gvhmr_obs_samples[warmup_ci, p]) >= 15
            and float(np.linalg.norm(b)) > 1e-6
            and float(cache.gvhmr_beta_std[warmup_ci, p].mean()) <= shape_gate_max_std
        )
        gallery_betas.append(b.copy() if b_ok else None)

    K = len(gallery_ids)
    eval_nodes: list[EvalNode] = []
    links: list[PlayerIdentityLink] = []

    # Diagnostics
    shape_used = 0
    shape_unavailable = 0
    temporal_used = 0
    temporal_unavailable = 0
    gallery_updates_total = 0
    gallery_updates_wrong = 0
    wrong_update_clip_idx: int | None = None
    post_contamination_fp_nodes = 0
    no_common_camera_nodes = 0

    # Chronological order for query clips
    for ci in sorted(clip_indices):
        cname = cache.clip_names[ci]
        block_id = split["clip_blocks"][cname]
        P = 4

        if method == "E0":
            for p in range(P):
                pid_str = f"{cname}_p{p}"
                gt_p = gt_map.get((cname, p))
                eval_nodes.append(
                    EvalNode(
                        clip_idx=ci,
                        clip_name=cname,
                        local_player_id=p,
                        gt_global_id=gt_p,
                        pred_global_id=pid_str,
                        status="accepted",
                        score=1.0,
                        margin=1.0,
                        block_id=block_id,
                        num_valid_frames=int(cache.obs_frames_pv[ci, p].max()),
                        is_pure=purity_map.get((cname, p), True),
                    )
                )
                links.append(
                    PlayerIdentityLink(
                        clip_id=f"video_000/{cname}",
                        local_player_id=p,
                        global_player_id=pid_str,
                        status="accepted",
                        score=1.0,
                        score_kind="none",
                        runner_up_score=None,
                        assignment_margin=None,
                        reason="clip_local_only_baseline",
                        evidence=LinkEvidence((), {}, {}),
                    )
                )
            continue

        # Compute P x K similarity matrix
        app_mat = np.full((P, K), np.nan, dtype=np.float64)
        shape_mat = np.zeros((P, K), dtype=np.float64)
        shape_valid_p = [False] * P
        common_cams_pk: dict[tuple[int, int], list[int]] = {}

        for p in range(P):
            # Check SMPL beta validity for E5
            if method == "E5":
                b = cache.smpl_betas[ci, p].astype(np.float64)
                b_ok = (
                    int(cache.smpl_valid_frames[ci, p]) >= 30
                    and int(cache.gvhmr_obs_samples[ci, p]) >= 15
                    and float(np.linalg.norm(b)) > 1e-6
                    and float(cache.gvhmr_beta_std[ci, p].mean()) <= shape_gate_max_std
                )
                if b_ok:
                    shape_valid_p[p] = True
                    shape_used += 1
                else:
                    shape_unavailable += 1

            # Check temporal continuity eligibility for E6
            if method == "E6":
                prev_ci = ci - 1
                if prev_ci >= 0:
                    dt = float(cache.manifest["clips"][ci]["global_start_sec"]) - float(
                        cache.manifest["clips"][prev_ci]["global_end_sec"]
                    )
                    if 0.0 <= dt <= short_gap_max_sec:
                        temporal_used += 1
                    else:
                        temporal_unavailable += 1
                else:
                    temporal_unavailable += 1

            for k in range(K):
                if method == "E1":
                    if pooled_valid[ci, p]:
                        app_mat[p, k] = float(np.clip(pooled[ci, p] @ gallery_pooled[k], -1.0, 1.0))
                        common_cams_pk[(p, k)] = [v for v in range(4) if per_cam_valid[ci, p, v]]
                else:
                    # Per-camera mean similarity over common observed physical cameras
                    sims = []
                    used_v = []
                    for v in range(4):
                        if per_cam_valid[ci, p, v] and v in gallery_per_cam[k] and gallery_per_cam[k][v]:
                            # Average prototype in that camera (for E7, normalized mean of gallery entries in cam v)
                            g_vecs = gallery_per_cam[k][v]
                            g_proto = _l2_norm(np.mean(g_vecs, axis=0))
                            if g_proto is not None:
                                sims.append(float(np.clip(per_cam[ci, p, v] @ g_proto, -1.0, 1.0)))
                                used_v.append(v)
                    if sims:
                        app_mat[p, k] = float(np.mean(sims))
                        common_cams_pk[(p, k)] = used_v
                    elif cross_cam_fallback and pooled_valid[ci, p]:
                        # Explicit optional cross-camera fallback (off by default in main ablation)
                        all_g = [vec for v_list in gallery_per_cam[k].values() for vec in v_list]
                        g_proto = _l2_norm(np.mean(all_g, axis=0)) if all_g else None
                        if g_proto is not None:
                            app_mat[p, k] = float(np.clip(pooled[ci, p] @ g_proto, -1.0, 1.0))
                            common_cams_pk[(p, k)] = []

                if method == "E5" and shape_valid_p[p] and gallery_betas[k] is not None:
                    d_beta = float(np.linalg.norm(cache.smpl_betas[ci, p].astype(np.float64) - gallery_betas[k]))
                    # Convert L2 beta distance into bounded affinity bonus/penalty: exp(-d/scale) - 0.5
                    shape_mat[p, k] = float(np.exp(-d_beta / 2.5) - 0.5)

        score_mat = app_mat + (shape_weight * shape_mat if method == "E5" else 0.0)

        # Assignment stage
        if method in {"E1", "E2", "E3"} and not force_hungarian:
            # Independent nearest-neighbor per query node
            assigned_k: list[int | None] = [None] * P
            best_s: list[float | None] = [None] * P
            runner_s: list[float | None] = [None] * P
            margins: list[float | None] = [None] * P
            for p in range(P):
                row = [(float(score_mat[p, k]), k) for k in range(K) if np.isfinite(score_mat[p, k])]
                if not row:
                    continue
                row.sort(reverse=True)
                s0, k0 = row[0]
                s1 = row[1][0] if len(row) > 1 else -1.0
                best_s[p] = s0
                runner_s[p] = s1
                margins[p] = s0 - s1
                if s0 >= min_score and (s0 - s1) >= min_margin:
                    assigned_k[p] = k0
        else:
            # Hungarian 1-to-1 with dummy unassigned slots (E4, E5, E6, E7)
            assigned_k, best_s, margins, _swap_m = _hungarian_with_dummy(
                score_mat, min_score, use_swap_margin=use_swap_margin
            )
            runner_s = [
                (best_s[p] - margins[p]) if (best_s[p] is not None and margins[p] is not None) else None
                for p in range(P)
            ]
            # Apply margin gate
            for p in range(P):
                if assigned_k[p] is not None and (margins[p] is None or margins[p] < min_margin):
                    assigned_k[p] = None

        # Emit links + EvalNodes and (for E7) perform causal high-confidence gallery updates
        for p in range(P):
            k_sel = assigned_k[p]
            gt_p = gt_map.get((cname, p))
            has_any_score = any(np.isfinite(score_mat[p, k]) for k in range(K))
            if not has_any_score:
                no_common_camera_nodes += 1
                status = "unresolved"
                pred_gid = None
                reason = "no_common_camera"
            elif k_sel is None:
                status = "unresolved"
                pred_gid = None
                if best_s[p] is not None and best_s[p] < min_score:
                    reason = "below_min_score"
                else:
                    reason = "below_min_margin"
            else:
                status = "accepted"
                pred_gid = gallery_ids[k_sel]
                reason = "matched"
                if (
                    wrong_update_clip_idx is not None
                    and ci > wrong_update_clip_idx
                    and gallery_gt_id.get(pred_gid) != gt_p
                ):
                    post_contamination_fp_nodes += 1

            cams_used = tuple(f"cam{v}" for v in common_cams_pk.get((p, k_sel if k_sel is not None else 0), []))
            s_frames = {
                f"cam{v}": [int(f) for f in cache.crop_frames[ci, p, v][cache.crop_valid[ci, p, v]].tolist()]
                for v in range(4)
                if camera_mask[v] and cache.crop_valid[ci, p, v].any()
            }
            s_rows = {
                f"cam{v}": [int(r) for r in cache.crop_source_rows[ci, p, v][cache.crop_valid[ci, p, v]].tolist()]
                for v in range(4)
                if camera_mask[v] and cache.crop_valid[ci, p, v].any()
            }
            ev = LinkEvidence(
                cameras_used=cams_used,
                sample_frames=s_frames,
                sample_source_rows=s_rows,
                appearance_term=float(app_mat[p, k_sel]) if (k_sel is not None and np.isfinite(app_mat[p, k_sel])) else best_s[p],
                shape_term=float(shape_mat[p, k_sel]) if (method == "E5" and k_sel is not None and shape_valid_p[p]) else None,
                temporal_term=None,
                shape_valid=shape_valid_p[p] if method == "E5" else False,
                temporal_valid=False,
                extra={
                    "shape_reason": None if (method != "E5" or shape_valid_p[p]) else "shape_unavailable",
                    "temporal_reason": "temporal_gap_too_large" if method == "E6" else None,
                },
            )
            links.append(
                PlayerIdentityLink(
                    clip_id=f"video_000/{cname}",
                    local_player_id=p,
                    global_player_id=pred_gid,
                    status=status,  # type: ignore[arg-type]
                    score=best_s[p],
                    score_kind="combined_score" if method == "E5" else "cosine_similarity",
                    runner_up_score=runner_s[p],
                    assignment_margin=margins[p],
                    reason=reason,
                    evidence=ev,
                )
            )
            # For precision-coverage sweep, keep top candidate gid even if unresolved when score exists
            top_k_cand = k_sel
            if top_k_cand is None and has_any_score:
                top_k_cand = int(np.nanargmax(score_mat[p]))
            eval_nodes.append(
                EvalNode(
                    clip_idx=ci,
                    clip_name=cname,
                    local_player_id=p,
                    gt_global_id=gt_p,
                    pred_global_id=pred_gid if status == "accepted" else (gallery_ids[top_k_cand] if top_k_cand is not None else None),
                    status=status,
                    score=best_s[p],
                    margin=margins[p],
                    block_id=block_id,
                    num_valid_frames=int(cache.obs_frames_pv[ci, p].max()),
                    is_pure=purity_map.get((cname, p), True),
                )
            )

        # Causal gallery update for E7 (strictly after clip assignment is finalized)
        if method == "E7":
            for p in range(P):
                k_sel = assigned_k[p]
                if k_sel is None or best_s[p] is None or margins[p] is None:
                    continue
                mean_q = float(all_q_weights[ci, p][cache.crop_valid[ci, p]].mean()) if cache.crop_valid[ci, p].any() else 0.0
                if best_s[p] >= update_min_score and margins[p] >= update_min_margin and mean_q >= update_min_quality:
                    updated_any = False
                    for v in range(4):
                        if per_cam_valid[ci, p, v]:
                            g_list = gallery_per_cam[k_sel].setdefault(v, [])
                            if len(g_list) < max_gallery_per_cam:
                                g_list.append(per_cam[ci, p, v].copy())
                                updated_any = True
                    if updated_any:
                        gallery_updates_total += 1
                        gid = gallery_ids[k_sel]
                        registry.record_gallery_update(
                            gid,
                            clip_id=f"video_000/{cname}",
                            local_player_id=p,
                            score=best_s[p],
                            margin=margins[p],
                        )
                        if gallery_gt_id.get(gid) != gt_map.get((cname, p)):
                            gallery_updates_wrong += 1
                            if wrong_update_clip_idx is None:
                                wrong_update_clip_idx = ci

    wall_ms = (time.perf_counter() - t0) * 1000.0
    cpu_ms = (time.process_time() - t0_cpu) * 1000.0
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # Metrics: for EvalNode, unresolved must not use top_k_cand as accepted; compute_scope_metrics checks status=='accepted'
    reg_gt_ids = {g for g in gallery_gt_id.values() if g is not None}
    metrics_all = compute_scope_metrics(eval_nodes, registered_gt_ids=reg_gt_ids, n_bootstrap=n_bootstrap)
    pure_nodes = [n for n in eval_nodes if n.is_pure]
    metrics_pure = compute_scope_metrics(pure_nodes, registered_gt_ids=reg_gt_ids, n_bootstrap=n_bootstrap)
    pc_curve = precision_coverage_curve(eval_nodes) if (method != "E0" and n_bootstrap > 0) else []

    # Build versioned contract document only for 1-to-1 compliant methods (E1-E3 can have co-occurring duplicates!)
    doc_dict = None
    duplicate_cooccurring_violations = 0
    by_clip_gid_chk: dict[tuple[str, str], int] = {}
    for lnk in links:
        if lnk.status == "accepted" and lnk.global_player_id is not None:
            k_dup = (lnk.clip_id, lnk.global_player_id)
            by_clip_gid_chk[k_dup] = by_clip_gid_chk.get(k_dup, 0) + 1
    duplicate_cooccurring_violations = sum(cnt - 1 for cnt in by_clip_gid_chk.values() if cnt > 1)
    if duplicate_cooccurring_violations == 0 and method != "E0":
        doc = PlayerIdentityLinksDocument(
            schema_version=SCHEMA_VERSION,
            identity_scope="meiji_4cam/video_000",
            registry_version=registry.registry_version,
            source_manifest_hashes={"multiview_v4": cache.manifest["benchmark_manifest_sha256"]},
            upstream_digests={"cache_npz": cache.manifest["stats"]["cache_npz_sha256"]},
            encoder_identity=cache.manifest["encoder"],
            preprocessing_identity=cache.manifest["sampling"],
            method=method,
            config={
                "min_score": min_score,
                "min_margin": min_margin,
                "shape_weight": shape_weight if method == "E5" else None,
                "camera_mask": list(camera_mask),
            },
            links=tuple(links),
        )
        doc_dict = doc.to_dict()

    return {
        "method": method,
        "params": {
            "min_score": min_score,
            "min_margin": min_margin,
            "shape_weight": shape_weight if method == "E5" else None,
            "update_min_score": update_min_score if method == "E7" else None,
            "update_min_margin": update_min_margin if method == "E7" else None,
            "camera_mask": list(camera_mask),
            "excluded_anchor_gt_id": excluded_anchor_gt_id,
        },
        "metrics_pure": metrics_pure,
        "metrics_all": metrics_all,
        "precision_coverage_curve": pc_curve,
        "diagnostics": {
            "duplicate_cooccurring_violations": duplicate_cooccurring_violations,
            "no_common_camera_nodes": no_common_camera_nodes,
            "shape_used": shape_used,
            "shape_unavailable": shape_unavailable,
            "temporal_used": temporal_used,
            "temporal_unavailable": temporal_unavailable,
            "gallery_updates_total": gallery_updates_total,
            "gallery_updates_wrong": gallery_updates_wrong,
            "wrong_update_clip_idx": wrong_update_clip_idx,
            "post_contamination_fp_nodes": post_contamination_fp_nodes,
            "wall_ms": round(wall_ms, 3),
            "cpu_ms": round(cpu_ms, 3),
            "peak_rss_mb": round(peak_rss_mb, 2),
        },
        "eval_nodes": eval_nodes,
        "links_document": doc_dict,
        "registry": registry.to_dict(),
    }


def calibrate_on_dev(
    method: str,
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    *,
    score_grid: tuple[float, ...] = (0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88),
    margin_grid: tuple[float, ...] = (0.00, 0.01, 0.02, 0.04, 0.06),
    shape_weight_grid: tuple[float, ...] = (0.04, 0.08, 0.15),
    fp_penalty: float = 0.5,
    use_swap_margin: bool = False,
) -> dict[str, Any]:
    """Fixed-budget grid calibration on dev split only (never touches test)."""
    if method == "E0":
        return {"min_score": 0.0, "min_margin": 0.0, "shape_weight": 0.0, "dev_objective": 0.0, "trials": 1}

    best_cfg: dict[str, Any] = {}
    best_obj = -1e9
    trials = 0
    s_weights = shape_weight_grid if method == "E5" else (0.0,)

    for sw in s_weights:
        for ms in score_grid:
            for mm in margin_grid:
                trials += 1
                res = run_method(
                    method,
                    cache,
                    gt_map,
                    purity_map,
                    split,
                    split["dev_indices"],
                    min_score=ms,
                    min_margin=mm,
                    shape_weight=sw,
                    use_swap_margin=use_swap_margin,
                    n_bootstrap=0,
                )
                m = res["metrics_pure"]
                # Objective rewards high F1 while explicitly penalizing false merges
                obj = float(m["cross_clip_pair_f1"]) - fp_penalty * float(m["false_merge_rate"])
                if obj > best_obj + 1e-9:
                    best_obj = obj
                    best_cfg = {
                        "min_score": ms,
                        "min_margin": mm,
                        "shape_weight": sw,
                        "dev_objective": round(obj, 6),
                        "dev_f1": m["cross_clip_pair_f1"],
                        "dev_false_merge_rate": m["false_merge_rate"],
                        "dev_coverage": m["coverage"],
                        "trials": trials,
                    }
    best_cfg["trials"] = trials
    return best_cfg
