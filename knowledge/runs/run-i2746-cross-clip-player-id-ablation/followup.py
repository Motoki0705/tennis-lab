"""Conditional follow-up experiments (Axis G: Multi-clip Global MILP vs Greedy,
Axis P: KPR Unprompted vs 2D-Keypoint Prompted) and Same-vs-Cross-Camera LLR calibration (Issue #2746).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from ablation import (  # noqa: E402
    CacheData,
    _hungarian_with_dummy,
    build_split_manifest,
    calibrate_on_dev,
    compute_crop_quality_weights,
    compute_node_descriptors,
    load_cache,
    run_method,
    select_warmup_clip_from_input_quality,
)
from metrics import (  # noqa: E402
    EvalNode,
    compute_scope_metrics,
    precision_coverage_curve,
)
from run_e0_e7 import load_gt, stratified_analysis  # noqa: E402

from src.tasks.player_association.appearance.affinity import (  # noqa: E402
    fit_cosine_log_likelihood_ratio,
)
from src.tasks.player_association.appearance.parts import (  # noqa: E402
    NativeParts,
    part_distance,
)
from src.utils.matching.multiview_clustering import (  # noqa: E402
    MAX_ITEMS,
    SolverTimeLimit,
    cluster_multiview,
)


def calibrate_same_vs_cross_camera_llr(
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    dev_indices: list[int],
) -> dict[str, Any]:
    """Calibrate separate LLR parameters (slope, center) on dev for:
    (1) same-camera cross-clip pairs, and
    (2) cross-camera cross-clip pairs,
    and mean-over-common-cameras node-pair cosine similarity (used for Axis G MILP).
    """
    qw = compute_crop_quality_weights(cache)
    _, _, per_cam, per_cam_valid = compute_node_descriptors(cache, quality_weights=qw)

    same_cam_pos, same_cam_neg = [], []
    cross_cam_pos, cross_cam_neg = [], []
    node_pos, node_neg = [], []

    nodes = [
        (ci, p, gt_map[(cache.clip_names[ci], p)])
        for ci in dev_indices
        for p in range(4)
        if purity_map.get((cache.clip_names[ci], p), False) and gt_map.get((cache.clip_names[ci], p)) is not None
    ]

    for i in range(len(nodes)):
        ci1, p1, g1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            ci2, p2, g2 = nodes[j]
            if ci1 == ci2:
                continue
            same_person = g1 == g2
            common_sims = []
            for v1 in range(4):
                if not per_cam_valid[ci1, p1, v1]:
                    continue
                for v2 in range(4):
                    if not per_cam_valid[ci2, p2, v2]:
                        continue
                    sim = float(np.clip(per_cam[ci1, p1, v1] @ per_cam[ci2, p2, v2], -1.0, 1.0))
                    if v1 == v2:
                        common_sims.append(sim)
                        (same_cam_pos if same_person else same_cam_neg).append(sim)
                    else:
                        (cross_cam_pos if same_person else cross_cam_neg).append(sim)
            if common_sims:
                (node_pos if same_person else node_neg).append(float(np.mean(common_sims)))

    sc_slope, sc_center = fit_cosine_log_likelihood_ratio(np.array(same_cam_pos), np.array(same_cam_neg))
    cc_slope, cc_center = fit_cosine_log_likelihood_ratio(np.array(cross_cam_pos), np.array(cross_cam_neg))
    np_slope, np_center = fit_cosine_log_likelihood_ratio(np.array(node_pos), np.array(node_neg))

    return {
        "same_camera_cross_clip": {
            "slope": round(sc_slope, 4),
            "center": round(sc_center, 4),
            "pos_mean": round(float(np.mean(same_cam_pos)), 4),
            "neg_mean": round(float(np.mean(same_cam_neg)), 4),
            "n_pos": len(same_cam_pos),
            "n_neg": len(same_cam_neg),
        },
        "cross_camera_cross_clip": {
            "slope": round(cc_slope, 4),
            "center": round(cc_center, 4),
            "pos_mean": round(float(np.mean(cross_cam_pos)), 4),
            "neg_mean": round(float(np.mean(cross_cam_neg)), 4),
            "n_pos": len(cross_cam_pos),
            "n_neg": len(cross_cam_neg),
        },
        "node_common_cam_mean": {
            "slope": round(np_slope, 4),
            "center": round(np_center, 4),
            "pos_mean": round(float(np.mean(node_pos)), 4),
            "neg_mean": round(float(np.mean(node_neg)), 4),
            "n_pos": len(node_pos),
            "n_neg": len(node_neg),
        },
    }


def _greedy_cluster_with_cannot_link(scores: np.ndarray, views: np.ndarray) -> np.ndarray:
    """Greedy correlation clustering with identical view-exclusivity (cannot-link) and transitivity."""
    n = len(views)
    clusters: list[set[int]] = [{i} for i in range(n)]
    owner = list(range(n))
    cluster_views: list[set[int]] = [{int(views[i])} for i in range(n)]

    # Sort all positive cross-view edges descending
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if views[i] != views[j] and scores[i, j] > 0.0:
                edges.append((float(scores[i, j]), i, j))
    edges.sort(reverse=True)

    for _w, i, j in edges:
        ci, cj = owner[i], owner[j]
        if ci == cj:
            continue
        # Cannot-link check: merged cluster must not contain two items from same clip/view
        if cluster_views[ci] & cluster_views[cj]:
            continue
        # Check net sum of cross-cluster edges is positive (consistent with correlation clustering)
        net = sum(float(scores[a, b]) for a in clusters[ci] for b in clusters[cj])
        if net <= 0.0:
            continue
        # Merge cj into ci
        for node in clusters[cj]:
            owner[node] = ci
        clusters[ci] |= clusters[cj]
        cluster_views[ci] |= cluster_views[cj]
        clusters[cj].clear()
        cluster_views[cj].clear()

    # Dense labels
    remap: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for idx in range(n):
        c = owner[idx]
        if c not in remap:
            remap[c] = len(remap)
        labels[idx] = remap[c]
    return labels


def run_axis_g(
    mode: str,  # "G_greedy" or "G_milp"
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    clip_indices: list[int],
    llr_params: dict[str, Any],
    *,
    window_clips: int = 6,
    overlap_clips: int = 1,
    max_abs_llr: float = 8.0,
    center_shift: float = 0.0,
    time_limit_s: float = 30.0,
    n_bootstrap: int = 400,
) -> dict[str, Any]:
    """Axis G: Fixed-window multi-clip global consistency (<=32 nodes, exact MILP vs same-window greedy).

    Every window consists of:
    - 4 fixed anchor nodes (from the warm-up clip, view=0)
    - up to `window_clips` query clips (4 nodes each, view=1..window_clips) -> max 28 nodes (<=32, <=MAX_ITEMS=40).
    Uses identical future information, identical pairwise LLR edges, identical cannot-link constraints.
    """
    assert mode in {"G_greedy", "G_milp"}
    t0 = time.perf_counter()
    t0_cpu = time.process_time()

    qw = compute_crop_quality_weights(cache)
    _, _, per_cam, per_cam_valid = compute_node_descriptors(cache, quality_weights=qw)

    warmup_ci = int(split["warmup_clip_idx"])
    slope = float(llr_params["slope"])
    center = float(llr_params["center"]) + center_shift

    sorted_clips = sorted(clip_indices)
    stride = max(1, window_clips - overlap_clips)
    windows: list[list[int]] = []
    pos = 0
    while pos < len(sorted_clips):
        win = sorted_clips[pos : pos + window_clips]
        windows.append(win)
        if pos + window_clips >= len(sorted_clips):
            break
        pos += stride

    # Per (ci, p) votes across overlapping windows (weighted by window objective / anchor consistency)
    # Note: anchors are persistent player_01..player_04 (registry indices 0..3)
    anchor_gids = [f"player_{p+1:02d}" for p in range(4)]
    warmup_cname = cache.clip_names[warmup_ci]
    anchor_gt_map = {anchor_gids[p]: gt_map.get((warmup_cname, p)) for p in range(4)}

    pred_by_node: dict[tuple[int, int], tuple[str | None, str, float | None]] = {}
    window_logs: list[dict[str, Any]] = []
    timeouts = 0
    max_nodes_seen = 0

    for w_idx, win_clips in enumerate(windows):
        # Build node list: first 4 are anchors (view 0), then each clip in win_clips (view 1..len)
        node_refs: list[tuple[int, int, int]] = [(warmup_ci, p, 0) for p in range(4)]
        for v_offset, ci in enumerate(win_clips, start=1):
            for p in range(4):
                node_refs.append((ci, p, v_offset))
        n_nodes = len(node_refs)
        max_nodes_seen = max(max_nodes_seen, n_nodes)
        assert n_nodes <= 32 and n_nodes <= MAX_ITEMS, f"Window node count {n_nodes} exceeds 32/MAX_ITEMS"

        views = np.array([v for _, _, v in node_refs], dtype=np.int64)
        scores = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for i in range(n_nodes):
            ci1, p1, v1 = node_refs[i]
            for j in range(i + 1, n_nodes):
                ci2, p2, v2 = node_refs[j]
                if v1 == v2:
                    continue
                sims = [
                    float(np.clip(per_cam[ci1, p1, c] @ per_cam[ci2, p2, c], -1.0, 1.0))
                    for c in range(4)
                    if per_cam_valid[ci1, p1, c] and per_cam_valid[ci2, p2, c]
                ]
                if sims:
                    mean_sim = float(np.mean(sims))
                    llr = float(np.clip(slope * (mean_sim - center), -max_abs_llr, max_abs_llr))
                    scores[i, j] = scores[j, i] = llr

        tw0 = time.perf_counter()
        timed_out = False
        labels: np.ndarray | None = None
        if mode == "G_greedy":
            labels = _greedy_cluster_with_cannot_link(scores, views)
        else:
            try:
                res_milp = cluster_multiview(scores, views, time_limit_s=time_limit_s)
                labels = res_milp.labels
            except SolverTimeLimit:
                timed_out = True
                timeouts += 1
        w_ms = (time.perf_counter() - tw0) * 1000.0
        window_logs.append(
            {
                "window_idx": w_idx,
                "clips": [cache.clip_names[c] for c in win_clips],
                "num_nodes": n_nodes,
                "timed_out": timed_out,
                "solve_ms": round(w_ms, 2),
            }
        )

        # Map clusters to anchor global IDs (view 0 items are indices 0..3)
        if timed_out or labels is None:
            for idx_n in range(4, n_nodes):
                ci, p, _ = node_refs[idx_n]
                if (ci, p) not in pred_by_node:
                    pred_by_node[(ci, p)] = (None, "unresolved", None)
            continue

        cluster_to_gid: dict[int, str] = {}
        for a_idx in range(4):
            cluster_to_gid[int(labels[a_idx])] = anchor_gids[a_idx]

        # Record predictions for query nodes (for overlapped clips, keep first window unless previous was unresolved)
        for idx_n in range(4, n_nodes):
            ci, p, _ = node_refs[idx_n]
            cid = int(labels[idx_n])
            gid = cluster_to_gid.get(cid)
            if gid is not None:
                a_idx = anchor_gids.index(gid)
                # Score = average LLR to cluster members
                members = [m for m in range(n_nodes) if int(labels[m]) == cid and m != idx_n]
                mean_llr = float(np.mean([scores[idx_n, m] for m in members])) if members else float(scores[idx_n, a_idx])
                if (ci, p) not in pred_by_node or pred_by_node[(ci, p)][1] != "accepted":
                    pred_by_node[(ci, p)] = (gid, "accepted", mean_llr)
            else:
                if (ci, p) not in pred_by_node:
                    pred_by_node[(ci, p)] = (None, "unresolved", None)

    eval_nodes: list[EvalNode] = []
    for ci in sorted_clips:
        cname = cache.clip_names[ci]
        block_id = split["clip_blocks"][cname]
        for p in range(4):
            gid, status, sc = pred_by_node.get((ci, p), (None, "unresolved", None))
            eval_nodes.append(
                EvalNode(
                    clip_idx=ci,
                    clip_name=cname,
                    local_player_id=p,
                    gt_global_id=gt_map.get((cname, p)),
                    pred_global_id=gid,
                    status=status,
                    score=sc,
                    margin=None,
                    block_id=block_id,
                    num_valid_frames=int(cache.obs_frames_pv[ci, p].max()),
                    is_pure=purity_map.get((cname, p), True),
                )
            )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    cpu_ms = (time.process_time() - t0_cpu) * 1000.0
    reg_gt_ids = {g for g in anchor_gt_map.values() if g is not None}
    metrics_pure = compute_scope_metrics([n for n in eval_nodes if n.is_pure], registered_gt_ids=reg_gt_ids, n_bootstrap=n_bootstrap)
    metrics_all = compute_scope_metrics(eval_nodes, registered_gt_ids=reg_gt_ids, n_bootstrap=n_bootstrap)

    return {
        "method": mode,
        "params": {
            "window_clips": window_clips,
            "overlap_clips": overlap_clips,
            "max_abs_llr": max_abs_llr,
            "center_shift": center_shift,
            "time_limit_s": time_limit_s,
            "llr_slope": slope,
            "llr_center": center,
        },
        "metrics_pure": metrics_pure,
        "metrics_all": metrics_all,
        "diagnostics": {
            "num_windows": len(windows),
            "max_nodes_per_window": max_nodes_seen,
            "timeouts": timeouts,
            "timeout_rate": round(timeouts / max(1, len(windows)), 6),
            "wall_ms": round(wall_ms, 2),
            "cpu_ms": round(cpu_ms, 2),
            "windows": window_logs,
        },
        "eval_nodes": eval_nodes,
    }


def _weighted_mean_parts(embs: np.ndarray, vis: np.ndarray, weights: np.ndarray) -> NativeParts | None:
    """Quality-weighted NativeParts mean over K crops in one camera -> (1, 6, 512)."""
    # embs: (K, 6, 512), vis: (K, 6), weights: (K,)
    w_vis = vis.astype(np.float64) * weights[:, None]
    w_sum = w_vis.sum(axis=0)  # (6,)
    part_vis = w_sum > 0
    if not part_vis.any():
        return None
    vec = (embs.astype(np.float64) * w_vis[:, :, None]).sum(axis=0)  # (6, 512)
    norms = np.linalg.norm(vec, axis=-1)
    part_vis = part_vis & (norms > 1e-12)
    if not part_vis.any():
        return None
    vec[part_vis] /= norms[part_vis, None]
    vec[~part_vis] = 0.0
    return NativeParts(vec[None].astype(np.float32), part_vis[None])


def run_axis_p_kpr(
    variant: str,  # "P_unprompted" or "P_prompted"
    cache: CacheData,
    kpr_npz: Any,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    clip_indices: list[int],
    *,
    min_score: float = 0.30,
    min_margin: float = 0.00,
    use_swap_margin: bool = False,
    n_bootstrap: int = 400,
) -> dict[str, Any]:
    """Axis P: KPR native-part similarity (1 - mean_euclidean_distance over common visible parts)
    under the exact same E4 quality-weighted per-camera Hungarian framework.
    """
    assert variant in {"P_unprompted", "P_prompted"}
    t0 = time.perf_counter()
    t0_cpu = time.process_time()

    emb_key = "emb_prompted" if variant == "P_prompted" else "emb_unprompted"
    vis_key = "vis_prompted" if variant == "P_prompted" else "vis_unprompted"
    kpr_emb = kpr_npz[emb_key]  # (N, 4, 4, K, 6, 512)
    kpr_vis = kpr_npz[vis_key]  # (N, 4, 4, K, 6)
    qw = compute_crop_quality_weights(cache)

    N = len(cache.clip_names)
    # Build per-(ci, p, v) NativeParts descriptors
    parts_pv: dict[tuple[int, int, int], NativeParts] = {}
    for ci in range(N):
        for p in range(4):
            for v in range(4):
                vk = cache.crop_valid[ci, p, v]
                if not vk.any():
                    continue
                mp = _weighted_mean_parts(kpr_emb[ci, p, v, vk], kpr_vis[ci, p, v, vk], qw[ci, p, v, vk])
                if mp is not None:
                    parts_pv[(ci, p, v)] = mp

    warmup_ci = int(split["warmup_clip_idx"])
    warmup_cname = cache.clip_names[warmup_ci]
    anchor_gids = [f"player_{p+1:02d}" for p in range(4)]
    anchor_gt_ids = {gt_map.get((warmup_cname, p)) for p in range(4) if gt_map.get((warmup_cname, p)) is not None}

    eval_nodes: list[EvalNode] = []
    for ci in sorted(clip_indices):
        cname = cache.clip_names[ci]
        block_id = split["clip_blocks"][cname]
        score_mat = np.full((4, 4), np.nan, dtype=np.float64)
        for p in range(4):
            for k in range(4):
                sims = []
                for v in range(4):
                    lp = parts_pv.get((ci, p, v))
                    rp = parts_pv.get((warmup_ci, k, v))
                    if lp is not None and rp is not None:
                        dist, valid = part_distance(lp, rp)
                        if valid[0, 0]:
                            sims.append(float(np.clip(1.0 - dist[0, 0], -1.0, 1.0)))
                if sims:
                    score_mat[p, k] = float(np.mean(sims))

        assigned_k, best_s, margins, _ = _hungarian_with_dummy(score_mat, min_score, use_swap_margin=use_swap_margin)
        for p in range(4):
            if assigned_k[p] is not None and (margins[p] is None or margins[p] < min_margin):
                assigned_k[p] = None
            k_sel = assigned_k[p]
            status = "accepted" if k_sel is not None else "unresolved"
            top_k = k_sel if k_sel is not None else (int(np.nanargmax(score_mat[p])) if np.isfinite(score_mat[p]).any() else None)
            eval_nodes.append(
                EvalNode(
                    clip_idx=ci,
                    clip_name=cname,
                    local_player_id=p,
                    gt_global_id=gt_map.get((cname, p)),
                    pred_global_id=anchor_gids[top_k] if top_k is not None else None,
                    status=status,
                    score=best_s[p],
                    margin=margins[p],
                    block_id=block_id,
                    num_valid_frames=int(cache.obs_frames_pv[ci, p].max()),
                    is_pure=purity_map.get((cname, p), True),
                )
            )

    wall_ms = (time.perf_counter() - t0) * 1000.0
    cpu_ms = (time.process_time() - t0_cpu) * 1000.0
    metrics_pure = compute_scope_metrics([n for n in eval_nodes if n.is_pure], registered_gt_ids=anchor_gt_ids, n_bootstrap=n_bootstrap)
    metrics_all = compute_scope_metrics(eval_nodes, registered_gt_ids=anchor_gt_ids, n_bootstrap=n_bootstrap)
    pc_curve = precision_coverage_curve(eval_nodes) if n_bootstrap > 0 else []

    return {
        "method": variant,
        "params": {"min_score": min_score, "min_margin": min_margin, "use_swap_margin": use_swap_margin},
        "metrics_pure": metrics_pure,
        "metrics_all": metrics_all,
        "precision_coverage_curve": pc_curve,
        "diagnostics": {"wall_ms": round(wall_ms, 2), "cpu_ms": round(cpu_ms, 2)},
        "eval_nodes": eval_nodes,
    }


def calibrate_axis_p_on_dev(
    variant: str,
    cache: CacheData,
    kpr_npz: Any,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    *,
    use_swap_margin: bool = False,
) -> dict[str, Any]:
    score_grid = (-0.20, -0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30)
    margin_grid = (0.00, 0.01, 0.02, 0.04, 0.06)
    best_cfg: dict[str, Any] = {}
    best_obj = -1e9
    for ms in score_grid:
        for mm in margin_grid:
            res = run_axis_p_kpr(
                variant,
                cache,
                kpr_npz,
                gt_map,
                purity_map,
                split,
                split["dev_indices"],
                min_score=ms,
                min_margin=mm,
                use_swap_margin=use_swap_margin,
                n_bootstrap=0,
            )
            m = res["metrics_pure"]
            obj = float(m["cross_clip_pair_f1"]) - 0.5 * float(m["false_merge_rate"])
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_cfg = {
                    "min_score": ms,
                    "min_margin": mm,
                    "dev_objective": round(obj, 6),
                    "dev_f1": m["cross_clip_pair_f1"],
                    "dev_fmr": m["false_merge_rate"],
                    "dev_cov": m["coverage"],
                }
    return best_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/i2746_cross_clip_player_id/cache"))
    parser.add_argument("--gt", type=Path, default=RUN_DIR / "gt_cross_clip.json")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/i2746_cross_clip_player_id/followup_results.json"))
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    cache = load_cache(args.cache_dir)
    gt_doc, gt_map, purity_map = load_gt(args.gt)
    warmup = select_warmup_clip_from_input_quality(cache)
    split = build_split_manifest(cache, warmup["warmup_clip_idx"])

    llr_cal = calibrate_same_vs_cross_camera_llr(cache, gt_map, purity_map, split["dev_indices"])
    print("=== Same-camera vs Cross-camera LLR calibration (dev) ===")
    print(json.dumps(llr_cal, indent=2))

    # Calibrate Axis G center_shift on dev (same shift grid for G_greedy and G_milp)
    g_out: dict[str, Any] = {"dev": {}, "test": {}}
    best_shifts: dict[str, float] = {}
    for g_mode in ("G_greedy", "G_milp"):
        best_obj = -1e9
        best_sh = 0.0
        for sh in (-0.04, -0.02, 0.0, 0.02, 0.04):
            r = run_axis_g(
                g_mode,
                cache,
                gt_map,
                purity_map,
                split,
                split["dev_indices"],
                llr_cal["node_common_cam_mean"],
                center_shift=sh,
                n_bootstrap=0,
            )
            m = r["metrics_pure"]
            obj = float(m["cross_clip_pair_f1"]) - 0.5 * float(m["false_merge_rate"])
            if obj > best_obj + 1e-9:
                best_obj = obj
                best_sh = sh
        best_shifts[g_mode] = best_sh
        r_dev = run_axis_g(
            g_mode,
            cache,
            gt_map,
            purity_map,
            split,
            split["dev_indices"],
            llr_cal["node_common_cam_mean"],
            center_shift=best_sh,
        )
        g_out["dev"][g_mode] = {
            "metrics_pure": r_dev["metrics_pure"],
            "metrics_all": r_dev["metrics_all"],
            "diagnostics": r_dev["diagnostics"],
            "params": r_dev["params"],
            "stratified": stratified_analysis(r_dev, cache, gt_doc, split["warmup_clip_idx"]),
        }
        mp = r_dev["metrics_pure"]
        dg = r_dev["diagnostics"]
        print(
            f"[DEV] {g_mode}: PairP={mp['cross_clip_pair_precision']:.4f} PairR={mp['cross_clip_pair_recall']:.4f} "
            f"PairF1={mp['cross_clip_pair_f1']:.4f} FMR={mp['false_merge_rate']:.4f} Cov={mp['coverage']:.4f} "
            f"ClipEx={mp['clip_exact_match_rate']:.4f} shift={best_sh:+.2f} maxN={dg['max_nodes_per_window']} "
            f"timeouts={dg['timeouts']} wall={dg['wall_ms']:.1f}ms"
        )
        if args.eval_test:
            r_test = run_axis_g(
                g_mode,
                cache,
                gt_map,
                purity_map,
                split,
                split["test_indices"],
                llr_cal["node_common_cam_mean"],
                center_shift=best_sh,
            )
            g_out["test"][g_mode] = {
                "metrics_pure": r_test["metrics_pure"],
                "metrics_all": r_test["metrics_all"],
                "diagnostics": r_test["diagnostics"],
                "params": r_test["params"],
                "stratified": stratified_analysis(r_test, cache, gt_doc, split["warmup_clip_idx"]),
            }
            mpt = r_test["metrics_pure"]
            dgt = r_test["diagnostics"]
            print(
                f"[TEST] {g_mode}: PairP={mpt['cross_clip_pair_precision']:.4f} PairR={mpt['cross_clip_pair_recall']:.4f} "
                f"PairF1={mpt['cross_clip_pair_f1']:.4f} FMR={mpt['false_merge_rate']:.4f} Cov={mpt['coverage']:.4f} "
                f"ClipEx={mpt['clip_exact_match_rate']:.4f} timeouts={dgt['timeouts']} wall={dgt['wall_ms']:.1f}ms"
            )

    # E4 swap-margin sensitivity check on dev (and test if enabled)
    swap_out: dict[str, Any] = {"dev": {}, "test": {}}
    for m_name in ("E4", "E7"):
        cfg_sw = calibrate_on_dev(m_name, cache, gt_map, purity_map, split, use_swap_margin=True)
        r_dev = run_method(
            m_name,
            cache,
            gt_map,
            purity_map,
            split,
            split["dev_indices"],
            min_score=cfg_sw["min_score"],
            min_margin=cfg_sw["min_margin"],
            use_swap_margin=True,
        )
        swap_out["dev"][f"{m_name}_swap_margin"] = {
            "calibrated_config": cfg_sw,
            "metrics_pure": r_dev["metrics_pure"],
            "metrics_all": r_dev["metrics_all"],
            "precision_coverage_curve": r_dev["precision_coverage_curve"],
            "diagnostics": r_dev["diagnostics"],
            "stratified": stratified_analysis(r_dev, cache, gt_doc, split["warmup_clip_idx"]),
        }
        mp = r_dev["metrics_pure"]
        print(
            f"[DEV] {m_name}_swap_margin: PairP={mp['cross_clip_pair_precision']:.4f} PairR={mp['cross_clip_pair_recall']:.4f} "
            f"PairF1={mp['cross_clip_pair_f1']:.4f} FMR={mp['false_merge_rate']:.4f} Cov={mp['coverage']:.4f} "
            f"ClipEx={mp['clip_exact_match_rate']:.4f} minS={cfg_sw['min_score']:.2f} minM={cfg_sw['min_margin']:.2f}"
        )
        if args.eval_test:
            r_test = run_method(
                m_name,
                cache,
                gt_map,
                purity_map,
                split,
                split["test_indices"],
                min_score=cfg_sw["min_score"],
                min_margin=cfg_sw["min_margin"],
                use_swap_margin=True,
            )
            swap_out["test"][f"{m_name}_swap_margin"] = {
                "frozen_config": cfg_sw,
                "metrics_pure": r_test["metrics_pure"],
                "metrics_all": r_test["metrics_all"],
                "precision_coverage_curve": r_test["precision_coverage_curve"],
                "diagnostics": r_test["diagnostics"],
                "stratified": stratified_analysis(r_test, cache, gt_doc, split["warmup_clip_idx"]),
            }
            mpt = r_test["metrics_pure"]
            print(
                f"[TEST] {m_name}_swap_margin: PairP={mpt['cross_clip_pair_precision']:.4f} PairR={mpt['cross_clip_pair_recall']:.4f} "
                f"PairF1={mpt['cross_clip_pair_f1']:.4f} FMR={mpt['false_merge_rate']:.4f} Cov={mpt['coverage']:.4f} "
                f"ClipEx={mpt['clip_exact_match_rate']:.4f}"
            )

    # Axis P (KPR) if kpr_cache.npz exists
    p_out: dict[str, Any] = {"available": False, "dev": {}, "test": {}}
    kpr_path = args.cache_dir / "kpr_cache.npz"
    if kpr_path.is_file():
        p_out["available"] = True
        kpr_npz = np.load(kpr_path, allow_pickle=False)
        for use_sw, tag in [(False, ""), (True, "_swap")]:
            for variant in ("P_unprompted", "P_prompted"):
                vkey = f"{variant}{tag}"
                cfg_p = calibrate_axis_p_on_dev(variant, cache, kpr_npz, gt_map, purity_map, split, use_swap_margin=use_sw)
                r_dev = run_axis_p_kpr(
                    variant,
                    cache,
                    kpr_npz,
                    gt_map,
                    purity_map,
                    split,
                    split["dev_indices"],
                    min_score=cfg_p["min_score"],
                    min_margin=cfg_p["min_margin"],
                    use_swap_margin=use_sw,
                )
                p_out["dev"][vkey] = {
                    "calibrated_config": cfg_p,
                    "metrics_pure": r_dev["metrics_pure"],
                    "metrics_all": r_dev["metrics_all"],
                    "precision_coverage_curve": r_dev["precision_coverage_curve"],
                    "diagnostics": r_dev["diagnostics"],
                    "stratified": stratified_analysis(r_dev, cache, gt_doc, split["warmup_clip_idx"]),
                }
                mp = r_dev["metrics_pure"]
                occ_f1 = p_out["dev"][vkey]["stratified"]["strata"].get("occlusion:overlap_iou>0.1", {}).get("pair_f1")
                print(
                    f"[DEV] {vkey}: PairP={mp['cross_clip_pair_precision']:.4f} PairR={mp['cross_clip_pair_recall']:.4f} "
                    f"PairF1={mp['cross_clip_pair_f1']:.4f} FMR={mp['false_merge_rate']:.4f} Cov={mp['coverage']:.4f} "
                    f"ClipEx={mp['clip_exact_match_rate']:.4f} occ_F1={occ_f1}"
                )
                if args.eval_test:
                    r_test = run_axis_p_kpr(
                        variant,
                        cache,
                        kpr_npz,
                        gt_map,
                        purity_map,
                        split,
                        split["test_indices"],
                        min_score=cfg_p["min_score"],
                        min_margin=cfg_p["min_margin"],
                        use_swap_margin=use_sw,
                    )
                    p_out["test"][vkey] = {
                        "frozen_config": cfg_p,
                        "metrics_pure": r_test["metrics_pure"],
                        "metrics_all": r_test["metrics_all"],
                        "precision_coverage_curve": r_test["precision_coverage_curve"],
                        "diagnostics": r_test["diagnostics"],
                        "stratified": stratified_analysis(r_test, cache, gt_doc, split["warmup_clip_idx"]),
                    }
                    mpt = r_test["metrics_pure"]
                    occt = p_out["test"][vkey]["stratified"]["strata"].get("occlusion:overlap_iou>0.1", {}).get("pair_f1")
                    print(
                        f"[TEST] {vkey}: PairP={mpt['cross_clip_pair_precision']:.4f} PairR={mpt['cross_clip_pair_recall']:.4f} "
                        f"PairF1={mpt['cross_clip_pair_f1']:.4f} FMR={mpt['false_merge_rate']:.4f} Cov={mpt['coverage']:.4f} "
                        f"ClipEx={mpt['clip_exact_match_rate']:.4f} occ_F1={occt}"
                    )

    out_bundle = {
        "schema_version": 1,
        "llr_calibration_dev": llr_cal,
        "axis_g": g_out,
        "swap_margin_comparison": swap_out,
        "axis_p_kpr": p_out,
    }
    args.out_json.write_text(json.dumps(out_bundle, indent=2, ensure_ascii=False))
    print(f"Saved follow-up results to {args.out_json}")


if __name__ == "__main__":
    main()
