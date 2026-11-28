"""Scope-level evaluation metrics for cross-clip player identity linking (Issue #2746).

Enforces the strict rules from Issue #2746:
- Never re-match anonymous global IDs per clip; any global label permutation is solved
  at most once across the entire identity_scope.
- Cross-clip pair precision / recall / F1 are the primary metrics, evaluated strictly
  on cross-clip node pairs (clip_i != clip_j).
- 'unresolved' nodes are never merged into one cluster; missed same-person links count as FN.
- Block bootstrap confidence intervals resample by game/time block, never by correlated pairs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class EvalNode:
    clip_idx: int
    clip_name: str
    local_player_id: int
    gt_global_id: str | None  # None if uncertain
    pred_global_id: str | None
    status: str  # accepted | unresolved | new_candidate | excluded
    score: float | None
    margin: float | None
    block_id: str  # e.g., game block for block bootstrap
    num_valid_frames: int = 1
    is_pure: bool = True
    tags: tuple[str, ...] = ()


def global_scope_label_map(nodes: list[EvalNode]) -> dict[str, str]:
    """Solve a single one-to-one mapping from predicted anonymous global IDs to GT IDs
    across the entire identity_scope (never per clip).
    """
    pred_ids = sorted({n.pred_global_id for n in nodes if n.status == "accepted" and n.pred_global_id})
    gt_ids = sorted({n.gt_global_id for n in nodes if n.gt_global_id})
    if not pred_ids or not gt_ids:
        return {}
    p_idx = {pid: i for i, pid in enumerate(pred_ids)}
    g_idx = {gid: j for j, gid in enumerate(gt_ids)}
    counts = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)
    for n in nodes:
        if n.status == "accepted" and n.pred_global_id and n.gt_global_id:
            counts[p_idx[n.pred_global_id], g_idx[n.gt_global_id]] += 1
    rows, cols = linear_sum_assignment(-counts)
    mapping: dict[str, str] = {}
    for r, c in zip(rows, cols, strict=True):
        if counts[r, c] > 0:
            mapping[pred_ids[r]] = gt_ids[c]
    return mapping


def compute_cross_clip_pair_metrics(nodes: list[EvalNode]) -> dict[str, Any]:
    """Primary metric: cross-clip pair precision / recall / F1, false merge rate, split rate."""
    valid_nodes = [n for n in nodes if n.gt_global_id is not None]
    n_count = len(valid_nodes)
    if n_count < 2:
        return {
            "cross_clip_pair_precision": 0.0,
            "cross_clip_pair_recall": 0.0,
            "cross_clip_pair_f1": 0.0,
            "false_merge_rate": 0.0,
            "split_rate": 0.0,
            "tp_pairs": 0,
            "fp_pairs": 0,
            "fn_pairs": 0,
            "tn_pairs": 0,
            "pred_pos_pairs": 0,
            "gt_pos_pairs": 0,
            "total_cross_clip_pairs": 0,
        }

    clip_ids = np.array([n.clip_name for n in valid_nodes], dtype=object)
    gt_ids = np.array([n.gt_global_id for n in valid_nodes], dtype=object)
    acc = np.array([n.status == "accepted" and n.pred_global_id is not None for n in valid_nodes], dtype=bool)
    pred_ids = np.array([n.pred_global_id if a else "" for n, a in zip(valid_nodes, acc, strict=True)], dtype=object)

    iu = np.triu_indices(n_count, k=1)
    diff_clip = clip_ids[iu[0]] != clip_ids[iu[1]]
    i0 = iu[0][diff_clip]
    i1 = iu[1][diff_clip]
    total_pairs = int(len(i0))

    gt_same = gt_ids[i0] == gt_ids[i1]
    pred_same = acc[i0] & acc[i1] & (pred_ids[i0] == pred_ids[i1])

    tp = int(np.sum(gt_same & pred_same))
    fp = int(np.sum((~gt_same) & pred_same))
    fn = int(np.sum(gt_same & (~pred_same)))
    tn = int(np.sum((~gt_same) & (~pred_same)))

    pred_pos = tp + fp
    gt_pos = tp + fn
    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / gt_pos if gt_pos > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    false_merge_rate = fp / pred_pos if pred_pos > 0 else 0.0
    split_rate = fn / gt_pos if gt_pos > 0 else 0.0

    return {
        "cross_clip_pair_precision": round(precision, 6),
        "cross_clip_pair_recall": round(recall, 6),
        "cross_clip_pair_f1": round(f1, 6),
        "false_merge_rate": round(false_merge_rate, 6),
        "split_rate": round(split_rate, 6),
        "tp_pairs": tp,
        "fp_pairs": fp,
        "fn_pairs": fn,
        "tn_pairs": tn,
        "pred_pos_pairs": pred_pos,
        "gt_pos_pairs": gt_pos,
        "total_cross_clip_pairs": total_pairs,
    }


def compute_scope_metrics(
    nodes: list[EvalNode],
    *,
    registered_gt_ids: set[str] | None = None,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Full scope-level evaluation bundle (primary + auxiliary + block bootstrap CI)."""
    valid_nodes = [n for n in nodes if n.gt_global_id is not None]
    reg_set = registered_gt_ids if registered_gt_ids is not None else {n.gt_global_id for n in valid_nodes if n.gt_global_id}

    pair_m = compute_cross_clip_pair_metrics(valid_nodes)
    label_map = global_scope_label_map(valid_nodes)

    # Coverage on registered GT players
    reg_nodes = [n for n in valid_nodes if n.gt_global_id in reg_set]
    accepted_reg = [n for n in reg_nodes if n.status == "accepted" and n.pred_global_id is not None]
    coverage = len(accepted_reg) / len(reg_nodes) if reg_nodes else 0.0

    # Global node-level accuracy (after single scope-level permutation)
    correct_nodes = sum(
        1 for n in reg_nodes if n.status == "accepted" and label_map.get(n.pred_global_id or "") == n.gt_global_id
    )
    accepted_correct = sum(
        1 for n in accepted_reg if label_map.get(n.pred_global_id or "") == n.gt_global_id
    )
    node_precision_at_accepted = accepted_correct / len(accepted_reg) if accepted_reg else 0.0
    node_accuracy = correct_nodes / len(reg_nodes) if reg_nodes else 0.0

    # Clip-level all-player exact match rate
    by_clip: dict[str, list[EvalNode]] = defaultdict(list)
    for n in reg_nodes:
        by_clip[n.clip_name].append(n)
    clip_exact = 0
    for _cname, cnodes in by_clip.items():
        if cnodes and all(n.status == "accepted" and label_map.get(n.pred_global_id or "") == n.gt_global_id for n in cnodes):
            clip_exact += 1
    clip_exact_rate = clip_exact / len(by_clip) if by_clip else 0.0

    # ID switch boundaries in chronological order (accepted predictions for the same GT player)
    switches = 0
    transitions = 0
    by_gt: dict[str, list[EvalNode]] = defaultdict(list)
    for n in sorted(reg_nodes, key=lambda x: (x.clip_idx, x.local_player_id)):
        if n.gt_global_id:
            by_gt[n.gt_global_id].append(n)
    per_player_stats: dict[str, dict[str, Any]] = {}
    for gid, gnodes in sorted(by_gt.items()):
        prev_pred: str | None = None
        p_switches = 0
        p_acc = 0
        p_corr = 0
        for n in gnodes:
            if n.status == "accepted" and n.pred_global_id is not None:
                p_acc += 1
                if label_map.get(n.pred_global_id) == gid:
                    p_corr += 1
                if prev_pred is not None:
                    transitions += 1
                    if n.pred_global_id != prev_pred:
                        switches += 1
                        p_switches += 1
                prev_pred = n.pred_global_id
        per_player_stats[gid] = {
            "total_nodes": len(gnodes),
            "accepted_nodes": p_acc,
            "correct_nodes": p_corr,
            "coverage": round(p_acc / len(gnodes), 6) if gnodes else 0.0,
            "accuracy": round(p_corr / len(gnodes), 6) if gnodes else 0.0,
            "switches": p_switches,
        }

    macro_acc = float(np.mean([v["accuracy"] for v in per_player_stats.values()])) if per_player_stats else 0.0
    macro_cov = float(np.mean([v["coverage"] for v in per_player_stats.values()])) if per_player_stats else 0.0

    # Scope-level frame-weighted IDF1 / AssA on unique (clip, frame, gt_id) keys (no 4-cam double counting)
    # Only valid for pure tracks; frame count = num_valid_frames per node.
    idtp_f = 0
    gt_f_total = 0
    pred_f_total = 0
    gt_pred_f: dict[tuple[str, str], int] = defaultdict(int)
    gt_f_by_id: dict[str, int] = defaultdict(int)
    pred_f_by_id: dict[str, int] = defaultdict(int)
    for n in reg_nodes:
        f = max(1, int(n.num_valid_frames))
        gid = n.gt_global_id or ""
        gt_f_total += f
        gt_f_by_id[gid] += f
        if n.status == "accepted" and n.pred_global_id is not None:
            pid = n.pred_global_id
            pred_f_total += f
            pred_f_by_id[pid] += f
            gt_pred_f[(gid, pid)] += f
            if label_map.get(pid) == gid:
                idtp_f += f
    scope_idf1 = (2.0 * idtp_f / (gt_f_total + pred_f_total)) if (gt_f_total + pred_f_total) > 0 else 0.0
    # AssA per HOTA [P11]: average association IoU over all TP frame observations
    assa_num = 0.0
    for (gid, pid), tpa in gt_pred_f.items():
        if label_map.get(pid) == gid:
            fna = gt_f_by_id[gid] - tpa
            fpa = pred_f_by_id[pid] - tpa
            assa_num += tpa * (tpa / max(1, tpa + fna + fpa))
    scope_assa = assa_num / max(1, gt_f_total)

    # Open-set unregistered player false-accept rate (if any unregistered GT nodes present)
    unreg_nodes = [n for n in valid_nodes if n.gt_global_id not in reg_set]
    unreg_fp = sum(1 for n in unreg_nodes if n.status == "accepted")
    unreg_fa_rate = (unreg_fp / len(unreg_nodes)) if unreg_nodes else None

    # Block bootstrap 95% CI on cross_clip_pair_f1 (resampling game/time blocks)
    blocks: dict[str, list[EvalNode]] = defaultdict(list)
    for n in valid_nodes:
        blocks[n.block_id].append(n)
    block_keys = sorted(blocks.keys())
    ci_f1 = None
    if len(block_keys) >= 2 and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        boot_f1s = []
        for _ in range(n_bootstrap):
            sampled_keys = rng.choice(block_keys, size=len(block_keys), replace=True)
            b_nodes: list[EvalNode] = []
            for bi, bk in enumerate(sampled_keys):
                for orig in blocks[bk]:
                    b_nodes.append(
                        EvalNode(
                            clip_idx=orig.clip_idx,
                            clip_name=f"b{bi}_{orig.clip_name}",
                            local_player_id=orig.local_player_id,
                            gt_global_id=orig.gt_global_id,
                            pred_global_id=orig.pred_global_id,
                            status=orig.status,
                            score=orig.score,
                            margin=orig.margin,
                            block_id=orig.block_id,
                            num_valid_frames=orig.num_valid_frames,
                            is_pure=orig.is_pure,
                        )
                    )
            bm = compute_cross_clip_pair_metrics(b_nodes)
            boot_f1s.append(bm["cross_clip_pair_f1"])
        ci_f1 = [round(float(np.percentile(boot_f1s, 2.5)), 6), round(float(np.percentile(boot_f1s, 97.5)), 6)]

    return {
        **pair_m,
        "coverage": round(coverage, 6),
        "accepted_nodes": len(accepted_reg),
        "total_registered_nodes": len(reg_nodes),
        "node_accuracy": round(node_accuracy, 6),
        "node_precision_at_accepted": round(node_precision_at_accepted, 6),
        "correct_nodes": correct_nodes,
        "clip_exact_match_rate": round(clip_exact_rate, 6),
        "clip_exact_matches": clip_exact,
        "total_eval_clips": len(by_clip),
        "id_switch_boundaries": switches,
        "id_transitions": transitions,
        "macro_player_accuracy": round(macro_acc, 6),
        "macro_player_coverage": round(macro_cov, 6),
        "scope_frame_idf1": round(scope_idf1, 6),
        "scope_frame_assa": round(scope_assa, 6),
        "unregistered_total_nodes": len(unreg_nodes),
        "unregistered_false_accepts": unreg_fp,
        "unregistered_false_accept_rate": round(unreg_fa_rate, 6) if unreg_fa_rate is not None else None,
        "block_bootstrap_f1_ci95": ci_f1,
        "num_bootstrap_blocks": len(block_keys),
        "per_player": per_player_stats,
        "scope_label_map": label_map,
    }


def precision_coverage_curve(nodes: list[EvalNode], n_points: int = 20) -> list[dict[str, Any]]:
    """Sweep acceptance threshold over node scores (using accepted+unresolved candidates with a score)
    to produce a precision-coverage & false-merge-rate curve.
    """
    scored = [n for n in nodes if n.gt_global_id is not None and n.score is not None and n.pred_global_id is not None]
    if not scored:
        return []
    scores = np.array([float(n.score) for n in scored], dtype=np.float64)
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 0.95, n_points)))
    curve = []
    for thr in thresholds:
        gated = []
        for n in nodes:
            if n.score is not None and n.pred_global_id is not None and n.score >= thr and n.status != "excluded":
                gated.append(
                    EvalNode(
                        clip_idx=n.clip_idx,
                        clip_name=n.clip_name,
                        local_player_id=n.local_player_id,
                        gt_global_id=n.gt_global_id,
                        pred_global_id=n.pred_global_id,
                        status="accepted",
                        score=n.score,
                        margin=n.margin,
                        block_id=n.block_id,
                        num_valid_frames=n.num_valid_frames,
                    )
                )
            else:
                gated.append(
                    EvalNode(
                        clip_idx=n.clip_idx,
                        clip_name=n.clip_name,
                        local_player_id=n.local_player_id,
                        gt_global_id=n.gt_global_id,
                        pred_global_id=None,
                        status="unresolved",
                        score=n.score,
                        margin=n.margin,
                        block_id=n.block_id,
                        num_valid_frames=n.num_valid_frames,
                    )
                )
        pm = compute_cross_clip_pair_metrics(gated)
        acc = sum(1 for g in gated if g.gt_global_id is not None and g.status == "accepted")
        tot = sum(1 for g in gated if g.gt_global_id is not None)
        curve.append(
            {
                "threshold": round(float(thr), 6),
                "coverage": round(acc / tot, 6) if tot else 0.0,
                "pair_precision": pm["cross_clip_pair_precision"],
                "pair_recall": pm["cross_clip_pair_recall"],
                "pair_f1": pm["cross_clip_pair_f1"],
                "false_merge_rate": pm["false_merge_rate"],
                "fp_pairs": pm["fp_pairs"],
                "tp_pairs": pm["tp_pairs"],
            }
        )
    return curve
