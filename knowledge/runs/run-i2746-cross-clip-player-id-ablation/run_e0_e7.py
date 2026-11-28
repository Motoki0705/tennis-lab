"""Run E0-E7 ablation study, stratified failure analysis, 1-cam dropout, and open-set tests (Issue #2746).

Calibrates strictly on dev; test evaluation is run once with frozen dev parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from ablation import (  # noqa: E402
    CacheData,
    analyze_smpl_beta_separability,
    analyze_temporal_gaps,
    build_split_manifest,
    calibrate_on_dev,
    load_cache,
    run_method,
    select_warmup_clip_from_input_quality,
)
from metrics import EvalNode, compute_scope_metrics  # noqa: E402

METHODS = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")


def load_gt(gt_path: Path) -> tuple[dict[str, Any], dict[tuple[str, int], str | None], dict[tuple[str, int], bool]]:
    gt_doc = json.loads(gt_path.read_text())
    gt_map: dict[tuple[str, int], str | None] = {}
    purity_map: dict[tuple[str, int], bool] = {}
    for cname, cdata in gt_doc["clips"].items():
        for p in cdata["players"]:
            lid = int(p["local_player_id"])
            gt_map[(cname, lid)] = p["gt_global_player_id"] if p["status"] == "confirmed" else None
            purity_map[(cname, lid)] = p["purity"] == "pure"
    return gt_doc, gt_map, purity_map


def stratified_analysis(
    res: dict[str, Any],
    cache: CacheData,
    gt_doc: dict[str, Any],
    warmup_ci: int,
) -> dict[str, Any]:
    """Break down node accuracy, coverage, false-merge errors across strata."""
    eval_nodes: list[EvalNode] = res["eval_nodes"]
    label_map = res["metrics_pure"]["scope_label_map"]
    warmup_cname = cache.clip_names[warmup_ci]
    warmup_end_flipped = bool(gt_doc["clips"][warmup_cname]["end_flipped_vs_clip000"])
    clips_meta = cache.manifest["clips"]

    # Precompute median beta_std on evaluated nodes for SMPL quality stratification
    beta_stds = [float(cache.gvhmr_beta_std[n.clip_idx, n.local_player_id].mean()) for n in eval_nodes]
    med_beta_std = float(np.median(beta_stds)) if beta_stds else 0.5

    strata: dict[str, list[EvalNode]] = defaultdict(list)
    error_breakdown = {
        "same_team_swap": 0,
        "cross_team_swap": 0,
        "unresolved_missed": 0,
        "total_errors": 0,
        "examples": [],
    }
    team_of = {"player_01": "A", "player_03": "A", "player_02": "B", "player_04": "B"}

    for n in eval_nodes:
        if not n.is_pure or n.gt_global_id is None:
            continue
        ci, p = n.clip_idx, n.local_player_id
        cgt = gt_doc["clips"][n.clip_name]

        # Wardrobe regime
        strata[f"wardrobe:{cgt['wardrobe_regime']}"].append(n)
        # End change vs warmup
        flipped = bool(cgt["end_flipped_vs_clip000"]) != warmup_end_flipped
        strata["end:flipped_vs_warmup" if flipped else "end:same_as_warmup"].append(n)
        # Gap vs previous clip
        if ci > 0:
            gap = float(clips_meta[ci]["global_start_sec"]) - float(clips_meta[ci - 1]["global_end_sec"])
            strata["gap:short_<12s" if gap < 12.0 else "gap:long_>=12s"].append(n)
        # Overlap (occlusion)
        max_iou = float(cache.crop_max_iou[ci, p][cache.crop_valid[ci, p]].max()) if cache.crop_valid[ci, p].any() else 0.0
        strata["occlusion:overlap_iou>0.1" if max_iou > 0.1 else "occlusion:clean"].append(n)
        # Facing proxy
        mean_facing = float(cache.crop_facing[ci, p][cache.crop_valid[ci, p]].mean()) if cache.crop_valid[ci, p].any() else 0.0
        strata["facing:front_dominant" if mean_facing >= 0.0 else "facing:back_dominant"].append(n)
        # SMPL beta std
        b_std = float(cache.gvhmr_beta_std[ci, p].mean())
        strata["smpl:high_std" if b_std > med_beta_std else "smpl:low_std"].append(n)
        # Role (net players who removed outerwear vs baseline players)
        role = "net_p01_p02" if n.gt_global_id in {"player_01", "player_02"} else "baseline_p03_p04"
        strata[f"role:{role}"].append(n)

        # Error breakdown
        if n.status != "accepted" or n.pred_global_id is None:
            error_breakdown["unresolved_missed"] += 1
            error_breakdown["total_errors"] += 1
        else:
            mapped_gt = label_map.get(n.pred_global_id)
            if mapped_gt != n.gt_global_id:
                error_breakdown["total_errors"] += 1
                if mapped_gt and team_of.get(mapped_gt) == team_of.get(n.gt_global_id):
                    error_breakdown["same_team_swap"] += 1
                else:
                    error_breakdown["cross_team_swap"] += 1
                if len(error_breakdown["examples"]) < 25:
                    error_breakdown["examples"].append(
                        {
                            "clip": n.clip_name,
                            "local_id": p,
                            "gt": n.gt_global_id,
                            "pred_mapped": mapped_gt,
                            "score": round(float(n.score), 4) if n.score is not None else None,
                            "margin": round(float(n.margin), 4) if n.margin is not None else None,
                            "wardrobe": cgt["wardrobe_regime"],
                            "end_flipped": flipped,
                            "max_iou": round(max_iou, 3),
                        }
                    )

    summary = {}
    for k, nodes in sorted(strata.items()):
        m = compute_scope_metrics(nodes, n_bootstrap=0)
        summary[k] = {
            "nodes": m["total_registered_nodes"],
            "coverage": m["coverage"],
            "node_accuracy": m["node_accuracy"],
            "pair_precision": m["cross_clip_pair_precision"],
            "pair_recall": m["cross_clip_pair_recall"],
            "pair_f1": m["cross_clip_pair_f1"],
            "false_merge_rate": m["false_merge_rate"],
            "fp_pairs": m["fp_pairs"],
            "tp_pairs": m["tp_pairs"],
            "fn_pairs": m["fn_pairs"],
        }
    return {"strata": summary, "error_breakdown": error_breakdown}


def run_open_set_leave_one_out(
    method: str,
    cfg: dict[str, Any],
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    clip_indices: list[int],
) -> dict[str, Any]:
    """Omit each of the 4 players from the initial gallery one at a time and measure false-accept rate."""
    per_omitted = {}
    tot_unreg = 0
    tot_fa = 0
    tot_reg_acc = 0
    tot_reg_n = 0
    for omitted in ("player_01", "player_02", "player_03", "player_04"):
        res = run_method(
            method,
            cache,
            gt_map,
            purity_map,
            split,
            clip_indices,
            min_score=cfg["min_score"],
            min_margin=cfg["min_margin"],
            shape_weight=cfg.get("shape_weight", 0.0),
            excluded_anchor_gt_id=omitted,
            n_bootstrap=0,
        )
        m = res["metrics_pure"]
        per_omitted[omitted] = {
            "unregistered_nodes": m["unregistered_total_nodes"],
            "unregistered_false_accepts": m["unregistered_false_accepts"],
            "unregistered_false_accept_rate": m["unregistered_false_accept_rate"],
            "registered_coverage": m["coverage"],
            "registered_pair_f1": m["cross_clip_pair_f1"],
        }
        tot_unreg += int(m["unregistered_total_nodes"])
        tot_fa += int(m["unregistered_false_accepts"])
        tot_reg_acc += int(m["accepted_nodes"])
        tot_reg_n += int(m["total_registered_nodes"])
    return {
        "mean_unregistered_false_accept_rate": round(tot_fa / max(1, tot_unreg), 6),
        "total_unregistered_false_accepts": tot_fa,
        "total_unregistered_nodes": tot_unreg,
        "mean_registered_coverage": round(tot_reg_acc / max(1, tot_reg_n), 6),
        "per_omitted_player": per_omitted,
    }


def run_camera_dropout_robustness(
    method: str,
    cfg: dict[str, Any],
    cache: CacheData,
    gt_map: dict[tuple[str, int], str | None],
    purity_map: dict[tuple[str, int], bool],
    split: dict[str, Any],
    clip_indices: list[int],
) -> dict[str, Any]:
    """Drop 1 camera at a time strictly via descriptor input mask (no upstream re-inference)."""
    out = {}
    for drop_v in range(4):
        mask = tuple(v != drop_v for v in range(4))
        res = run_method(
            method,
            cache,
            gt_map,
            purity_map,
            split,
            clip_indices,
            min_score=cfg["min_score"],
            min_margin=cfg["min_margin"],
            shape_weight=cfg.get("shape_weight", 0.0),
            camera_mask=mask,  # type: ignore[arg-type]
            n_bootstrap=0,
        )
        m = res["metrics_pure"]
        out[f"drop_cam{drop_v}"] = {
            "pair_f1": m["cross_clip_pair_f1"],
            "pair_precision": m["cross_clip_pair_precision"],
            "pair_recall": m["cross_clip_pair_recall"],
            "false_merge_rate": m["false_merge_rate"],
            "coverage": m["coverage"],
            "clip_exact_match_rate": m["clip_exact_match_rate"],
        }
    out["mean_pair_f1"] = round(float(np.mean([v["pair_f1"] for v in out.values() if isinstance(v, dict)])), 6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/i2746_cross_clip_player_id/cache"))
    parser.add_argument("--gt", type=Path, default=RUN_DIR / "gt_cross_clip.json")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/i2746_cross_clip_player_id/e0_e7_results.json"))
    parser.add_argument("--eval-test", action="store_true", help="Also run frozen dev configs on test split")
    args = parser.parse_args()

    cache = load_cache(args.cache_dir)
    gt_doc, gt_map, purity_map = load_gt(args.gt)

    warmup_info = select_warmup_clip_from_input_quality(cache)
    split = build_split_manifest(cache, warmup_info["warmup_clip_idx"])
    beta_diag_dev = analyze_smpl_beta_separability(cache, gt_map, split["dev_indices"])
    gap_diag = analyze_temporal_gaps(cache, short_gap_max_sec=3.0)

    print("=== Warm-up & Split ===")
    print(f"Warm-up clip: {warmup_info['warmup_clip_name']} (input-quality only)")
    print(f"Dev clips: {len(split['dev_indices'])}, Test clips: {len(split['test_indices'])}, Overlaps: {split['source_frame_overlaps_verified']}")
    print("=== SMPL Beta Dev Diagnostics ===")
    print(json.dumps(beta_diag_dev, indent=2))
    print("=== Inter-clip Temporal Gap Audit (E6) ===")
    print(json.dumps(gap_diag, indent=2))

    dev_calibrated: dict[str, Any] = {}
    dev_results: dict[str, Any] = {}
    test_results: dict[str, Any] = {}

    for method in METHODS:
        cfg = calibrate_on_dev(method, cache, gt_map, purity_map, split)
        dev_calibrated[method] = cfg
        res_dev = run_method(
            method,
            cache,
            gt_map,
            purity_map,
            split,
            split["dev_indices"],
            min_score=cfg["min_score"],
            min_margin=cfg["min_margin"],
            shape_weight=cfg.get("shape_weight", 0.0),
        )
        strat_dev = stratified_analysis(res_dev, cache, gt_doc, split["warmup_clip_idx"])
        dev_results[method] = {
            "calibrated_config": cfg,
            "metrics_pure": res_dev["metrics_pure"],
            "metrics_all": res_dev["metrics_all"],
            "precision_coverage_curve": res_dev["precision_coverage_curve"],
            "diagnostics": res_dev["diagnostics"],
            "stratified": strat_dev,
        }
        if args.eval_test:
            res_test = run_method(
                method,
                cache,
                gt_map,
                purity_map,
                split,
                split["test_indices"],
                min_score=cfg["min_score"],
                min_margin=cfg["min_margin"],
                shape_weight=cfg.get("shape_weight", 0.0),
            )
            strat_test = stratified_analysis(res_test, cache, gt_doc, split["warmup_clip_idx"])
            test_results[method] = {
                "frozen_config": cfg,
                "metrics_pure": res_test["metrics_pure"],
                "metrics_all": res_test["metrics_all"],
                "precision_coverage_curve": res_test["precision_coverage_curve"],
                "diagnostics": res_test["diagnostics"],
                "stratified": strat_test,
            }

    # Also compute open-set & 1-cam dropout for E1, E2, E3, E4, E7 on dev (and test if enabled)
    open_set_dev = {
        m: run_open_set_leave_one_out(m, dev_calibrated[m], cache, gt_map, purity_map, split, split["dev_indices"])
        for m in ("E1", "E2", "E3", "E4", "E5", "E7")
    }
    cam_drop_dev = {
        m: run_camera_dropout_robustness(m, dev_calibrated[m], cache, gt_map, purity_map, split, split["dev_indices"])
        for m in ("E1", "E2", "E3", "E4")
    }
    open_set_test = {}
    cam_drop_test = {}
    if args.eval_test:
        open_set_test = {
            m: run_open_set_leave_one_out(m, dev_calibrated[m], cache, gt_map, purity_map, split, split["test_indices"])
            for m in ("E1", "E2", "E3", "E4", "E5", "E7")
        }
        cam_drop_test = {
            m: run_camera_dropout_robustness(m, dev_calibrated[m], cache, gt_map, purity_map, split, split["test_indices"])
            for m in ("E1", "E2", "E3", "E4")
        }

    # Print compact dev comparison table
    print("\n=== DEV Results (Pure tracks, 34 clips = 136 nodes) ===")
    print(f"{'Method':<6} {'PairP':>7} {'PairR':>7} {'PairF1':>7} {'FMR':>7} {'Cov':>7} {'ClipEx':>7} {'Sw':>4} {'DupV':>4} {'minS':>5} {'minM':>5}")
    for m in METHODS:
        mp = dev_results[m]["metrics_pure"]
        dg = dev_results[m]["diagnostics"]
        cfg = dev_calibrated[m]
        print(
            f"{m:<6} {mp['cross_clip_pair_precision']:7.4f} {mp['cross_clip_pair_recall']:7.4f} "
            f"{mp['cross_clip_pair_f1']:7.4f} {mp['false_merge_rate']:7.4f} {mp['coverage']:7.4f} "
            f"{mp['clip_exact_match_rate']:7.4f} {mp['id_switch_boundaries']:4d} {dg['duplicate_cooccurring_violations']:4d} "
            f"{cfg['min_score']:5.2f} {cfg['min_margin']:5.2f}"
        )

    if args.eval_test:
        print("\n=== TEST Results (Pure tracks, 34 clips; 1 mixed node excluded from pure) ===")
        print(f"{'Method':<6} {'PairP':>7} {'PairR':>7} {'PairF1':>7} {'FMR':>7} {'Cov':>7} {'ClipEx':>7} {'Sw':>4} {'DupV':>4}")
        for m in METHODS:
            mp = test_results[m]["metrics_pure"]
            dg = test_results[m]["diagnostics"]
            print(
                f"{m:<6} {mp['cross_clip_pair_precision']:7.4f} {mp['cross_clip_pair_recall']:7.4f} "
                f"{mp['cross_clip_pair_f1']:7.4f} {mp['false_merge_rate']:7.4f} {mp['coverage']:7.4f} "
                f"{mp['clip_exact_match_rate']:7.4f} {mp['id_switch_boundaries']:4d} {dg['duplicate_cooccurring_violations']:4d}"
            )

    bundle = {
        "schema_version": 1,
        "warmup": warmup_info,
        "split": split,
        "smpl_beta_dev_diagnostics": beta_diag_dev,
        "temporal_gap_audit": gap_diag,
        "dev": dev_results,
        "dev_open_set": open_set_dev,
        "dev_camera_dropout": cam_drop_dev,
        "test": test_results,
        "test_open_set": open_set_test,
        "test_camera_dropout": cam_drop_test,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    print(f"\nSaved full E0-E7 results to {args.out_json}")


if __name__ == "__main__":
    main()
