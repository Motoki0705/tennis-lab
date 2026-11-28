"""Generate compact summary JSON/CSV artifacts, leave-one-out integration check,
versioned player_identity_links example, and downstream #1722/#2545 join proof (Issue #2746).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from ablation import (  # noqa: E402
    build_split_manifest,
    load_cache,
    run_method,
    select_warmup_clip_from_input_quality,
)
from contract import join_events_with_global_ids  # noqa: E402
from run_e0_e7 import load_gt, run_open_set_leave_one_out  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/i2746_cross_clip_player_id/cache"))
    parser.add_argument("--e0-e7", type=Path, default=Path("outputs/i2746_cross_clip_player_id/e0_e7_results.json"))
    parser.add_argument("--followup", type=Path, default=Path("outputs/i2746_cross_clip_player_id/followup_results.json"))
    parser.add_argument("--gt", type=Path, default=RUN_DIR / "gt_cross_clip.json")
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()

    cache = load_cache(args.cache_dir)
    gt_doc, gt_map, purity_map = load_gt(args.gt)
    e0_e7 = json.loads(args.e0_e7.read_text())
    followup = json.loads(args.followup.read_text())
    warmup = select_warmup_clip_from_input_quality(cache)
    split = build_split_manifest(cache, warmup["warmup_clip_idx"])

    # 1. Leave-one-out integration ablation on TEST (and DEV) around adopted E4_swap_margin:
    # Adopted elements: (a) per-camera descriptor, (b) quality weighting, (c) Hungarian 1-to-1 + swap margin.
    # Dropping one element at a time:
    # - Full adopted: E4 (per-cam + quality + Hungarian + swap_margin)
    # - Drop swap_margin (use row_margin): E4 (row_margin)
    # - Drop Hungarian 1-to-1 (use independent NN): E3
    # - Drop quality weighting (uniform per-cam + Hungarian swap_margin): E2_hungarian_swap
    # - Drop per-camera (all-cam pooled + Hungarian swap_margin): E1_hungarian_swap
    cfg_sw = followup["swap_margin_comparison"]["dev"]["E4_swap_margin"]["calibrated_config"]
    loo_results: dict[str, Any] = {}
    for split_name, indices in [("dev", split["dev_indices"]), ("test", split["test_indices"])]:
        loo_split: dict[str, Any] = {}
        # Full adopted (E4_swap_margin)
        r_full = run_method(
            "E4", cache, gt_map, purity_map, split, indices,
            min_score=cfg_sw["min_score"], min_margin=cfg_sw["min_margin"], use_swap_margin=True, n_bootstrap=400,
        )
        loo_split["adopted_E4_swap"] = {
            "metrics_pure": r_full["metrics_pure"],
            "metrics_all": r_full["metrics_all"],
            "open_set": run_open_set_leave_one_out("E4", cfg_sw, cache, gt_map, purity_map, split, indices),
        }
        # Open-set specifically with swap_margin=True (separate from row_margin!)
        tot_u, tot_fa = 0, 0
        for om in ("player_01", "player_02", "player_03", "player_04"):
            ro = run_method(
                "E4", cache, gt_map, purity_map, split, indices,
                min_score=cfg_sw["min_score"], min_margin=cfg_sw["min_margin"],
                use_swap_margin=True, excluded_anchor_gt_id=om, n_bootstrap=0,
            )
            tot_u += ro["metrics_pure"]["unregistered_total_nodes"]
            tot_fa += ro["metrics_pure"]["unregistered_false_accepts"]
        loo_split["adopted_E4_swap"]["open_set_swap_margin"] = {
            "unreg_false_accepts": tot_fa,
            "unreg_total": tot_u,
            "unreg_false_accept_rate": round(tot_fa / max(1, tot_u), 6),
        }
        # Also test strict score-gated E4_swap (min_score=0.84) for open-set safety comparison
        r_strict = run_method(
            "E4", cache, gt_map, purity_map, split, indices,
            min_score=0.84, min_margin=0.02, use_swap_margin=True, n_bootstrap=400,
        )
        tot_u2, tot_fa2 = 0, 0
        for om in ("player_01", "player_02", "player_03", "player_04"):
            ro = run_method(
                "E4", cache, gt_map, purity_map, split, indices,
                min_score=0.84, min_margin=0.02,
                use_swap_margin=True, excluded_anchor_gt_id=om, n_bootstrap=0,
            )
            tot_u2 += ro["metrics_pure"]["unregistered_total_nodes"]
            tot_fa2 += ro["metrics_pure"]["unregistered_false_accepts"]
        loo_split["E4_swap_strict_open_set_gate"] = {
            "min_score": 0.84,
            "min_margin": 0.02,
            "metrics_pure": r_strict["metrics_pure"],
            "open_set_swap_margin": {
                "unreg_false_accepts": tot_fa2,
                "unreg_total": tot_u2,
                "unreg_false_accept_rate": round(tot_fa2 / max(1, tot_u2), 6),
            },
        }
        # LOO - drop swap_margin (use row_margin)
        loo_split["minus_swap_margin_E4_row"] = e0_e7[split_name]["E4"]["metrics_pure"]
        # LOO - drop Hungarian 1-to-1 (use independent NN = E3)
        loo_split["minus_hungarian_E3"] = e0_e7[split_name]["E3"]["metrics_pure"]
        # LOO - drop quality weighting (uniform per-cam + Hungarian + swap_margin)
        r_no_qual = run_method(
            "E2", cache, gt_map, purity_map, split, indices,
            min_score=cfg_sw["min_score"], min_margin=cfg_sw["min_margin"],
            use_swap_margin=True, force_hungarian=True, n_bootstrap=400,
        )
        loo_split["minus_quality_E2_hungarian_swap"] = r_no_qual["metrics_pure"]
        # LOO - drop per-camera descriptor (all-cam pooled + Hungarian + swap_margin)
        r_no_percam = run_method(
            "E1", cache, gt_map, purity_map, split, indices,
            min_score=cfg_sw["min_score"], min_margin=cfg_sw["min_margin"],
            use_swap_margin=True, force_hungarian=True, n_bootstrap=400,
        )
        loo_split["minus_per_cam_E1_hungarian_swap"] = r_no_percam["metrics_pure"]
        loo_results[split_name] = loo_split

    # 2. Emit versioned player_identity_links sample (6 representative clips incl. anchor, end change, outerwear change, test, mixed clip_062)
    sample_clips = [split["warmup_clip_idx"], 5, 19, 39, 62, 63]
    r_sample = run_method(
        "E4", cache, gt_map, purity_map, split, [i for i in sample_clips if i != split["warmup_clip_idx"]],
        min_score=cfg_sw["min_score"], min_margin=cfg_sw["min_margin"], use_swap_margin=True, n_bootstrap=0,
    )
    doc_dict = r_sample["links_document"]
    # Verify downstream #1722/#2545 join on sample clip_039 events without touching SceneResult
    from contract import LinkEvidence, PlayerIdentityLink, PlayerIdentityLinksDocument

    reconstructed_links = tuple(
        PlayerIdentityLink(
            clip_id=lk["clip_id"],
            local_player_id=lk["local_player_id"],
            global_player_id=lk["global_player_id"],
            status=lk["status"],
            score=lk["score"],
            score_kind=lk["score_kind"],
            runner_up_score=lk["runner_up_score"],
            assignment_margin=lk["assignment_margin"],
            reason=lk["reason"],
            evidence=LinkEvidence(
                cameras_used=tuple(lk["evidence"]["cameras_used"]),
                sample_frames=lk["evidence"]["sample_frames"],
                sample_source_rows=lk["evidence"]["sample_source_rows"],
                appearance_term=lk["evidence"]["appearance_term"],
                shape_term=lk["evidence"]["shape_term"],
                temporal_term=lk["evidence"]["temporal_term"],
                shape_valid=lk["evidence"]["shape_valid"],
                temporal_valid=lk["evidence"]["temporal_valid"],
                gallery_update_sources=tuple(lk["evidence"]["gallery_update_sources"]),
                extra=lk["evidence"]["extra"],
            ),
            frame_interval=tuple(lk["frame_interval"]) if lk["frame_interval"] is not None else None,
        )
        for lk in doc_dict["links"]
    )
    doc_obj = PlayerIdentityLinksDocument(
        schema_version=doc_dict["schema_version"],
        identity_scope=doc_dict["identity_scope"],
        registry_version=doc_dict["registry_version"],
        source_manifest_hashes=doc_dict["source_manifest_hashes"],
        upstream_digests=doc_dict["upstream_digests"],
        encoder_identity=doc_dict["encoder_identity"],
        preprocessing_identity=doc_dict["preprocessing_identity"],
        method=doc_dict["method"],
        config=doc_dict["config"],
        links=reconstructed_links,
    )
    demo_events = [
        {"clip_id": "video_000/clip_005", "frame": 120, "event_kind": "hit", "local_player_id": 0},
        {"clip_id": "video_000/clip_019", "frame": 200, "event_kind": "hit", "local_player_id": 1},
        {"clip_id": "video_000/clip_039", "frame": 150, "event_kind": "hit", "local_player_id": 0},
        {"clip_id": "video_000/clip_063", "frame": 180, "event_kind": "hit", "local_player_id": 2},
    ]
    joined_demo = join_events_with_global_ids(demo_events, doc_obj, require_accepted=True)
    sample_out = {
        "player_identity_links_example": doc_dict,
        "registry_example": r_sample["registry"],
        "downstream_event_join_example": joined_demo,
    }
    (args.out_dir / "player_identity_links_sample.json").write_text(json.dumps(sample_out, indent=2, ensure_ascii=False))

    # 3. Compact summary CSV of all conditions (dev + test)
    rows = []
    def add_row(split_name: str, name: str, m: dict[str, Any], m_all: dict[str, Any], dg: dict[str, Any]) -> None:
        rows.append(
            {
                "split": split_name,
                "method": name,
                "pair_precision": m["cross_clip_pair_precision"],
                "pair_recall": m["cross_clip_pair_recall"],
                "pair_f1": m["cross_clip_pair_f1"],
                "pair_f1_ci95_lo": m["block_bootstrap_f1_ci95"][0] if m.get("block_bootstrap_f1_ci95") else "",
                "pair_f1_ci95_hi": m["block_bootstrap_f1_ci95"][1] if m.get("block_bootstrap_f1_ci95") else "",
                "false_merge_rate": m["false_merge_rate"],
                "split_rate": m["split_rate"],
                "tp_pairs": m["tp_pairs"],
                "fp_pairs": m["fp_pairs"],
                "fn_pairs": m["fn_pairs"],
                "coverage": m["coverage"],
                "accepted_nodes": m["accepted_nodes"],
                "total_nodes": m["total_registered_nodes"],
                "clip_exact_match_rate": m["clip_exact_match_rate"],
                "id_switch_boundaries": m["id_switch_boundaries"],
                "scope_frame_idf1": m["scope_frame_idf1"],
                "scope_frame_assa": m["scope_frame_assa"],
                "all_nodes_pair_f1": m_all["cross_clip_pair_f1"],
                "wall_ms": dg.get("wall_ms", ""),
            }
        )

    for sp in ("dev", "test"):
        for mname in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"):
            entry = e0_e7[sp][mname]
            add_row(sp, mname, entry["metrics_pure"], entry["metrics_all"], entry["diagnostics"])
        for mname in ("E4_swap_margin", "E7_swap_margin"):
            entry = followup["swap_margin_comparison"][sp][mname]
            add_row(sp, mname, entry["metrics_pure"], entry["metrics_all"], entry["diagnostics"])
        for mname in ("G_greedy", "G_milp"):
            entry = followup["axis_g"][sp][mname]
            add_row(sp, mname, entry["metrics_pure"], entry["metrics_all"], entry["diagnostics"])
        if followup["axis_p_kpr"].get("available"):
            for mname in ("P_unprompted", "P_prompted", "P_unprompted_swap", "P_prompted_swap"):
                if mname in followup["axis_p_kpr"][sp]:
                    entry = followup["axis_p_kpr"][sp][mname]
                    add_row(sp, mname, entry["metrics_pure"], entry["metrics_all"], entry["diagnostics"])

    csv_path = args.out_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 4. Compact metrics.json (<256 KiB) with all provenance, digests, split, diagnostics, and leave-one-out
    kpr_manifest = json.loads((args.cache_dir / "kpr_manifest.json").read_text()) if (args.cache_dir / "kpr_manifest.json").is_file() else None
    compact = {
        "schema_version": 1,
        "issue": 2746,
        "dataset_id": "meiji_4cam",
        "video_id": "video_000",
        "benchmark_version": "multiview-v4",
        "benchmark_manifest_sha256": cache.manifest["benchmark_manifest_sha256"],
        "projects_json_sha256": cache.manifest["projects_json_sha256"],
        "gt_audit": gt_doc["audit"],
        "warmup": e0_e7["warmup"],
        "split": {
            "warmup_clip_name": e0_e7["split"]["warmup_clip_name"],
            "dev_count": len(e0_e7["split"]["dev_indices"]),
            "test_count": len(e0_e7["split"]["test_indices"]),
            "dev_clip_names": e0_e7["split"]["dev_clip_names"],
            "test_clip_names": e0_e7["split"]["test_clip_names"],
            "source_frame_overlaps_verified": e0_e7["split"]["source_frame_overlaps_verified"],
        },
        "cache_provenance": {
            "clip_reid": {
                "encoder": cache.manifest["encoder"],
                "sampling": cache.manifest["sampling"],
                "stats": cache.manifest["stats"],
            },
            "kpr": kpr_manifest,
        },
        "smpl_beta_dev_diagnostics": e0_e7["smpl_beta_dev_diagnostics"],
        "temporal_gap_audit": e0_e7["temporal_gap_audit"],
        "llr_calibration_dev": followup["llr_calibration_dev"],
        "dev_open_set": e0_e7["dev_open_set"],
        "test_open_set": e0_e7["test_open_set"],
        "dev_camera_dropout": e0_e7["dev_camera_dropout"],
        "test_camera_dropout": e0_e7["test_camera_dropout"],
        "leave_one_out_integration": loo_results,
        "axis_g_diagnostics": {
            "dev": {k: v["diagnostics"] for k, v in followup["axis_g"]["dev"].items()},
            "test": {k: v["diagnostics"] for k, v in followup["axis_g"]["test"].items()},
        },
        "e7_gallery_diagnostics": {
            "dev": e0_e7["dev"]["E7"]["diagnostics"],
            "test": e0_e7["test"]["E7"]["diagnostics"],
        },
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"Wrote {csv_path} ({csv_path.stat().st_size/1024:.1f} KiB), {metrics_path} ({metrics_path.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
