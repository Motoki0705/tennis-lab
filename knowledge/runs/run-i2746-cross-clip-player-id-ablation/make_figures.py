"""Generate compact (<512 KiB each) publication-quality figures and 4-view identity montage (Issue #2746)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN_DIR = Path(__file__).resolve().parent
if str(RUN_DIR) not in sys.path:
    sys.path.insert(0, str(RUN_DIR))

from ablation import (  # noqa: E402
    build_split_manifest,
    load_cache,
    run_method,
    select_warmup_clip_from_input_quality,
)
from run_e0_e7 import load_gt  # noqa: E402

PLAYER_COLORS_BGR = {
    "player_01": (60, 76, 231),   # Red
    "player_02": (219, 152, 52),  # Blue
    "player_03": (113, 204, 46),  # Green
    "player_04": (15, 196, 241),  # Gold/Yellow
}


def plot_ablation_and_pc(e0_e7: dict[str, Any], followup: dict[str, Any], out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.8), dpi=135)

    # Panel A: Dev vs Test Pair F1 across E0..E7 + swap-margin + G + P(swap)
    labels = ["E0", "E1", "E2", "E3", "E4\nrow", "E5\n+beta", "E6\n+3D*", "E7\n+upd", "E4\nswap", "E7\nswap"]
    f1_dev = [e0_e7["dev"][m]["metrics_pure"]["cross_clip_pair_f1"] for m in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")]
    f1_test = [e0_e7["test"][m]["metrics_pure"]["cross_clip_pair_f1"] for m in ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")]
    for mname in ("E4_swap_margin", "E7_swap_margin"):
        f1_dev.append(followup["swap_margin_comparison"]["dev"][mname]["metrics_pure"]["cross_clip_pair_f1"])
        f1_test.append(followup["swap_margin_comparison"]["test"][mname]["metrics_pure"]["cross_clip_pair_f1"])

    if "G_greedy" in followup["axis_g"]["test"]:
        labels += ["G\ngreedy", "G\nMILP"]
        f1_dev += [
            followup["axis_g"]["dev"]["G_greedy"]["metrics_pure"]["cross_clip_pair_f1"],
            followup["axis_g"]["dev"]["G_milp"]["metrics_pure"]["cross_clip_pair_f1"],
        ]
        f1_test += [
            followup["axis_g"]["test"]["G_greedy"]["metrics_pure"]["cross_clip_pair_f1"],
            followup["axis_g"]["test"]["G_milp"]["metrics_pure"]["cross_clip_pair_f1"],
        ]
    if followup["axis_p_kpr"].get("available") and "P_unprompted_swap" in followup["axis_p_kpr"]["test"]:
        labels += ["KPR\nno-prm", "KPR\n2D-prm"]
        f1_dev += [
            followup["axis_p_kpr"]["dev"]["P_unprompted_swap"]["metrics_pure"]["cross_clip_pair_f1"],
            followup["axis_p_kpr"]["dev"]["P_prompted_swap"]["metrics_pure"]["cross_clip_pair_f1"],
        ]
        f1_test += [
            followup["axis_p_kpr"]["test"]["P_unprompted_swap"]["metrics_pure"]["cross_clip_pair_f1"],
            followup["axis_p_kpr"]["test"]["P_prompted_swap"]["metrics_pure"]["cross_clip_pair_f1"],
        ]

    x = np.arange(len(labels))
    w = 0.37
    ax = axes[0]
    ax.bar(x - w / 2, f1_dev, w, label="Dev (Games 1-4, 34 clips)", color="#4C78A8", alpha=0.88)
    ax.bar(x + w / 2, f1_test, w, label="Test (Games 5-9, 34 clips)", color="#F58518", alpha=0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Cross-clip Pair F1 (pure tracks)")
    ax.set_title("A. Cross-clip Pair F1: E0-E7, Swap-Margin, Axis G & Axis P")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, loc="lower right")
    for xi, v in zip(x + w / 2, f1_test, strict=True):
        if v > 0:
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6)

    # Panel B: Test Precision-Coverage curves (E1..E4 row-margin + E4_swap + E7_swap + KPR prompted swap)
    ax2 = axes[1]
    curve_specs = [
        ("E1 (all-cam NN)", e0_e7["test"]["E1"]["precision_coverage_curve"], "#E45756", "--"),
        ("E2 (per-cam NN)", e0_e7["test"]["E2"]["precision_coverage_curve"], "#72B7B2", "-."),
        ("E3 (qual per-cam NN)", e0_e7["test"]["E3"]["precision_coverage_curve"], "#54A24B", ":"),
        ("E4 (Hungarian row-margin)", e0_e7["test"]["E4"]["precision_coverage_curve"], "#4C78A8", "-"),
    ]
    if "E4_swap_margin" in followup["swap_margin_comparison"]["test"]:
        curve_specs.append(
            ("E4 (Hungarian swap-margin)", followup["swap_margin_comparison"]["test"]["E4_swap_margin"]["precision_coverage_curve"], "#F58518", "-")
        )
    if "E7_swap_margin" in followup["swap_margin_comparison"]["test"]:
        curve_specs.append(
            ("E7 (+causal update, swap)", followup["swap_margin_comparison"]["test"]["E7_swap_margin"]["precision_coverage_curve"], "#2CA02C", "-")
        )
    if followup["axis_p_kpr"].get("available") and "P_prompted_swap" in followup["axis_p_kpr"]["test"]:
        curve_specs.append(
            ("P: KPR 2D-prompted (swap)", followup["axis_p_kpr"]["test"]["P_prompted_swap"]["precision_coverage_curve"], "#B279A2", "-.")
        )
    for cname, pts, col, ls in curve_specs:
        if not pts:
            continue
        covs = [p["coverage"] for p in pts]
        precs = [p["pair_precision"] for p in pts]
        ax2.plot(covs, precs, marker="o", markersize=3.0, linewidth=1.8, linestyle=ls, color=col, label=cname)
    ax2.set_xlim(0.0, 1.04)
    ax2.set_ylim(0.60, 1.03)
    ax2.set_xlabel("Coverage (accepted registered nodes / total)")
    ax2.set_ylabel("Cross-clip Pair Precision (1 - False Merge Rate)")
    ax2.set_title("B. Test Precision-Coverage Trade-off (All Post-Wardrobe-Change)")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=7.6, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=135, pil_kwargs={"optimize": True})
    plt.close(fig)
    print(f"Wrote {out_png} ({out_png.stat().st_size / 1024:.1f} KiB)")


def build_4view_consistency_montage(
    cache_dir: Path,
    gt_path: Path,
    bench_root: Path,
    e0_e7: dict[str, Any],
    followup: dict[str, Any],
    out_png: Path,
) -> None:
    """Create a 4-row (player_01..04) x 5-clip montage showing consistent global IDs across:
    anchor (clip_002), first end-change (clip_005), outerwear removal (clip_019),
    test post-removal + end-change (clip_039), and final game long-gap (clip_063).
    Each cell displays a 2x2 4-view thumbnail (cam0..cam3) framed in the predicted global ID color.
    """
    cache = load_cache(cache_dir)
    crop_boxes = np.load(cache_dir / "clip_reid_cache.npz")["crop_boxes"]
    gt_doc, gt_map, purity_map = load_gt(gt_path)
    warmup = select_warmup_clip_from_input_quality(cache)
    split = build_split_manifest(cache, warmup["warmup_clip_idx"])

    # Run E4_swap_margin (adopted) on all clips (except warmup) using frozen dev config
    cfg_sw = followup["swap_margin_comparison"]["dev"]["E4_swap_margin"]["calibrated_config"]
    all_query = [i for i in range(len(cache.clip_names)) if i != split["warmup_clip_idx"]]
    res_e4 = run_method(
        "E4",
        cache,
        gt_map,
        purity_map,
        split,
        all_query,
        min_score=cfg_sw["min_score"],
        min_margin=cfg_sw["min_margin"],
        use_swap_margin=True,
        n_bootstrap=0,
    )
    lmap = res_e4["metrics_pure"]["scope_label_map"]
    pred_lookup: dict[tuple[str, int], tuple[str | None, str]] = {}
    for n in res_e4["eval_nodes"]:
        mapped = lmap.get(n.pred_global_id or "") if n.status == "accepted" else None
        pred_lookup[(n.clip_name, n.local_player_id)] = (mapped, n.status)
    # Warmup anchor itself
    w_cname = cache.clip_names[split["warmup_clip_idx"]]
    for p in range(4):
        pred_lookup[(w_cname, p)] = (gt_map[(w_cname, p)], "anchor")

    selected_clips = [
        (w_cname, "Warm-up Anchor\n(Game 1)"),
        ("clip_005", "End Change #1\n(Game 2, +132s)"),
        ("clip_019", "Outerwear Off + EC\n(Game 4, +494s)"),
        ("clip_039", "Test: Post-Change\n(Game 6, +1003s)"),
        ("clip_063", "Test: Final Game\n(Game 9, +1626s)"),
    ]

    # Build 2x2 4-view mini-tiles per (gt_player, selected_clip)
    sub_w, sub_h = 68, 102
    cell_w, cell_h = sub_w * 2 + 10, sub_h * 2 + 34
    left_w, top_h = 128, 56
    canvas_w = left_w + len(selected_clips) * cell_w
    canvas_h = top_h + 4 * cell_h
    canvas = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)

    # Header
    for ci_col, (cname, subtitle) in enumerate(selected_clips):
        x0 = left_w + ci_col * cell_w
        cv2.putText(canvas, cname, (x0 + 8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        for li, ln in enumerate(subtitle.split("\n")):
            cv2.putText(canvas, ln, (x0 + 8, 38 + li * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 215, 245), 1, cv2.LINE_AA)

    gt_players = ["player_01", "player_02", "player_03", "player_04"]
    role_labels = {
        "player_01": "P01 (Net A)\nRed->Blk T",
        "player_02": "P02 (Net B)\nBlk G->Blk T",
        "player_03": "P03 (Base A)\nBlk+LtBlue",
        "player_04": "P04 (Base B)\nWht+Blk",
    }

    for r_idx, gid in enumerate(gt_players):
        y0 = top_h + r_idx * cell_h
        col_bgr = PLAYER_COLORS_BGR[gid]
        cv2.rectangle(canvas, (4, y0 + 4), (left_w - 6, y0 + cell_h - 4), col_bgr, 2)
        for li, ln in enumerate(role_labels[gid].split("\n")):
            cv2.putText(canvas, ln, (12, y0 + 92 + li * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    # Decode best frame per camera for each selected clip
    for ci_col, (cname, _) in enumerate(selected_clips):
        ci = cache.clip_names.index(cname)
        clip_dir = bench_root / "videos/video_000/clips" / cname
        wanted_frames: dict[int, set[int]] = {v: set() for v in range(4)}
        best_k_pv: dict[tuple[int, int], int] = {}
        for p in range(4):
            for v in range(4):
                vk = np.flatnonzero(cache.crop_valid[ci, p, v])
                if len(vk) > 0:
                    # Pick crop with highest box height and lowest overlap
                    scores_k = cache.crop_heights[ci, p, v, vk] * np.exp(-3.0 * cache.crop_max_iou[ci, p, v, vk])
                    bk = int(vk[int(np.argmax(scores_k))])
                    best_k_pv[(p, v)] = bk
                    wanted_frames[v].add(int(cache.crop_frames[ci, p, v, bk]))

        decoded: dict[int, dict[int, np.ndarray]] = {v: {} for v in range(4)}
        for v in range(4):
            if not wanted_frames[v]:
                continue
            cap = cv2.VideoCapture(str(clip_dir / f"media/cam{v}.mp4"))
            max_f = max(wanted_frames[v])
            fidx = 0
            while fidx <= max_f:
                ok, frame = cap.read()
                if not ok:
                    break
                if fidx in wanted_frames[v]:
                    decoded[v][fidx] = frame
                fidx += 1
            cap.release()

        for p in range(4):
            gt_gid = gt_map[(cname, p)]
            if gt_gid not in gt_players:
                continue
            r_idx = gt_players.index(gt_gid)
            x0 = left_w + ci_col * cell_w
            y0 = top_h + r_idx * cell_h
            pred_gid, st = pred_lookup.get((cname, p), (None, "unresolved"))
            frame_color = PLAYER_COLORS_BGR.get(pred_gid or "", (120, 120, 120))
            cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x0 + cell_w - 3, y0 + cell_h - 3), frame_color, 2)
            side = int(cache.sides[ci, p])
            ok_str = "OK" if pred_gid == gt_gid else f"ERR:{pred_gid}"
            lbl = f"loc={p} s={side:+d} -> {pred_gid or 'UNR'} ({ok_str})"
            cv2.putText(canvas, lbl, (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)

            for v in range(4):
                bk = best_k_pv.get((p, v))
                if bk is None:
                    continue
                f = int(cache.crop_frames[ci, p, v, bk])
                fr = decoded[v].get(f)
                if fr is None:
                    continue
                bx = crop_boxes[ci, p, v, bk]
                ix1, iy1, ix2, iy2 = max(0, int(bx[0])), max(0, int(bx[1])), min(fr.shape[1], int(bx[2])), min(fr.shape[0], int(bx[3]))
                patch = fr[iy1:iy2, ix1:ix2]
                if patch.size == 0:
                    continue
                tile = cv2.resize(patch, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
                gx = x0 + 5 + (v % 2) * sub_w
                gy = y0 + 26 + (v // 2) * sub_h
                canvas[gy : gy + sub_h, gx : gx + sub_w] = tile
                cv2.putText(canvas, f"c{v}", (gx + 3, gy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 140), 1, cv2.LINE_AA)

    # Save PNG under 512 KiB budget (use palette quantization if needed via PIL)
    from PIL import Image

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.save(out_png, format="PNG", optimize=True)
    if out_png.stat().st_size > 500 * 1024:
        img_q = img.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        img_q.save(out_png, format="PNG", optimize=True)
    print(f"Wrote {out_png} ({out_png.stat().st_size / 1024:.1f} KiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/i2746_cross_clip_player_id/cache"))
    parser.add_argument("--e0-e7", type=Path, default=Path("outputs/i2746_cross_clip_player_id/e0_e7_results.json"))
    parser.add_argument("--followup", type=Path, default=Path("outputs/i2746_cross_clip_player_id/followup_results.json"))
    parser.add_argument("--gt", type=Path, default=RUN_DIR / "gt_cross_clip.json")
    parser.add_argument("--bench-root", type=Path, default=Path("data/tennis_multiview/processed/meiji_4cam/benchmarks/multiview-v4"))
    parser.add_argument("--out-dir", type=Path, default=RUN_DIR)
    args = parser.parse_args()

    e0_e7 = json.loads(args.e0_e7.read_text())
    followup = json.loads(args.followup.read_text())
    plot_ablation_and_pc(e0_e7, followup, args.out_dir / "ablation_comparison.png")
    build_4view_consistency_montage(
        args.cache_dir, args.gt, args.bench_root, e0_e7, followup, args.out_dir / "montage_cross_clip_consistency.png"
    )


if __name__ == "__main__":
    main()
