"""Compare the three CPU media interventions to the frozen CUDA observations."""

from __future__ import annotations

import argparse
import json
import runpy
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from audit_frames import (
    GATE,
    ROOT,
    Inputs,
    anchor_order,
    csv_rows,
    match_candidates,
    read,
    write,
)

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(output: Path | None = None) -> None:
    torch.set_num_threads(1)
    out = ROOT / "conclusion" if output is None else output
    if not out.is_absolute():
        raise ValueError("Output must be an absolute new directory")
    out.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    gate_api = runpy.run_path(str(GATE.parent / "evaluate_gate.py"))
    labels = dict(np.load(inputs.pin(GATE / "cam2-labels.npz"), allow_pickle=False))
    target, observed = labels["target_uv"], labels["target_reason"] == 0
    stored = dict(np.load(inputs.pin(ROOT / "frames/cam2.npz"), allow_pickle=False))
    cache_winner = stored["cached_refiner_mixture_logits"].argmax(-1)
    source_winner = stored["source_refiner_mixture_logits"].argmax(-1)
    metrics, frame_rows = [], []
    timeline: dict[str, Any] = {}
    for name in ("cached", "source"):
        means = stored[f"{name}_refiner_means"]
        winner = cache_winner if name == "cached" else source_winner
        timeline[name] = np.full(270, np.nan)
        timeline[name][observed] = np.linalg.norm((means[np.arange(270), winner][observed].astype(np.float64)
                                                   - target[observed]) * [1919, 1079], axis=-1)
    for mode in ("raw", "resize_only", "reencoded"):
        base = ROOT / "isolation/cpu-v3" / f"cam2-{mode}"
        arrays = dict(np.load(inputs.pin(base / "predictions.npz"), allow_pickle=False))
        summary = read(inputs.pin(base / "summary.json"))
        if summary["status"] != "complete" or summary["cuda_initialized"]:
            raise ValueError("Require completed CPU-only isolation")
        for variant in ("pipeline", "store_coordinate_convention", "cached_evidence_replay"):
            prediction = BallGMM2D(**{f.name: torch.from_numpy(arrays[f"{variant}_{f.name}"][None]) for f in fields(BallGMM2D)})
            scores = gate_api["score"](prediction, target, observed, (1920, 1080))
            winner = prediction.mixture_logits[0].argmax(-1).numpy()
            metrics.append({"mode": mode, "refiner_input": variant, **gate_api["summarize"]([scores]),
                "winner_changes_from_cache_cuda": int((winner != cache_winner).sum()),
                "winner_changes_from_source_cuda": int((winner != source_winner).sum()),
                "free_winner_frames": int((winner == 3).sum()),
                "free_winner_frames_217_265": int((winner[217:266] == 3).sum())})
            if variant == "pipeline":
                timeline[mode] = np.full(270, np.nan)
                timeline[mode][observed] = scores["error_px"]
            for index, f in enumerate(np.flatnonzero(observed)):
                frame_rows.append({"mode": mode, "refiner_input": variant, "camera": "cam2", "frame": int(f),
                    "pts": int(labels["pts"][f]), "error_px": float(scores["error_px"][index]),
                    "nll_px": float(scores["nll_px"][index]), "winner": int(winner[f])})
    # How much of the huge same-component mean delta involves reassigning its anchor?
    anchor_rows = []
    for cam in ("cam0", "cam1", "cam2"):
        a = dict(np.load(inputs.pin(ROOT / f"frames/{cam}.npz"), allow_pickle=False))
        cp, sp = a["cache_candidate_coords"] * [1919, 1079], a["source_candidate_uv_px"]
        ca = anchor_order(cp, a["cache_candidate_scores"], a["cache_candidate_valid"])
        sa = anchor_order(sp, a["source_candidate_scores"], a["source_candidate_valid"])
        big, identity_changed, paired_mean_deltas = 0, 0, []
        for f in range(270):
            mapping = match_candidates(cp[f], sp[f], a["cache_candidate_valid"][f], a["source_candidate_valid"][f])
            for k in range(3):
                delta = float(np.linalg.norm((a["cached_refiner_means"][f, k] - a["source_refiner_means"][f, k]) * [1919, 1079]))
                if delta > 100:
                    big += 1
                    identity_changed += int(mapping[ca[f, k]] != sa[f, k])
                matched = np.flatnonzero(sa[f, :3] == mapping[ca[f, k]])
                if len(matched):
                    paired_mean_deltas.append(float(np.linalg.norm((a["cached_refiner_means"][f, k] - a["source_refiner_means"][f, matched[0]]) * [1919, 1079])))
        anchor_rows.append({"camera": cam, "anchored_component_frame_pairs": 810,
            "same_component_mean_distance_over_100px": big, "among_those_anchor_not_spatially_same_20px": identity_changed,
            "spatially_matched_anchors_in_top3": len(paired_mean_deltas),
            "spatially_matched_component_mean_distance_px_median_p90": np.quantile(paired_mean_deltas, [.5, .9]).tolist()})
    csv_rows(out / "cam2-isolation-observed.csv", frame_rows)
    noise = read(inputs.pin(ROOT / "noise-balanced-v2/summary.json"))
    write(out / "summary.json", {"status": "complete", "metrics": metrics, "anchor_reassignment": anchor_rows,
        "scope": "cam2 270 frames, 217 observed, CPU media interventions; no pipeline/gate/default/calibration change",
        "causal_limit": "Resize and JPEG are isolated as sequential interventions, not additive effects; CPU/CUDA differences reported separately; no new independent clip tested"})
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), layout="constrained")
    for name, color, style in (("cached", "#008855", "-"), ("source", "#cc3322", "-"),
                               ("reencoded", "#2255cc", "--"), ("resize_only", "#aa7700", ":")):
        axes[0].plot(np.arange(270), timeline[name], style, color=color, linewidth=1.4, label=name)
    axes[0].axvspan(217, 265, color="gray", alpha=.12)
    axes[0].set(title="cam2: observed error, unchanged top-weight component rule", ylabel="source pixels", xlabel="original frame")
    axes[0].legend(ncol=4)
    axes[1].plot(source_winner, ".", label="source CUDA", color="#cc3322")
    axes[1].plot(cache_winner, "+", label="cache CUDA", color="#008855")
    axes[1].axvspan(217, 265, color="gray", alpha=.12)
    axes[1].set(yticks=[0, 1, 2, 3], yticklabels=["anchor 0", "anchor 1", "anchor 2", "free"],
                ylabel="heaviest component", xlabel="original frame")
    axes[1].legend()
    selected = [r for r in noise["results"] if r["partition_offset"] == 0 and not r["camera_blocks_synchronized"]]
    for i, r in enumerate(selected):
        lo, hi = r["pooled"]["percentile_95_interval_px"]
        med = r["pooled"]["median_px"]
        axes[2].plot([lo, hi], [i, i], "-", color="#444444", linewidth=3)
        axes[2].plot(med, i, "o", color="#444444")
    axes[2].axvline(0, color="black", linestyle=":")
    axes[2].axvline(noise["pooled_original_delta_px"], color="#cc3322", label="observed +46.34 px")
    axes[2].set(yticks=range(len(selected)), yticklabels=[f"{r['block_frames']}-frame blocks" for r in selected],
                xlabel="pooled p90 source - cache (px)", title="Paired block bootstrap: 95% percentile intervals, 10,000 draws")
    axes[2].legend()
    figure.savefig(out / "tail-audit.png", dpi=130)
    plt.close(figure)
    inputs.finish(out / "input_sha256.json")
    print(json.dumps({"metrics": metrics, "anchor_reassignment": anchor_rows}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    main(parser.parse_args().output)
