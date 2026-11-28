"""Compare wrist reach and score margins using only labels from V4 validation."""

import argparse
import json
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

from src.tennis_scene.analytics.hitter import HitterThresholds, estimate_hitter
from src.tennis_scene.analytics.hitter_eval import (
    audit_splits,
    labeled_shots,
    read_labels,
)
from src.tennis_scene.analytics.hitter_v4 import V4Benchmark

BASELINE = HitterThresholds(wrist_reach_m=1.15, min_margin=0.10)
WRIST_REACH_M = (1.00, 1.15, 1.30, 1.45)
MARGINS = (0.06, 0.10, 0.14, 0.18)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("manifest_sha256")
    parser.add_argument("gt", type=Path)
    args = parser.parse_args()
    benchmark = V4Benchmark.load(args.benchmark, args.manifest_sha256)
    splits = audit_splits(benchmark)
    if any(
        splits[key]
        for key in (
            "cross_split_time_overlaps_sec",
            "cross_split_source_frame_overlaps_inclusive",
            "cross_split_duplicate_media_sha256",
        )
    ):
        raise ValueError("Resolve split overlaps before calibration")
    labels, gt_sha = read_labels(args.gt, benchmark)
    validation = list(labeled_shots(benchmark, labels, "val"))
    rows = []
    for reach, margin in product(WRIST_REACH_M, MARGINS):
        thresholds = replace(BASELINE, wrist_reach_m=reach, min_margin=margin)
        correct = wrong = resolved = 0
        for label, shot, scene in validation:
            estimate = estimate_hitter(
                scene,
                frame=shot.frame,
                frame_uncertainty=shot.frame_uncertainty,
                court_side=shot.court_side,
                thresholds=thresholds,
            )
            if estimate.resolution_status == "resolved":
                resolved += 1
                if label.gt_local_player_id is not None:
                    if estimate.local_player_id == label.gt_local_player_id:
                        correct += 1
                    else:
                        wrong += 1
        rows.append(
            {
                "thresholds": asdict(thresholds),
                "rule_version": thresholds.rule_version,
                "resolved": resolved,
                "correct": correct,
                "wrong": wrong,
            }
        )
    selected = min(
        rows,
        key=lambda row: (
            row["wrong"],
            -row["correct"],
            abs(row["thresholds"]["wrist_reach_m"] - BASELINE.wrist_reach_m),
            -row["thresholds"]["min_margin"],
        ),
    )
    print(
        json.dumps(
            {
                "split": "val",
                "clips": splits["clip_counts"]["val"],
                "shots": len(validation),
                "manifest_sha256": benchmark.manifest_sha256,
                "ground_truth_sha256": gt_sha,
                "selection": "fewest wrong, then most correct, then closest to original wrist reach, then larger margin",
                "selected": selected,
                "grid": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
