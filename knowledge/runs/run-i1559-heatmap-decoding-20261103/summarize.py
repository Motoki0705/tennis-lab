"""Render saved foundation metrics as Markdown; performs no scoring."""

import argparse
import json
from pathlib import Path


def metric(row):
    if row["median"] is None:
        return "n/a"
    return f"{row['median']:.3f} / {row['p90']:.3f} / {100 * row['above_threshold_rate']:.2f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    scores = json.loads((args.comparison / "scores.json").read_text())
    selection = json.loads((args.comparison / "selection.json").read_text())
    lines = ["# clip_000 heatmap復号の指標", "", "値は median px / p90 px / >20px率。coverageは採点数/対象数。", "",
             "## 主評価 [505,1010)", "", "| 方式 | LOVO | coverage | 全view median px | 有効3D率 |", "|---|---:|---:|---:|---:|"]
    for name, method in scores["methods"].items():
        row = method["lovo_reprojection_px"]["evaluation"]
        full = method["folds"]["all_views"]["evaluation"]
        lines.append(f"| {name} | {metric(row)} | {row['count']}/{row['label_count']} ({100 * row['scored_label_coverage']:.2f}%) | {full['reprojection_px']['median']:.3f} | {100 * full['valid_3d_rate']:.2f}% |")
    chosen = list(dict.fromkeys(["pipeline_baseline", "without_loess", selection["base_winner"], selection["selected"], selection["selected"] + "_raw", "legacy_stride", "legacy_affine", "annotation_oracle"]))
    for segment, label in (("tuning", "前半 [0,505)"), ("all", "全区間 [0,1010)")):
        lines += ["", "## " + label, "", "| 方式 | LOVO | coverage |", "|---|---:|---:|"]
        for name in chosen:
            row = scores["methods"][name]["lovo_reprojection_px"][segment]
            lines.append(f"| {name} | {metric(row)} | {row['count']}/{row['label_count']} ({100 * row['scored_label_coverage']:.2f}%) |")
    lines += ["", "## LOVO別・区間別", "", "| 方式 | 除外camera | 区間 | median / p90 / >20px | 採点数/対象数 |", "|---|---|---|---:|---:|"]
    for name in chosen:
        for fold in ("without_cam0", "without_cam1", "without_cam2"):
            for segment in ("evaluation", "tuning", "all"):
                row = scores["methods"][name]["folds"][fold][segment]
                lines.append(f"| {name} | {fold[8:]} | {segment} | {metric(row['reprojection_px'])} | {row['scored_label_count']}/{row['scoring_label_count']} |")
    paired = json.loads((args.comparison / "paired-scores.json").read_text())
    lines += ["", "## baselineと選定方式の共通採点点", "", "| 方式 | 区間 | LOVO | 採点数 |", "|---|---|---:|---:|"]
    for name, method in paired["methods"].items():
        for segment in ("evaluation", "tuning", "all"):
            row = method["common_lovo_reprojection_px"][segment]
            lines.append(f"| {name} | {segment} | {metric(row)} | {row['count']} |")
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
