"""Verify the complete output inventory, capture only small research receipts."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    target = Path(__file__).parent
    inventory = json.loads((root/"artifacts.json").read_text())
    declared = set(inventory["files"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if actual != declared | {"artifacts.json"}:
        raise ValueError("Output inventory has undeclared or missing files")
    for relative, spec in inventory["files"].items():
        path = root/relative
        if path.stat().st_size != spec["size_bytes"] or dual_sha256(path) != spec["sha256"]:
            raise ValueError(f"Artifact checksum/size mismatch: {relative}")
    copied = []
    for filename in ("artifacts.json", "frozen-calibration.json", "uncertainty.json", "metrics-table.png",
                     "uncertainty.png", "ellipsoids.png", "trajectories.png", "extremes.json"):
        path = root/filename
        limit = 512*1024 if path.suffix == ".png" else 256*1024
        if path.stat().st_size > limit:
            raise ValueError(f"Knowledge receipt exceeds storage limit: {filename}")
        shutil.copyfile(path, target/filename)
        copied.append(filename)
    scores = json.loads((root/"scores.json").read_text())
    paired = json.loads((root/"paired-scores.json").read_text())
    verification = json.loads((root/"verification.json").read_text())
    compact = {"common_lovo_cohort": scores["common_lovo_cohort"], "methods": {
        name: {key: metrics[key] for key in ("lovo_reprojection_by_reference_kind", "lovo_reprojection_px",
            "common_lovo_reprojection_px", "runtime_seconds_all_four_runs")}
        for name, metrics in scores["methods"].items()},
        "paired": {"common_lovo_cohort": paired["common_lovo_cohort"], "methods": {
            name: value["common_lovo_reprojection_px"] for name, value in paired["methods"].items()}}}
    write_json_atomic(target/"metrics.json", compact)
    verification.pop("uncertainty")
    verification["comparison_video"].pop("source_frames")
    write_json_atomic(target/"verification.json", verification)
    with (target/"fold-metrics.csv").open("w") as stream:
        fields = ["segment", "method", "fold", "median_px", "p90_px", "above_20px_rate", "scored", "eligible", "coverage", "valid_3d_frames", "valid_3d_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, metrics in scores["methods"].items():
            for segment in ("evaluation", "tuning", "all"):
                for fold, results in metrics["folds"].items():
                    values = results[segment]
                    error = values["reprojection_px"]
                    writer.writerow(dict(zip(fields, (segment, name, fold, error["median"], error["p90"], error["above_threshold_rate"],
                        values["scored_label_count"], values["scoring_label_count"], values["scored_label_coverage"], values["valid_3d_frames"], values["valid_3d_rate"]), strict=True)))
    uncertainty = json.loads((root/"uncertainty.json").read_text())
    with (target/"uncertainty-coverage.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "fold", "segment", "reference_kind", "count", "ellipse_50", "ellipse_90", "ellipse_95"])
        for name, value in uncertainty["methods"].items():
            for fold, segments in {"lovo_pool": value["lovo"], **value["folds"]}.items():
                for segment, kinds in segments.items():
                    for kind, spec in kinds.items():
                        writer.writerow([name, fold, segment, kind, spec["count"],
                            *(spec["ellipse_coverage"][str(p)] for p in (.5, .9, .95))])
    write_json_atomic(target/"inventory-verification.json", {"output_dir": str(root.resolve()),
        "verified_files": len(declared), "manifest_sha256": dual_sha256(root/"artifacts.json"),
        "copied_small_receipts": copied, "prediction_arrays_video_html": "remain in output directory"})
    print(f"Verified all {len(declared)} files; large arrays/media remain in {root}")


if __name__ == "__main__":
    main()
