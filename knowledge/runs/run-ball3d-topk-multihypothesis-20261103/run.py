"""Pinned clip_000 CPU replay. All loading/scoring/media use the foundation API."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison, save_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    annotation_oracle,
    score_comparison,
)
from src.tennis_scene.evaluation.ball3d.topk import TopKMultiHypothesis
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.visualization.ball3d.rendering import (
    BallComparisonRenderer,
    render_video,
)
from src.tennis_scene.visualization.ball3d.report import (
    broken_html_references,
    plot_comparison,
    plot_metrics_table,
    render_extremes,
    write_report,
)
from src.utils.checksum import dual_sha256
from src.utils.paths import PROJECT_ROOT

BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
REFERENCE_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prior-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if args.output.exists():
        raise ValueError("Use a new output directory")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", str(Path(__file__).relative_to(Path.cwd()))], text=True):
        raise ValueError("Commit replay source before recording a run")
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    inputs = bundle.inputs
    if inputs.clip_id != "meiji_3cam/video_000/clip_000" or inputs.num_frames != 1010 or inputs.camera_ids != ("cam0", "cam1", "cam2"):
        raise ValueError("Only the agreed clip_000 is supported by this run script")
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.bundle, args.videos, args.annotations.parent, args.prior_baseline)):
        raise ValueError("Output overlaps immutable inputs")
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline_receipt = verify_baseline(bundle)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    args.output.mkdir(parents=True)
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"):
        composed = compose(config_name="pipeline", overrides=["ball_reconstruction.method=topk_multihypothesis"])
        runtime = PipelineRuntimeConfig.from_config(composed, bind_inputs=False)
    config = runtime.ball_topk
    # Fixed without inspecting ANY clip labels. No post-evaluation parameter search.
    write_json_atomic(args.output / "frozen-config.json", {
        "config": asdict(config), "code_commit": commit,
        "selection": "predeclared defaults before loading annotations; no label-based tuning",
        "tuning_frames": [0, 505], "primary_frames": [505, 1010],
    })
    method = TopKMultiHypothesis(config, min_frames=runtime.ball_min_frames)
    methods = {m.name: m for m in (baseline_from_bundle(bundle), method)}
    runs = run_comparison(methods, inputs, args.output / "results", bundle_sha256=BUNDLE_SHA,
                          code_commit=commit, code_fingerprint=fingerprint)
    runs = {name: tuple(load_run(args.output / "results" / name / run.key) for run in rows) for name, rows in runs.items()}
    for run in runs["pipeline_baseline"]:
        before = load_run(args.prior_baseline / "results/pipeline_baseline" / run.key)
        if before.bundle_sha256 != BUNDLE_SHA:
            raise ValueError("Prior baseline used a different fixed input")
        for field in ("positions", "valid", "used_views"):
            np.testing.assert_array_equal(getattr(run.result, field), getattr(before.result, field))
    print("CPU replay complete; baseline arrays equal foundation in all four folds", flush=True)
    annotations = load_annotations(args.annotations, bundle, expected_sha256=REFERENCE_SHA)
    # First report the permitted tuning half, with configuration already frozen.
    # score_comparison is the sole scoring implementation for every report.
    paired = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "paired-scores.json", paired)
    oracle = annotation_oracle(bundle, annotations, code_commit=commit, code_fingerprint=fingerprint)
    for run in oracle:
        save_run(args.output / "scoring_oracle" / run.key, run)
    runs["annotation_oracle"] = tuple(load_run(args.output / "scoring_oracle" / run.key) for run in oracle)
    scores = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "scores.json", scores)
    # Format the foundation's summaries; no residuals or new scorer are computed here.
    with (args.output / "fold-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "segment", "fold", "median_px", "p90_px",
            "above20_rate", "scored", "eligible", "coverage", "valid_3d_rate"])
        writer.writeheader()
        for name, values in scores["methods"].items():
            for segment in ("evaluation", "tuning", "all"):
                for fold, segments in values["folds"].items():
                    row = segments[segment]
                    errors = row["reprojection_px"]
                    writer.writerow({"method": name, "segment": segment, "fold": fold,
                        "median_px": errors["median"], "p90_px": errors["p90"],
                        "above20_rate": errors["above_threshold_rate"], "scored": row["scored_label_count"],
                        "eligible": row["scoring_label_count"], "coverage": row["scored_label_coverage"],
                        "valid_3d_rate": row["valid_3d_rate"]})
    print(json.dumps({name: row["lovo_reprojection_px"]["evaluation"] for name, row in scores["methods"].items()}), flush=True)
    plot_metrics_table(scores, args.output / "metrics-table.png")
    plot_comparison(runs, bundle, args.output / "trajectories.png", annotations)
    full = {name: rows[0] for name, rows in runs.items() if name in methods}
    with BallComparisonRenderer(inputs, full, videos, annotations=annotations, show_candidates=True) as renderer:
        media = render_video(renderer, args.output / "comparison.mp4", step=1)
    print("Rendered and fully decoded 1010 comparison frames with candidate overlays", flush=True)
    extremes = render_extremes({name: runs[name] for name in methods}, bundle, videos,
                               args.output / "extremes.png", annotations)
    write_json_atomic(args.output / "extremes.json", extremes)
    verification = {
        "schema": "tennis_lab_ball3d_topk_replay.v1", "code_commit": commit,
        "code_fingerprint": fingerprint, "script_sha256": dual_sha256(Path(__file__)),
        "command": sys.argv, "bundle_sha256": BUNDLE_SHA, "annotations_sha256": REFERENCE_SHA,
        "source_code_commit": bundle.manifest["source_code_commit"],
        "baseline": baseline_receipt, "baseline_four_folds_bitwise_equal_foundation": True,
        "video_inputs": video_receipt, "comparison_video": media,
        "gpu_used": False, "inference_rerun": False,
        "reference_counts": {k: int((annotations.reference_kind == k).sum()) for k in np.unique(annotations.reference_kind)},
        "configuration_selection": "predeclared defaults; no labels used for tuning or prediction",
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "limitations": ["single clip and approximate fixed calibration", "pair-seeded proposals and bounded beam are approximate",
                         "raw scores uncalibrated; no 2D LOESS after selection", "missing frames remain missing; motion memory expires after configured gap"],
        "human_adoption": "pending; do not merge before human integrated comparison decision",
    }
    page = write_report(args.output, scores, video=args.output / "comparison.mp4",
        figure=args.output / "trajectories.png", crops=args.output / "extremes.png" if extremes["frames"] else None,
        verification=verification, table_figure=args.output / "metrics-table.png")
    verification["broken_html_references"] = broken_html_references(page)
    if verification["broken_html_references"]:
        raise ValueError("Broken HTML references")
    write_json_atomic(args.output / "verification.json", verification)
    summary = {"code_commit": commit, "output": str(args.output.resolve()),
               "config": asdict(config), "elapsed_seconds": time.perf_counter() - started,
               "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_json_atomic(args.output / "run-summary.json", summary)
    inventory = {"schema": "tennis_lab_ball3d_artifact_inventory.v1", "code_commit": commit,
        "inputs": {"bundle_sha256": BUNDLE_SHA, "annotations_sha256": REFERENCE_SHA},
        "files": {str(p.relative_to(args.output)): {"sha256": dual_sha256(p), "size_bytes": p.stat().st_size}
                  for p in sorted(args.output.rglob("*")) if p.is_file()}}
    write_json_atomic(args.output / "artifacts.json", inventory)
    print(f"Finished: {page} ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
