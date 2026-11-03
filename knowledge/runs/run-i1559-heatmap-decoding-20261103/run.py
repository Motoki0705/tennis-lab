"""Pinned clip_000 CPU comparison; selection reads tuning metrics only."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

from src.tasks.ball_detection.inference.heatmap_decoding import (
    HeatmapDecodingConfig,
    decode_native_sequence,
)
from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.heatmap_decoding import HeatmapDecodingMethod
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison
from src.tennis_scene.evaluation.ball3d.scoring import score_comparison
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

BUNDLE_SHA256 = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
ANNOTATIONS_SHA256 = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"


def candidate_configs():
    return {
        "legacy_redecoded": HeatmapDecodingConfig(),
        "axis_logit": HeatmapDecodingConfig(method="axis_logit"),
        **{f"{method}_{patch}": HeatmapDecodingConfig(method=method, patch_size=patch)
           for method in ("quadratic_log", "quadratic_logit", "centroid") for patch in (3, 5)},
        **{f"softargmax_t{str(t).replace('.', 'p')}": HeatmapDecodingConfig(method="softargmax", temperature=t)
           for t in (.5, 2.)},  # T=1 is exactly the background-subtracted centroid.
        "gaussian_5": HeatmapDecodingConfig(method="gaussian"),
    }


def fit_front_affine(inputs, annotations, decoding):
    """Fit global XY affine constants to front-half inlier correspondences only."""
    predicted, target, counts = [], [], {}
    for i, view in enumerate(inputs.views):
        evidence = view.detections.evidence
        uv, score, _ = decode_native_sequence(evidence.heatmaps[:505], decoding, subpixel_refine=view.detector_config.subpixel_refine)
        scale = np.array(inputs.source_size_wh) - 1
        truth = annotations.uv_px[i, :505]
        valid = annotations.valid_for_scoring[i, :505] & (score >= view.detector_config.score_threshold)
        valid &= np.linalg.norm(uv * scale - truth, axis=1) <= 20
        predicted.append(uv[valid])
        target.append(truth[valid] / scale)
        counts[view.camera.camera_id] = int(valid.sum())
    x, y = np.concatenate(predicted), np.concatenate(target)
    if len(x) < 50:
        raise ValueError("Insufficient front-half correspondences for affine calibration")
    scales, offsets = [], []
    for axis, pixels in enumerate(np.array(inputs.source_size_wh) - 1):
        fit = least_squares(lambda p, axis=axis, pixels=pixels: (x[:, axis] * p[0] + p[1] - y[:, axis]) * pixels,
            [1., 0.], bounds=([.97, -.02], [1.03, .02]), loss="soft_l1", f_scale=5.)
        if not fit.success:
            raise ValueError("Front-half affine optimization failed")
        scales.append(float(fit.x[0]))
        offsets.append(float(fit.x[1]))
    return replace(decoding, coordinate_correction="affine", scale_xy=tuple(scales), offset_xy=tuple(offsets)), {
        "fit_frames": [0, 505], "input_inlier_max_px": 20, "loss": "soft_l1", "f_scale_px": 5,
        "global_across_cameras": True, "counts": counts, "config": {"scale_xy": scales, "offset_xy": offsets},
    }


def tuning_choice(scores, names):
    baseline = scores["methods"]["pipeline_baseline"]["lovo_reprojection_px"]["tuning"]
    eligible = [n for n in names if scores["methods"][n]["lovo_reprojection_px"]["tuning"]["scored_label_coverage"] >= baseline["scored_label_coverage"] - .03]
    if not eligible:
        raise ValueError("No candidate satisfies the predeclared tuning coverage guard")
    def key(name):
        row = scores["methods"][name]["lovo_reprojection_px"]["tuning"]
        return row["median"], row["p90"], row["above_threshold_rate"], name
    return min(eligible, key=key)


def compact_scores(scores):
    return {name: {
        "lovo": row["lovo_reprojection_px"],
        "folds": {fold: {segment: {k: r[k] for k in (
            "reprojection_px", "scored_label_count", "scoring_label_count", "scored_label_coverage", "valid_3d_rate")}
            for segment, r in segments.items()} for fold, segments in row["folds"].items()},
        "runtime_seconds": row["runtime_seconds_all_four_runs"],
    } for name, row in scores["methods"].items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("bundle", "annotations", "videos", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA256)
    inputs = bundle.inputs
    if inputs.clip_id != "meiji_3cam/video_000/clip_000" or inputs.num_frames != 1010 or inputs.camera_ids != ("cam0", "cam1", "cam2"):
        raise ValueError("Only the agreed pinned clip_000 is supported")
    if args.output.exists():
        raise ValueError("Use a new output directory; prior results are immutable")
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in (args.bundle, args.videos, args.annotations.parent)):
        raise ValueError("Output must be separate from the immutable inputs")
    script = Path(__file__).relative_to(Path.cwd())
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", str(script)], text=True):
        raise ValueError("Commit source and replay entrypoint before executing")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline_receipt = verify_baseline(bundle)
    annotations = load_annotations(args.annotations, bundle, expected_sha256=ANNOTATIONS_SHA256)
    args.output.mkdir(parents=True)
    kwargs = dict(bundle_sha256=bundle.manifest_sha256, code_commit=commit, code_fingerprint=fingerprint)
    baseline = baseline_from_bundle(bundle)
    configs = candidate_configs()
    methods = {m.name: m for m in (baseline, baseline_from_bundle(bundle, without_loess=True))}
    for name, config in configs.items():
        for raw in (False, True):
            label = name + ("_raw" if raw else "")
            methods[label] = HeatmapDecodingMethod(label, config, replace(baseline, without_loess=raw))
    runs = {}
    for name, method in methods.items():
        runs.update(run_comparison({name: method}, inputs, args.output / "results", **kwargs))
        print(f"Replayed {name} ({time.perf_counter() - started:.1f}s)", flush=True)
    # The shared scorer computes its fixed segments. Only its tuning entries are
    # read for selection; evaluation entries are never inspected before freezing.
    scores = score_comparison(runs, inputs, annotations)
    winner = tuning_choice(scores, list(configs))
    corrections = {
        "legacy_stride": replace(configs["legacy_redecoded"], coordinate_correction="stride"),
        "selected_stride": replace(configs[winner], coordinate_correction="stride"),
    }
    fit_receipts = {}
    for label, original in (("legacy_affine", configs["legacy_redecoded"]), ("selected_affine", configs[winner])):
        corrections[label], fit_receipts[label] = fit_front_affine(inputs, annotations, original)
    for name, config in corrections.items():
        for raw in (False, True):
            label = name + ("_raw" if raw else "")
            method = HeatmapDecodingMethod(label, config, replace(baseline, without_loess=raw))
            runs.update(run_comparison({label: method}, inputs, args.output / "results", **kwargs))
            print(f"Replayed {label} ({time.perf_counter() - started:.1f}s)", flush=True)
    scores = score_comparison(runs, inputs, annotations)
    selected = tuning_choice(scores, [*configs, *corrections])
    selection = {
        "schema": "heatmap_decoding_selection.v1", "code_commit": commit,
        "selection_frames": [0, 505], "primary_evaluation_frames": [505, 1010],
        "rule": "LOESS-on only; tuning LOVO coverage >= baseline minus 3 percentage points; lexicographic median/p90/>20px/name",
        "base_winner": winner, "selected": selected,
        "configs": {n: asdict(c) for n, c in {**configs, **corrections}.items()},
        "affine_fit": fit_receipts,
        "tuning_scores": {n: m["lovo_reprojection_px"]["tuning"] for n, m in scores["methods"].items()},
        "evaluation_used_for_selection": False,
    }
    write_json_atomic(args.output / "selection.json", selection)
    selection_digest = dual_sha256(args.output / "selection.json")
    print(f"Frozen front-half choice: {selected}; selection SHA256 {selection_digest}", flush=True)
    # Re-open every serialized result through the foundation checksum boundary.
    runs = {name: tuple(load_run(args.output / "results" / name / r.key) for r in values) for name, values in runs.items()}
    scores = score_comparison(runs, inputs, annotations)
    write_json_atomic(args.output / "scores.json", scores)
    write_json_atomic(args.output / "metrics-summary.json", compact_scores(scores))
    plot_metrics_table(scores, args.output / "metrics-table.png")
    # Compact video: baseline, the frozen winner, its raw ablation, the winning
    # uncorrected decoder, and both uncorrected coordinate hypotheses.
    display_names = list(dict.fromkeys(["pipeline_baseline", selected, selected + "_raw", winner, "legacy_stride", "legacy_affine"]))
    display_runs = {name: runs[name] for name in display_names}
    with BallComparisonRenderer(inputs, {n: rs[0] for n, rs in display_runs.items()}, videos, annotations=annotations, width=1200) as renderer:
        media = render_video(renderer, args.output / "comparison.mp4", step=1)
    plot_comparison(display_runs, inputs, args.output / "trajectories.png", annotations)
    extremes = render_extremes(display_runs, inputs, videos, args.output / "extremes.png", annotations)
    write_json_atomic(args.output / "extremes.json", extremes)
    verification = {
        "bundle_sha256": bundle.manifest_sha256, "annotations_sha256": annotations.reference.sha256,
        "code_commit": commit, "code_fingerprint": fingerprint, "script_sha256": dual_sha256(Path(__file__)),
        "selection_sha256": selection_digest, "selection": selection,
        "baseline": baseline_receipt, "videos": video_receipt, "comparison_video": media,
        "video_methods": display_names, "all_methods_in_scores_and_main_table": list(runs),
        "gpu_used": False, "inference_rerun": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["one clip; approximate fixed cameras; varying coverage", "CPU reconstruction and repeated re-decoding are not an optimized runtime comparison", "video shows selected variants; all variants have saved all-view/LOVO results", "stride correction is a half-pixel-centre hypothesis; clipped sigmoid logits cannot be recovered"],
        "human_adoption": "pending human comparison; do not merge",
    }
    page = write_report(args.output, scores, video=args.output / "comparison.mp4", figure=args.output / "trajectories.png",
        crops=args.output / "extremes.png", verification=verification, table_figure=args.output / "metrics-table.png")
    verification["broken_html_references"] = broken_html_references(page)
    write_json_atomic(args.output / "verification.json", verification)
    manifest = {"schema": "tennis_lab_ball3d_artifact_inventory.v1", "code_commit": commit,
        "files": {str(p.relative_to(args.output)): {"sha256": dual_sha256(p), "size_bytes": p.stat().st_size}
                  for p in sorted(args.output.rglob("*")) if p.is_file()}}
    write_json_atomic(args.output / "artifacts.json", manifest)
    (args.output / "artifacts.sha256").write_text(dual_sha256(args.output / "artifacts.json") + "\n")
    print(f"Comparison: {page}; {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
