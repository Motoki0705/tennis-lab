"""Reproduce #1558 diagnostics on the pinned clip_000 bundle, CPU only."""

from __future__ import annotations

import argparse
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
from diagnostics import (
    candidate_oracle,
    candidate_report,
    decode_native,
    decode_report,
    fit_offsets,
    segments,
    smoothing_report,
    timing_oracle,
)
from plots import diagnostic_plots, extend_report

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison, save_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    annotation_oracle,
    paired_error_summary,
    score_comparison,
)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "annotations", "videos", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--annotations-sha256", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    if args.output.exists():
        raise ValueError("Use a new output directory; previous artifacts are immutable")
    if any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (args.bundle, args.annotations.parent, args.videos)
    ):
        raise ValueError("Output must be separate from fixed inputs")
    run_dir = Path(__file__).resolve().parent
    relative = run_dir.relative_to(Path.cwd())
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", str(relative)], text=True
    ):
        raise ValueError("Commit all execution code before the recorded run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest(
        {
            str(p): dual_sha256(p)
            for root in (Path("src"), run_dir)
            for p in sorted(root.rglob("*.py"))
        }
    )
    bundle = load_bundle(args.bundle, expected_sha256=args.bundle_sha256)
    inputs = bundle.inputs
    if (
        inputs.clip_id != "meiji_3cam/video_000/clip_000"
        or inputs.num_frames != 1010
        or inputs.camera_ids != ("cam0", "cam1", "cam2")
    ):
        raise ValueError("Run is restricted to the agreed 3-camera 1010-frame clip_000")
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline = verify_baseline(bundle)
    annotations = load_annotations(
        args.annotations, bundle, expected_sha256=args.annotations_sha256
    )
    args.output.mkdir(parents=True)
    identity = dict(
        bundle_sha256=bundle.manifest_sha256,
        code_commit=commit,
        code_fingerprint=fingerprint,
    )
    methods = {
        method.name: method
        for method in (
            baseline_from_bundle(bundle),
            baseline_from_bundle(bundle, without_loess=True),
        )
    }
    runs = run_comparison(methods, inputs, args.output / "results", **identity)
    print("Baseline and without-LOESS replayed (all-view and LOVO)", flush=True)
    oracle = annotation_oracle(inputs, annotations, **identity)
    refined, integer, confidence = decode_native(inputs)
    offsets, timing = fit_offsets(inputs, annotations)
    diagnostic_runs = {
        "annotation_oracle": oracle,
        "annotation_time_oracle": timing_oracle(inputs, annotations, offsets, oracle),
        "candidate1_oracle": candidate_oracle(inputs, annotations, oracle, k=1),
        "candidate8_oracle": candidate_oracle(inputs, annotations, oracle, k=8),
    }
    for name, values in diagnostic_runs.items():
        for run in values:
            save_run(args.output / "scoring_diagnostics_only" / name / run.key, run)
    runs.update(diagnostic_runs)
    runs = {
        name: tuple(
            load_run(
                args.output
                / ("results" if name in methods else "scoring_diagnostics_only")
                / name
                / run.key
            )
            for run in values
        )
        for name, values in runs.items()
    }
    print(
        "Scoring-only geometry/candidate/clock diagnostics saved and reloaded",
        flush=True,
    )
    scores = score_comparison(runs, inputs, annotations)
    decode = decode_report(inputs, annotations, refined, integer, confidence)
    candidates = candidate_report(inputs, annotations, refined)
    smoothing = smoothing_report(inputs, annotations)
    comparisons = {
        "remove_loess": ("pipeline_baseline", "without_loess"),
        "remove_detector_errors": ("pipeline_baseline", "annotation_oracle"),
        "fit_annotation_clocks": ("annotation_oracle", "annotation_time_oracle"),
        "nearest_top8_vs_top1": ("candidate1_oracle", "candidate8_oracle"),
    }
    budget = {}
    for comparison, (left, right) in comparisons.items():
        budget[comparison] = {
            "reference": left,
            "candidate": right,
            "segments": {
                segment: {
                    "pooled" if c is None else c: paired_error_summary(
                        runs[left],
                        runs[right],
                        inputs,
                        annotations,
                        start=start,
                        stop=stop,
                        held_out_camera=c,
                    )
                    for c in (None, *inputs.camera_ids)
                }
                for segment, (start, stop) in segments(inputs.num_frames).items()
            },
        }
    for name, value in (
        ("scores", scores),
        ("decode", decode),
        ("candidates", candidates),
        ("smoothing", smoothing),
        ("timing", timing),
        ("paired-budget", budget),
    ):
        write_json_atomic(args.output / f"{name}.json", value)
    figures = diagnostic_plots(args.output, inputs, decode, candidates, smoothing)
    plot_metrics_table(scores, args.output / "metrics-table.png")
    plot_comparison(runs, inputs, args.output / "trajectories.png", annotations)
    displayed = {
        name: values[0]
        for name, values in runs.items()
        if not name.startswith("candidate")
    }
    with BallComparisonRenderer(
        inputs, displayed, videos, annotations=annotations
    ) as renderer:
        for frame in (0, 252, 505, 757, 1009):
            if not cv2.imwrite(
                str(args.output / f"preview-{frame:04d}.png"), renderer.render(frame)
            ):
                raise RuntimeError("Failed to save representative preview")
        media = render_video(renderer, args.output / "comparison.mp4")
    extremes = render_extremes(
        {name: runs[name] for name in displayed},
        inputs,
        videos,
        args.output / "extremes.png",
        annotations,
    )
    write_json_atomic(args.output / "extremes.json", extremes)
    verification = {
        "schema": "tennis_lab_ball3d_error_decomposition.v1",
        "execution_commit": commit,
        "code_fingerprint": fingerprint,
        "script_sha256": {p.name: dual_sha256(p) for p in sorted(run_dir.glob("*.py"))},
        "source_store_commit": bundle.manifest["source_code_commit"],
        "bundle_sha256": bundle.manifest_sha256,
        "annotations_sha256": annotations.reference.sha256,
        "videos": video_receipt,
        "baseline": baseline,
        "comparison_video": media,
        "gpu_used": False,
        "inference_rerun": False,
        "production_method_added": False,
        "fit_frames": [0, 505],
        "primary_evaluation_frames": [505, 1010],
        "error_sign": "prediction minus reference",
        "position_regression": "OLS, source-time 25-frame cluster bootstrap, 499 repeats, seed1558",
        "all_peaks": "pre-threshold/pre-gate production log-parabola; integer argmax separately",
        "near_correct_peaks": "conditional error<=20px analysis; selection restricts inference",
        "diagnostic_oracles": "scoring only; never supplied to method runner/MethodInput; not deployable or additive causal error components",
        "calibration_limit": "no independent 3D/camera truth; residual after timing mixes calibration, label errors, interpolation and unmodelled motion",
        "clock_fit": timing,
        "elapsed_seconds_before_report": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
    }
    page = write_report(
        args.output,
        scores,
        video=args.output / "comparison.mp4",
        figure=args.output / "trajectories.png",
        crops=args.output / "extremes.png",
        table_figure=args.output / "metrics-table.png",
        verification=verification,
    )
    extend_report(page, figures, budget, timing)
    verification["broken_html_references"] = broken_html_references(page)
    if verification["broken_html_references"]:
        raise ValueError("Diagnostic HTML has broken references")
    verification["elapsed_seconds"] = time.perf_counter() - started
    write_json_atomic(args.output / "verification.json", verification)
    manifest = {
        "schema": "tennis_lab_ball3d_artifact_inventory.v1",
        "code_commit": commit,
        "files": {
            str(p.relative_to(args.output)): {
                "sha256": dual_sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    write_json_atomic(args.output / "artifacts.json", manifest)
    (args.output / "artifacts.sha256").write_text(
        dual_sha256(args.output / "artifacts.json") + "\n"
    )
    print(f"Complete: {page} ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
