"""Restricted clip_000 replay; fixed foundation input, scoring and media APIs."""

from __future__ import annotations

import argparse
import base64
import platform
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch
from diagnostics import diagnostic_summary, export_tables, plot_diagnostics
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.ball_trajectory.config import SegmentPhysicsConfig
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
from src.tennis_scene.evaluation.ball3d.segment_physics import SegmentPhysicsMethod
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "annotations", "videos", "prior-comparison", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("bundle-sha256", "annotations-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    if args.output.exists():
        raise ValueError("Use a new output directory")
    if any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (
            args.bundle,
            args.annotations.parent,
            args.videos,
            args.prior_comparison,
        )
    ):
        raise ValueError("Output must be separate from immutable inputs")
    run_dir = Path(__file__).parent
    if subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            str(run_dir / "run.py"),
            str(run_dir / "diagnostics.py"),
        ],
        text=True,
    ):
        raise ValueError("Commit method and replay code before recording the run")
    bundle = load_bundle(args.bundle, expected_sha256=args.bundle_sha256)
    inputs = bundle.inputs
    if (
        inputs.clip_id != "meiji_3cam/video_000/clip_000"
        or inputs.num_frames != 1010
        or inputs.camera_ids != ("cam0", "cam1", "cam2")
    ):
        raise ValueError("This experiment is restricted to the agreed clip_000")
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline_receipt = verify_baseline(bundle)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest(
        {str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))}
    )
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        composed = compose(config_name="pipeline")
    settings_values = OmegaConf.to_container(
        composed.ball_reconstruction.segment_physics, resolve=True
    )
    if not isinstance(settings_values, dict):
        raise ValueError("Composed physics configuration must be a mapping")
    settings = SegmentPhysicsConfig(**settings_values)
    baseline = baseline_from_bundle(bundle)
    methods = {
        method.name: method
        for method in (
            baseline,
            SegmentPhysicsMethod(settings, baseline),
            SegmentPhysicsMethod(replace(settings, observation="heatmap"), baseline),
        )
    }
    print(
        "Inputs verified. Running blind full-view/LOVO physics; labels remain unloaded.",
        flush=True,
    )
    args.output.mkdir(parents=True)
    runs = run_comparison(
        methods,
        inputs,
        args.output / "results",
        bundle_sha256=bundle.manifest_sha256,
        code_commit=commit,
        code_fingerprint=fingerprint,
    )
    runs = {
        name: tuple(
            load_run(args.output / "results" / name / run.key) for run in values
        )
        for name, values in runs.items()
    }
    unchanged = []
    for run in runs[baseline.name]:
        prior = load_run(args.prior_comparison / "results" / baseline.name / run.key)
        if prior.bundle_sha256 != bundle.manifest_sha256:
            raise ValueError("Prior baseline belongs to another bundle")
        for field in ("positions", "valid", "used_views"):
            np.testing.assert_array_equal(
                getattr(prior.result, field), getattr(run.result, field)
            )
        unchanged.append(run.key)
    prediction_finished_seconds = time.perf_counter() - started
    print(
        "All predictions saved and reloaded. Loading labels only for scoring.",
        flush=True,
    )
    annotations = load_annotations(
        args.annotations, bundle, expected_sha256=args.annotations_sha256
    )
    oracle = annotation_oracle(
        bundle, annotations, code_commit=commit, code_fingerprint=fingerprint
    )
    for run in oracle:
        save_run(args.output / "scoring_oracle" / run.key, run)
    runs["annotation_oracle"] = oracle
    scores = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "scores.json", scores)
    rows = export_tables(scores, args.output)
    summary = diagnostic_summary(runs, inputs.fps)
    write_json_atomic(args.output / "diagnostics.json", summary)
    # The #1558 label-derived diagnostic is compared only after all estimates.
    # It is never passed to a method or used to select/tune a configuration.
    oracle_clock = {
        "cam0": 0.0,
        "cam1": -0.24927409120949945,
        "cam2": -0.774857871707072,
    }
    write_json_atomic(
        args.output / "clock-diagnostic-comparison.json",
        {
            "source": "PR #1670 / issue #1558; annotation-derived tuning-half diagnostic, not physical truth or method input",
            "annotation_clock_frames": oracle_clock,
            "method_all_views": [r for r in summary if r["fold"] == "all_views"],
        },
    )
    plot_metrics_table(scores, args.output / "metrics-table.png")
    plot_comparison(runs, bundle, args.output / "trajectories.png", annotations)
    plot_diagnostics(runs, inputs.fps, args.output / "physics-diagnostics.png")
    full = {
        name: values[0] for name, values in runs.items() if name != "annotation_oracle"
    }
    print(
        "Scores and diagnostics saved. Rendering one source frame at a time.",
        flush=True,
    )
    with BallComparisonRenderer(
        inputs, full, videos, annotations=annotations, width=1200
    ) as renderer:
        media = render_video(renderer, args.output / "comparison.mp4")
    extremes = render_extremes(
        {name: runs[name] for name in methods},
        bundle,
        videos,
        args.output / "extremes.png",
        annotations,
    )
    write_json_atomic(args.output / "extremes.json", extremes)
    verification = {
        "code_commit": commit,
        "code_fingerprint": fingerprint,
        "source_code_commit": bundle.manifest["source_code_commit"],
        "bundle_sha256": bundle.manifest_sha256,
        "annotations_sha256": annotations.reference.sha256,
        "baseline": baseline_receipt,
        "unchanged_baseline_folds": unchanged,
        "videos": video_receipt,
        "comparison_video": media,
        "predictions_before_label_load_seconds": prediction_finished_seconds,
        "gpu_used": False,
        "inference_rerun": False,
        "label_tuning": False,
        "settings_frozen_commit": commit,
        "clock_gauge": "lexicographically first retained camera; heldout offset remains zero in foundation scorer",
        "diagnostics": summary,
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
        crops=args.output / "extremes.png" if extremes["frames"] else None,
        table_figure=args.output / "metrics-table.png",
        verification=verification,
    )
    # Keep the common report intact and embed this experiment's extra diagnostic.
    png = base64.b64encode(
        (args.output / "physics-diagnostics.png").read_bytes()
    ).decode("ascii")
    page.write_text(
        page.read_text().replace(
            "<h2>Provenance and verification</h2>",
            f'<h2>Segments, physical residual and relative clocks</h2><img alt="Physical diagnostics" src="data:image/png;base64,{png}"><h2>Provenance and verification</h2>',
        )
    )
    verification["broken_html_references"] = broken_html_references(page)
    if verification["broken_html_references"]:
        raise ValueError("Broken comparison HTML references")
    verification["elapsed_seconds"] = time.perf_counter() - started
    write_json_atomic(args.output / "verification.json", verification)
    write_json_atomic(
        args.output / "metrics-summary.json",
        {
            "rows": rows,
            "diagnostics": summary,
            "common_cohort": {
                k: v
                for k, v in scores["common_lovo_cohort"].items()
                if k != "camera_frame_pairs"
            },
            "runtime_seconds": verification["elapsed_seconds"],
        },
    )
    inventory = {
        "schema": "tennis_lab_ball3d_artifact_inventory.v1",
        "code_commit": commit,
        "inputs": {
            "bundle_sha256": bundle.manifest_sha256,
            "annotations_sha256": annotations.reference.sha256,
        },
        "files": {
            str(p.relative_to(args.output)): {
                "sha256": dual_sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    write_json_atomic(args.output / "artifacts.json", inventory)
    (args.output / "artifacts.sha256").write_text(
        dual_sha256(args.output / "artifacts.json") + "\n"
    )
    print(f"Saved {page}; {verification['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
