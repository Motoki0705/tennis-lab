"""Pinned clip_000 candidate GMM experiment; no independent loader/scorer."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from calibration import fit_calibration
from plots import ellipsoid_figure, uncertainty_figure

from src.tennis_scene.ball_gmm.config import CandidateGMMConfig
from src.tennis_scene.ball_gmm.heatmap import candidate_gaussians
from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.candidate_gmm import CandidateGMM
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    annotation_oracle,
    score_comparison,
)
from src.tennis_scene.evaluation.ball3d.uncertainty import score_uncertainty
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

BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
REFERENCE_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"


class FoldCalibrated:
    """Only scalar frozen settings survive calibration; labels are never retained."""
    requires_court_lines = False

    def __init__(self, settings, estimator, temporal):
        self.settings = {ids: replace(value, estimator=estimator, temporal=temporal) for ids, value in settings.items()}
        self.name = f"candidate_gmm_{estimator}_{temporal}"
        self.config = {"fold_calibrations": {",".join(ids): asdict(value) for ids, value in self.settings.items()},
            "fit_frames": [0, 505], "requires_court_lines": False}

    def __call__(self, inputs):
        return CandidateGMM(self.settings[inputs.camera_ids])(inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "videos", "annotations", "prior_baseline", "output"):
        parser.add_argument("--"+name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--reuse-predictions", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if args.output.exists():
        raise ValueError("Use a new output directory; existing experiments are immutable")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", str(Path(__file__).parent)], text=True):
        raise ValueError("Commit all replay code before executing this experiment")
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    inputs = bundle.inputs
    if inputs.clip_id != "meiji_3cam/video_000/clip_000" or inputs.num_frames != 1010 or inputs.camera_ids != ("cam0", "cam1", "cam2"):
        raise ValueError("The experiment is restricted to the pinned 3-camera 1010-frame clip")
    videos = {c: args.videos/f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    annotations = load_annotations(args.annotations, bundle, expected_sha256=REFERENCE_SHA)
    base = CandidateGMMConfig()
    candidates = tuple(candidate_gaussians(v.detections.evidence, base) for v in inputs.views)
    settings, calibration = {}, {}
    for held_out in (None, *inputs.camera_ids):
        ids = tuple(c for c in inputs.camera_ids if c != held_out)
        rows = [inputs.camera_ids.index(c) for c in ids]
        config, receipt = fit_calibration(tuple(candidates[i] for i in rows), annotations.uv_px[rows],
            annotations.valid_for_scoring[rows], stop=505, base=base)
        settings[ids] = config
        calibration["all_views" if held_out is None else f"without_{held_out}"] = {
            "retained_camera_ids": ids, "config": asdict(config), **receipt}
    args.output.mkdir(parents=True)
    write_json_atomic(args.output/"frozen-calibration.json", calibration)
    print("Frozen calibration from retained cameras and frames [0,505) only", flush=True)
    if args.calibrate_only:
        return
    baseline_verification = verify_baseline(bundle)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    baseline = baseline_from_bundle(bundle)
    methods = [baseline, *(FoldCalibrated(settings, estimator, temporal)
        for estimator, temporal in (("map", "none"), ("mean", "none"), ("map", "rts"), ("mean", "rts")))]
    runs = {}
    for method in methods:
        print(f"Running {method.name}: all views and three LOVO folds", flush=True)
        if args.reuse_predictions is None:
            result = run_method(method, inputs,
                bundle_sha256=bundle.manifest_sha256, code_commit=commit, code_fingerprint=fingerprint)
        else:
            from src.tennis_scene.evaluation.ball3d.scoring import validate_scoring_run
            previous_calibration = json.loads((args.reuse_predictions/"frozen-calibration.json").read_text())
            if previous_calibration != json.loads((args.output/"frozen-calibration.json").read_text()):
                raise ValueError("Cannot reuse predictions with different frozen calibration")
            result = tuple(load_run(args.reuse_predictions/"results"/method.name/key) for key in
                ("all_views", *(f"without_{c}" for c in inputs.camera_ids)))
            for run in result:
                validate_scoring_run(run, bundle)
                if run.config != json.loads(json.dumps(dict(method.config))):
                    raise ValueError("Reused method configuration differs")
        for run in result:
            save_run(args.output/"results"/method.name/run.key, run)
        runs[method.name] = tuple(load_run(args.output/"results"/method.name/r.key) for r in result)
        print(f"Saved {method.name}: {[int(r.result.valid.sum()) for r in runs[method.name]]} valid frames", flush=True)
    unchanged = []
    for run in runs[baseline.name]:
        prior = load_run(args.prior_baseline/"results"/baseline.name/run.key)
        if prior.bundle_sha256 != bundle.manifest_sha256:
            raise ValueError("Prior baseline input digest differs")
        for field in ("positions", "valid", "used_views"):
            np.testing.assert_array_equal(getattr(prior.result, field), getattr(run.result, field))
        unchanged.append(run.key)
    # Primary scoring begins only after parameters and every prediction are saved.
    oracle = annotation_oracle(bundle, annotations, code_commit=commit, code_fingerprint=fingerprint)
    for run in oracle:
        save_run(args.output/"scoring_oracle"/run.key, run)
    runs["annotation_oracle"] = oracle
    scores = score_comparison(runs, bundle, annotations)
    paired = score_comparison({name: runs[name] for name in (baseline.name, "candidate_gmm_map_none")}, bundle, annotations)
    write_json_atomic(args.output/"scores.json", scores)
    write_json_atomic(args.output/"paired-scores.json", paired)
    uncertainty = score_uncertainty(runs, bundle, annotations)
    write_json_atomic(args.output/"uncertainty.json", uncertainty)
    uncertainty_figure(uncertainty, args.output/"uncertainty.png")
    full = {name: values[0] for name, values in runs.items() if name != "annotation_oracle"}
    ellipsoid_figure(full, args.output/"ellipsoids.png")
    plot_metrics_table(scores, args.output/"metrics-table.png")
    plot_comparison(runs, bundle, args.output/"trajectories.png", annotations)
    print("Scored all folds/splits, generated uncertainty and comparison figures", flush=True)
    with BallComparisonRenderer(inputs, full, videos, annotations=annotations) as renderer:
        media = render_video(renderer, args.output/"comparison.mp4")
        available = np.flatnonzero(full["candidate_gmm_map_none"].result.valid)
        example_frame = int(available[np.argmin(np.abs(available-505))])
        if not cv2.imwrite(str(args.output/"uncertainty-example.png"), renderer.render(example_frame)):
            raise RuntimeError("Failed to save uncertainty example")
    extremes = render_extremes({name: runs[name] for name in full}, bundle, videos, args.output/"extremes.png", annotations)
    write_json_atomic(args.output/"extremes.json", extremes)
    verification = {"bundle_sha256": bundle.manifest_sha256, "annotations_sha256": REFERENCE_SHA,
        "replay_code_commit": commit, "replay_code_fingerprint": fingerprint,
        "prediction_code_commits": sorted({r.code_commit for values in runs.values() for r in values}),
        "reused_predictions_from": str(args.reuse_predictions) if args.reuse_predictions else None,
        "uncertainty_example_frame": example_frame,
        "videos": video_receipt, "baseline": baseline_verification,
        "baseline_bitwise_equal_foundation_v2": unchanged, "calibration": calibration,
        "uncertainty": uncertainty, "comparison_video": media,
        "reference_counts": {kind: int((annotations.reference_kind == kind).sum()) for kind in
            ("observed", "estimated", "excluded_extrapolation", "occluded", "unknown")},
        "diagnostics": {name: {run.key: {"valid": int(run.result.valid.sum()),
            "solver_failure_frames": len(run.result.diagnostics["solver_failures"]),
            "bounded_peaks": int(run.result.diagnostics["bounded_curvature"].sum()),
            "rejected_curvature_count": int(run.result.diagnostics["rejected_curvature_count"].sum())}
            for run in values} for name, values in runs.items() if name.startswith("candidate_gmm")},
        "gpu_used": False, "detector_inference_rerun": False,
        "elapsed_seconds": time.perf_counter()-started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "ellipses": "95% 2D Gaussian projection in RGB/top/side; 3D ellipsoids use chi-square df=3",
        "limitations": "Laplace/local Gaussian approximation; posterior-only covariance; usable-support proxy is not physical existence; adoption pending human comparison"}
    page = write_report(args.output, scores, video=args.output/"comparison.mp4", figure=args.output/"trajectories.png",
        crops=args.output/"extremes.png" if extremes["frames"] else None, verification=verification,
        table_figure=args.output/"metrics-table.png", supplementary_figures={
            "LOVO uncertainty coverage (conditional on scored predictions)": args.output/"uncertainty.png",
            "Representative 95% 3D ellipsoids (availability-selected frames)": args.output/"ellipsoids.png",
            "95% ellipses: camera, top and side views": args.output/"uncertainty-example.png"})
    verification["broken_html_references"] = broken_html_references(page)
    write_json_atomic(args.output/"verification.json", verification)
    metrics = {"methods": scores["methods"], "common_lovo_cohort": scores.get("common_lovo_cohort")}
    write_json_atomic(args.output/"metrics.json", metrics)
    files = {str(p.relative_to(args.output)): {"size_bytes": p.stat().st_size, "sha256": dual_sha256(p)}
        for p in sorted(args.output.rglob("*")) if p.is_file()}
    write_json_atomic(args.output/"artifacts.json", {"inputs": {"bundle_sha256": BUNDLE_SHA, "annotations_sha256": REFERENCE_SHA},
        "replay_code_commit": commit, "files": files})
    print(json.dumps({"output": str(args.output), "artifacts_sha256": dual_sha256(args.output/"artifacts.json"),
        "elapsed_seconds": time.perf_counter()-started}), flush=True)


if __name__ == "__main__":
    main()
