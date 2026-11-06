"""Pinned clip_000 CPU replay. All IO, folds, scores and media use the foundation."""

from __future__ import annotations

import argparse
import base64
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import scipy
import torch
from hydra import compose, initialize_config_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reuse import reuse

from src.tennis_scene.ball_likelihood import DenseLikelihoodConfig
from src.tennis_scene.ball_likelihood.reconstruction import (
    candidate_seeds,
    estimate_frame,
    make_frame_maps,
)
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.contracts import Ball3DResult
from src.tennis_scene.evaluation.ball3d.dense_likelihood import DenseHeatmapMethod
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_method, save_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    annotation_oracle,
    score_comparison,
)
from src.tennis_scene.pipeline.artifacts import document_digest, write_json_atomic
from src.tennis_scene.visualization.ball3d.rendering import (
    BallComparisonRenderer,
    render_video,
    validate_video,
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
TUNING_FRAMES = tuple(range(0, 505, 5))


@dataclass
class TuningMethod:
    """Predict only prespecified tuning frames. Every evaluation point stays invalid."""
    parameters: DenseLikelihoodConfig
    name: str
    requires_court_lines: bool = False

    @property
    def config(self):
        return {**asdict(self.parameters), "tuning_frames": TUNING_FRAMES}

    def __call__(self, inputs):
        xyz = np.zeros((inputs.num_frames, 3), np.float32)
        valid = np.zeros(inputs.num_frames, bool)
        used = np.zeros((len(inputs.views), inputs.num_frames), bool)
        evidence = tuple(v.detections.evidence for v in inputs.views)
        cameras = tuple(v.camera for v in inputs.views)
        for frame in TUNING_FRAMES:
            maps = make_frame_maps(evidence, cameras, frame, self.parameters)
            if sum(m.informative for m in maps) < 2:
                continue
            seeds = candidate_seeds(evidence, cameras, frame)
            point, support, _, success = estimate_frame(maps, seeds, self.parameters)
            if success:
                xyz[frame], valid[frame], used[:, frame] = point, True, support
        return Ball3DResult(inputs.camera_ids, inputs.frame_indices, xyz, valid, used)


def tune(bundle, annotations, output, provenance):
    rows = []
    cache = {}
    possible = int(annotations.valid_for_scoring[:, TUNING_FRAMES].sum())

    def evaluate(parameters):
        key = document_digest(asdict(parameters))
        if key in cache:
            return cache[key]
        name = f"tuning_{len(rows):02d}"
        print(f"Tuning {name}: {asdict(parameters)}", flush=True)
        method = TuningMethod(parameters, name)
        runs = run_method(method, bundle.inputs, **provenance)
        for run in runs:
            save_run(output / "tuning" / name / run.key, run)
        report = score_comparison({name: runs}, bundle, annotations)
        # Evaluation/full scores are never accessed in parameter selection. These
        # predictions have zero evaluation coverage, including in saved arrays.
        metrics = report["methods"][name]["lovo_reprojection_px"]["tuning"]
        if metrics["count"]:
            value = metrics["median"] + .2 * metrics["p90"] + 30 * (1 - metrics["count"] / possible)
        else:
            value = 1e9
        row = {"name": name, "config": asdict(parameters), "metrics": metrics, "sampled_eligible_count": possible,
               "selection_objective": value}
        rows.append(row)
        write_json_atomic(output / "tuning-progress.json", {"rows": rows})
        print(f"Tuning {name}: objective {value:.4f}, {metrics}", flush=True)
        cache[key] = value
        return value

    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        composed = compose(config_name="pipeline")
    parameters = PipelineRuntimeConfig.from_config(composed, bind_inputs=False).dense_likelihood
    best = evaluate(parameters)
    # One prespecified coordinate pass, only over camera-specific T and epsilon.
    for key, alternative in (("view_temperatures", 2.0), ("view_backgrounds", .2)):
        for camera_id in bundle.inputs.camera_ids:
            override = dict(getattr(parameters, key))
            override[camera_id] = alternative
            candidate = replace(parameters, **{key: tuple(override.items())})
            score = evaluate(candidate)
            if score < best:
                parameters, best = candidate, score
    # Compare interpolation with the same fitted per-camera robust scales.
    for interpolation in ("quadratic_logit", "quadratic_log"):
        candidate = replace(parameters, interpolation=interpolation)
        score = evaluate(candidate)
        if score < best:
            parameters, best = candidate, score
    selection = {"schema": "dense_likelihood_tuning.v1", "frames": TUNING_FRAMES,
                 "frame_rule": "every fifth source frame in [0,505), fixed before scoring",
                 "objective": "median + 0.2*p90 + 30*(1-scored/sampled_eligible)",
                 "rows": rows, "selected": asdict(parameters), "evaluation_labels_used": False,
                 "temporal_weight": .5, "temporal_weight_policy": "prespecified, not fit using either scoring half"}
    write_json_atomic(output / "selection.json", selection)
    return parameters


def slice_plots(bundle, runs, annotations, parameters, output):
    dense = runs["dense_selected"][0].result
    baseline = runs["pipeline_baseline"][0].result
    available = dense.valid & baseline.valid
    raw_errors = np.zeros((3, 1010))
    for i, view in enumerate(bundle.inputs.views):
        raw_errors[i] = np.linalg.norm(view.detections.evidence.candidate_uv_px[:, 0] - annotations.uv_px[i], axis=-1)
    normal = np.flatnonzero(available & annotations.valid_for_scoring.all(0) & (np.nanmax(raw_errors, axis=0) < 10))
    occluded = np.flatnonzero(available & (annotations.reference_kind == "occluded").any(0))
    confident_peak = np.stack([v.detections.evidence.candidate_valid[:, 0] &
                                (v.detections.evidence.candidate_scores[:, 0] >= v.detector_config.score_threshold)
                                for v in bundle.inputs.views])
    bad = np.flatnonzero(available & (annotations.valid_for_scoring & confident_peak & (raw_errors > 40)).any(0))
    groups = [("normal", normal), ("reference_occluded", occluded), ("raw_peak_misdetection", bad)]
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
    records = []
    evidence = tuple(v.detections.evidence for v in bundle.inputs.views)
    cameras = tuple(v.camera for v in bundle.inputs.views)
    offset = np.linspace(-.75, .75, 121)
    dx, dz = np.meshgrid(offset, offset)
    for row, (kind, frames) in enumerate(groups):
        if not len(frames):
            raise ValueError(f"No scored representative for likelihood slice: {kind}")
        frame = int(frames[len(frames) // 2])
        centre = dense.positions[frame].astype(np.float64)
        points = np.tile(centre, (dx.size, 1))
        points[:, 0] += dx.ravel()
        points[:, 2] += dz.ravel()
        maps = make_frame_maps(evidence, cameras, frame, parameters)
        robust = np.sum([m.evaluate(points)[0] for m in maps], axis=0)
        naive = np.zeros(len(points))
        for m in maps:
            uv, front = m.camera.project(points)
            xy = uv * m.scale
            inside = front & (xy >= 0).all(-1) & (xy <= np.array(m.shape[::-1]) - 1).all(-1)
            values = np.full(len(points), -50.)
            values[inside] = m.interpolate(xy[inside])[0] / m.temperature - m.normalizer
            naive += values
        for col, (label, values) in enumerate((("uniform-background mixture", robust), ("no robust background", naive))):
            ax = axes[row, col]
            artist = ax.imshow(values.reshape(dx.shape), extent=(-.75, .75, -.75, .75), origin="lower", aspect="equal")
            ax.plot(0, 0, "wx", label="dense mode")
            b = baseline.positions[frame] - centre
            ax.plot(b[0], b[2], "ro", fillstyle="none", label="baseline (at different y)")
            ax.set(xlabel="delta court x (m)", ylabel="delta height z (m)", title=f"{kind}, frame {frame}: {label}")
            figure.colorbar(artist, ax=ax, label="log likelihood relative to uniform")
            ax.legend(fontsize=7)
        records.append({"kind": kind, "frame": frame, "centre_court_m": centre.tolist(),
                        "reference_kinds": annotations.reference_kind[:, frame].tolist(),
                        "raw_peak_error_px": [float(e) if np.isfinite(e) else None for e in raw_errors[:, frame]],
                        "selection": "post-selection diagnostic using references; never parameter tuning"})
    figure.savefig(output / "likelihood-slices.png", dpi=115)
    plt.close(figure)
    write_json_atomic(output / "likelihood-slices.json", {"plane": "XZ at selected dense mode Y", "representatives": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("bundle", "videos", "annotations", "baseline", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--reuse-results", type=Path)
    parser.add_argument("--reuse-inventory", type=Path)
    parser.add_argument("--reuse-inventory-sha256")
    args = parser.parse_args()
    if any((args.reuse_results, args.reuse_inventory, args.reuse_inventory_sha256)) and not all((args.reuse_results, args.reuse_inventory, args.reuse_inventory_sha256)):
        raise ValueError("Reusing predictions requires source, pinned inventory and its digest")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    if args.output.exists():
        raise ValueError("Use a new output directory; previous results are immutable")
    script = Path(__file__).relative_to(Path.cwd())
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", str(script)], text=True):
        raise ValueError("Commit replay code before recording a reproducible run")
    started = time.perf_counter()
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    inputs = bundle.inputs
    if inputs.clip_id != "meiji_3cam/video_000/clip_000" or inputs.camera_ids != ("cam0", "cam1", "cam2") or inputs.num_frames != 1010:
        raise ValueError("This run is restricted to the agreed clip_000 fixed bundle")
    args.output.mkdir(parents=True)
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline_receipt = verify_baseline(bundle)
    provenance = {"bundle_sha256": bundle.manifest_sha256,
                  "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "code_fingerprint": document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})}
    annotations = load_annotations(args.annotations, bundle, expected_sha256=REFERENCE_SHA)
    reused = None
    if args.reuse_results is None:
        parameters = tune(bundle, annotations, args.output, provenance)
    else:
        reused = reuse(args.reuse_results, args.reuse_inventory, args.reuse_inventory_sha256, args.output)
        values = json.loads((args.output / "selection.json").read_text())["selected"]
        for key in ("x_bounds", "y_bounds", "z_bounds"):
            values[key] = tuple(values[key])
        for key in ("view_temperatures", "view_backgrounds"):
            values[key] = tuple(tuple(row) for row in values[key])
        parameters = DenseLikelihoodConfig(**values)
    selection_sha = dual_sha256(args.output / "selection.json")
    print(f"FROZEN selection {selection_sha}: {parameters}", flush=True)
    methods = [baseline_from_bundle(bundle), baseline_from_bundle(bundle, without_loess=True)]
    for interpolation in ("bicubic_logit", "quadratic_logit", "quadratic_log"):
        name = "dense_selected" if interpolation == parameters.interpolation else f"dense_{interpolation}"
        methods.append(DenseHeatmapMethod(replace(parameters, interpolation=interpolation), name))
    methods.extend([DenseHeatmapMethod(replace(parameters, initialization="coarse"), "dense_coarse"),
                    DenseHeatmapMethod(replace(parameters, temporal_weight=.5), "dense_temporal")])
    runs = {}
    for method in methods:
        if reused is None:
            print(f"Full replay {method.name}", flush=True)
            values = run_method(method, inputs, **provenance)
            for run in values:
                save_run(args.output / "results" / method.name / run.key, run)
            keys = [r.key for r in values]
        else:
            keys = ["all_views", *(f"without_{c}" for c in inputs.camera_ids)]
        runs[method.name] = tuple(load_run(args.output / "results" / method.name / key) for key in keys)
        for run in runs[method.name]:
            if run.bundle_sha256 != bundle.manifest_sha256 or document_digest(run.config) != document_digest(method.config):
                raise ValueError("Reused prediction input/config mismatch")
        values = runs[method.name]
        print(f"Completed {method.name}: {sum(r.elapsed_seconds for r in values):.1f}s, valid frames {[int(r.result.valid.sum()) for r in values]}", flush=True)
    unchanged = []
    for name in ("pipeline_baseline", "without_loess"):
        for run in runs[name]:
            before = load_run(args.baseline / "results" / name / run.key)
            for field in ("positions", "valid", "used_views"):
                np.testing.assert_array_equal(getattr(run.result, field), getattr(before.result, field))
            unchanged.append(f"{name}/{run.key}")
    # Every selected configuration is now fixed and all predictions persisted.
    assert dual_sha256(args.output / "selection.json") == selection_sha
    oracle = annotation_oracle(bundle, annotations, code_commit=provenance["code_commit"], code_fingerprint=provenance["code_fingerprint"])
    for run in oracle:
        save_run(args.output / "scoring_oracle" / run.key, run)
    runs["annotation_oracle"] = oracle
    scores = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "scores.json", scores)
    paired = score_comparison({name: runs[name] for name in ("pipeline_baseline", "dense_selected")}, bundle, annotations)
    write_json_atomic(args.output / "paired-scores.json", paired)
    plot_metrics_table(scores, args.output / "metrics-table.png")
    plot_comparison(runs, bundle, args.output / "trajectories.png", annotations)
    extremes = render_extremes({name: runs[name] for name in ("pipeline_baseline", "dense_selected", "dense_coarse", "dense_temporal")},
                               bundle, videos, args.output / "extremes.png", annotations)
    write_json_atomic(args.output / "extremes.json", extremes)
    slice_plots(bundle, runs, annotations, parameters, args.output)
    full = {name: runs[name][0] for name in ("pipeline_baseline", "without_loess", "dense_selected", "dense_coarse", "dense_temporal")}
    with BallComparisonRenderer(inputs, full, videos, annotations=annotations, width=1200) as renderer:
        media = render_video(renderer, args.output / "comparison.mp4", step=1)
    source_decode = {camera: validate_video(path, frames=1010, fps=inputs.fps, size=inputs.source_size_wh) for camera, path in videos.items()}
    verification = {"bundle_sha256": BUNDLE_SHA, "annotations_sha256": REFERENCE_SHA, "baseline": baseline_receipt,
                    "source_code_commit": bundle.manifest["source_code_commit"], "replay_provenance": provenance,
                    "reused_predictions": reused,
                    "prediction_code_commits": sorted({r.code_commit for name, values in runs.items() if name != "annotation_oracle" for r in values}),
                    "baseline_unchanged_from_foundation_v2": unchanged, "selection_sha256": selection_sha,
                    "script_sha256": dual_sha256(Path(__file__)), "source_decode": source_decode, "comparison_video": media,
                    "gpu_used": False, "inference_rerun": False, "elapsed_seconds": time.perf_counter() - started,
                    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
                    "limitations": "single clip, approximate fixed calibration, no independent 3D truth; method adoption pending human comparison"}
    page = write_report(args.output, scores, video=args.output / "comparison.mp4", figure=args.output / "trajectories.png",
                        crops=args.output / "extremes.png", verification=verification, table_figure=args.output / "metrics-table.png")
    section = '<section><h2>Dense likelihood slices</h2><p>Post-selection XZ diagnostics. Occlusion is the reference status; distractor selection uses raw-peak residuals. These frames never selected parameters.</p><img style="max-width:100%" src="data:image/png;base64,' + base64.b64encode((args.output / "likelihood-slices.png").read_bytes()).decode() + '"></section>'
    page.write_text(page.read_text().replace("</body>", section + "</body>"))
    verification["broken_html_references"] = broken_html_references(page)
    write_json_atomic(args.output / "verification.json", verification)
    manifest = {"schema": "tennis_lab_ball3d_artifact_inventory.v1", "code_commit": provenance["code_commit"],
                "inputs": {"bundle_sha256": BUNDLE_SHA, "annotations_sha256": REFERENCE_SHA, "videos": video_receipt},
                "files": {str(p.relative_to(args.output)): {"sha256": dual_sha256(p), "size_bytes": p.stat().st_size}
                          for p in sorted(args.output.rglob("*")) if p.is_file()}}
    write_json_atomic(args.output / "artifacts.json", manifest)
    print(f"DONE {page}, artifact manifest {dual_sha256(args.output / 'artifacts.json')}", flush=True)


if __name__ == "__main__":
    main()
