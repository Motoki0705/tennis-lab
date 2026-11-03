"""CPU-only clip_000 comparison using the shared frozen-input foundation."""

from __future__ import annotations

import argparse
import base64
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch

from src.tennis_scene.ball_calibration.bundle_adjustment import court_world_points
from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle, verify_videos
from src.tennis_scene.evaluation.ball3d.court_ball_ba import CourtBallBAMethod
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison, save_run
from src.tennis_scene.evaluation.ball3d.scoring import (
    annotation_oracle,
    distribution,
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
from src.utils.schema.court import COURT_SKELETON
from src.utils.video.reader import RandomAccessVideoReader

BUNDLE_SHA = "7aede6d7410f6a3f8e153d231d08b74c2bf7067819cdb82b3e7515b379752b8a"
REFERENCE_SHA = "dc7cf0809cee56d10c58ea3cfd661e958756590666de6843cd6c0e4ce6ce9c09"


def court_report(runs, inputs, videos, output):
    """Raw-court-only diagnostics; ball scoring stays entirely in the foundation."""
    report = {}
    for run in runs:
        rows = {}
        for view, fitted in zip((v for v in inputs.views if v.camera.camera_id != run.held_out_camera), run.result.cameras, strict=True):
            xyz = court_world_points(view.half_turn)
            before, _ = view.camera.project(xyz)
            after, _ = fitted.project(xyz)
            source = view.court
            masks = {
                "fit": source.valid & source.selected_for_homography,
                "holdout": source.valid & ~source.selected_for_homography,
                "holdout_score_ge_0_5": source.valid & ~source.selected_for_homography & (source.scores >= .5),
            }
            rows[fitted.camera_id] = {key: {
                "baseline_px": distribution(np.linalg.norm(before[None] - source.keypoints_px, axis=-1)[mask], threshold=20.),
                "refined_px": distribution(np.linalg.norm(after[None] - source.keypoints_px, axis=-1)[mask], threshold=20.),
                "keypoint_indices": np.argwhere(mask).tolist(),
            } for key, mask in masks.items()}
        report[run.key] = rows
    full = next(run for run in runs if run.held_out_camera is None)
    canvas = np.full((3 * 390, 1280, 3), 22, np.uint8)
    for v, (view, fitted) in enumerate(zip(inputs.views, full.result.cameras, strict=True)):
        reader = RandomAccessVideoReader(videos[view.camera.camera_id])
        try:
            frame = reader.read(0)
        finally:
            reader.close()
        for column, (label, camera) in enumerate((("baseline", view.camera), ("court_ball_ba", fitted))):
            image = frame.copy()
            xyz = court_world_points(view.half_turn)
            uv, front = camera.project(xyz)
            for a, b in COURT_SKELETON:
                if max(a, b) < 14 and front[a] and front[b]:
                    cv2.line(image, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), (255, 180, 0), 3, cv2.LINE_AA)
            for kp in range(14):
                if view.court.valid[0, kp]:
                    color = (0, 255, 80) if view.court.selected_for_homography[0, kp] else (255, 255, 255)
                    pixel = tuple(np.rint(view.court.keypoints_px[0, kp]).astype(int))
                    cv2.drawMarker(image, pixel, color, cv2.MARKER_CROSS, 17, 2)
            y, x = v * 390 + 30, column * 640
            canvas[y:y+360, x:x+640] = cv2.resize(image, (640, 360))
            cv2.putText(canvas, f"{view.camera.camera_id} {label}: projected lines; raw KP +", (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output / "court-overlay.png"), canvas):
        raise RuntimeError("Court overlay could not be written")
    write_json_atomic(output / "court-holdout.json", report)
    return report


def metric_markdown(scores):
    lines = ["# clip_000: median px / p90 px / >20px rate; scored/eligible (coverage)", ""]
    for segment in ("evaluation", "tuning", "all"):
        lines += [f"## {segment}", "", "| Method | LOVO pool | without cam0 | without cam1 | without cam2 |", "|---|---:|---:|---:|---:|"]
        def cell(row, fold=None):
            eligible = row["label_count"] if "label_count" in row else fold["scoring_label_count"]
            coverage = row["scored_label_coverage"] if "scored_label_coverage" in row else fold["scored_label_coverage"]
            return f"{row['median']:.3f}/{row['p90']:.3f}/{100*row['above_threshold_rate']:.2f}%; {row['count']}/{eligible} ({100*coverage:.2f}%)"
        for name, metrics in scores["methods"].items():
            columns = [cell(metrics["lovo_reprojection_px"][segment])]
            columns += [cell(metrics["folds"][f"without_cam{camera}"][segment]["reprojection_px"], metrics["folds"][f"without_cam{camera}"][segment]) for camera in range(3)]
            lines.append(f"| {name} | " + " | ".join(columns) + " |")
        lines += ["", "| Method | All-view reprojection | Valid 3D frame rate | Common LOVO |", "|---|---:|---:|---:|"]
        for name, metrics in scores["methods"].items():
            full = metrics["folds"]["all_views"][segment]
            lines.append(f"| {name} | {cell(full['reprojection_px'], full)} | {full['valid_3d_rate']*100:.2f}% | {cell(metrics['common_lovo_reprojection_px'][segment])} |")
        lines += [""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    started = time.perf_counter()
    if args.output.exists():
        raise ValueError("Use a new versioned output directory")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "src", str(Path(__file__).relative_to(Path.cwd()))], text=True):
        raise ValueError("Commit implementation and replay script before recording evidence")
    bundle = load_bundle(args.bundle, expected_sha256=BUNDLE_SHA)
    inputs = bundle.inputs
    if inputs.clip_id != "meiji_3cam/video_000/clip_000" or inputs.camera_ids != ("cam0", "cam1", "cam2") or inputs.num_frames != 1010:
        raise ValueError("Restricted to the human-agreed clip_000")
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    baseline_receipt = verify_baseline(bundle)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    fingerprint = document_digest({str(p): dual_sha256(p) for p in sorted(Path("src").rglob("*.py"))})
    baseline = baseline_from_bundle(bundle)
    method = CourtBallBAMethod(reprojection_px=baseline.reprojection_px, min_frames=baseline.min_frames, ball_threshold=baseline.ball_threshold)
    methods = {m.name: m for m in (baseline, method)}
    args.output.mkdir(parents=True)
    write_json_atomic(args.output / "fixed-config.json", {name: dict(m.config) for name, m in methods.items()})
    print("Predicting all-view and LOVO before reading ground truth", flush=True)
    runs = run_comparison(methods, inputs, args.output / "results", bundle_sha256=bundle.manifest_sha256,
                          code_commit=commit, code_fingerprint=fingerprint)
    runs = {name: tuple(load_run(args.output / "results" / name / r.key) for r in values) for name, values in runs.items()}
    unchanged = []
    for run in runs[baseline.name]:
        old = load_run(args.prior / "results" / baseline.name / run.key)
        if old.bundle_sha256 != bundle.manifest_sha256:
            raise ValueError("Prior results are from another bundle")
        for key in ("positions", "valid", "used_views"):
            np.testing.assert_array_equal(getattr(run.result, key), getattr(old.result, key))
        unchanged.append(run.key)
    court = court_report(runs[method.name], inputs, videos, args.output)
    print("Predictions saved/reloaded; baseline arrays match foundation; reading scoring reference", flush=True)
    annotations = load_annotations(args.annotations, bundle, expected_sha256=REFERENCE_SHA)
    paired = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "paired-scores.json", paired)
    oracle = annotation_oracle(bundle, annotations, code_commit=commit, code_fingerprint=fingerprint)
    for run in oracle:
        save_run(args.output / "scoring_oracle" / run.key, run)
    runs["annotation_oracle"] = tuple(load_run(args.output / "scoring_oracle" / r.key) for r in oracle)
    scores = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "scores.json", scores)
    (args.output / "metrics.md").write_text(metric_markdown(scores))
    plot_metrics_table(scores, args.output / "metrics-table.png")
    plot_comparison(runs, bundle, args.output / "trajectories.png", annotations)
    full = {name: values[0] for name, values in runs.items() if name in methods}
    with BallComparisonRenderer(inputs, full, videos, annotations=annotations, width=1200) as renderer:
        media = render_video(renderer, args.output / "comparison.mp4")
        for frame in (0, 252, 505, 757, 1009):
            if not cv2.imwrite(str(args.output / f"preview-{frame:04d}.png"), renderer.render(frame)):
                raise RuntimeError("Preview could not be written")
    extremes = render_extremes({name: runs[name] for name in methods}, bundle, videos, args.output / "extremes.png", annotations)
    write_json_atomic(args.output / "extremes.json", extremes)
    verification = {
        "source_code_commit": bundle.manifest["source_code_commit"], "replay_code_commit": commit,
        "bundle_sha256": bundle.manifest_sha256, "annotations_sha256": annotations.reference.sha256,
        "replay_script_sha256": dual_sha256(Path(__file__)), "code_fingerprint": fingerprint,
        "baseline": baseline_receipt, "baseline_bitwise_equal_to_foundation_v2": unchanged,
        "videos": video_receipt, "comparison_video": media,
        "court_holdout": court, "raw_line_observations": "unavailable; no reinference; KP-only BA",
        "config_selection": "fixed before scoring; no tuning on either label split",
        "gpu_used": False, "detector_inference_rerun": False,
        "seconds": time.perf_counter() - started,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
    }
    page = write_report(args.output, scores, video=args.output / "comparison.mp4",
        figure=args.output / "trajectories.png", crops=args.output / "extremes.png",
        table_figure=args.output / "metrics-table.png", verification=verification)
    overlay = base64.b64encode((args.output / "court-overlay.png").read_bytes()).decode("ascii")
    extra = '<figure><img style="max-width:100%" src="data:image/png;base64,' + overlay + '"><figcaption>Projected court lines; raw measured KP crosses (green fit, white holdout). Raw LINE observations are unavailable.</figcaption></figure>'
    page.write_text(page.read_text().replace("</body>", extra + "</body>"))
    verification["broken_html_references"] = broken_html_references(page)
    write_json_atomic(args.output / "verification.json", verification)
    manifest = {"schema": "tennis_lab_ball3d_artifact_inventory.v1", "code_commit": commit,
        "inputs": {"bundle_sha256": bundle.manifest_sha256, "annotations_sha256": annotations.reference.sha256},
        "files": {str(p.relative_to(args.output)): {"sha256": dual_sha256(p), "size_bytes": p.stat().st_size}
                  for p in sorted(args.output.rglob("*")) if p.is_file()}}
    write_json_atomic(args.output / "artifacts.json", manifest)
    (args.output / "artifacts.sha256").write_text(dual_sha256(args.output / "artifacts.json") + "\n")
    print(json.dumps({"output": str(args.output), "artifacts_sha256": dual_sha256(args.output / "artifacts.json"), "elapsed_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
