"""clip_000-only reproduction entrypoint; reusable code lives in src."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
import torch

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.baseline import (
    baseline_from_bundle,
    verify_baseline,
)
from src.tennis_scene.evaluation.ball3d.bundle import (
    create_bundle,
    load_bundle,
    validate_source_candidate,
    verify_videos,
)
from src.tennis_scene.evaluation.ball3d.runner import load_run, run_comparison, save_run
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

CLIP_ID = "meiji_3cam/video_000/clip_000"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("bundle")
    build.add_argument("--store", type=Path, required=True)
    build.add_argument("--candidate", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--source-code-commit", required=True)
    build.add_argument("--source-code-evidence", required=True)
    build.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--bundle", type=Path, required=True)
    compare.add_argument("--bundle-sha256", required=True)
    compare.add_argument("--annotations", type=Path, required=True)
    compare.add_argument("--annotations-sha256", required=True)
    compare.add_argument("--checkpoints", type=Path, required=True)
    compare.add_argument("--prior-comparison", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--video-step", type=int, default=1)
    compare.add_argument("--width", type=int, default=1200)
    for command in (build, compare):
        command.add_argument("--videos", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    if args.action == "bundle":
        candidate = validate_source_candidate(args.candidate, args.store)
        if candidate["source_code_commit"] != args.source_code_commit:
            raise ValueError("Candidate and supplied historical code commit disagree")
        bundle = create_bundle(
            args.store,
            args.output,
            clip_id=CLIP_ID,
            source_code_commit=args.source_code_commit,
            source_code_evidence=args.source_code_evidence,
            source_archive=args.archive,
        )
        videos = {c: args.videos / f"{c}.mp4" for c in bundle.inputs.camera_ids}
        receipt = {
            "candidate": candidate,
            "bundle_sha256": bundle.manifest_sha256,
            "videos": verify_videos(bundle, videos),
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json_atomic(
            args.output.parent / f"{args.output.name}-verification.json", receipt
        )
        print(receipt, flush=True)
        return
    bundle = load_bundle(
        args.bundle, expected_sha256=args.bundle_sha256, require_court_lines=False
    )
    inputs = bundle.inputs
    if (
        inputs.clip_id != CLIP_ID
        or inputs.num_frames != 1010
        or inputs.camera_ids != ("cam0", "cam1", "cam2")
    ):
        raise ValueError(
            "This reproduction script is restricted to the agreed clip_000 inputs"
        )
    if args.output.exists():
        raise ValueError(
            "Use a new output directory; never overwrite a prior comparison"
        )
    if any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (args.bundle, args.videos, args.annotations.parent)
    ):
        raise ValueError("Comparison output must be separate from the immutable inputs")
    if subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            str(Path(__file__).relative_to(Path.cwd())),
        ],
        text=True,
    ):
        raise ValueError("Commit the replay code before recording a reproducible run")
    videos = {c: args.videos / f"{c}.mp4" for c in inputs.camera_ids}
    video_receipt = verify_videos(bundle, videos)
    checkpoint_receipt = {}
    index = json.loads((args.bundle / "scene.json").read_text())
    for component in (
        f"{kind}/{camera}"
        for kind in ("ball_detection", "court_detection")
        for camera in inputs.camera_ids
    ):
        # Court detection is an audit-only transitive descriptor, not an array input.
        descriptor = args.bundle / index["artifacts"][component]["path"]
        settings = json.loads(descriptor.read_text())["identity"]["settings"]
        relative = settings["config"]["checkpoint"]
        path = args.checkpoints / relative
        if Path(relative).is_absolute() or not path.resolve().is_relative_to(
            args.checkpoints.resolve()
        ):
            raise ValueError("Checkpoint path escapes explicitly supplied root")
        if relative not in checkpoint_receipt:
            checkpoint_receipt[relative] = {
                "sha256": dual_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        checksum = checkpoint_receipt[relative]["sha256"]
        if checksum != settings["checkpoint"]["sha256"]:
            raise ValueError(f"Historical checkpoint checksum mismatch: {relative}")
    baseline = verify_baseline(bundle)
    annotations = load_annotations(
        args.annotations, bundle, expected_sha256=args.annotations_sha256
    )
    counts = {
        kind: int((annotations.reference_kind == kind).sum())
        for kind in (
            "observed",
            "estimated",
            "excluded_extrapolation",
            "occluded",
            "unknown",
        )
    }
    if counts != {
        "observed": 2589,
        "estimated": 370,
        "excluded_extrapolation": 28,
        "occluded": 26,
        "unknown": 17,
    }:
        raise ValueError("The pinned clip_000 reference counts changed")
    print("Verified fixed inputs and baseline", flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    production_paths = [
        "src/tennis_scene/pipeline/components/ball_reconstruction.py",
        "src/tennis_scene/pipeline/components/ball_points.py",
        "src/utils/geometry/triangulation.py",
        "src/utils/geometry/temporal_smoothing.py",
        "src/tasks/ball_detection/inference/trajectory_gate.py",
        "src/utils/data/heatmaps.py",
    ]
    production_changes = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            bundle.manifest["source_code_commit"],
            commit,
            "--",
            *production_paths,
        ],
        text=True,
    ).splitlines()
    src = Path("src")
    fingerprint = document_digest(
        {str(p.relative_to(src)): dual_sha256(p) for p in sorted(src.rglob("*.py"))}
    )
    args.output.mkdir(parents=True)
    methods = {
        method.name: method
        for method in (
            baseline_from_bundle(bundle),
            baseline_from_bundle(bundle, without_loess=True),
        )
    }
    runs = run_comparison(
        methods,
        inputs,
        args.output / "results",
        bundle_sha256=bundle.manifest_sha256,
        code_commit=commit,
        code_fingerprint=fingerprint,
    )
    # Load every published result through the real checksum/type boundary.
    runs = {
        name: tuple(
            load_run(args.output / "results" / name / run.key) for run in values
        )
        for name, values in runs.items()
    }
    old_index = json.loads((args.prior_comparison / "artifacts.json").read_text())
    if old_index["inputs"]["bundle_sha256"] != bundle.manifest_sha256:
        raise ValueError("The prior comparison used different fixed inputs")
    unchanged = []
    for name, values in runs.items():
        for run in values:
            before = load_run(args.prior_comparison / "results" / name / run.key)
            for field in ("positions", "valid", "used_views"):
                np.testing.assert_array_equal(
                    getattr(run.result, field), getattr(before.result, field)
                )
            unchanged.append(f"{name}/{run.key}")
    print(
        "Saved and reloaded all eight CPU runs; predictions unchanged from v1",
        flush=True,
    )
    oracle = annotation_oracle(
        bundle,
        annotations,
        code_commit=commit,
        code_fingerprint=fingerprint,
    )
    for run in oracle:
        save_run(args.output / "scoring_oracle" / run.key, run)
    runs["annotation_oracle"] = tuple(
        load_run(args.output / "scoring_oracle" / run.key) for run in oracle
    )
    scores = score_comparison(runs, bundle, annotations)
    write_json_atomic(args.output / "scores.json", scores)
    plot_metrics_table(scores, args.output / "metrics-table.png")
    full = {
        name: next(r for r in values if r.held_out_camera is None)
        for name, values in runs.items()
        if name != "annotation_oracle"
    }
    with BallComparisonRenderer(
        inputs, full, videos, annotations=annotations, width=args.width
    ) as renderer:
        media = render_video(
            renderer, args.output / "comparison.mp4", step=args.video_step
        )
    print("Rendered and decoded every comparison frame", flush=True)
    plot_comparison(runs, bundle, args.output / "trajectories.png", annotations)
    extremes = render_extremes(
        {name: runs[name] for name in methods},
        bundle,
        videos,
        args.output / "extremes.png",
        annotations,
    )
    write_json_atomic(args.output / "extremes.json", extremes)
    source_decode = {
        camera: validate_video(
            path, frames=inputs.num_frames, fps=inputs.fps, size=inputs.source_size_wh
        )
        for camera, path in videos.items()
    }
    verification = {
        "schema": "tennis_lab_ball3d_qualification.v1",
        "review_findings": ["pr-1669-1", "pr-1669-2", "pr-1669-3"],
        "unchanged_predictions_from_foundation_v1": unchanged,
        "prior_artifact_manifest_sha256": dual_sha256(
            args.prior_comparison / "artifacts.json"
        ),
        "court_lines_required_by_baseline": False,
        "court_observations": {
            v.camera.camera_id: {
                "measured_frame_indices": v.court.frame_indices.tolist(),
                "raw_keypoints": int(v.court.valid.sum()),
                "selected_for_source_homography": int(
                    v.court.selected_for_homography.sum()
                ),
                "line_observations_available": v.court.lines is not None,
                "line_missing_reason": v.court.line_missing_reason,
            }
            for v in inputs.views
        },
        "bundle_sha256": bundle.manifest_sha256,
        "source_code_commit": bundle.manifest["source_code_commit"],
        "source_archive": bundle.manifest["source_archive"],
        "source_index_sha256": bundle.manifest["source_index_sha256"],
        "replay_code_commit": commit,
        "production_reconstruction_code_paths_checked": production_paths,
        "production_reconstruction_code_changes": production_changes,
        "replay_code_fingerprint": fingerprint,
        "replay_script_sha256": dual_sha256(Path(__file__)),
        "baseline": baseline,
        "videos": video_receipt,
        "historical_checkpoints": checkpoint_receipt,
        "source_decode": source_decode,
        "comparison_video": media,
        "annotation_status": "scored on human-provided observed and approved estimated references",
        "annotations_sha256": annotations.reference.sha256,
        "reference_counts": counts,
        "scoring_reference_count": int(annotations.valid_for_scoring.sum()),
        "event_source_rows": len(annotations.reference.events),
        "physical_event_count": None,
        "estimation_segments": len(annotations.reference.estimation_segments),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "gpu_used": False,
        "inference_rerun": False,
        "elapsed_seconds": time.perf_counter() - started,
        "human_adoption": "method adoption pending; foundation merge belongs to parent after explicit review and CI",
    }
    page = write_report(
        args.output,
        scores,
        video=args.output / "comparison.mp4",
        figure=args.output / "trajectories.png",
        crops=args.output / "extremes.png" if extremes["frames"] else None,
        verification=verification,
        table_figure=args.output / "metrics-table.png",
    )
    verification["broken_html_references"] = broken_html_references(page)
    verification["artifact_sha256"] = {
        p.name: dual_sha256(p)
        for p in (
            page,
            args.output / "scores.json",
            args.output / "comparison.mp4",
            args.output / "trajectories.png",
            args.output / "extremes.json",
        )
    }
    write_json_atomic(args.output / "verification.json", verification)
    # Manifest covers every result array, receipt, preview and media file. Its own
    # digest lives separately to avoid a self-referential checksum.
    manifest = {
        "schema": "tennis_lab_ball3d_artifact_inventory.v1",
        "code_commit": commit,
        "inputs": {
            "bundle_sha256": bundle.manifest_sha256,
            "annotations_sha256": annotations.reference.sha256,
            "videos": video_receipt,
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
    write_json_atomic(args.output / "artifacts.json", manifest)
    (args.output / "artifacts.sha256").write_text(
        dual_sha256(args.output / "artifacts.json") + "\n"
    )
    print(f"Comparison: {page} ({time.perf_counter() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
