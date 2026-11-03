"""Rescore the immutable v1 results after the foundation review; no prediction replay."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from plots import extend_report
from run import paired_budget

from src.tennis_scene.evaluation.ball3d.annotations import load_annotations
from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import score_comparison
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.visualization.ball3d.report import (
    broken_html_references,
    plot_metrics_table,
    write_report,
)
from src.utils.checksum import dual_sha256

METHODS = (
    "pipeline_baseline",
    "without_loess",
    "annotation_oracle",
    "annotation_time_oracle",
    "candidate1_oracle",
    "candidate8_oracle",
)
ORIGINAL_FOUNDATION = "3abc5e7595bccabf4ed558ce209a9b4bc117a73c"
PREDICTION_PATHS = (
    "src/tennis_scene/pipeline/components/ball_reconstruction.py",
    "src/tennis_scene/pipeline/components/ball_points.py",
    "src/utils/geometry/triangulation.py",
    "src/utils/geometry/temporal_smoothing.py",
    "src/tasks/ball_detection/inference/trajectory_gate.py",
    "src/utils/data/heatmaps.py",
)


def verified_inventory(directory, expected_sha256):
    inventory = directory / "artifacts.json"
    if dual_sha256(inventory) != expected_sha256:
        raise ValueError("Prior comparison inventory digest mismatch")
    receipt = json.loads(inventory.read_text())
    for relative, spec in receipt["files"].items():
        path = directory / relative
        if (
            path.stat().st_size != spec["size_bytes"]
            or dual_sha256(path) != spec["sha256"]
        ):
            raise ValueError(f"Prior artifact changed: {relative}")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "annotations", "prior-comparison", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("bundle-sha256", "annotations-sha256", "prior-manifest-sha256"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    run_dir = Path(__file__).resolve().parent
    if args.output.exists():
        raise ValueError("Use a new output directory; previous artifacts are immutable")
    if any(
        args.output.resolve().is_relative_to(p.resolve())
        for p in (
            args.bundle,
            args.annotations.parent,
            args.prior_comparison,
        )
    ):
        raise ValueError("Output must be separate from immutable inputs")
    if subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src",
            str(run_dir.relative_to(Path.cwd())),
        ],
        text=True,
    ):
        raise ValueError("Commit scoring code before recording a reproducible rescore")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    inventory = verified_inventory(args.prior_comparison, args.prior_manifest_sha256)
    prior = json.loads((args.prior_comparison / "verification.json").read_text())
    if (
        dual_sha256(run_dir / "diagnostics.py")
        != prior["script_sha256"]["diagnostics.py"]
    ):
        raise ValueError(
            "Diagnostic calculations changed; rescore cannot establish equivalence"
        )
    changed_prediction_paths = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            ORIGINAL_FOUNDATION,
            commit,
            "--",
            *PREDICTION_PATHS,
        ],
        text=True,
    ).splitlines()
    if changed_prediction_paths:
        raise ValueError("Prediction calculations changed; use a full replay")
    bundle = load_bundle(args.bundle, expected_sha256=args.bundle_sha256)
    inputs = bundle.inputs
    if (
        inputs.clip_id != "meiji_3cam/video_000/clip_000"
        or inputs.num_frames != 1010
        or inputs.camera_ids != ("cam0", "cam1", "cam2")
    ):
        raise ValueError("This rescore is restricted to the agreed clip_000")
    annotations = load_annotations(
        args.annotations, bundle, expected_sha256=args.annotations_sha256
    )
    runs = {}
    for name in METHODS:
        parent = "results" if name in METHODS[:2] else "scoring_diagnostics_only"
        runs[name] = tuple(
            load_run(args.prior_comparison / parent / name / key)
            for key in ("all_views", *(f"without_{c}" for c in inputs.camera_ids))
        )
    scores = score_comparison(runs, bundle, annotations)
    old_scores = json.loads((args.prior_comparison / "scores.json").read_text())
    for name, old in old_scores["methods"].items():
        if {key: scores["methods"][name][key] for key in old} != old:
            raise ValueError(f"Legacy scoring values changed: {name}")
    budget = paired_budget(runs, bundle, annotations)
    if budget != json.loads((args.prior_comparison / "paired-budget.json").read_text()):
        raise ValueError("Pairwise error decomposition changed")

    # Check the foundation's updated stores independently, without any replay.
    foundation_checks = []
    foundation = args.prior_comparison.parent / "foundation_v2"
    for name in METHODS[:2]:
        for run in runs[name]:
            updated = load_run(foundation / "results" / name / run.key)
            for field in ("positions", "valid", "used_views"):
                np.testing.assert_array_equal(
                    getattr(run.result, field), getattr(updated.result, field)
                )
            foundation_checks.append(f"{name}/{run.key}")
    args.output.mkdir(parents=True)
    replaced = {
        "scores.json",
        "paired-budget.json",
        "metrics-table.png",
        "index.html",
        "verification.json",
    }
    for relative in inventory["files"]:
        if relative not in replaced:
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.prior_comparison / relative, destination)
    write_json_atomic(args.output / "scores.json", scores)
    write_json_atomic(args.output / "paired-budget.json", budget)
    plot_metrics_table(scores, args.output / "metrics-table.png")
    verification = {
        "schema": "tennis_lab_ball3d_error_decomposition_rescore.v1",
        "scoring_commit": commit,
        "prediction_execution_commit": prior["execution_commit"],
        "prior_comparison": str(args.prior_comparison.resolve()),
        "prior_manifest_sha256": args.prior_manifest_sha256,
        "verified_prior_files": len(inventory["files"]),
        "bundle_sha256": bundle.manifest_sha256,
        "annotations_sha256": annotations.reference.sha256,
        "prediction_replay_rerun": False,
        "diagnostic_fit_rerun": False,
        "video_render_encode_rerun": False,
        "gpu_used": False,
        "inference_rerun": False,
        "loaded_result_count": sum(map(len, runs.values())),
        "legacy_method_scores_exact": True,
        "pairwise_error_budget_exact": True,
        "diagnostics_script_bytes_exact": True,
        "prediction_code_base": ORIGINAL_FOUNDATION,
        "prediction_code_paths_checked": list(PREDICTION_PATHS),
        "changed_prediction_paths": changed_prediction_paths,
        "foundation_v2_positions_valid_used_views_exact": foundation_checks,
        "source_result_sha256": {
            relative: spec["sha256"]
            for relative, spec in inventory["files"].items()
            if relative.startswith(("results/", "scoring_diagnostics_only/"))
        },
        "comparison_video": {
            "reused_without_change": True,
            "sha256": prior["comparison_video"]["sha256"],
            "prior_decoded_frames": prior["comparison_video"]["decoded_frames"],
            "decode_repeated": False,
        },
        "common_lovo_cohort": scores["common_lovo_cohort"],
        "common_lovo_reprojection_px": {
            name: row["common_lovo_reprojection_px"]
            for name, row in scores["methods"].items()
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
    extend_report(
        page,
        [
            args.output / name
            for name in (
                "position-bias.png",
                "candidate-recovery.png",
                "regression.png",
                "smoothing.png",
            )
        ],
        budget,
        json.loads((args.output / "timing.json").read_text()),
    )
    verification["broken_html_references"] = broken_html_references(page)
    if verification["broken_html_references"]:
        raise ValueError("Generated HTML has broken references")
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
    print(
        f"Rescored {sum(map(len, runs.values()))} results; existing scores and paired budget unchanged: {page}",
        flush=True,
    )


if __name__ == "__main__":
    main()
