"""Verify every output and retain small measurements; arrays/video stay in outputs."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.tennis_scene.evaluation.ball3d.bundle import load_bundle
from src.tennis_scene.evaluation.ball3d.runner import load_run
from src.tennis_scene.evaluation.ball3d.scoring import validate_scoring_run
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--artifacts-sha256", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    args = parser.parse_args()
    output = args.comparison.resolve()
    target = Path(__file__).resolve().parent
    digest = dual_sha256(output / "artifacts.json")
    if digest != args.artifacts_sha256 or digest != (output / "artifacts.sha256").read_text().strip():
        raise ValueError("Artifact inventory checksum mismatch")
    inventory = json.loads((output / "artifacts.json").read_text())
    for relative, record in inventory["files"].items():
        path = output / relative
        if dual_sha256(path) != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Artifact changed: {relative}")
    if {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()} != set(inventory["files"]) | {"artifacts.json", "artifacts.sha256"}:
        raise ValueError("Undeclared output files")
    bundle = load_bundle(args.bundle, expected_sha256=args.bundle_sha256)
    calibrations = {}
    typed = []
    for method in ("pipeline_baseline", "court_ball_ba", "annotation_oracle"):
        for key in ("all_views", "without_cam0", "without_cam1", "without_cam2"):
            path = output / "results" / method / key if method != "annotation_oracle" else output / "scoring_oracle" / key
            run = load_run(path)
            validate_scoring_run(run, bundle)
            typed.append(f"{method}/{key}")
            if method != "court_ball_ba":
                continue
            rows = []
            for camera in run.result.cameras:
                v = run.result.camera_ids.index(camera.camera_id)
                before = bundle.inputs.views[bundle.inputs.camera_ids.index(camera.camera_id)].camera
                rows.append({
                    "camera_id": camera.camera_id,
                    "focal_before_px": float(before.intrinsic[0, 0]),
                    "focal_after_px": float(camera.intrinsic[0, 0]),
                    "centre_before_m": before.center.tolist(), "centre_after_m": camera.center.tolist(),
                    "rotation_change_degrees": float(np.rad2deg(Rotation.from_matrix(camera.rotation @ before.rotation.T).magnitude())),
                    "offset_frames": float(run.result.time_offsets_seconds[v] * bundle.inputs.fps),
                    "offset_seconds": float(run.result.time_offsets_seconds[v]),
                })
            diagnostic = run.result.diagnostics
            calibrations[key] = {"cameras": rows,
                **{k: diagnostic[k] for k in ("reference_camera", "nfev", "cost", "optimality",
                    "solver_message", "parameter_count", "residual_count", "court_fit_count",
                    "court_holdout_count", "court_camera_jacobian_rank", "ball_fit_count",
                    "camera_bounds_hit", "clock_bounds_hit")},
                "fit_frame_count": len(diagnostic["fit_frame_indices"]),
                "fit_frame_min_max": [int(np.min(diagnostic["fit_frame_indices"])), int(np.max(diagnostic["fit_frame_indices"]))],
            }
    write_json_atomic(target / "calibration-summary.json", calibrations)
    scores = json.loads((output / "scores.json").read_text())
    paired = json.loads((output / "paired-scores.json").read_text())
    metrics = {"output_dir": str(output), "artifacts_sha256": digest,
        "verified_file_count": len(inventory["files"]), "verified_typed_runs": typed,
        "common_lovo_cohort": scores["common_lovo_cohort"], "methods": {}}
    for name, method in scores["methods"].items():
        metrics["methods"][name] = {key: method[key] for key in (
            "lovo_reprojection_by_reference_kind", "common_lovo_reprojection_by_reference_kind",
            "runtime_seconds_all_four_runs")}
        metrics["methods"][name]["folds"] = {
            fold: {segment: {key: data[key] for key in (
                "reprojection_px", "scoring_label_count", "scored_label_coverage", "valid_3d_rate", "valid_3d_frames")}
                for segment, data in segments.items()} for fold, segments in method["folds"].items()}
        if name in paired["methods"]:
            metrics["methods"][name]["two_method_common_lovo"] = paired["methods"][name]["common_lovo_reprojection_px"]
    write_json_atomic(target / "metrics.json", metrics)
    for filename in ("artifacts.json", "verification.json", "court-holdout.json", "metrics.md", "metrics-table.png", "trajectories.png"):
        source = output / filename
        limit = 512 * 1024 if source.suffix == ".png" else 256 * 1024
        if source.stat().st_size > limit:
            raise ValueError(f"Knowledge receipt exceeds size limit: {filename}")
        shutil.copyfile(source, target / filename)
    shutil.copyfile(output.parent / f"{output.name}-resource-usage.txt", target / "resource-usage.txt")
    shutil.copyfile(output.parent / f"{output.name}-run.log", target / "execution.log")
    print(f"Verified {len(typed)} typed runs and {len(inventory['files'])} output files. Predictions remain in {output}.")


if __name__ == "__main__":
    main()
