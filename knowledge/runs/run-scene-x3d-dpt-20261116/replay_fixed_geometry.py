"""Explicit CPU comparison with X3D's accepted cameras held fixed for both models.

This is a diagnostic replay, never an adopted pipeline artifact or a replacement
for i935's failed court-side decision. Source stores stay immutable.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.ball_points import BallPointsOutput
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import CourtCalibrationOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationModule
from src.tennis_scene.pipeline.contracts import AssemblyContext
from src.tennis_scene.pipeline.input_assembly.reconstruction import BallTriangulationInputAssembler
from src.tennis_scene.pipeline.source import build_clip_source
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256


def store(directory):
    source = json.loads((directory / "scene.json").read_text())["source"]
    return ClipStore(directory, source)


def load(original, name, cls):
    ref = original.active(name)
    if ref is None:
        raise ValueError(f"Missing source artifact: {name}")
    return original.load(ref, ArtifactCodec(cls)), json_value(ref)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clip = "clip_000"
    cameras = ("cam1", "cam2", "cam3")
    media = args.dataset / "videos/video_000/clips" / clip / "media"
    source = build_clip_source(tuple((media / f"{camera}.mp4").resolve() for camera in cameras),
                               cameras, clip_id=f"video_000/{clip}")
    candidate = store(args.candidate / "store")
    if json_value(source) != candidate.source:
        raise ValueError("Fixed geometry source identity differs")
    calibration, calibration_ref = load(candidate, "court_calibration", CourtCalibrationOutput)
    alignment, alignment_ref = load(candidate, "camera_alignment", CameraAlignmentOutput)
    if alignment.geometry is None:
        raise ValueError("Candidate did not accept aligned cameras")
    records = {}
    for variant, directory in (("i935", args.baseline), ("x3d", args.candidate)):
        original = store(directory / "store")
        if original.source != candidate.source:
            raise ValueError("Comparison source/camera identities differ")
        config = json.loads((directory / "config.json").read_text())
        artifacts = dict(calibration=calibration, alignment=alignment)
        references = {}
        for camera in cameras:
            point, ref = load(original, f"ball_points/{camera}", BallPointsOutput)
            artifacts[f"ball_{camera}"] = point
            references[camera] = ref
        inputs = BallTriangulationInputAssembler(float(config["ball_detection"]["score_threshold"])).assemble(
            AssemblyContext(source), artifacts)
        settings = config["ball_reconstruction"]
        module = BallTriangulationModule(cameras, reprojection_px=settings["reprojection_px"], min_frames=settings["min_frames"])
        begin = time.perf_counter()
        result = module.process(inputs).ball
        if result is None:
            raise ValueError("Replay did not reconstruct a ball result")
        trajectory = result.trajectory
        with np.load(directory / "comparison-arrays.npz") as arrays:
            data = {key: arrays[key] for key in ("uv_px", "observed", "confidence")}
            if variant == "x3d":
                np.testing.assert_array_equal(trajectory.valid, arrays["ball_3d_valid"])
                np.testing.assert_allclose(trajectory.positions, arrays["ball_3d"], rtol=0, atol=1e-5)
        data.update(ball_3d=trajectory.positions, ball_3d_valid=trajectory.valid)
        errors = trajectory.reprojection_px[trajectory.inliers & trajectory.valid[None]]
        record = json.loads((directory / "metrics.json").read_text())
        record.update(status="fixed_geometry_replay", canonical_pipeline_status=record["status"],
            canonical_pipeline_failure=record["failure"], failure=None,
            fixed_geometry_replay=True,
            comparison_scope="Explicit fixed X3D camera geometry for both sets of measured 2D points. i935's original court-side failure remains unchanged. Not an independent accuracy test.",
            ball_3d_valid_frames=int(trajectory.valid.sum()), reprojection_inlier_count=len(errors),
            reprojection_mean_px=float(errors.mean()) if len(errors) else None,
            reprojection_median_px=float(np.median(errors)) if len(errors) else None,
            reprojection_p95_px=float(np.quantile(errors, .95)) if len(errors) else None,
            replay_seconds=time.perf_counter() - begin,
            geometry_source=dict(directory=str(args.candidate), source_index_sha256=dual_sha256(args.candidate / "store/scene.json"),
                                 calibration=calibration_ref, alignment=alignment_ref),
            measured_ball_points=references)
        output = args.output / variant / clip
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "comparison-arrays.npz", **data)
        (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        records[variant] = {key: record[key] for key in ("canonical_pipeline_status", "ball_3d_valid_frames", "reprojection_mean_px", "reprojection_median_px", "reprojection_p95_px", "replay_seconds")}
    (args.output / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
