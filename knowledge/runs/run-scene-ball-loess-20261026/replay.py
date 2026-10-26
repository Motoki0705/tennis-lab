"""CPU-only ball replay of a frozen, checksummed component-store export.

This experiment does not impersonate a current full pipeline run: cameras, side
decisions and all people remain from the source run. Only ball_points and
ball_triangulation run again. The source index/artifacts are never opened for write.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np

from src.tennis_scene.archive import load_scene_result, save_scene_result
from src.tennis_scene.pipeline.artifacts import document_digest, json_value, write_json_atomic
from src.tennis_scene.pipeline.components.ball_detection import BallDetectionOutput
from src.tennis_scene.pipeline.components.ball_points import BallPointsInput, BallPointsModule
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.court_calibration import CourtCalibrationOutput
from src.tennis_scene.pipeline.components.triangulation import BallTriangulationModule, BallTriangulationOutput
from src.tennis_scene.pipeline.contracts import AssemblyContext, ClipSource, SourceVideo
from src.tennis_scene.pipeline.input_assembly.reconstruction import BallTriangulationInputAssembler
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path, read_component_descriptor
from src.utils.checksum import dual_sha256
from src.utils.geometry.keypoints import normalize_keypoints
from src.utils.geometry.temporal_smoothing import DEFAULT_TEMPORAL_SMOOTHING
from src.utils.paths import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True, help="Explicit remapping: CAMERA.mp4 under this directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.store.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Use a new output directory separate from the immutable input store")
    index_path = root / "scene.json"
    index_hash = dual_sha256(index_path)
    index = json.loads(index_path.read_text())
    if document_digest(index["source"]) != index["source_sha256"]:
        raise ValueError("Store source document checksum mismatch")
    source_scene = indexed_scene_path(index_path)  # Checks full adopted descriptor lineage and export checksums.
    scene = load_scene_result(source_scene)
    source = ClipSource(index["source"]["clip_id"], tuple(SourceVideo(
        **{**row, "path": (args.videos / f"{row['camera_id']}.mp4").resolve()}
    ) for row in index["source"]["videos"]))
    for video in source.videos:
        if dual_sha256(video.path) != video.sha256:
            raise ValueError(f"Explicit source video remapping checksum mismatch: {video.camera_id}")
    if (scene.num_frames, scene.fps, scene.width, scene.height) != (source.num_frames, source.fps, *source.size):
        raise ValueError("Frozen scene and store source timeline differ")
    if scene.metadata["camera_ids"] != list(source.camera_ids):
        raise ValueError("Frozen scene camera axis differs from source")
    used = {}

    def load(node, schema, version, output_type):
        ref = index["artifacts"][node]
        if (ref["schema"], ref["version"]) != (schema, version):
            raise ValueError(f"Unsupported replay input schema: {node}")
        descriptor = read_component_descriptor(root, ref, node=node, source_sha256=index["source_sha256"])
        used[node] = ref
        return ArtifactCodec(output_type).load(descriptor["payload"], (root / ref["path"]).parent, descriptor["arrays"])

    alignment = load("camera_alignment", "aligned_cameras", 1, CameraAlignmentOutput)
    calibration = load("court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
    if alignment.geometry is None:
        raise ValueError("Frozen-camera replay requires resolved geometry")
    active = tuple(v.source_index for v in calibration.calibration.views)
    if tuple(source.camera_ids[i] for i in active) != alignment.geometry.camera_ids:
        raise ValueError("Calibration and aligned camera order mismatch")
    baseline = load("ball_triangulation", "ball_trajectory", 1, BallTriangulationOutput).ball
    if baseline is None or not np.array_equal(baseline.trajectory.valid, scene.ball_3d_valid):
        raise ValueError("Saved ball trajectory and frozen scene disagree")
    original_triangulation = read_component_descriptor(root, index["artifacts"]["ball_triangulation"], node="ball_triangulation", source_sha256=index["source_sha256"])
    settings = original_triangulation["identity"]["settings"]
    triangulation = BallTriangulationModule(source.camera_ids, **settings["config"])
    points_module = BallPointsModule(DEFAULT_TEMPORAL_SMOOTHING)
    rows, counts = {}, {}
    for camera in source.camera_ids:
        detections = load(f"ball_detection/{camera}", "ball_detections", 2, BallDetectionOutput)
        if detections.camera_id != camera or not np.array_equal(detections.frame_indices, np.arange(source.num_frames)):
            raise ValueError("Saved detector camera/timeline mismatch")
        row = points_module.process(BallPointsInput(detections, source.size, source.fps))
        rows[f"ball_{camera}"] = row
        counts[camera] = {name: int(getattr(row.smoothing, name).sum()) for name in (
            "source_observed", "accepted", "rejected", "interpolated", "smoothed", "restored_turn")}
    inputs = BallTriangulationInputAssembler(settings["threshold"]).assemble(
        AssemblyContext(source), {"alignment": alignment, "calibration": calibration, **rows})
    result = triangulation.process(inputs)
    ball = result.ball
    if ball is None:
        raise ValueError("Replay produced no ball result")
    ball_uv = np.zeros((len(source.camera_ids), source.num_frames, 2), np.float32)
    ball_vis = np.zeros(ball_uv.shape[:2], bool)
    ball_uv[list(active)] = normalize_keypoints(ball.uv_px, *source.size)
    ball_vis[list(active)] = ball.visibility
    if np.any(ball.trajectory.valid & (ball_vis.sum(0) < 2)):
        raise AssertionError("A filled point entered 3D geometry")
    code = PROJECT_ROOT / "src"
    provenance = json_value({
        "schema": "frozen_camera_ball_replay_v1", "source_index": str(index_path), "source_index_sha256": index_hash,
        "source_scene": str(source_scene), "source_scene_sha256": dual_sha256(source_scene),
        "source_artifacts": used, "source_ball_triangulation": index["artifacts"]["ball_triangulation"],
        "source_video_remapping": {v.camera_id: {"path": str(v.path), "sha256": v.sha256} for v in source.videos},
        "code_sha256": document_digest({str(p.relative_to(code)): dual_sha256(p) for p in sorted(code.rglob("*.py"))}),
        "replay_script_sha256": dual_sha256(Path(__file__)),
        "smoothing": points_module.config, "triangulation_settings": settings,
        "rerun": ["ball_points", "ball_triangulation"],
        "frozen": ["camera_alignment", "court_calibration", "court_side", "all_player_and_body_arrays"],
        "full_pipeline_rerun": False,
    })
    meta = copy.deepcopy(scene.metadata)
    meta["video_paths"] = [str(v.path) for v in source.videos]
    meta["ball_replay"] = provenance
    meta["ball_observation_contract"] = "motion_consensus_loess_accepted_real_observations_only"
    meta["validity_statistics"]["ball_3d_frames"] = int(ball.trajectory.valid.sum())
    updated = replace(scene, ball_uv=ball_uv, ball_vis=ball_vis, ball_3d=ball.trajectory.positions,
                      ball_3d_valid=ball.trajectory.valid, ball_rejection_code=ball.trajectory.reasons, metadata=meta)
    frozen_arrays = []
    for field in fields(scene):
        if field.name.startswith("ball_") or field.name == "metadata":
            continue
        before, after = getattr(scene, field.name), getattr(updated, field.name)
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
            frozen_arrays.append(field.name)
        elif before != after:
            raise AssertionError(f"Frozen scene field changed: {field.name}")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in {**rows, "ball_triangulation": result}.items():
        directory = output / name
        directory.mkdir()
        payload, arrays = ArtifactCodec(type(value)).dump(value, directory)
        write_json_atomic(directory / "artifact.json", {"payload": payload, "arrays": arrays, "replay": provenance,
                          "output_schema": "ball_points" if name.startswith("ball_") and name != "ball_triangulation" else "ball_trajectory",
                          "output_version": 4 if name in rows else 1})
    save_scene_result(updated, output / "scene.npz")
    valid = ball.trajectory.valid
    paired = valid[1:] & valid[:-1]
    jumps = np.linalg.norm(np.diff(ball.trajectory.positions, axis=0), axis=1)[paired]
    old_valid = scene.ball_3d_valid
    old_paired = old_valid[1:] & old_valid[:-1]
    old_jumps = np.linalg.norm(np.diff(scene.ball_3d, axis=0), axis=1)[old_paired]
    errors = ball.trajectory.reprojection_px[ball.trajectory.inliers & valid[None]]
    old_errors = baseline.trajectory.reprojection_px[baseline.trajectory.inliers & old_valid[None]]
    summary = {
        "source_frames": source.num_frames, "per_camera": counts, "valid_before": int(old_valid.sum()),
        "valid_after": int(valid.sum()), "lost_valid": int((old_valid & ~valid).sum()), "gained_valid": int((~old_valid & valid).sum()),
        "median_reprojection_px": float(np.median(errors)) if len(errors) else None,
        "p95_reprojection_px": float(np.quantile(errors, .95)) if len(errors) else None,
        "before_median_reprojection_px": float(np.median(old_errors)),
        "before_p95_reprojection_px": float(np.quantile(old_errors, .95)),
        "before_p95_adjacent_step_m": float(np.quantile(old_jumps, .95)),
        "after_p95_adjacent_step_m": float(np.quantile(jumps, .95)),
        "before_max_adjacent_step_m": float(old_jumps.max()), "after_max_adjacent_step_m": float(jumps.max()),
        "valid_3d_with_fewer_than_two_real_views": int((valid & (ball_vis.sum(0) < 2)).sum()),
        "unchanged_scene_arrays": frozen_arrays, "ground_truth_accuracy_evaluated": False,
    }
    if dual_sha256(index_path) != index_hash:
        raise ValueError("Source store changed during replay")
    write_json_atomic(output / "replay.json", {"provenance": provenance, "summary": summary,
        "outputs": {p.name: dual_sha256(p) for p in (output / "scene.npz", output / "scene.metadata.json")}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
