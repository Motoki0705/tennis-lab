"""Planning probe for #2444. Read saved all-row poses and score relative court turns on CPU."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from src.tasks.court_side.hypothesis import half_turn_hypotheses
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_features import (
    PersonPoseFeaturesOutput,
    require_alignment,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec, restore_type
from src.tennis_scene.pipeline.storage.scene_index import (
    assert_current_component_lineage,
    read_component_descriptor,
)
from src.tennis_scene.pipeline.video_court_side.artifact import VideoCourtSideArtifact
from src.tennis_scene.pipeline.video_court_side.inputs import PinnedSideInputs
from src.utils.geometry.triangulation import PinholeCamera

TORSO = [5, 6, 11, 12]
MIN_CONFIDENCE = 0.25
MIN_TORSO_JOINTS = 3
REPROJECTION_AT_1080P = 24.0
MIN_ANGLE_SIN_SQUARED = np.sin(np.deg2rad(1.0)) ** 2
MIN_EDGE_FRAMES = 8
MIN_SCORE = 0.05
MIN_MARGIN = 0.02
Output = TypeVar("Output")
TorsoObservations = tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_output(
    directory: Path, index: dict[str, Any], node: str, schema: str, version: int, output_type: type[Output]
) -> tuple[Output, dict[str, Any]]:
    ref = index["artifacts"][node]
    if (ref["schema"], ref["version"]) != (schema, version):
        raise ValueError(f"Unexpected input schema: {directory} {node}")
    descriptor = read_component_descriptor(directory, ref, node=node, source_sha256=index["source_sha256"])
    value = ArtifactCodec(output_type).load(descriptor["payload"], (directory / ref["path"]).parent, descriptor["arrays"])
    return value, descriptor["dependencies"]


def read_person_inputs(
    directory: Path, index: dict[str, Any], source: ClipSource
) -> tuple[CourtCalibrationOutput, list[PersonPoseFeaturesOutput], dict[str, Any]]:
    calibration, _ = read_output(directory, index, "court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
    if calibration.calibration.camera_ids != source.camera_ids or calibration.calibration.excluded:
        raise ValueError(f"Probe requires local calibrations for all source views: {source.clip_id}")
    poses = []
    names = ["court_calibration"]
    for video in source.videos:
        detection_node = f"person_detection/{video.camera_id}"
        pose_node = f"person_pose_features/{video.camera_id}"
        detection, detection_deps = read_output(directory, index, detection_node, "person_detections", 2, PersonDetectionOutput)
        pose, pose_deps = read_output(directory, index, pose_node, "person_pose_features", 2, PersonPoseFeaturesOutput)
        rows = require_alignment(video, detection, pose)
        if (detection_deps or pose_deps != {"detections": index["artifacts"][detection_node]}
                or detection.duplicate_merges or not np.array_equal(rows, np.arange(len(rows)))):
            raise ValueError(f"Expected every detection row and its own pose: {source.clip_id} {video.camera_id}")
        poses.append(pose)
        names.extend((detection_node, pose_node))
    refs = {name: index["artifacts"][name] for name in names}
    allowed = set(names) | {f"court_detection/{camera}" for camera in source.camera_ids}
    if not assert_current_component_lineage(index, directory, refs) <= allowed:
        raise ValueError(f"Unexpected input lineage: {source.clip_id}")
    return calibration, poses, {"source_sha256": index["source_sha256"], "artifacts": refs}


def torso_observations(poses: NDArray[np.float32], camera: PinholeCamera, size: tuple[int, int]) -> TorsoObservations:
    keypoints = poses[:, TORSO].astype(np.float64)
    uv = keypoints[..., :2]
    visible = (keypoints[..., 2] >= MIN_CONFIDENCE) & ((uv >= 0) & (uv < size)).all(axis=-1)
    homogeneous = np.concatenate((uv, np.ones_like(uv[..., :1])), axis=-1)
    rays = homogeneous @ np.linalg.inv(camera.intrinsic).T @ camera.rotation
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return uv, visible, rays


def pair_weights(
    observations_a: TorsoObservations, observations_b: TorsoObservations,
    camera_a: PinholeCamera, camera_b: PinholeCamera, threshold_px: float,
) -> NDArray[np.float64]:
    uv_a, mask_a, rays_a = observations_a
    uv_b, mask_b, rays_b = observations_b
    ray_a, ray_b = rays_a[:, None], rays_b[None, :]
    start_a, start_b = camera_a.center, camera_b.center
    separation = start_a - start_b
    dot = (ray_a * ray_b).sum(axis=-1)
    divisor = 1.0 - dot ** 2
    good_angle = divisor >= MIN_ANGLE_SIN_SQUARED
    divisor = np.where(good_angle, divisor, 1.0)
    offset_a, offset_b = ray_a @ separation, ray_b @ separation
    depth_a = (dot * offset_b - offset_a) / divisor
    depth_b = (offset_b - dot * offset_a) / divisor
    midpoint = (start_a + depth_a[..., None] * ray_a + start_b + depth_b[..., None] * ray_b) / 2.0
    projected_a, front_a = camera_a.project(midpoint)
    projected_b, front_b = camera_b.project(midpoint)
    squared_error = np.maximum(((projected_a - uv_a[:, None]) ** 2).sum(axis=-1),
                               ((projected_b - uv_b[None, :]) ** 2).sum(axis=-1))
    physical = (good_angle & (depth_a > 0) & (depth_b > 0) & front_a & front_b
                & (np.abs(midpoint[..., :2]) <= 40).all(axis=-1)
                & (midpoint[..., 2] >= -0.2) & (midpoint[..., 2] <= 3.0))
    supported = mask_a[:, None] & mask_b[None, :] & physical & (squared_error < threshold_px ** 2)
    joint_credit = np.where(supported, 1.0 - squared_error / threshold_px ** 2, 0.0)
    return np.where(supported.sum(axis=-1) >= MIN_TORSO_JOINTS, joint_credit.mean(axis=-1), 0.0)


def score_clip(
    source: ClipSource, calibration: CourtCalibrationOutput, poses: list[PersonPoseFeaturesOutput],
    pairs: list[tuple[int, int]], stride: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int]:
    cameras = [view.camera for view in calibration.calibration.views]
    turned = [camera.half_turned(True) for camera in cameras]
    data = [torso_observations(pose.keypoints, camera, source.size) for pose, camera in zip(poses, cameras, strict=True)]
    scores: NDArray[np.float64] = np.zeros((len(pairs), 2), np.float64)
    edge_frames: NDArray[np.int64] = np.zeros(scores.shape, np.int64)
    possible = 0
    threshold_px = REPROJECTION_AT_1080P * source.pixel_threshold_scale
    for frame in range(0, source.num_frames, stride):
        frame_data = []
        for (uv, visible, rays), pose in zip(data, poses, strict=True):
            start, end = pose.frame_offsets[frame:frame + 2]
            usable = visible[start:end].sum(axis=-1) >= MIN_TORSO_JOINTS
            frame_data.append((uv[start:end][usable], visible[start:end][usable], rays[start:end][usable]))
        for edge, (left, right) in enumerate(pairs):
            first, second = frame_data[left], frame_data[right]
            capacity = min(len(first[0]), len(second[0]))
            if not capacity:
                continue
            possible += capacity
            # Globally turning both cameras leaves this geometric score invariant.
            for parity, (right_camera, right_rays) in enumerate(
                ((cameras[right], second[2]), (turned[right], second[2] * [-1.0, -1.0, 1.0]))
            ):
                weights = pair_weights(first, (second[0], second[1], right_rays), cameras[left], right_camera, threshold_px)
                rows, columns = linear_sum_assignment(weights, maximize=True)
                credit = float(weights[rows, columns].sum())
                scores[edge, parity] += credit
                edge_frames[edge, parity] += credit > 0
    return scores, edge_frames, possible, len(range(0, source.num_frames, stride))


def rank_hypotheses(
    pairs: list[tuple[int, int]], scores: NDArray[np.float64], edge_frames: NDArray[np.int64], possible: int,
    camera_ids: tuple[str, ...], reference: str,
) -> dict[str, Any]:
    ref = camera_ids.index(reference)
    ranking = []
    for turns in half_turn_hypotheses(len(camera_ids), ref):
        parity = [int(turns[left] != turns[right]) for left, right in pairs]
        score = float(scores[np.arange(len(pairs)), parity].sum() / max(possible, 1))
        support = edge_frames[np.arange(len(pairs)), parity]
        connected = {ref}
        for _ in camera_ids:
            for (left, right), frames in zip(pairs, support, strict=True):
                if frames >= MIN_EDGE_FRAMES and (left in connected or right in connected):
                    connected.update((left, right))
        ranking.append({"turns": list(turns), "score": score, "edge_frames": support.tolist(),
                        "connected": len(connected) == len(camera_ids)})
    ranking.sort(key=lambda item: item["score"], reverse=True)
    best, second = ranking[:2]
    margin = best["score"] - second["score"]
    reason = None
    if not best["connected"]:
        reason = "disconnected_support"
    elif best["score"] < MIN_SCORE:
        reason = "insufficient_score"
    elif margin < MIN_MARGIN:
        reason = "ambiguous_margin"
    return {"ranking": ranking, "margin": margin, "reason": reason}


def verify_ball_scope(
    directory: Path, pin_digest: str, source: ClipSource, calibration: CourtCalibrationOutput,
    artifact: VideoCourtSideArtifact,
) -> None:
    pin = PinnedSideInputs.from_store(directory)
    if (pin.digest != pin_digest or pin_digest != artifact.decision.input_revisions[source.clip_id]
            or pin.source.clip_id != source.clip_id or pin.source.camera_ids != source.camera_ids):
        raise ValueError(f"Different ball evidence scope: {source.clip_id}")
    for first, second in zip(source.videos, pin.source.videos, strict=True):
        if (first.sha256, first.num_frames, first.fps, first.width, first.height) != (
            second.sha256, second.num_frames, second.fps, second.width, second.height
        ):
            raise ValueError(f"Different source media for ball evidence: {source.clip_id} {first.camera_id}")
    ball_index = read_json(directory / "scene.json")
    ball_court, _ = read_output(directory, ball_index, "court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
    for current in (calibration, ball_court):
        signature = document_digest({"cameras": [v.camera for v in current.calibration.views], "reference": current.reference_camera})
        if signature != artifact.decision.calibration_digest:
            raise ValueError(f"Different calibration from ball evidence: {source.clip_id}")
    current_diagnostics = calibration.court.diagnostics or {}
    ball_diagnostics = ball_court.court.diagnostics or {}
    if any(current_diagnostics[field] != ball_diagnostics[field] for field in ("shared_courts", "drift")):
        raise ValueError(f"Different shared court/drift provenance: {source.clip_id}")


def run(dataset: Path, artifact_root: Path, ball_run: Path, stride: int) -> dict[str, Any]:
    started = perf_counter()
    ball = VideoCourtSideArtifact.load(ball_run / "side-store", read_json(ball_run / "summary.json")["side_digest"])
    pins = read_json(ball_run / "side-inputs.json")["clips"]
    if ball.decision.evidence != "ball" or set(pins) != set(ball.clips):
        raise ValueError("Expected the published ball-based video-side evidence")
    clips = {}
    input_pins = {}
    camera_pairs = list(combinations(range(len(ball.decision.camera_ids)), 2))
    total_scores: NDArray[np.float64] = np.zeros((len(camera_pairs), 2), np.float64)
    total_edge_frames: NDArray[np.int64] = np.zeros(total_scores.shape, np.int64)
    total_possible = 0
    for clip_id in sorted(ball.clips):
        video_id, clip = clip_id.split("/")
        directory = (dataset / "videos" / video_id / "clips" / clip / "annotations" / "tennis_scene").resolve()
        index = read_json(directory / "scene.json")
        source = restore_type(index["source"], ClipSource)
        if (index["schema"] != "tennis_scene_index_v1" or document_digest(source) != index["source_sha256"]
                or source.clip_id != clip_id or source.camera_ids != ball.decision.camera_ids):
            raise ValueError(f"Unexpected i2363 source: {clip_id}")
        calibration, poses, input_pins[clip_id] = read_person_inputs(directory, index, source)
        scores, edge_frames, possible, sampled = score_clip(source, calibration, poses, camera_pairs, stride)
        ranked = rank_hypotheses(camera_pairs, scores, edge_frames, possible, source.camera_ids, calibration.reference_camera)
        total_scores += scores
        total_edge_frames += edge_frames
        total_possible += possible
        pin = pins[clip_id]
        verify_ball_scope((artifact_root / pin["directory"]).resolve(), pin["inputs_digest"], source, calibration, ball)
        old, old_margin = None, None
        if "court_side" in index["artifacts"]:
            old_result, _ = read_output(directory, index, "court_side", "court_side", 3, CourtSideOutput)
            if (old_result.camera_ids, old_result.reference_camera) != (source.camera_ids, calibration.reference_camera):
                raise ValueError(f"Mismatched legacy comparison scope: {clip_id}")
            old, old_margin = list(old_result.view_half_turns), old_result.margin
        clips[clip_id] = {**ranked, "i2363_turns": old, "i2363_margin": old_margin, "frames": source.num_frames,
                          "sampled": sampled, "detection_rows": [len(pose.source_rows) for pose in poses],
                          "max_rows_per_frame": [int(np.diff(pose.frame_offsets).max()) for pose in poses],
                          "old_alignment_present": "camera_alignment" in index["artifacts"],
                          "finalized_scene_present": "scene" in index["exports"],
                          "ball_points_present": any(f"ball_points/{camera}" in index["artifacts"] for camera in source.camera_ids)}
    shared = rank_hypotheses(camera_pairs, total_scores, total_edge_frames, total_possible, ball.decision.camera_ids,
                             ball.decision.reference_camera)
    return {"stride": stride, "camera_ids": ball.decision.camera_ids, "reference_camera": ball.decision.reference_camera,
            "matching_threshold_px_at_1080p": REPROJECTION_AT_1080P, "min_torso_joint_score": MIN_CONFIDENCE,
            "min_torso_joints": MIN_TORSO_JOINTS, "min_edge_frames": MIN_EDGE_FRAMES, "min_score": MIN_SCORE,
            "min_margin": MIN_MARGIN, "camera_pairs": [[ball.decision.camera_ids[i] for i in pair] for pair in camera_pairs],
            "source_inputs_digest": document_digest(input_pins),
            "ball_digest": ball.digest, "ball_court_digest": ball.decision.court_digest,
            "ball_drift_digest": ball.decision.court_drift_digest, "calibration_digest": ball.decision.calibration_digest,
            "ball_shared_turns": ball.output().view_half_turns, "person_shared": shared, "clips": clips,
            "wall_seconds": perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="The i2363 run's saved inputs/dataset directory")
    parser.add_argument("--artifact-root", type=Path, required=True, help="Root for the saved court-side stores")
    parser.add_argument("--ball-run", type=Path, required=True, help="Location of the #2241 ball side run")
    parser.add_argument("--stride", type=int, default=12, help="Sample every N source frames, starting at frame zero")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    print(json.dumps(run(args.dataset, args.artifact_root, args.ball_run, args.stride), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
