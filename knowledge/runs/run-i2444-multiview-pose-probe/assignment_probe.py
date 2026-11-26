"""Compare frame-level greedy and MILP row assignment on the four labelled 3cam dev clips."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray
from probe import (
    REPROJECTION_AT_1080P,
    pair_weights,
    read_json,
    read_output,
    torso_observations,
)
from scipy.optimize import Bounds, LinearConstraint, milp

from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tennis_scene.pipeline.artifacts import document_digest
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.storage.codec import restore_type
from src.utils.checksum import dual_sha256

DEV_CLIPS = ("video_000/clip_000", "video_000/clip_007", "video_001/clip_001", "video_002/clip_013")
R7 = Path("person_tracking/evaluate/dev_features/i964-features-r7-20260930")
R6 = Path("person_tracking/evaluate/court_selection/i964-fullframe-r6-20260929/sources/coco_fullframe_0.30")
OBSERVE = Path("player_association/evaluate/meiji_clips/i933-observe-v1-20260927/stores")
SIDE_REPORT = Path("court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json")
R7_EVIDENCE = Path("knowledge/runs/run-i964-coco-person-features-r7-20260930")
ENCODER_ARCHIVE = "clipreid_vitb16_market1501"


@dataclass(frozen=True)
class Hypothesis:
    rows: tuple[tuple[int, int], ...]  # camera index and frame-local row position
    score: float


def read_all_pose_rows(
    archives: Path, clip_id: str, video: SourceVideo, plan: dict[str, Any], features: dict[str, Any], plan_sha: str,
) -> tuple[DetectionArchive, NDArray[np.float32]]:
    key = f"{clip_id}/{video.camera_id}"
    inputs = {item["key"]: item for item in plan["inputs"]}[key]
    record = features["records"][ENCODER_ARCHIVE][key]
    path = archives / R7 / ENCODER_ARCHIVE / clip_id / f"{video.camera_id}.features.npz"
    if dual_sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Archived pose checksum mismatch: {key}")
    detection = DetectionArchive.load({**inputs["detection"], "path": archives / R6 / clip_id / f"{video.camera_id}.npz"})
    with np.load(path, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"]))
        poses = artifact["poses"]
        provenance = metadata["provenance"]
        if (metadata.get("schema") != "person_detection_features_v2" or provenance["source"] != inputs["video"]
                or document_digest(video) != document_digest(restore_type(inputs["video"], SourceVideo))
                or provenance["plan_sha256"] != plan_sha or provenance["detection"] != inputs["detection"]
                or provenance["models"]["pose"] != plan["models"]["pose"]
                or not np.array_equal(artifact["rows"], np.arange(len(detection.boxes)))
                or not np.array_equal(artifact["boxes"], detection.boxes)
                or not np.array_equal(artifact["scores"], detection.scores)
                or not np.array_equal(artifact["offsets"], detection.offsets)
                or len(detection.offsets) != video.num_frames + 1
                or len(poses) != record["detections"] or poses.shape != (len(detection.boxes), 17, 3)
                or poses.dtype != np.float32 or not np.isfinite(poses).all()):
            raise ValueError(f"Not every raw detection row has its own pose: {key}")
    return detection, poses


def hypotheses(weights: dict[tuple[int, int], NDArray[np.float64]]) -> list[Hypothesis]:
    result = [Hypothesis(((a, int(left)), (b, int(right))), float(matrix[left, right]))
              for (a, b), matrix in weights.items() for left, right in np.argwhere(matrix > 0)]
    first, second, third = weights[(0, 1)], weights[(0, 2)], weights[(1, 2)]
    for left, middle in np.argwhere(first > 0):
        for right in np.flatnonzero((second[left] > 0) & (third[middle] > 0)):
            weight = first[left, middle] + second[left, right] + third[middle, right]
            result.append(Hypothesis(((0, int(left)), (1, int(middle)), (2, int(right))), float(weight)))
    return result


def greedy_assignment(candidates: list[Hypothesis]) -> list[Hypothesis]:
    used: set[tuple[int, int]] = set()
    selected = []
    for candidate in sorted(candidates, key=lambda h: (-h.score, h.rows)):
        if used.isdisjoint(candidate.rows):
            selected.append(candidate)
            used.update(candidate.rows)
    return selected


def milp_assignment(candidates: list[Hypothesis]) -> list[Hypothesis]:
    if not candidates:
        return []
    rows = {row: i for i, row in enumerate(sorted({row for h in candidates for row in h.rows}))}
    usage: NDArray[np.float64] = np.zeros((len(rows), len(candidates)), dtype=np.float64)
    for column, candidate in enumerate(candidates):
        for row in candidate.rows:
            usage[rows[row], column] = 1
    result = milp(-np.array([h.score for h in candidates]), integrality=np.ones(len(candidates), np.int32),
                  bounds=Bounds(0, 1), constraints=LinearConstraint(usage, 0, 1), options={"mip_rel_gap": 0.0})
    if not result.success or result.x is None or not np.allclose(result.x, np.round(result.x), atol=1e-6):
        raise ValueError(f"Frame MILP did not return an integer optimum: {result.message}")
    return [h for h, selected in zip(candidates, result.x, strict=True) if selected > 0.5]


def player_only(labels: ClipLabels) -> ClipLabels:
    non_players = np.flatnonzero(labels.roles == "non_player")
    cameras = {camera: replace(view, person_index=np.where(np.isin(view.person_index, non_players), AMBIGUOUS, view.person_index))
               for camera, view in labels.cameras.items()}
    return replace(labels, cameras=cameras)


def evaluate_clip(
    source: ClipSource, calibration: CourtCalibrationOutput, turns: list[bool],
    inputs: list[tuple[DetectionArchive, NDArray[np.float32]]], labels_path: Path,
) -> dict[str, Any]:
    cameras = [view.camera.half_turned(turn) for view, turn in zip(calibration.calibration.views, turns, strict=True)]
    data = [torso_observations(poses, camera, source.size) for (_, poses), camera in zip(inputs, cameras, strict=True)]
    predictions: dict[str, dict[str, CameraPrediction]] = {"greedy": {}, "milp": {}}
    for camera, (det, _) in zip(source.camera_ids, inputs, strict=True):
        rows = int(np.diff(det.offsets).max())
        observed = np.arange(rows)[:, None] < np.diff(det.offsets)
        boxes = np.zeros((*observed.shape, 4), np.float64)
        boxes.transpose(1, 0, 2)[observed.T] = det.boxes
        for mode in predictions:
            predictions[mode][camera] = CameraPrediction(np.arange(rows, dtype=np.int64), boxes, observed,
                                                          np.full(observed.shape, -1, np.int64))
    threshold_px = REPROJECTION_AT_1080P * source.pixel_threshold_scale
    candidate_counts = []
    changed_frames = higher_score_frames = 0
    timings: dict[str, list[float]] = {"greedy": [], "milp": []}
    for frame in range(source.num_frames):
        uv_at_frame = []
        for (det, _), (uv, visible, rays) in zip(inputs, data, strict=True):
            start, end = det.offsets[frame:frame + 2]
            uv_at_frame.append((uv[start:end], visible[start:end], rays[start:end]))
        weights = {(a, b): pair_weights(uv_at_frame[a], uv_at_frame[b], cameras[a], cameras[b], threshold_px)
                   for a, b in ((0, 1), (0, 2), (1, 2))}
        candidates = hypotheses(weights)
        candidate_counts.append(len(candidates))
        selected = {}
        for mode, solve in (("greedy", greedy_assignment), ("milp", milp_assignment)):
            started = perf_counter()
            selected[mode] = solve(candidates)
            timings[mode].append((perf_counter() - started) * 1e3)
            for local_id, candidate in enumerate(selected[mode]):
                for camera_idx, row in candidate.rows:
                    camera = source.camera_ids[camera_idx]
                    ids = predictions[mode][camera].player_ids
                    if ids[row, frame] >= 0 or not predictions[mode][camera].observed[row, frame]:
                        raise ValueError(f"Invalid selected hypothesis: {source.clip_id} {frame} {camera} {row}")
                    ids[row, frame] = local_id
        gain = sum(h.score for h in selected["milp"]) - sum(h.score for h in selected["greedy"])
        if gain < -1e-6:
            raise ValueError(f"MILP scored below greedy: {source.clip_id} {frame}")
        higher_score_frames += gain > 1e-6
        changed_frames += {h.rows for h in selected["milp"]} != {h.rows for h in selected["greedy"]}
    labels = ClipLabels.load(labels_path)
    if labels.clip_id != source.clip_id or labels.num_frames != source.num_frames:
        raise ValueError(f"Wrong evaluation labels: {labels_path}")
    scores = {}
    for mode, prediction in predictions.items():
        legacy = evaluate(labels, prediction)
        scores[mode] = {"pairs_legacy": legacy["pairs"], "coverage": legacy["coverage"],
                        "pairs_player_only": evaluate(player_only(labels), prediction)["pairs"],
                        "solver_ms_p50": float(np.median(timings[mode])), "solver_ms_p95": float(np.percentile(timings[mode], 95))}
    return {"scores": scores, "source_frames": source.num_frames, "camera_rows": [len(det.scores) for det, _ in inputs],
            "label_sha256": dual_sha256(labels_path), "turns": turns, "changed_frames": changed_frames,
            "higher_geometry_score_frames": higher_score_frames, "max_candidates": max(candidate_counts),
            "p95_candidates": float(np.percentile(candidate_counts, 95)), "empty_frames": candidate_counts.count(0)}


def run(dataset: Path, archives: Path) -> dict[str, Any]:
    started = perf_counter()
    plan = read_json(R7_EVIDENCE / "plan.json")
    features = read_json(R7_EVIDENCE / "features.json")
    plan_sha = dual_sha256(R7_EVIDENCE / "plan.json")
    side_sha = dual_sha256(archives / SIDE_REPORT)
    if plan["source_name"] != "coco_fullframe_0.30" or plan_sha != features["plan_sha256"]:
        raise ValueError("Expected the saved r7 all-person feature plan")
    side = {entry["clip_id"]: entry for entry in read_json(archives / SIDE_REPORT)["clips"]}
    clips = {}
    for clip_id in DEV_CLIPS:
        directory = (archives / OBSERVE / clip_id).resolve()
        index = read_json(directory / "scene.json")
        source = restore_type(index["source"], ClipSource)
        calibration, _ = read_output(directory, index, "court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
        reference = side[clip_id]
        if (source.clip_id != clip_id or document_digest(source) != index["source_sha256"]
                or tuple(reference["camera_ids"]) != source.camera_ids or len(source.camera_ids) != 3
                or calibration.calibration.camera_ids != source.camera_ids or calibration.calibration.excluded
                or calibration.reference_camera != reference["reference_camera"] or not reference["annotation"]["decided"]):
            raise ValueError(f"Missing scoped court and reviewed-ball side: {clip_id}")
        inputs = [read_all_pose_rows(archives, clip_id, video, plan, features, plan_sha) for video in source.videos]
        for video in source.videos:
            path = label_path(dataset, clip_id).parents[2] / "media" / video.path.name
            if dual_sha256(path) != video.sha256:
                raise ValueError(f"Label dataset video differs from the observations: {path}")
        clips[clip_id] = {"calibration_artifact_id": index["artifacts"]["court_calibration"]["artifact_id"],
                          **evaluate_clip(source, calibration, reference["annotation"]["view_half_turns"], inputs, label_path(dataset, clip_id))}
    return {"r7_plan_sha256": plan_sha, "r7_features_sha256": dual_sha256(R7_EVIDENCE / "features.json"),
            "side_report_sha256": side_sha, "clips": clips, "wall_seconds": perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="The meiji_3cam dataset directory with player labels")
    parser.add_argument("--archives", required=True, type=Path, help="The archived output root containing the #932, #933, and #964 runs")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.archives), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
