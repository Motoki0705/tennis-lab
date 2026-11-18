"""Compare clip artifacts by source row, frame and matched identity on CPU."""
from __future__ import annotations

import argparse
import json
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from benchmark import CLIP_IDS, CONDITIONS
from PIL import Image
from scipy.optimize import linear_sum_assignment

from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.components.identity import PlayerIdentitiesOutput
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import PlayerSelectionOutput
from src.tennis_scene.pipeline.components.triangulation import PlayerTriangulationOutput
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video import read_video_frame


class Case:
    def __init__(self, path):
        self.path = path
        root = path / "store"
        document = json.loads((root / "scene.json").read_text())
        self.source = document["source"]
        self.videos = {video["camera_id"]: video for video in self.source["videos"]}
        self.store = ClipStore(root, self.source, memory_entries=2)
        self.nominal = json.loads((path / "nominal.json").read_text())
        diagnostic = path / "diagnostic.json"
        self.diagnostic = json.loads(diagnostic.read_text()) if diagnostic.exists() else None

    def load(self, name, output_type, *, downstream=False):
        measurements = self.diagnostic if downstream and self.diagnostic is not None else self.nominal
        node = measurements.get(name, {})
        if "reference" not in node:
            return None
        return self.store.load(ArtifactRef(**node["reference"]), ArtifactCodec(output_type))

    def poses(self, camera, detections):
        with np.load(self.path / f"pose-{camera}.npz", allow_pickle=False) as payload:
            if not np.array_equal(detections.source_rows, payload["rows"]):
                raise ValueError(f"{self.path}/{camera}: pose rows no longer align to DINO")
            return payload["keypoints"]

    @cached_property
    def association(self):
        return self.load("player_association", PlayerIdentitiesOutput, downstream=True)

    @cached_property
    def triangulation(self):
        return self.load("player_triangulation", PlayerTriangulationOutput, downstream=True)

    def timings(self):
        return {phase: {
            stage: {k: v for k, v in metrics.items() if k != "reference"}
            for stage, metrics in data.items()
        } for phase, data in (("nominal", self.nominal), ("diagnostic", self.diagnostic)) if data is not None}


def stats(values):
    if not values.size:
        return {"count": 0}
    return {"count": int(values.size), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99)), "max": float(values.max())}


def iou(a, b):
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    overlap = np.prod(np.maximum(0, bottom_right - top_left), axis=-1)
    union = np.prod(a[:, 2:] - a[:, :2], axis=-1)[:, None] + np.prod(b[:, 2:] - b[:, :2], axis=-1)[None, :] - overlap
    return np.divide(overlap, union, where=union > 0, out=np.zeros_like(overlap))


def match_detections(base, candidate):
    paired_base, paired_candidate, overlaps = [], [], []
    frames_with_count_change = 0
    for l0, l1, r0, r1 in zip(base.frame_offsets[:-1], base.frame_offsets[1:],
                             candidate.frame_offsets[:-1], candidate.frame_offsets[1:], strict=True):
        left, right = base.boxes_xyxy[l0:l1], candidate.boxes_xyxy[r0:r1]
        frames_with_count_change += len(left) != len(right)
        overlap = iou(left, right)
        if np.array_equal(left, right):
            i = j = np.arange(len(left))
        else:
            i, j = linear_sum_assignment(np.where(overlap >= .5, 1 - overlap, 1000))
        good = overlap[i, j] >= .5
        paired_base.append(l0 + i[good])
        paired_candidate.append(r0 + j[good])
        overlaps.append(overlap[i[good], j[good]])
    i, j, overlaps = (np.concatenate(axis) for axis in (paired_base, paired_candidate, overlaps))
    report = {
        "fp32_rows": len(base.confidence), "candidate_rows": len(candidate.confidence),
        "fp32_unmatched": len(base.confidence) - len(i), "candidate_unmatched": len(candidate.confidence) - len(j),
        "frames_with_count_change": frames_with_count_change, "matched_iou": stats(overlaps),
        "matched_box_abs_px": stats(np.abs(base.boxes_xyxy[i] - candidate.boxes_xyxy[j])),
        "matched_score_abs": stats(np.abs(base.confidence[i] - candidate.confidence[j])),
        "exact_ordered_output": (np.array_equal(base.frame_offsets, candidate.frame_offsets)
                                 and np.array_equal(base.boxes_xyxy, candidate.boxes_xyxy)
                                 and np.array_equal(base.confidence, candidate.confidence)),
    }
    return i, j, report


def labels_by_detection(detections, rows, labels):
    source_rows = detections.source_rows
    if source_rows is None:
        raise ValueError("Require detector source rows")
    keep = (rows >= 0) & (labels >= 0)
    matched_rows = rows[keep]
    index = np.searchsorted(source_rows, matched_rows)
    if (index >= len(source_rows)).any() or not np.array_equal(source_rows[index], matched_rows) or len(np.unique(index)) != len(index):
        raise ValueError("A tracked detection row is missing or duplicated")
    result = np.full(len(source_rows), -1, dtype=np.int64)
    result[index] = labels[keep]
    return result


def compare_labels(left, right):
    a, b = left >= 0, right >= 0
    common = a & b
    left_ids, rows = np.unique(left[common], return_inverse=True)
    right_ids, columns = np.unique(right[common], return_inverse=True)
    counts = np.zeros((len(left_ids), len(right_ids)), dtype=np.int64)
    np.add.at(counts, (rows, columns), 1)
    i, j = linear_sum_assignment(counts, maximize=True)
    matched = counts[i, j] > 0
    mapping = {int(left_ids[x]): int(right_ids[y]) for x, y in zip(i[matched], j[matched], strict=True)}
    return {"jointly_assigned": int(common.sum()), "lost_assignment": int((a & ~b).sum()),
            "gained_assignment": int((~a & b).sum()), "numeric_id_changes": int((common & (left != right)).sum()),
            "assignment_changes_after_best_id_mapping": int(common.sum() - counts[i, j].sum()),
            "mapping_fp32_to_candidate": mapping}


def association_labels(case, camera, detections, selection):
    result = case.association
    if result is None:
        return None
    if camera not in result.camera_ids:
        return np.full(len(detections.confidence), -1, np.int64)
    view = result.camera_ids.index(camera)
    tracks = selection.tracks
    count = len(tracks.track_ids)
    if not np.array_equal(result.local_track_ids[view, :count], tracks.track_ids):
        raise ValueError("Association output disagrees with the selected track group axis")
    return labels_by_detection(detections, tracks.evidence.detection_rows, result.player_ids[view, :count])


def overlay(frame, selection, index, color, *, thickness, threshold):
    tracks = selection.tracks
    for row, points in enumerate(tracks.evidence.require_poses()[:, index]):
        if not tracks.observed[row, index]:
            continue
        p0, p1 = np.rint(tracks.boxes_xyxy[row, index].reshape(2, 2)).astype(int)
        cv2.rectangle(frame, tuple(p0), tuple(p1), color, thickness, cv2.LINE_AA)
        locations = np.rint(points[:, :2]).astype(int)
        visible = points[:, 2] >= threshold
        for a, b in COCO17_SKELETON:
            if visible[a] and visible[b]:
                cv2.line(frame, tuple(locations[a]), tuple(locations[b]), color, thickness, cv2.LINE_AA)
        for joint in locations[visible]:
            cv2.circle(frame, tuple(joint), thickness, color, -1, cv2.LINE_AA)


def compare_camera(base, candidate, camera, threshold, overlay_image):
    detection_a, detection_b = (
        case.load(f"person_detection/{camera}", PersonDetectionOutput) for case in (base, candidate)
    )
    track_a, track_b = (case.load(f"person_tracking/{camera}", PersonTrackingOutput) for case in (base, candidate))
    select_a, select_b = (case.load(f"player_selection/{camera}", PlayerSelectionOutput) for case in (base, candidate))
    i, j, detection = match_detections(detection_a, detection_b)
    poses_a, poses_b = base.poses(camera, detection_a)[i], candidate.poses(camera, detection_b)[j]
    pose_delta = np.linalg.norm(poses_a[..., :2] - poses_b[..., :2], axis=-1)
    raw_a, raw_b = [labels_by_detection(d, t.evidence.detection_rows,
                                       np.broadcast_to(t.track_ids[:, None], t.observed.shape))
                    for d, t in ((detection_a, track_a), (detection_b, track_b))]
    sel_a, sel_b = [
        labels_by_detection(d, np.where(s.selected, t.evidence.detection_rows, -1),
                            np.broadcast_to(t.track_ids[:, None], t.observed.shape))
        for d, t, s in ((detection_a, track_a, select_a), (detection_b, track_b, select_b))
    ]
    grouped_a, grouped_b = [
        labels_by_detection(d, s.tracks.evidence.detection_rows,
                            np.broadcast_to(s.tracks.track_ids[:, None], s.tracks.observed.shape))
        for d, s in ((detection_a, select_a), (detection_b, select_b))
    ]
    labelled_a = association_labels(base, camera, detection_a, select_a)
    labelled_b = association_labels(candidate, camera, detection_b, select_b)
    association = (labelled_a[i], labelled_b[j]) if labelled_a is not None and labelled_b is not None else None
    selected_both = (grouped_a[i] >= 0) & (grouped_b[j] >= 0)
    common_visible = (poses_a[..., 2] >= threshold) & (poses_b[..., 2] >= threshold)
    image = None
    if overlay_image:
        video = base.videos[camera]
        index = video["num_frames"] // 2
        path = Path(video["path"])
        if dual_sha256(path) != video["sha256"]:
            raise ValueError(f"Overlay video changed: {path}")
        frame = read_video_frame(path, index).frame
        frame = cv2.convertScaleAbs(frame, alpha=.72)
        overlay(frame, select_a, index, (255, 230, 70), thickness=6, threshold=threshold)
        overlay(frame, select_b, index, (90, 160, 255), thickness=3, threshold=threshold)
        image = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(image, f"{camera}  frame {index}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return ({
        "dino": detection,
        "vitpose": {"xy_distance_px_all": stats(pose_delta), "xy_distance_px_both_visible": stats(pose_delta[common_visible]),
                    "xy_distance_px_selected_both_visible": stats(pose_delta[selected_both[:, None] & common_visible]),
                    "confidence_abs": stats(np.abs(poses_a[..., 2] - poses_b[..., 2])),
                    "threshold_changes": int(((poses_a[..., 2] >= threshold) != (poses_b[..., 2] >= threshold)).sum())},
        "tracking": {"fp32_tracks": len(track_a.track_ids), "candidate_tracks": len(track_b.track_ids),
                     "fp32_observations": int(track_a.observed.sum()), "candidate_observations": int(track_b.observed.sum()),
                     "fp32_unmatched_tracked_rows": int((raw_a >= 0).sum() - (raw_a[i] >= 0).sum()),
                     "candidate_unmatched_tracked_rows": int((raw_b >= 0).sum() - (raw_b[j] >= 0).sum()),
                     **compare_labels(raw_a[i], raw_b[j])},
        "selection": {"fp32_selected_detection_rows": int((sel_a >= 0).sum()), "candidate_selected_detection_rows": int((sel_b >= 0).sum()),
                      "fp32_group_detection_rows": int((grouped_a >= 0).sum()), "candidate_group_detection_rows": int((grouped_b >= 0).sum()),
                      "fp32_group_count": len(select_a.tracks.track_ids), "candidate_group_count": len(select_b.tracks.track_ids),
                      "lost_selection": int(((sel_a[i] >= 0) & (sel_b[j] < 0)).sum()),
                      "gained_selection": int(((sel_a[i] < 0) & (sel_b[j] >= 0)).sum()),
                      "groups": compare_labels(grouped_a[i], grouped_b[j])},
    }, association, image)


def skeleton_identity_axis(case, threshold):
    identity = case.association
    kept = []
    for view, camera in enumerate(identity.camera_ids):
        pose = case.load(f"pose_estimation/{camera}", ObjectObservations)
        visibility = pose.visibility(threshold)[0].any(axis=-1).T
        labels = identity.player_ids[view, :len(visibility)][visibility]
        kept.append(labels[labels >= 0])
    return np.unique(np.concatenate(kept))


def triangulation(base, candidate, mapping, threshold):
    a, b = base.triangulation, candidate.triangulation
    if a is None or b is None or a.skeleton is None or b.skeleton is None:
        return None
    a, b = a.skeleton, b.skeleton
    ids_a, ids_b = [skeleton_identity_axis(case, threshold) for case in (base, candidate)]
    if len(ids_a) != len(a.valid) or len(ids_b) != len(b.valid):
        raise ValueError("Triangulated joint axes disagree with the associated IDs")
    paired = [(a, b) for a, b in mapping.items() if a in ids_a and b in ids_b]
    i = np.array([np.searchsorted(ids_a, key) for key, _ in paired], dtype=int)
    j = np.array([np.searchsorted(ids_b, value) for _, value in paired], dtype=int)
    valid_a, valid_b = a.valid[i], b.valid[j]
    both = valid_a & valid_b
    return {
        "fp32_valid_joints": int(a.valid.sum()), "candidate_valid_joints": int(b.valid.sum()),
        "jointly_valid": int(both.sum()), "lost_valid_joints": int(a.valid.sum() - both.sum()),
        "gained_valid_joints": int(b.valid.sum() - both.sum()),
        "reasons_changed_on_matched_players": int((a.reasons[i] != b.reasons[j]).sum()),
        "3d_distance_m": stats(np.linalg.norm(a.positions[i][both] - b.positions[j][both], axis=-1)),
    }


def contact_sheet(images, output, condition):
    result = np.concatenate([np.concatenate(images[:2], axis=1), np.concatenate(images[2:], axis=1)], axis=0)
    result = cv2.copyMakeBorder(result, 42, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(result, "FP32", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 230, 70), 2)
    cv2.putText(result, condition, (130, 28), cv2.FONT_HERSHEY_SIMPLEX, .8, (90, 160, 255), 2)
    cv2.putText(result, "Meiji clip_037  |  selected groups  |  COCO17", (570, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .72, (230, 230, 230), 1)
    Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).quantize(colors=96).save(output, optimize=True)


def compare_clip(base, candidate, image_path, condition, joint_confidence, human_visibility_threshold):
    if base.source != candidate.source:
        raise ValueError("Conditions must have exactly the same source")
    reports, association, images = {}, [], []
    for camera in base.videos:
        reports[camera], labels, frame = compare_camera(base, candidate, camera, joint_confidence, image_path is not None)
        if labels is not None:
            association.append(labels)
        if frame is not None:
            images.append(frame)
    if image_path is not None:
        contact_sheet(images, image_path, condition)
    result = {"cameras": reports, "association": None, "triangulation": None,
              "fp32_downstream": "diagnostic" if base.diagnostic is not None else "nominal",
              "candidate_downstream": "diagnostic" if candidate.diagnostic is not None else "nominal"}
    if association:
        comparison = compare_labels(*(np.concatenate(labels) for labels in zip(*association, strict=True)))
        result["association"] = comparison
        result["triangulation"] = triangulation(
            base, candidate, comparison["mapping_fp32_to_candidate"], human_visibility_threshold,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--pin", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pin = json.loads(args.pin.read_text())
    write_json_atomic(args.output / "comparison-method.json", {
        "detection": "per-frame one-to-one maximum matches at IoU>=0.5, then maximize IoU",
        "pose": "2D Euclidean distance in source pixels over matched detection rows; raw heatmap peak confidence",
        "joint_confidence": pin["joint_confidence"],
        "track_group_and_association": "matched detection rows; numeric IDs and best overlap permutation",
        "skeleton": "court-frame distance in metres for paired players and jointly valid joints",
    })
    for clip_id in CLIP_IDS:
        clip = clip_id.split("/")[-1]
        base_path = args.run / "fp32" / clip
        if not (base_path / "complete.json").exists():
            continue
        base = Case(base_path)
        write_json_atomic(args.output / f"{clip}-fp32.json", {"timing": base.timings()})
        for condition in CONDITIONS:
            if condition == "fp32":
                continue
            path = args.run / condition / clip
            if not (path / "complete.json").exists():
                continue
            case = Case(path)
            image = args.output / f"{clip}-{condition}.png" if clip == "clip_037" else None
            result = compare_clip(base, case, image, condition, pin["joint_confidence"], pin["human_visibility_threshold"])
            result["timing"] = case.timings()
            write_json_atomic(args.output / f"{clip}-{condition}.json", result)


if __name__ == "__main__":
    main()
