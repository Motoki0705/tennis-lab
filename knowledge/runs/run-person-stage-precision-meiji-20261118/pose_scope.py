"""Replay camera-local tracking with pose weight zero on saved FP32 CLIP rows."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from benchmark import config, prepare_cases
from compare import Case, compare_labels, labels_by_detection

from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.sequence import DatasetTrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.court_side import CourtSideOutput
from src.tennis_scene.pipeline.components.identity import PlayerAssociationInput
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.components.player_selection import (
    PlayerSelectionInput,
    PlayerSelectionOutput,
)
from src.tennis_scene.pipeline.definition import standard_definition
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable


def features(directory, camera, detections):
    with np.load(directory / f"appearance-{camera}.npz", allow_pickle=False) as payload:
        rows, embeddings, valid = payload["rows"], payload["embeddings"], payload["valid"]
    if not np.array_equal(rows, detections.source_rows):
        raise ValueError("CLIP rows must exactly match FP32 DINO source rows")
    for frame, (start, stop) in enumerate(zip(detections.frame_offsets[:-1], detections.frame_offsets[1:], strict=True)):
        item = slice(start, stop)
        yield DetectionFeatures(frame, rows[item], detections.boxes_xyxy[item], detections.confidence[item],
                                None, embeddings[item], valid[item])


def tracking_labels(detections, tracks, keep):
    rows = np.where(keep, tracks.evidence.detection_rows, -1)
    return labels_by_detection(detections, rows, np.broadcast_to(tracks.track_ids[:, None], rows.shape))


def association_labels(identities, case_selection, detections):
    assigned = []
    for view, camera in enumerate(identities.camera_ids):
        tracks = case_selection[camera].tracks
        count = len(tracks.track_ids)
        if not np.array_equal(identities.local_track_ids[view, :count], tracks.track_ids):
            raise ValueError("Camera association and selected groups disagree")
        assigned.append(labels_by_detection(detections[camera], tracks.evidence.detection_rows,
                                             identities.player_ids[view, :count]))
    return np.concatenate(assigned)


def replay(base, source, cfg, nodes):
    result = {"cameras": {}}
    calibration = base.load("court_calibration", CourtCalibrationOutput)
    selections = {}
    baseline_selections = {}
    detections_by_camera = {}
    for video in source.videos:
        camera = video.camera_id
        detections = base.load(f"person_detection/{camera}", PersonDetectionOutput)
        baseline_tracks = base.load(f"person_tracking/{camera}", PersonTrackingOutput)
        baseline_selection = base.load(f"player_selection/{camera}", PlayerSelectionOutput)
        tracked = track_sequence(features(base.path, camera, detections), fps=video.fps,
                                 config=DatasetTrackingConfig(features=cfg.tracking.features),
                                 aflink=AFLink(cfg.aflink_checkpoint))
        tracks = PersonTrackingOutput(camera, tracked.track_ids, tracked.boxes, tracked.observed,
                                      tracked.source_track_ids, (), tracked.evidence, tracked.reconstruction,
                                      tracked.link_candidates)
        selection = nodes[f"player_selection/{camera}"].process(PlayerSelectionInput(video, calibration, tracks))
        selections[camera] = selection
        baseline_selections[camera] = baseline_selection
        detections_by_camera[camera] = detections
        left = tracking_labels(detections, baseline_tracks, baseline_selection.selected)
        right = tracking_labels(detections, tracks, selection.selected)
        result["cameras"][camera] = {
            "fp32_pose_input_rows": len(detections.confidence),
            "fp32_selected_group_rows": int(baseline_selection.tracks.observed.sum()),
            "appearance_only_selected_group_rows": int(selection.tracks.observed.sum()),
            "selected_lost": int(((left >= 0) & (right < 0)).sum()),
            "selected_gained": int(((left < 0) & (right >= 0)).sum()),
            "tracking": compare_labels(tracking_labels(detections, baseline_tracks, baseline_tracks.observed),
                                       tracking_labels(detections, tracks, tracks.observed)),
        }
    side = base.load("court_side", CourtSideOutput, downstream=True)
    result["association"] = {"status": "blocked", "reason": "court_side_unavailable"}
    if side is not None:
        try:
            output = nodes["player_association"].process(PlayerAssociationInput(
                source, calibration, side, tuple(selections[c].tracks for c in source.camera_ids),
            ))
        except ReconstructionUnavailable as error:
            result["association"] = {"status": "unavailable", "reason": error.reason, "message": str(error)}
        else:
            result["association"] = {"status": "executed", "comparison": None}
            if base.association is not None:
                result["association"]["comparison"] = compare_labels(
                    association_labels(base.association, baseline_selections, detections_by_camera),
                    association_labels(output, selections, detections_by_camera),
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--shared-root", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.dataset, args.shared_root, args.output = args.dataset.resolve(), args.shared_root.resolve(), args.output.resolve()
    _, clips, identity, shared = prepare_cases(args)
    cfg = replace(config(args, "fp32"), device="cpu")
    result = {}
    for clip_id, (_, inputs) in clips.items():
        path = args.run / "fp32" / clip_id.split("/")[-1]
        if not (path / "complete.json").exists():
            continue
        source = inputs.source
        nodes = {node.name: node.component for node in standard_definition(
            cfg, source, code_identity=identity, video_court=shared,
        )}
        result[clip_id] = replay(Case(path), source, cfg, nodes)
        write_json_atomic(args.output / "pose-scope.json", result)


if __name__ == "__main__":
    main()
