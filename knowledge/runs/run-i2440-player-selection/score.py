"""Score the selection masks on the reviewed Meiji 3cam development clips."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from inputs import (
    DEV_CLIPS,
    SELECTION,
    read_artifact,
    read_fields,
    source_from_index,
    write_json,
)
from numpy.typing import NDArray

from src.tasks.person_tracking.court_linking import select_linked_candidates
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.dataset_labels import label_path
from src.tasks.player_association.evaluation.labels import AMBIGUOUS, ClipLabels
from src.tasks.player_association.evaluation.metrics import (
    CameraPrediction,
    match_to_labels,
)
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tasks.player_association.review.reader import observation_location
from src.tennis_scene.generate_dataset.manifest import (
    ClipManifest,
    load_dataset_manifest,
)
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.checksum import dual_sha256

Masks = dict[str, dict[str, NDArray[np.bool_]]]


def raw_prediction(data: dict[str, Any]) -> CameraPrediction:
    return CameraPrediction(
        data["track_ids"], data["boxes_xyxy"].astype(np.float64), data["observed"],
        np.full(data["observed"].shape, -1, np.int64),
    )


def score_frames(
    labels: ClipLabels,
    raw: dict[str, CameraPrediction],
    masks: Masks,
    frames: tuple[int, ...],
) -> dict[str, dict[str, int]]:
    if not frames or tuple(sorted(set(frames))) != frames or frames[0] < 0 or frames[-1] >= labels.num_frames:
        raise ValueError("Scoring requires a nonempty ordered frame set on the label timeline")
    included = np.isin(np.arange(labels.num_frames), frames)
    predictions = {
        camera: replace(prediction, observed=prediction.observed & included)
        for camera, prediction in raw.items()
    }
    matching = match_to_labels(labels, predictions, min_iou=.5)
    player_roles = np.flatnonzero(labels.roles == "player")
    non_player_roles = np.flatnonzero(labels.roles == "non_player")
    per_condition = {
        name: Counter({"frames": len(frames), "player_boxes": 0, "non_player_boxes": 0,
                       "tracked_player": 0, "tracked_non_player": 0, "tp": 0, "fp": 0,
                       "fn": 0, "selected_ambiguous": 0, "selected_unmatched": 0})
        for name in masks
    }
    for camera, prediction in raw.items():
        camera_labels = labels.cameras[camera]
        people = camera_labels.person_index[included[camera_labels.frames]]
        positive = int(np.isin(people, player_roles).sum())
        negative = int(np.isin(people, non_player_roles).sum())
        matched = matching[camera]
        player = np.isin(matched, player_roles)
        non_player = np.isin(matched, non_player_roles)
        for name, selection in masks.items():
            mask = selection[camera]
            if mask.shape != prediction.observed.shape or mask.dtype != np.bool_ or (mask & ~prediction.observed).any():
                raise ValueError(f"{labels.clip_id}/{camera}: invalid selection mask for {name}")
            accepted = mask & included
            hits = int((accepted & player).sum())
            per_condition[name].update({
                "player_boxes": positive, "non_player_boxes": negative,
                "tracked_player": int(player.sum()), "tracked_non_player": int(non_player.sum()),
                "tp": hits, "fn": positive - hits, "fp": int((accepted & non_player).sum()),
                "selected_ambiguous": int((accepted & (matched == AMBIGUOUS)).sum()),
                "selected_unmatched": int((accepted & (matched == -2)).sum()),
            })
    return {name: dict(counts) for name, counts in per_condition.items()}


def add_rates(counts: Counter[str]) -> dict[str, float | int | None]:
    tp, fp, fn = (counts[key] for key in ("tp", "fp", "fn"))
    return {
        **counts,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def summarize(clips: dict[str, Any]) -> dict[str, dict[str, dict[str, float | int | None]]]:
    totals: dict[str, Counter[str]] = {}
    for clip in clips.values():
        for name, counts in clip["scores"]["all"].items():
            totals.setdefault(name, Counter()).update(counts)
    return {"all": {name: add_rates(counts) for name, counts in totals.items()}}


def singles(dataset: Path, artifact_root: Path) -> dict[str, Any]:
    manifest = load_dataset_manifest(dataset)
    if manifest.dataset_id != "meiji_3cam":
        raise ValueError("The singles comparison uses the Meiji 3cam dev references")
    clips = {}
    for clip_id in DEV_CLIPS:
        clip = ClipManifest.load(dataset / manifest.clips[clip_id].path)
        path = label_path(dataset, clip_id)
        labels = ClipLabels.load(path)
        review = path.with_name("review.yaml")
        if (
            labels.clip_id != clip_id
            or labels.num_frames != clip.num_frames
            or labels.provenance["review"]["sha256"] != dual_sha256(review)
        ):
            raise ValueError(f"{clip_id}: label timeline or review provenance changed")
        store_path, layout = observation_location(clip, artifact_root)
        if store_path is None or layout != "historical_observe_v1":
            raise ValueError(f"{clip_id}: the reviewed person_tracks v3 archive is required")
        store = StoreReader(store_path / "scene.json")
        source = source_from_index(store)
        if (source.clip_id != clip_id or source.camera_ids != clip.camera_ids or source.num_frames != clip.num_frames
                or source.size != (clip.width, clip.height) or abs(source.fps - clip.fps) > 1e-5):
            raise ValueError(f"{clip_id}: archive and labels have different recording grids")
        calibration = read_artifact(store, "court_calibration", "local_court_calibration", 1, CourtCalibrationOutput)
        cameras = {view.camera.camera_id: view.camera for view in calibration.calibration.views}
        raw = {}
        masks: Masks = {name: {} for name in SELECTION}
        for camera in source.camera_ids:
            raw[camera] = raw_prediction(read_fields(
                store, f"person_tracking/{camera}", "person_tracks", 3, "track_ids", "boxes_xyxy", "observed",
            ))
            pred = raw[camera]
            tracks = CameraTracks(cameras[camera], source.size, pred.track_ids, pred.boxes_xyxy, pred.observed)
            footpoints = FootpointConfig(4. * source.pixel_threshold_scale)
            for name, config in SELECTION.items():
                masks[name][camera], _ = select_linked_candidates(tracks, source.fps, config, footpoints)
        frames = tuple(sorted({int(frame) for camera in labels.cameras.values() for frame in camera.frames}))
        clips[clip_id] = {
            "scene_index_sha256": dual_sha256(store.index_path), "labels_sha256": dual_sha256(path),
            "scores": {"all": score_frames(labels, raw, masks, frames)},
        }
    return {"dataset": manifest.dataset_id, "summary": summarize(clips), "clips": clips}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dev = commands.add_parser("singles")
    dev.add_argument("dataset", type=Path)
    dev.add_argument("archive_outputs", type=Path)
    dev.add_argument("output", type=Path)
    args = parser.parse_args()
    write_json(args.output, singles(args.dataset, args.archive_outputs))


if __name__ == "__main__":
    main()
