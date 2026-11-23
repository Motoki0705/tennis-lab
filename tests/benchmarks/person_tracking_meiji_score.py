"""Apply the common metrics to saved and replayed Meiji tracking routes."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import person_tracking_meiji_features as appearance  # type: ignore[import-not-found]
from person_tracking_meiji_inputs import (  # type: ignore[import-not-found]
    CLIP,
    load_detections,
)
from person_tracking_meiji_replay import (  # type: ignore[import-not-found]
    DEFAULT,
    HYBRID,
    read_tracks,
    track_path,
)

from src.tasks.person_tracking.comparison import TrackingPrediction
from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.feature_tracks import sampled_appearance
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.evaluation.labels import PSEUDO_PROVENANCE, ClipLabels
from src.tasks.player_association.pseudo_annotation.frozen_run import (
    FrozenClip,
    FrozenRun,
)
from src.tasks.player_association.pseudo_annotation.sampling import DOUBT, STRATA
from src.tennis_scene.pipeline.components.player_reconstruction import (
    triangulate_players,
)
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.utils.checksum import dual_sha256
from src.utils.geometry.triangulation import PinholeCamera

ASSOCIATION_CONFIG = Path(__file__).resolve().parents[2] / "src/tasks/player_association/configs/association.yaml"
SCORE_STRATA = ("all", "doubt_any", *STRATA)


def label_strata(
    labels: ClipLabels, clip: FrozenClip, frozen: FrozenRun, plan: dict[str, Any], *, require_full: bool,
) -> dict[str, list[int]]:
    frames = labels.annotated_frames
    record = plan["clips"][clip.clip.clip_id]
    meta = labels.provenance[PSEUDO_PROVENANCE]
    if (labels.clip_id != clip.clip.clip_id or labels.num_frames != clip.clip.num_frames
            or set(labels.cameras) != set(clip.clip.camera_ids)
            or meta["input_snapshot_sha256"] != frozen.sha256 or meta["show_prior_association"]
            or frames is None or not set(frames).issubset(record["sampled_frames"])
            or (require_full and list(frames) != record["sampled_frames"])):
        raise ValueError(f"Labels differ from the pinned frame plan for {clip.clip.clip_id}")
    for frame in frames:
        expected = {name for name in STRATA if frame in record["sampled"][name]}
        if not expected or set(meta["frame_strata"][str(frame)]) != expected:
            raise ValueError(f"Label stratum mismatch at {clip.clip.clip_id}/{frame}")
    for camera, labelled in labels.cameras.items():
        offsets, boxes, _ = clip.detections(camera)
        for frame in frames:
            observed = labelled.boxes_xyxy[labelled.at(frame)]
            source = boxes[offsets[frame]:offsets[frame + 1]].astype(np.float64)
            np.testing.assert_array_equal(observed, np.round(source, 1))
    strata = {name: [frame for frame in frames if frame in record["sampled"][name]] for name in STRATA}
    return dict(all=list(frames), doubt_any=sorted({f for s in DOUBT for f in strata[s]}), **strata)


def _from_group_ids(raw: CameraTracks, origins: np.ndarray, group_ids: np.ndarray) -> np.ndarray:
    if (origins.shape != group_ids.shape or (origins < -1).any() or (origins >= len(raw.track_ids)).any()
            or ((group_ids >= 0) & (origins < 0)).any()):
        raise ValueError("Invalid group origin or assignment on the raw track grid")
    result: np.ndarray = np.full(raw.observed.shape, -1, np.int64)
    group, frame = np.nonzero(group_ids >= 0)
    source = origins[group, frame]
    if len(set(zip(source, frame, strict=True))) != len(source):
        raise ValueError("A raw box cannot carry multiple group assignments")
    result[source, frame] = group_ids[group, frame]
    return result


def baseline(
    clip: FrozenClip, local: dict[str, PinholeCamera],
) -> tuple[TrackingPrediction, dict[str, Any]]:
    size = (clip.clip.width, clip.clip.height)
    result = clip.assignments()
    raw, groups, selected, ids = {}, {}, {}, {}
    for name, camera in local.items():
        value, selection = clip.raw(name), clip.selection(name, origins=True)
        track = CameraTracks(camera, size, value["track_ids"], value["boxes_xyxy"], value["observed"])
        np.testing.assert_array_equal(selection["raw_track_ids"], track.track_ids)
        local_ids, observed = selection["tracks.track_ids"], selection["tracks.observed"]
        origins = selection["origin_rows"]
        if (origins.shape != observed.shape or not np.array_equal(origins >= 0, observed)
                or (origins >= len(track.track_ids)).any()):
            raise ValueError(f"{name} has invalid saved group origins")
        group_boxes = np.zeros((*observed.shape, 4), dtype=np.float32)
        group_rows, frames = np.nonzero(observed)
        group_boxes[group_rows, frames] = track.boxes_xyxy[origins[group_rows, frames], frames]
        groups[name] = CameraTracks(camera, size, local_ids, group_boxes, observed)
        assigned = {} if result is None else result.get(name, {})
        unknown = set(assigned) - set(local_ids.tolist())
        if unknown:
            raise ValueError(f"{name} has assignments to absent groups: {sorted(unknown)}")
        player_ids = np.full(observed.shape, -1, np.int64)
        for row, identity in enumerate(local_ids):
            if identity in assigned:
                player_ids[row] = assigned[identity]
        ids[name] = _from_group_ids(track, origins, player_ids)
        raw[name], selected[name] = track, selection["selected"]
    valid = {}
    stop = None
    if clip.outcome["status"] != "generated":
        stop = clip.outcome["reason"]
    if "player_triangulation" in clip.store.references:
        skeleton = clip.read("player_triangulation", "player_skeletons", 1, "skeleton.valid")["skeleton.valid"]
        people = sorted({int(p) for values in ids.values() for p in np.unique(values) if p >= 0})
        valid = dict(zip(people, skeleton.any(-1), strict=True))
    elif stop is None:
        raise ValueError(f"Generated clip lacks triangulation: {clip.clip.clip_id}")
    return TrackingPrediction(raw, selected, groups, ids, valid, stop), {"camera_stops": {}, "stop_reason": stop}


def _triangulated(
    cameras: list[CameraTracks], poses: list[np.ndarray], group_ids: Sequence[np.ndarray], frames: list[int], scale: float,
) -> dict[int, np.ndarray]:
    people = sorted({int(p) for values in group_ids for p in np.unique(values) if p >= 0})
    if not people:
        return {}
    size = cameras[0].image_size
    uv: np.ndarray = np.zeros((len(people), len(cameras), len(frames), 17, 2), np.float32)
    confidence = np.zeros(uv.shape[:-1], np.float32)
    observed = np.full(uv.shape[:3], -1, np.int64)
    for view, (pose, ids) in enumerate(zip(poses, group_ids, strict=True)):
        for p, person in enumerate(people):
            row, index = np.nonzero(ids[:, frames] == person)
            if len(index) != len(np.unique(index)):
                raise ValueError("Multiple boxes occupy a player/view/frame in triangulation")
            uv[p, view, index] = pose[row, index, :, :2]
            confidence[p, view, index] = np.clip(pose[row, index, :, 2], 0, 1)
            observed[p, view, index] = row
    visible = ((observed[..., None] >= 0) & (confidence >= .3) & (uv >= 0).all(-1) & (uv < size).all(-1))
    grouped = GroupedObservations(np.asarray(people, np.int64), uv, confidence, visible, observed)
    result = triangulate_players(grouped, tuple(track.camera for track in cameras), reprojection_px=40 * scale)
    mask = np.zeros((len(people), cameras[0].observed.shape[1]), dtype=bool)
    mask[:, frames] = result.valid.any(-1)
    return dict(zip(people, mask, strict=True))


def candidate(
    frozen: FrozenRun, clip: FrozenClip, adopted: FrozenClip, local: dict[str, PinholeCamera],
    report: Path, variant: str, encoders: Sequence[str], frames: list[int], identity: str, *, saved_clip: bool,
) -> dict[str, tuple[TrackingPrediction, dict[str, Any]]]:
    size = (clip.clip.width, clip.clip.height)
    scale = math.hypot(*size) / math.hypot(1920, 1080)
    config = load_association_config(ASSOCIATION_CONFIG, players_per_side=2)
    config = replace(config, footpoints=replace(config.footpoints, bottom_border_px=config.footpoints.bottom_border_px * scale))
    if variant in (DEFAULT, HYBRID) and tuple(encoders) != (CLIP,):
        raise ValueError("Run 10 hybrids only used CLIP for cross-camera association")
    if not config.appearance:
        raise ValueError("Comparison requires the fixed appearance scale")
    raw, selected, groups, parents, track_inputs = {}, {}, {}, {}, {}
    all_groups: dict[str, list[CameraTracks]] = {name: [] for name in encoders}
    all_poses = []
    scene = dual_sha256(adopted.store.index_path)
    side = None
    if "court_side" in clip.store.references:
        side = clip.read("court_side", "court_side", 3, "camera_ids", "view_half_turns")
        turns = dict(zip(side["camera_ids"], side["view_half_turns"], strict=True))
    for name, camera in local.items():
        target = track_path(report, variant, clip.clip.clip_id, name)
        track, rows, receipt = read_tracks(target, camera, size)
        expected = {"comparison_identity": identity, "variant": variant, "camera": name, "scene": scene}
        if any(receipt["input"].get(key) != value for key, value in expected.items()):
            raise ValueError(f"Cannot score a different comparison input: {target}")
        track_inputs[name] = {"stop_reason": receipt["stop_reason"], "sha256": receipt["sha256"]}
        data = load_detections(adopted, name)
        clip_features = appearance.load_encoder_frames(frozen, adopted, name, data, report, CLIP, saved_clip=saved_clip)
        sample = tuple(sampled_appearance(track, rows, clip_features))
        original = replace(track, appearance=sample)
        raw[name] = track
        selected[name], diagnostics = select_linked_candidates(original, clip.clip.fps, LinkingConfig(), config.footpoints)
        groups[name], parents[name] = linked_timeline(original, diagnostics)
        group_rows = np.full(parents[name].shape, -1, np.int64)
        group, source_frame = np.nonzero(parents[name] >= 0)
        group_rows[group, source_frame] = rows[parents[name][group, source_frame], source_frame]
        poses: np.ndarray = np.zeros((len(groups[name].track_ids), len(frames), 17, 3), np.float32)
        group, sample_frame = np.nonzero(group_rows[:, frames] >= 0)
        source_rows = group_rows[group, np.asarray(frames)[sample_frame]]
        indices = np.searchsorted(data.rows, source_rows)
        if (indices >= len(data.rows)).any() or not np.array_equal(data.rows[indices], source_rows):
            raise ValueError(f"Group pose origin differs from the pinned detection row: {target}")
        poses[group, sample_frame] = data.keypoints[indices]
        all_poses.append(poses)
        if side is not None:
            for encoder in encoders:
                item = groups[name]
                if encoder != CLIP:
                    features = appearance.load_encoder_frames(frozen, adopted, name, data, report, encoder, saved_clip=saved_clip)
                    # Use the same raw-track sampling frames as CLIP, even at
                    # overlapping handoffs where the group retains one box.
                    alternate = sampled_appearance(track, rows, features)
                    item, _ = linked_timeline(replace(track, appearance=tuple(alternate)), diagnostics)
                all_groups[encoder].append(replace(item, camera=camera.half_turned(turns[name])))
    outcomes = {}
    for encoder, camera_tracks in all_groups.items():
        stop = None
        global_ids = {name: np.full(track.observed.shape, -1, np.int64) for name, track in raw.items()}
        valid = {}
        if side is None:
            stop = "court_side_unavailable"
        else:
            try:
                decision = associate(camera_tracks, clip.clip.fps, config)
            except AssociationUndecided as error:
                stop = f"player_association_{error.reason}"
            else:
                observed_ids = tuple(np.where(track.observed, ids, -1)
                                     for track, ids in zip(camera_tracks, decision.player_ids, strict=True))
                for name, group_ids in zip(local, observed_ids, strict=True):
                    global_ids[name] = _from_group_ids(raw[name], parents[name], group_ids)
                valid = _triangulated(camera_tracks, all_poses, observed_ids, frames, scale)
        evidence = {"camera_stops": {name: item["stop_reason"] for name, item in track_inputs.items() if item["stop_reason"]},
                    "stop_reason": stop, "tracks": track_inputs}
        outcomes[encoder] = TrackingPrediction(raw, selected, groups, global_ids, valid, stop), evidence
    return outcomes


def calibration_errors(path: Path | None, snapshot: str, selection: str) -> dict[str, Any]:
    if path is None:
        return {}
    records = json.loads(path.read_text())
    if (records.get("schema_version") != 1 or records["input_snapshot_sha256"] != snapshot
            or records["selection_sha256"] != selection or not records["strata"].keys() <= set(SCORE_STRATA)):
        raise ValueError("Unsupported annotation audit or stratum name")
    result = {}
    for stratum, record in records["strata"].items():
        checked, errors = record["checked_boxes"], record["error_boxes"]
        if type(checked) is not int or type(errors) is not int or not 0 <= errors <= checked:
            raise ValueError(f"Invalid human-audit counts in {stratum}")
        result[stratum] = {**record, "box_error_rate": errors / checked if checked else None}
    return result
