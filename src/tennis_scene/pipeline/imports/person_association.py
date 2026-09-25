"""Import a confirmed historical player association as the Re-ID output schema.

Historical assignments address GVHMR player axes, not current tracker IDs. Their
bbox traces must identify the current tracks before any assignment is published.
The actual model embeddings stay unchanged; only their identity labels change.
"""

from __future__ import annotations

import hashlib
import json
import mmap
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tennis_scene.pipeline.artifact_schemas.manual_association import (
    PlayerAssociationResult,
)
from src.tennis_scene.pipeline.artifacts import json_value
from src.tennis_scene.pipeline.components.person_association import PlayerReIDOutput
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256

MIN_MATCH_OBSERVATIONS = 30
MAX_MEDIAN_CENTER_DISTANCE_BOX_SIZES = .25
MIN_NEXT_BEST_MARGIN_BOX_SIZES = .5
MIN_NON_TARGET_DISTANCE_BOX_SIZES = .5


def _legacy_numeric_field(path: Path, name: str, *, max_bytes: int) -> Any:
    """Read one small top-level field without loading a 1.5 GB legacy mesh JSON."""
    marker = f'"{name}"'.encode()
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        location = mapped.find(marker)
        if location < 0 or mapped.find(marker, location + len(marker)) >= 0:
            raise ValueError(f"Legacy GVHMR file must contain one {name}: {path}")
        colon = mapped.find(b":", location + len(marker))
        if colon < 0:
            raise ValueError(f"Legacy GVHMR field has no value: {name}")
        snippet = mapped[colon + 1:colon + 1 + max_bytes].decode("utf-8").lstrip()
        try:
            value, _ = json.JSONDecoder().raw_decode(snippet)
        except json.JSONDecodeError as error:
            raise ValueError(f"Legacy GVHMR field is invalid or exceeds {max_bytes} bytes: {name}") from error
        return value


def _legacy_tracks(path: Path, frames: int) -> tuple[tuple[int, ...], np.ndarray, str]:
    ids = _legacy_numeric_field(path, "track_ids", max_bytes=4096)
    boxes = np.asarray(_legacy_numeric_field(path, "bbx_xys", max_bytes=2_000_000), np.float64)
    if not isinstance(ids, list) or any(type(track_id) is not int or track_id < 0 for track_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Legacy GVHMR track IDs must be unique nonnegative integers")
    if boxes.shape != (len(ids), frames, 3) or not np.isfinite(boxes).all() or (boxes[..., 2] <= 0).any():
        raise ValueError("Legacy GVHMR bbox axes do not match the source clip")
    digest = hashlib.sha256(boxes.tobytes()).hexdigest()
    return tuple(ids), boxes, digest


def _match_current_tracks(legacy_boxes: np.ndarray, current: PersonTrackingOutput) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Require a unique trace match for every historical GVHMR player axis."""
    selected: list[int] = []
    evidence: list[dict[str, Any]] = []
    for axis, old in enumerate(legacy_boxes):
        candidates: list[tuple[float, int, int, float]] = []
        for row, track_id in enumerate(current.track_ids):
            observed = current.observed[row]
            frames = int(observed.sum())
            if frames < MIN_MATCH_OBSERVATIONS:
                continue
            center = (current.boxes_xyxy[row, :, :2] + current.boxes_xyxy[row, :, 2:]) / 2
            distance = np.linalg.norm(old[:, :2] - center, axis=1) / old[:, 2]
            candidates.append((float(np.median(distance[observed])), int(track_id), frames,
                               float(np.percentile(distance[observed], 95))))
        candidates.sort()
        if not candidates or candidates[0][0] > MAX_MEDIAN_CENTER_DISTANCE_BOX_SIZES:
            raise ValueError(f"Historical player axis {axis} has no close current track in {current.camera_id}")
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] < MIN_NEXT_BEST_MARGIN_BOX_SIZES:
            raise ValueError(f"Historical player axis {axis} matches multiple current tracks in {current.camera_id}")
        score, track_id, frames, p95 = candidates[0]
        selected.append(track_id)
        evidence.append({"historical_axis": axis, "current_track_id": track_id,
                         "observed_frames": frames, "median_center_distance_box_sizes": score,
                         "p95_center_distance_box_sizes": p95,
                         "next_best_median": None if len(candidates) < 2 else candidates[1][0]})
    if len(set(selected)) != len(selected):
        raise ValueError(f"Historical axes map to the same current track in {current.camera_id}")
    return tuple(selected), evidence


def _nearest_legacy_axis_distance(legacy_boxes: np.ndarray, current: PersonTrackingOutput, row: int) -> float:
    observed = current.observed[row]
    if int(observed.sum()) < MIN_MATCH_OBSERVATIONS:
        raise ValueError("A model-valid non-target track needs enough observations for identity exclusion")
    center = (current.boxes_xyxy[row, :, :2] + current.boxes_xyxy[row, :, 2:]) / 2
    return min(float(np.median((np.linalg.norm(old[:, :2] - center, axis=1) / old[:, 2])[observed]))
               for old in legacy_boxes)


def import_confirmed_person_association(
    node: ComponentNode,
    store: ClipStore,
    source: ClipSource,
    *,
    model_reference: ArtifactRef,
    historical_association: Path,
    legacy_gvhmr_directory: Path,
) -> tuple[ArtifactRef, dict[str, Any]]:
    """Publish confirmed IDs while retaining model outputs for numeric review.

    This fixture is only for the explicitly selected clip. It never claims that
    the Re-ID model inferred the imported assignments.
    """
    if node.name != "person_reid" or node.source != "load" or node.io.output_schema != "person_identities":
        raise ValueError("Confirmed person IDs require a load-only person_reid node")
    if source.camera_ids != tuple(video["camera_id"] for video in store.source["videos"]):
        raise ValueError("Confirmed person IDs require the store's source camera order")
    association = PlayerAssociationResult.load(historical_association)
    if tuple(association.camera_ids) != source.camera_ids or len(association.segments) != 1:
        raise ValueError("Confirmed person IDs require one full-clip historical assignment in source camera order")
    if association.reference_camera != source.camera_ids[0]:
        raise ValueError("Historical reference camera differs from this clip's declared reference")
    dependencies: dict[str, ArtifactRef] = {}
    for port, producer in node.bindings.items():
        reference = store.active(producer)
        if reference is None:
            raise ValueError(f"Missing association dependency: {producer}")
        dependencies[port] = reference
    model_descriptor = store.descriptor(model_reference)
    if model_descriptor["node"] != node.name or model_descriptor["provenance"].get("origin") != "component" or model_descriptor["dependencies"] != json_value(dependencies):
        raise ValueError("Model Re-ID artifact must come from the current component inputs")
    model = store.load(model_reference, ArtifactCodec(PlayerReIDOutput))
    if model.camera_ids != source.camera_ids or model.result is None:
        raise ValueError("Current model Re-ID artifact has no complete camera output")
    result = model.result
    current: dict[str, PersonTrackingOutput] = {}
    track_refs: dict[str, ArtifactRef] = {}
    legacy_ids: dict[str, tuple[int, ...]] = {}
    matched: dict[str, tuple[int, ...]] = {}
    comparisons: dict[str, list[dict[str, Any]]] = {}
    legacy_digests: dict[str, str] = {}
    legacy_bboxes: dict[str, np.ndarray] = {}
    for camera in source.camera_ids:
        reference = store.active(f"person_tracking/{camera}")
        if reference is None:
            raise ValueError(f"Missing current track artifact: {camera}")
        track_refs[camera] = reference
        current[camera] = store.load(reference, ArtifactCodec(PersonTrackingOutput))
        path = legacy_gvhmr_directory / f"gvhmr_result_{camera}.json"
        legacy_ids[camera], legacy_bboxes[camera], legacy_digests[camera] = _legacy_tracks(path, source.num_frames)
        matched[camera], comparisons[camera] = _match_current_tracks(legacy_bboxes[camera], current[camera])
    valid, errors = association.validate(num_frames=source.num_frames,
        local_player_counts=[len(legacy_ids[camera]) for camera in source.camera_ids])
    if not valid or len(np.unique(association.canonical_player_ids)) != len(association.canonical_player_ids) or (association.canonical_player_ids < 0).any():
        raise ValueError(f"Historical person association is invalid: {errors}")
    assignments = association.segments[0].assignments
    label_by_track: dict[str, dict[int, int]] = {camera: {} for camera in source.camera_ids}
    for player, global_id in enumerate(association.canonical_player_ids):
        for view, camera in enumerate(source.camera_ids):
            axis = int(assignments[player, view])
            track_id = matched[camera][axis]
            if track_id in label_by_track[camera]:
                raise ValueError("Historical assignment reuses one local track for multiple players")
            label_by_track[camera][track_id] = int(global_id)

    slots = result.local_track_ids.detach().cpu().numpy()
    slot_valid = result.track_valid.detach().cpu().numpy()
    raw_ids = np.full(result.raw_track_ids.shape, -1, np.int64)
    slot_ids = np.full(result.slot_global_ids.shape, -1, np.int64)
    extras: dict[str, list[dict[str, Any]]] = {}
    for view, camera in enumerate(source.camera_ids):
        tracks = current[camera]
        if tracks.camera_id != camera or len(tracks.track_ids) > raw_ids.shape[1]:
            raise ValueError("Current tracker IDs do not fit model Re-ID carrier slots")
        extras[camera] = []
        for row, track_id_value in enumerate(tracks.track_ids):
            track_id = int(track_id_value)
            matches = np.flatnonzero(slots[view] == track_id)
            if len(matches) != 1:
                raise ValueError(f"Model slots do not identify current {camera} track {track_id}")
            slot = int(matches[0])
            has_pose = bool(slot_valid[view, slot])
            if track_id in label_by_track[camera]:
                if not has_pose:
                    raise ValueError(f"Confirmed {camera} track {track_id} has no model-valid pose")
                label = label_by_track[camera][track_id]
            elif has_pose:
                nearest = _nearest_legacy_axis_distance(legacy_bboxes[camera], tracks, row)
                if nearest < MIN_NON_TARGET_DISTANCE_BOX_SIZES:
                    raise ValueError(f"Unconfirmed {camera} track {track_id} resembles a target player")
                extras[camera].append({"track_id": track_id, "global_id": -1, "model_valid": True,
                                       "disposition": "excluded_non_target", "nearest_legacy_axis_median": nearest})
                continue
            else:
                extras[camera].append({"track_id": track_id, "global_id": -1, "model_valid": False,
                                       "disposition": "excluded_no_valid_pose"})
                continue
            raw_ids[view, row] = label
            slot_ids[view, slot] = label
    confirmed = PlayerReIDOutput(source.camera_ids,
        replace(result, raw_track_ids=torch.from_numpy(raw_ids), slot_global_ids=torch.from_numpy(slot_ids)))
    document: dict[str, Any] = {
        "schema": "confirmed_person_association_import_v1",
        "source_clip_id": source.clip_id,
        "source_sha256": store.source_key,
        "historical_association_path": str(historical_association.resolve()),
        "historical_association_sha256": dual_sha256(historical_association),
        "historical_reference_camera": association.reference_camera,
        "historical_player_ids": association.canonical_player_ids.tolist(),
        "historical_gvhmr_track_ids": {camera: list(ids) for camera, ids in legacy_ids.items()},
        "historical_bbox_sha256": legacy_digests,
        "current_track_artifacts": json_value(track_refs),
        "axis_to_current_track": {camera: list(ids) for camera, ids in matched.items()},
        "trace_comparisons": comparisons,
        "confirmed_assignments": {camera: {str(key): value for key, value in labels.items()} for camera, labels in label_by_track.items()},
        "confirmed_target_ids": association.canonical_player_ids.tolist(),
        "unassigned_track_policy": "Only confirmed target tracks receive global IDs; other tracks remain -1 with explicit exclusion evidence",
        "extra_tracks": extras,
        "model_artifact": json_value(model_reference),
        "model_raw_track_ids": result.raw_track_ids.tolist(),
        "confirmed_raw_track_ids": raw_ids.tolist(),
        "embedding_policy": "preserve_model_track_embedding_track_valid_and_cosine_threshold_unchanged",
    }
    reference = store.publish(node.name, confirmed, ArtifactCodec(PlayerReIDOutput),
        schema=node.io.output_schema, version=node.io.version,
        identity={"importer": "confirmed_historical_person_association", "version": 1,
                  "source_sha256": store.source_key, "historical_association_sha256": document["historical_association_sha256"],
                  "legacy_bbox_sha256": legacy_digests, "model_artifact_id": model_reference.artifact_id,
                  "confirmed_raw_track_ids": raw_ids.tolist()},
        dependencies=dependencies,
        provenance={"origin": "confirmed_person_association", "model_inference": "preserved_in_separate_artifact",
                    "model_artifact_id": model_reference.artifact_id,
                    "model_raw_track_ids": result.raw_track_ids.tolist(),
                    "confirmation": document})
    return reference, document
