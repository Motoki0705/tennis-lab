"""A confirmed historical player association as ``PlayerIdentitiesOutput``.

The historical record (``PlayerAssociationResult``) assigns players to the
player axes of the legacy GVHMR run, not to current tracker IDs. Each legacy
axis is therefore matched to exactly one current pose carrier by its bbox
trace before any ID is published; every other carrier receives ``-1`` with
explicit exclusion evidence. Nothing is inferred by a model.
"""

from __future__ import annotations

import hashlib
import json
import mmap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.plcs.data.manual_association import PlayerAssociationResult
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.identity import (
    PLAYER_ASSOCIATION,
    PlayerIdentitiesOutput,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.pipeline.imports.publish import bind_import, publish_import
from src.tennis_scene.pipeline.input_assembly.observations import gather_people
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.tennis_scene.pipeline.runner import ComponentNode
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.utils.checksum import dual_sha256

IMPORTER = "confirmed_historical_person_association"
IMPORTER_VERSION = 2
# Distances are bbox-centre distances divided by the legacy box size.
MIN_MATCH_OBSERVATIONS = 30
MAX_MEDIAN_CENTER_DISTANCE_BOX_SIZES = .25
# The margin also keeps every other eligible carrier at least this far from each axis.
MIN_NEXT_BEST_MARGIN_BOX_SIZES = .5


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


def legacy_player_boxes(path: Path, frames: int) -> tuple[tuple[int, ...], NDArray[np.float64]]:
    """The legacy run's player axes: track IDs and ``(P, T, 3)`` centre-x, centre-y, size boxes."""
    ids = _legacy_numeric_field(path, "track_ids", max_bytes=4096)
    boxes = np.asarray(_legacy_numeric_field(path, "bbx_xys", max_bytes=2_000_000), np.float64)
    if not isinstance(ids, list) or any(type(track_id) is not int or track_id < 0 for track_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Legacy GVHMR track IDs must be unique nonnegative integers: {path}")
    if boxes.shape != (len(ids), frames, 3) or not np.isfinite(boxes).all() or (boxes[..., 2] <= 0).any():
        raise ValueError(f"Legacy GVHMR bbox axes do not match the source clip: {path}")
    return tuple(ids), boxes


def _median_distances(legacy: NDArray[np.float64], poses: ObjectObservations, carrier: int) -> NDArray[np.float64]:
    """Median normalized centre distance from each legacy axis to one carrier's observed frames."""
    if poses.boxes_xys is None:
        raise ValueError("Pose artifacts must retain source boxes")
    observed = poses.observed[0, :, carrier]
    centre = poses.boxes_xys[0, :, carrier, :2].astype(np.float64)
    distance = np.linalg.norm(legacy[:, :, :2] - centre[None], axis=-1) / legacy[:, :, 2]
    medians: NDArray[np.float64] = np.median(distance[:, observed], axis=1)
    return medians


def match_legacy_axes(legacy: NDArray[np.float64], poses: ObjectObservations) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """The one pose carrier (column index) that follows each legacy axis, or ``ValueError``."""
    counts = poses.observed[0].sum(0)
    eligible = [c for c in range(poses.observed.shape[2]) if counts[c] >= MIN_MATCH_OBSERVATIONS]
    medians = {c: _median_distances(legacy, poses, c) for c in eligible}
    camera = poses.camera_ids[0]
    selected: list[int] = []
    evidence: list[dict[str, Any]] = []
    for axis in range(len(legacy)):
        ranked = sorted((float(medians[c][axis]), c) for c in eligible)
        if not ranked or ranked[0][0] > MAX_MEDIAN_CENTER_DISTANCE_BOX_SIZES:
            raise ValueError(f"Legacy player axis {axis} has no close current track in {camera}")
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < MIN_NEXT_BEST_MARGIN_BOX_SIZES:
            raise ValueError(f"Legacy player axis {axis} matches multiple current tracks in {camera}")
        score, carrier = ranked[0]
        selected.append(carrier)
        evidence.append({"legacy_axis": axis, "track_id": int(poses.local_track_ids[0, carrier]),
            "observed_frames": int(counts[carrier]), "median_center_distance_box_sizes": score,
            "next_best_median": None if len(ranked) < 2 else ranked[1][0]})
    if len(set(selected)) != len(selected):
        raise ValueError(f"Legacy axes map to the same current track in {camera}")
    return tuple(selected), evidence


def _load_association(path: Path, source: ClipSource) -> PlayerAssociationResult:
    association = PlayerAssociationResult.load(path)
    if tuple(association.camera_ids) != source.camera_ids or len(association.segments) != 1:
        raise ValueError("Confirmed person IDs require one full-clip historical assignment in source camera order")
    segment = association.segments[0]
    if (segment.start_frame, segment.end_frame) != (0, source.num_frames):
        raise ValueError("The historical assignment must cover the whole source timeline")
    ids = association.canonical_player_ids
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or (ids < 0).any():
        raise ValueError("Historical player IDs must be unique and nonnegative")
    if segment.assignments.shape != (len(ids), len(source.camera_ids)):
        raise ValueError("Historical assignments must be (players, cameras)")
    return association


def confirmed_identities(source: ClipSource, calibration: CourtCalibrationOutput, poses: dict[str, Any],
                         association: PlayerAssociationResult, legacy: dict[str, NDArray[np.float64]]
                         ) -> tuple[PlayerIdentitiesOutput, dict[str, Any]]:
    """Identities on the calibrated cameras' pose carriers, in ``gather_people`` order.

    ``poses`` maps ``pose_<camera>`` to every source camera's pose artifact and
    ``legacy`` maps each source camera to its legacy ``(P, T, 3)`` boxes.
    """
    active = tuple(view.source_index for view in calibration.calibration.views)
    carriers = gather_people(source, poses).select_views(active)
    player_ids = np.full(carriers.local_track_ids.shape, -1, np.int64)
    assignments = association.segments[0].assignments
    matches: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, list[dict[str, Any]]] = {}
    for row, camera in enumerate(carriers.camera_ids):
        column = source.camera_ids.index(camera)
        own = poses[f"pose_{camera}"]
        axes, matches[camera] = match_legacy_axes(legacy[camera], own)
        targets: dict[int, int] = {}
        for player, global_id in enumerate(association.canonical_player_ids.tolist()):
            axis = int(assignments[player, column])
            if not 0 <= axis < len(axes):
                raise ValueError(f"Historical assignment names missing legacy axis {axis} in {camera}")
            if axes[axis] in targets:
                raise ValueError(f"Historical assignment gives one {camera} track to two players")
            targets[axes[axis]] = int(global_id)
        counts = own.observed[0].sum(0)
        exclusions[camera] = []
        for carrier in range(own.observed.shape[2]):
            if carrier in targets:
                player_ids[row, carrier] = targets[carrier]
                continue
            if carrier in axes:
                disposition = "excluded_unassigned_legacy_axis"
            elif counts[carrier] >= MIN_MATCH_OBSERVATIONS:
                disposition = "excluded_non_target"
            else:
                disposition = "excluded_insufficient_observations" if counts[carrier] else "excluded_unobserved"
            exclusions[camera].append({"track_id": int(own.local_track_ids[0, carrier]), "observed_frames": int(counts[carrier]),
                "nearest_legacy_axis_median": float(_median_distances(legacy[camera], own, carrier).min()) if counts[carrier] else None,
                "disposition": disposition})
    identities = PlayerIdentitiesOutput(carriers.camera_ids, carriers.local_track_ids.copy(), player_ids)
    document = {"reference_camera": association.reference_camera,
        "player_ids": association.canonical_player_ids.tolist(), "excluded_cameras": sorted(set(source.camera_ids) - set(carriers.camera_ids)),
        "legacy_axis_matches": matches, "unassigned_tracks": exclusions,
        "unassigned_track_policy": "only confirmed target tracks receive player IDs; other tracks stay -1 with exclusion evidence"}
    return identities, document


def import_confirmed_person_association(nodes: Sequence[ComponentNode], store: ClipStore, source: ClipSource, *,
                                        historical_association: Path, legacy_gvhmr_directory: Path
                                        ) -> tuple[ArtifactRef, dict[str, Any]]:
    """Publish the load-only ``player_association`` node from a historical record.

    ``legacy_gvhmr_directory`` holds ``gvhmr_result_<camera>.json`` of the run
    whose player axes the record addresses.
    """
    association = _load_association(historical_association, source)
    bound = bind_import(nodes, PLAYER_ASSOCIATION, store)
    calibration: CourtCalibrationOutput = bound.artifacts["calibration"]
    poses = {port: value for port, value in bound.artifacts.items() if port.startswith("pose_")}
    legacy = {camera: legacy_player_boxes(legacy_gvhmr_directory / f"gvhmr_result_{camera}.json", source.num_frames)[1]
              for camera in source.camera_ids}
    identities, document = confirmed_identities(source, calibration, poses, association, legacy)
    legacy_digests = {camera: hashlib.sha256(boxes.tobytes()).hexdigest() for camera, boxes in legacy.items()}
    document = {**document, "historical_association_path": str(historical_association.resolve()),
                "legacy_gvhmr_directory": str(legacy_gvhmr_directory.resolve()), "player_id_matrix": identities.player_ids.tolist()}
    reference = publish_import(bound, store, identities, importer=IMPORTER, version=IMPORTER_VERSION,
        identity={"historical_association_sha256": dual_sha256(historical_association), "legacy_bbox_sha256": legacy_digests,
                  "matching": {"min_observations": MIN_MATCH_OBSERVATIONS, "max_median": MAX_MEDIAN_CENTER_DISTANCE_BOX_SIZES,
                               "min_margin": MIN_NEXT_BEST_MARGIN_BOX_SIZES}},
        provenance={"model_inference": False, "confirmation": document})
    return reference, document
