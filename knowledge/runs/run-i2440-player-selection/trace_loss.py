"""Reproduce the #2440 attribution on the adopted i2363 stores."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from inputs import SELECTION, ClipRecord, SavedRun, read_fields, read_store, write_json
from numpy.typing import NDArray

from src.tennis_scene.visualization.review.store import StoreReader
from src.utils.schema.court import HALF_DOUBLES_WIDTH

CORE = SELECTION["baseline"]
DETECTION_DISTANCE_M = 2.0
BOTTOM_BORDER_PX = 4.0
TARGET_STOPS = ("video_000/clip_020", "video_000/clip_058")


def foot(boxes: NDArray[np.floating]) -> NDArray[np.floating]:
    return np.stack(((boxes[..., 0] + boxes[..., 2]) / 2, boxes[..., 3]), axis=-1)


def transform(homography: NDArray[np.float64], points: NDArray[np.floating]) -> NDArray[np.float64]:
    homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
    projected = homogeneous @ homography.T
    return projected[..., :2] / projected[..., 2:3]


def homographies(
    store: StoreReader, camera_ids: tuple[str, ...], turns: list[bool]
) -> dict[str, NDArray[np.float64]]:
    output = read_fields(store, "court_calibration", "local_court_calibration", 1, "court.diagnostics")
    result = {}
    for camera in output["court.diagnostics"]["cameras"]:
        (frame,) = camera["frames"]
        result[camera["camera_id"]] = np.asarray(frame["homography_court_metres_to_image_pixels"], float)
    return {
        camera: result[camera] @ np.diag([-1., -1., 1.]) if turn else result[camera]
        for camera, turn in zip(camera_ids, turns, strict=True)
    }


def camera_observations(store: StoreReader, camera: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = read_fields(
        store, f"person_tracking/{camera}", "person_tracks", 7,
        "track_ids", "boxes_xyxy", "observed", "evidence.detection_rows",
    )
    selection = read_fields(
        store, f"player_selection/{camera}", "selected_player_tracks", 3,
        "selected", "origin_rows", "raw_track_ids", "tracks.track_ids",
        "tracks.boxes_xyxy", "tracks.observed", "diagnostics.linking",
    )
    np.testing.assert_array_equal(raw["track_ids"], selection["raw_track_ids"])
    return raw, selection


def outside_core(xy: NDArray[np.float64]) -> bool:
    return bool(abs(float(xy[0])) > CORE.core_half_width_m or abs(float(xy[1])) > CORE.baseline_limit_m)


def selection_loss(
    counts: Counter[str],
    diagnostic: dict[str, Any],
    track_id: int,
    frame: int,
    point: NDArray[np.float64],
    reference: NDArray[np.float64],
    box_bottom: float,
    height: int,
) -> None:
    if box_bottom > height - BOTTOM_BORDER_PX:
        counts["cut_off_at_bottom"] += 1
    counts["outside_core_unrounded" if outside_core(point) else "inside_core_unrounded"] += 1
    # The published breakdown was aggregated after mm rounding in scratch.
    rounded = np.round(point, 3)
    if outside_core(rounded):
        counts["outside_core"] += 1
        counts["reference_also_outside_core"] += outside_core(np.round(reference, 3))
        counts["outside_laterally"] += abs(float(rounded[0])) > CORE.core_half_width_m
        return
    counts["inside_core"] += 1
    fragments = diagnostic["fragments"]
    fragment = next(
        i for i, value in enumerate(fragments)
        if value["track_id"] == track_id and value["start"] <= frame < value["end"]
    )
    group = next((group for group in diagnostic["groups"] if fragment in group["fragments"]), None)
    if group is None:
        counts["inside_core_no_linked_group"] += 1
    observed = 0 if group is None else sum(fragments[i]["observed_frames"] for i in group["fragments"])
    counts["inside_core_chain_observations_below_requirement"] += observed < diagnostic["required_frames"]


def analyse(record: ClipRecord) -> dict[str, Any]:
    store = read_store(record)
    asn = read_fields(
        store, "player_association", "person_identities", 3,
        "camera_ids", "local_track_ids", "player_ids", "diagnostics.view_half_turns",
    )
    cameras = store.camera_ids
    if tuple(asn["camera_ids"]) != cameras:
        raise ValueError(f"{record.clip_id}: association and inlier camera axes must agree")
    world_to_image = homographies(store, cameras, asn["diagnostics.view_half_turns"])
    image_to_world = {camera: np.linalg.inv(matrix) for camera, matrix in world_to_image.items()}
    tri = read_fields(store, "player_triangulation", "player_skeletons", 1, "skeleton.inliers")
    inliers = tri["skeleton.inliers"].any(axis=-1)
    players, views, frames = inliers.shape
    if views != len(cameras) or frames != store.frame_count:
        raise ValueError(f"{record.clip_id}: unexpected archived 3D axes")
    missing = inliers.sum(axis=1) < 2
    stages: Counter[str] = Counter()
    split: Counter[str] = Counter()
    observations = {camera: camera_observations(store, camera) for camera in cameras}
    positions = np.full((views, players, frames, 2), np.nan)
    assignments = {}
    for i, camera in enumerate(cameras):
        _, selected = observations[camera]
        groups = selected["tracks.track_ids"]
        assignment = np.full((len(groups), frames), -1, np.int64)
        for group, ids in zip(asn["local_track_ids"][i], asn["player_ids"][i], strict=True):
            if group >= 0:
                (row,) = np.flatnonzero(groups == group)
                assignment[row] = ids
        assignments[camera] = assignment
        boxes = selected["tracks.boxes_xyxy"].astype(float)
        xy = transform(image_to_world[camera], foot(boxes))
        tracks, valid_frames = np.nonzero(selected["tracks.observed"] & (assignment >= 0))
        positions[i, assignment[tracks, valid_frames], valid_frames] = xy[tracks, valid_frames]
    have = ~np.isnan(positions[..., 0])
    number = have.sum(axis=0)
    references = np.divide(
        np.nansum(positions, axis=0), number[..., None],
        where=number[..., None] > 0, out=np.full((players, frames, 2), np.nan),
    )
    lost_players, lost_frames = np.nonzero(missing)
    for i, camera in enumerate(cameras):
        detections = read_fields(
            store, f"person_detection/{camera}", "person_detections", 2, "boxes_xyxy", "frame_offsets",
        )
        offsets = detections["frame_offsets"]
        detection_points = transform(image_to_world[camera], foot(detections["boxes_xyxy"].astype(float)))
        raw, selected = observations[camera]
        rows = raw["evidence.detection_rows"]
        raw_lookup: NDArray[np.int64] = np.full(len(detection_points), -1, np.int64)
        tracks, times = np.nonzero(rows >= 0)
        indices = rows[tracks, times]
        if not ((offsets[times] <= indices) & (indices < offsets[times + 1])).all() or len(np.unique(indices)) != len(indices):
            raise ValueError(f"{record.clip_id}/{camera}: tracking rows must map to unique detection rows")
        raw_lookup[indices] = tracks
        groups, times = np.nonzero(selected["origin_rows"] >= 0)
        group_lookup = np.full(rows.shape, -1, np.int64)
        group_lookup[selected["origin_rows"][groups, times], times] = groups
        width, height = store.size
        for player, frame in zip(lost_players, lost_frames, strict=True):
            reference = references[player, frame]
            if have[i, player, frame]:
                stages["assigned_without_second_inlier_view"] += 1
                continue
            if not number[player, frame]:
                stages["no_reference"] += 1
                continue
            x, y = transform(world_to_image[camera], reference)
            if not 0 <= x < width or not 0 <= y < height:
                stages["outside_image"] += 1
                continue
            lo, hi = offsets[frame:frame + 2]
            distances = np.linalg.norm(detection_points[lo:hi] - reference, axis=-1)
            if not len(distances):
                stages["no_detection_in_radius"] += 1
                continue
            nearest = int(distances.argmin())
            distance = distances[nearest]
            if distance > DETECTION_DISTANCE_M:
                stages["no_detection_in_radius"] += 1
                continue
            raw_row = raw_lookup[lo + nearest]
            group = -1 if raw_row < 0 else group_lookup[raw_row, frame]
            assigned = -1 if group < 0 else assignments[camera][group, frame]
            closer = np.linalg.norm(detection_points[lo + nearest] - references[:, frame], axis=-1) < distance
            # As in the original analysis, avoid assigning a box carried by a closer player.
            closer[player] = False
            if closer.any() and assigned >= 0 and assigned != player:
                stages["occluded_by_closer_player"] += 1
            elif raw_row < 0:
                stages["not_tracked"] += 1
            elif group < 0:
                stages["not_selected"] += 1
                split["selected_but_group_origin_missing"] += bool(selected["selected"][raw_row, frame])
                box = raw["boxes_xyxy"][raw_row, frame].astype(float)
                selection_loss(
                    split, selected["diagnostics.linking"], int(raw["track_ids"][raw_row]), frame,
                    transform(image_to_world[camera], foot(box)), reference, float(box[3]), height,
                )
            else:
                stages["not_assigned" if assigned < 0 else "assigned_to_other_player"] += 1
    return {
        "clip_id": record.clip_id, "player_frames": players * frames, "missing_player_frames": len(lost_frames),
        "stages": dict(stages), "selection_loss": dict(split),
    }


def stopped_clip(record: ClipRecord) -> dict[str, Any]:
    if record.reason != "player_association_player_not_found" or not record.error or "y>0" not in record.error:
        raise ValueError(f"{record.clip_id}: unexpected archived stop")
    store = read_store(record)
    side = read_fields(store, "court_side", "court_side", 3, "camera_ids", "view_half_turns")
    matrices = homographies(store, tuple(side["camera_ids"]), side["view_half_turns"])
    cameras = {}
    for camera in store.camera_ids:
        raw, selection = camera_observations(store, camera)
        diag = selection["diagnostics.linking"]
        xy = transform(np.linalg.inv(matrices[camera]), foot(raw["boxes_xyxy"].astype(float)))
        # Use the long rejected track with the most observations in the alley near the
        # positive baseline. The 0.5 m tolerance here is for diagnosing projection noise.
        alley = raw["observed"] & (np.abs(xy[..., 0]) > CORE.core_half_width_m)
        alley &= (np.abs(xy[..., 0]) <= HALF_DOUBLES_WIDTH + .5) & (xy[..., 1] >= 8) & (xy[..., 1] <= CORE.baseline_limit_m)
        eligible = (
            (raw["observed"].sum(axis=1) >= diag["required_frames"])
            & ~selection["selected"].any(axis=1)
        )
        evidence_count = np.where(eligible, alley.sum(axis=1), 0)
        row = int(evidence_count.argmax())
        if evidence_count[row] <= 0:
            raise ValueError(f"{record.clip_id}/{camera}: no long rejected alley track")
        track_id = int(raw["track_ids"][row])
        positions = xy[row, raw["observed"][row]]
        cameras[camera] = {
            "track_id": track_id, "observed_frames": len(positions), "selected_frames": 0,
            "max_chain_core_frames": max(
                g["in_core_frames"] for g in diag["groups"] if track_id in g["track_ids"]
            ),
            "required_frames": diag["required_frames"], "near_alley_frames": int(evidence_count[row]),
            "median_abs_x_m": float(np.median(np.abs(positions[:, 0])).round(3)),
            "median_y_m": float(np.median(positions[:, 1]).round(3)),
        }
    return {"clip_id": record.clip_id, "frames": store.frame_count, "error": record.error, "cameras": cameras}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("i2363_run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run = SavedRun(args.i2363_run)
    generated = [record for record in run.records if record.status == "generated"]
    if len(generated) != 55:
        raise ValueError("This reproduction requires all 55 archived successful clips")
    clips = [analyse(record) for record in generated]
    stages: Counter[str] = Counter({"occluded_by_closer_player": 0, "not_assigned": 0})
    split: Counter[str] = Counter({"cut_off_at_bottom": 0})
    for clip in clips:
        stages.update(clip["stages"])
        split.update(clip["selection_loss"])
    stopped = [stopped_clip(record) for record in run.records if record.clip_id in TARGET_STOPS]
    write_json(args.output, {
        "source": run.receipt(),
        "analysed_clips": len(clips),
        "player_frames": sum(clip["player_frames"] for clip in clips),
        "missing_player_frames": sum(clip["missing_player_frames"] for clip in clips),
        "camera_frames": stages.total(),
        "stages": dict(stages),
        "selection_loss": dict(split),
        "stopped": stopped,
        "clips": clips,
    })


if __name__ == "__main__":
    main()
