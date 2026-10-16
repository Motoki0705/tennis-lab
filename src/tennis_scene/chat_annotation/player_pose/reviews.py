from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .selection import load_campaign
from .storage import (
    clip_root,
    digest,
    lock,
    read_json,
    verify_record,
    write_json,
    write_npz,
)

SEGMENT_PROPERTIES: dict[str, Any] = {
    "raw_track_id": {"type": "integer"},
    "start_frame": {"type": "integer"},
    "stop_frame": {"type": "integer"},
    "role": {
        "type": "string",
        "enum": ["player", "non_player", "duplicate", "unknown"],
    },
    "player_id": {"type": ["string", "null"]},
    "evidence_frames": {"type": "array", "items": {"type": "integer"}},
    "reason": {"type": "string"},
}
REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "clip_id",
        "raw_tracks_sha256",
        "status",
        "reviewed_sheets",
        "segments",
        "notes",
    ],
    "properties": {
        "clip_id": {"type": "string"},
        "raw_tracks_sha256": {"type": "string"},
        "status": {"type": "string", "enum": ["approved", "needs_review"]},
        "reviewed_sheets": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(SEGMENT_PROPERTIES),
                "properties": SEGMENT_PROPERTIES,
            },
        },
    },
}


def validate_and_remap(
    raw: dict[str, Any],
    review: dict[str, Any],
    *,
    clip_id: str,
    raw_hash: str,
    required_sheets: list[str],
) -> dict[str, Any]:
    """Review every emitted observation; changes cannot invent detections or poses."""
    if (
        set(review) != set(REVIEW_SCHEMA["required"])
        or review["clip_id"] != clip_id
        or review["raw_tracks_sha256"] != raw_hash
    ):
        raise ValueError("Review identity/schema mismatch")
    if review["status"] not in ("approved", "needs_review") or not isinstance(
        review["notes"], str
    ):
        raise ValueError("Invalid review status/notes")
    if sorted(review["reviewed_sheets"]) != sorted(required_sheets):
        raise ValueError("All frame sheets must be included in the review")
    track_ids = raw["track_ids"]
    observed = raw["detection_rows"] >= 0
    p, frames = observed.shape
    if track_ids.shape != (p,) or len(raw["frame_index"]) != frames:
        raise ValueError("Raw tracking axes are inconsistent")
    lookup = {int(t): i for i, t in enumerate(track_ids)}
    coverage = np.zeros((p, frames), np.int32)
    assignments: dict[tuple[int, int], tuple[str, str | None]] = {}
    players: set[str] = set()
    for segment in review["segments"]:
        if set(segment) != set(SEGMENT_PROPERTIES):
            raise ValueError("Segment schema mismatch")
        t, start, stop = (
            segment[k] for k in ("raw_track_id", "start_frame", "stop_frame")
        )
        if (
            any(type(v) is not int for v in (t, start, stop))
            or t not in lookup
            or not 0 <= start < stop <= frames
        ):
            raise ValueError("Invalid track/frame interval")
        role, player = segment["role"], segment["player_id"]
        if role not in ("player", "non_player", "duplicate", "unknown"):
            raise ValueError("Invalid role")
        if role in ("player", "duplicate"):
            if not isinstance(player, str) or not re.fullmatch(
                r"player_[1-9][0-9]*", player
            ):
                raise ValueError("Player IDs must be anonymous player_N identifiers")
            players.add(player)
        elif player is not None:
            raise ValueError("Excluded/unknown observations cannot receive a player ID")
        if not isinstance(segment["reason"], str) or not segment["reason"].strip():
            raise ValueError("Every decision needs an explanation")
        evidence = segment["evidence_frames"]
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(type(f) is not int or not start <= f < stop for f in evidence)
        ):
            raise ValueError("Decision evidence must fall within its interval")
        at = lookup[t]
        if not observed[at, start:stop].any():
            raise ValueError("Review segment has no real observations")
        if not any(observed[at, f] for f in evidence):
            raise ValueError(
                "At least one evidence frame must contain the reviewed track"
            )
        coverage[at, start:stop] += 1
        for frame in np.flatnonzero(observed[at, start:stop]) + start:
            assignments[at, int(frame)] = role, player
    if (coverage > 1).any() or (coverage[observed] != 1).any():
        raise ValueError(
            "Every raw observation needs exactly one decision; overlapping intervals are forbidden"
        )
    selected: dict[tuple[str, int], int] = {}
    for (at, frame), (role, player) in assignments.items():
        if role == "unknown" and review["status"] == "approved":
            raise ValueError("Approved reviews cannot contain unresolved identities")
        if role == "player":
            assert player is not None
            key = player, frame
            if key in selected:
                raise ValueError(
                    "Two raw tracks assigned to one player at the same frame; mark the duplicate explicitly"
                )
            selected[key] = at
    for (_, frame), (role, player) in assignments.items():
        if role == "duplicate" and (player, frame) not in selected:
            raise ValueError(
                "A duplicate needs a retained observation of the same player/frame"
            )
    ids = sorted(players, key=lambda value: int(value.split("_")[1]))
    columns = {player: i for i, player in enumerate(ids)}
    boxes = np.zeros((frames, len(ids), 4), np.float32)
    poses = np.zeros((frames, len(ids), 17, 3), np.float32)
    valid = np.zeros((frames, len(ids)), bool)
    origins = np.full((frames, len(ids)), -1, np.int64)
    detection_rows = np.full_like(origins, -1)
    for (player, frame), at in selected.items():
        column = columns[player]
        boxes[frame, column] = raw["boxes"][at, frame]
        if "keypoints" in raw:
            poses[frame, column] = raw["keypoints"][at, frame]
        valid[frame, column] = True
        origins[frame, column] = track_ids[at]
        detection_rows[frame, column] = raw["detection_rows"][at, frame]
    result = {
        "frame_index": raw["frame_index"],
        "pts": raw["pts"],
        "player_ids": np.asarray(ids, dtype=np.str_),
        "boxes_xyxy": boxes,
        "observed": valid,
        "raw_track_ids": origins,
        "detection_rows": detection_rows,
    }
    if "keypoints" in raw:
        result["keypoints"] = poses
    if len(np.unique(detection_rows[valid])) != int(valid.sum()):
        raise ValueError("Selected detection rows must be unique real observations")
    return result


def accept_review(
    campaign: Path, index: int, review_path: Path, required_sheets: list[str]
) -> dict[str, Any]:
    """Freeze selected real observations; approval does not imply generated poses."""
    from .dataset import update_entry

    config, plan, _ = load_campaign(campaign)
    if not plan["clips"][index]["selected"]:
        raise ValueError("Cannot approve a skipped clip")
    root = clip_root(campaign, index)
    verify_record(root / "generation.json")
    review = read_json(review_path)
    with np.load(root / "tracks.npz", allow_pickle=False) as data:
        raw = {name: data[name] for name in data.files}
    arrays = validate_and_remap(
        raw,
        review,
        clip_id=plan["clips"][index]["clip_id"],
        raw_hash=digest(root / "tracks.npz"),
        required_sheets=required_sheets,
    )
    if "keypoints" in raw:
        raise ValueError("New campaigns require pose-free tracking artifacts")
    with lock(root / "selection.lock"):
        if (root / "review.json").exists():
            verify_record(root / "review.json")
            if read_json(root / "decision.json") != review:
                raise ValueError("Approved selection is immutable; use a new campaign")
            manifest = read_json(Path(config["dataset"]) / "manifest.json")
            if manifest["clips"][index]["pose_status"] != "approved":
                update_entry(Path(config["dataset"]), index, pose_status="pose_pending")
            write_json(
                root / "review_status.json",
                {"status": "approved", "pose_status": "pending"},
            )
            return {"status": "approved", "pose_status": "pending"}
        if review["status"] != "approved":
            result = {
                "status": "needs_review",
                "review": str(review_path),
                "notes": review["notes"],
            }
            write_json(root / "review_status.json", result)
            update_entry(Path(config["dataset"]), index, pose_status="needs_review")
            return result
        write_npz(root / "selection.npz", **arrays)
        write_json(root / "decision.json", review)
        write_json(
            root / "review.json",
            {
                "status": "approved",
                "raw_tracks_sha256": review["raw_tracks_sha256"],
                "required_sheets": required_sheets,
                "selected_crops": int(arrays["observed"].sum()),
                "files": {
                    name: digest(root / name)
                    for name in (
                        "tracks.npz",
                        "generation.json",
                        "selection.npz",
                        "decision.json",
                    )
                },
            },
        )
        update_entry(Path(config["dataset"]), index, pose_status="pose_pending")
        result = {
            "status": "approved",
            "pose_status": "pending",
            "selected_crops": int(arrays["observed"].sum()),
        }
        write_json(root / "review_status.json", result)
        return result


def load_selection(campaign: Path, index: int) -> dict[str, Any]:
    """Revalidate review-to-observation identity before pose or publication."""
    _, plan, store = load_campaign(campaign)
    if not plan["clips"][index]["selected"]:
        raise ValueError("Skipped clips cannot have selected poses")
    root = clip_root(campaign, index)
    verify_record(root / "generation.json")
    receipt = verify_record(root / "review.json")
    review = read_json(root / "decision.json")
    if receipt["status"] != "approved" or review["status"] != "approved":
        raise ValueError("Pose inference requires an approved review")
    with np.load(root / "tracks.npz", allow_pickle=False) as data:
        raw = dict(data)
    expected = validate_and_remap(
        raw,
        review,
        clip_id=plan["clips"][index]["clip_id"],
        raw_hash=digest(root / "tracks.npz"),
        required_sheets=receipt["required_sheets"],
    )
    with np.load(root / "selection.npz", allow_pickle=False) as data:
        arrays = dict(data)
    if set(arrays) != set(expected) or any(
        not np.array_equal(arrays[k], expected[k]) for k in expected
    ):
        raise ValueError("Selection differs from the approved raw detection rows")
    clip = store.clips[index]
    rows = store.clip_rows(clip)
    if any(
        not np.array_equal(arrays[k], store.frames[k][rows])
        for k in ("frame_index", "pts")
    ):
        raise ValueError("Selected observation frame/PTS differs from ball store")
    return arrays
