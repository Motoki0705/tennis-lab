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
        poses[frame, column] = raw["keypoints"][at, frame]
        valid[frame, column] = True
        origins[frame, column] = track_ids[at]
        detection_rows[frame, column] = raw["detection_rows"][at, frame]
    return {
        "frame_index": raw["frame_index"],
        "pts": raw["pts"],
        "player_ids": np.asarray(ids, dtype=np.str_),
        "boxes_xyxy": boxes,
        "keypoints": poses,
        "observed": valid,
        "raw_track_ids": origins,
        "detection_rows": detection_rows,
    }


def publish(
    campaign: Path, index: int, review_path: Path, required_sheets: list[str]
) -> dict[str, Any]:
    config, plan, _ = load_campaign(campaign)
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
    if review["status"] != "approved":
        write_json(
            root / "review_status.json",
            {
                "status": "needs_review",
                "review": str(review_path),
                "notes": review["notes"],
            },
        )
        return {"status": "needs_review"}
    dataset = Path(config["dataset"])
    review_hash = digest(review_path)
    artifact = f"clip-{index:05d}-{review_hash[:16]}"
    dataset.mkdir(parents=True, exist_ok=True)
    with lock(dataset / ".publish.lock"):
        target = dataset / "clips" / f"{artifact}.npz"
        # Idempotent recovery: deterministic files are written before the manifest reference.
        write_npz(target, **arrays)
        decision = dataset / "reviews" / f"{artifact}.json"
        write_json(decision, review)
        manifest_path = dataset / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest["ball_store"]["hashes"] != config["store_hashes"]:
                raise ValueError("Existing pose dataset belongs to another ball store")
        else:
            manifest = {
                "schema": "ball_detection_player_poses.v1",
                "coordinate_system": "stored_jpeg_pixels",
                "ball_store": {
                    "directory": config["store"],
                    "hashes": config["store_hashes"],
                },
                "campaign": str(campaign),
                "court_policy": "disabled",
                "clips": [
                    {**c, "pose_status": "pending" if c["selected"] else "skipped"}
                    for c in plan["clips"]
                ],
            }
        manifest["clips"][index].update(
            pose_status="approved",
            file=str(target.relative_to(dataset)),
            sha256=digest(target),
            review_file=str(decision.relative_to(dataset)),
            review_sha256=digest(decision),
            raw_tracks_sha256=review["raw_tracks_sha256"],
            players=len(arrays["player_ids"]),
        )
        manifest["status"] = (
            "complete"
            if all(
                c["pose_status"] in ("approved", "skipped") for c in manifest["clips"]
            )
            else "partial"
        )
        write_json(manifest_path, manifest)
        write_json(
            root / "review_status.json",
            {
                "status": "approved",
                "file": str(target),
                "sha256": digest(target),
                "review_sha256": review_hash,
            },
        )
    return {
        "status": "approved",
        "players": len(arrays["player_ids"]),
        "file": str(target),
    }
