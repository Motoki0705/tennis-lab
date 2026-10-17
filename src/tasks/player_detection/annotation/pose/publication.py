"""Publish only verified completed pose artifacts into the bound dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset import update_entry, validate_pose_arrays
from .reviews import load_selection
from .selection import load_campaign
from .storage import clip_root, digest, read_json, verify_record, write_json, write_npz


def publish(campaign: Path, index: int) -> dict[str, Any]:
    config, _, store = load_campaign(campaign)
    root = clip_root(campaign, index)
    selected = load_selection(campaign, index)
    receipt = verify_record(root / "pose.json")
    if receipt["status"] != "complete":
        raise ValueError("Cannot publish ungenerated player poses")
    with np.load(root / "poses.npz", allow_pickle=False) as data:
        arrays = dict(data)
    if set(arrays) != {*selected, "keypoints"} or any(
        not np.array_equal(arrays[name], value) for name, value in selected.items()
    ):
        raise ValueError(
            "Pose artifact differs from the approved observation selection"
        )
    points = arrays["keypoints"]
    if (
        points.shape != (*selected["observed"].shape, 17, 3)
        or points.dtype != np.float32
        or not np.isfinite(points).all()
        or points[~selected["observed"]].any()
    ):
        raise ValueError("Invalid generated poses/missing mask")
    validate_pose_arrays(arrays, store, store.clips[index])
    dataset = Path(config["dataset"])
    manifest = read_json(dataset / "manifest.json")
    if manifest["ball_store"]["hashes"] != config["store_hashes"]:
        raise ValueError("Pose dataset belongs to another ball store")
    review = read_json(root / "decision.json")
    name = f"clip-{index:05d}-{digest(root / 'pose.json')[:16]}"
    target = dataset / "clips" / f"{name}.npz"
    decision = dataset / "reviews" / f"{name}.json"
    write_npz(target, **arrays)
    write_json(decision, review)
    provenance = {
        "generation_mode": config["generation_mode"],
        "campaign": str(campaign),
        "campaign_identity": read_json(campaign / "identity.json"),
        "pose_receipt_sha256": digest(root / "pose.json"),
        "review_receipt_sha256": digest(root / "review.json"),
        "selection_sha256": digest(root / "selection.npz"),
        "shard_sha256": read_json(root / "input.json")["shard_sha256"],
        "vitpose_sha256": receipt["vitpose_sha256"],
    }
    update_entry(
        dataset,
        index,
        pose_status="approved",
        file=str(target.relative_to(dataset)),
        sha256=digest(target),
        review_file=str(decision.relative_to(dataset)),
        review_sha256=digest(decision),
        raw_tracks_sha256=review["raw_tracks_sha256"],
        players=len(arrays["player_ids"]),
        provenance=provenance,
    )
    result = {
        "status": "published",
        "file": str(target),
        "pose_crops": receipt["pose_crops"],
        "files": {str(target): digest(target), str(decision): digest(decision)},
    }
    write_json(root / "publication.json", result)
    return result
