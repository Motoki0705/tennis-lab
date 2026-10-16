"""Bind approved legacy poses to an expanded store by content, never path substitution."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.store import (
    SHARDS_DIR,
    BallFrameStore,
    ClipRecord,
    shard_name,
)

from .dataset import PlayerPoseStore, update_entry, validate_pose_arrays
from .storage import clip_root, digest, read_json, verify_record, write_json, write_npz


def clip_identity(clip: ClipRecord) -> dict[str, Any]:
    values = asdict(clip)
    values.pop("index")
    # JSON round-trip keeps tuple fields stable in the frozen plan.
    values["track_ids"] = list(clip.track_ids)
    return values


def plan_expansion(
    config: dict[str, Any], entries: list[dict[str, Any]], store: BallFrameStore
) -> dict[str, int]:
    source = (
        PlayerPoseStore(Path(config["reuse_dataset"]))
        if config.get("reuse_dataset")
        else None
    )
    if source is not None:
        config["reuse_manifest_sha256"] = digest(source.directory / "manifest.json")
    counts = {
        "existing": 0,
        "added": 0,
        "changed": 0,
        "reuse": 0,
        "generate": 0,
        "skip": 0,
    }
    for entry, clip in zip(entries, store.clips, strict=True):
        entry["identity"] = clip_identity(clip)
        old = (
            source.ball_store.clip_by_id(clip.clip_id)
            if source is not None and clip.clip_id in source.clips
            else None
        )
        category = (
            "added"
            if old is None
            else "existing"
            if clip_identity(old) == clip_identity(clip)
            else "changed"
        )
        entry["expansion"] = category
        counts[category] += 1
        entry["action"] = "generate" if entry["selected"] else "skip"
        if (
            source is not None
            and old is not None
            and category == "existing"
            and entry["selected"]
            and source.clips[clip.clip_id]["pose_status"] == "approved"
        ):
            old_entry = source.clips[clip.clip_id]
            arrays = source.read_clip(clip.clip_id)
            assert arrays is not None
            rows = store.clip_rows(clip)
            old_rows = source.ball_store.clip_rows(old)
            # Byte identity covers RGB; metadata identity covers annotation and scale.
            for key in ("frame_index", "pts", "is_target", "annotated", "inst_count"):
                if not np.array_equal(
                    store.frames[key][rows], source.ball_store.frames[key][old_rows]
                ):
                    raise ValueError(
                        f"Legacy clip frame/PTS/annotation mismatch: {clip.clip_id} {key}"
                    )
            new_shard = store.directory / SHARDS_DIR / shard_name(clip.index)
            old_shard = source.ball_store.directory / SHARDS_DIR / shard_name(old.index)
            image_hash = digest(new_shard)
            if not new_shard.samefile(old_shard) and digest(old_shard) != image_hash:
                raise ValueError(
                    f"Legacy JPEG coordinates/content changed: {clip.clip_id}"
                )
            root = clip_root(Path(source.manifest["campaign"]), old.index)
            generation = verify_record(root / "generation.json")
            if generation["files"]["tracks.npz"] != old_entry["raw_tracks_sha256"]:
                raise ValueError(
                    f"Legacy review/raw tracking hash mismatch: {clip.clip_id}"
                )
            if read_json(root / "input.json")["shard_sha256"] != image_hash:
                raise ValueError(
                    f"Legacy generation used different JPEG bytes: {clip.clip_id}"
                )
            from .reviews import validate_and_remap

            with np.load(root / "tracks.npz", allow_pickle=False) as data:
                raw = dict(data)
            review = read_json(source.directory / old_entry["review_file"])
            packet = read_json(root / "evidence/packet.json")
            expected = validate_and_remap(
                raw,
                review,
                clip_id=clip.clip_id,
                raw_hash=old_entry["raw_tracks_sha256"],
                required_sheets=packet["required_frame_sheets"],
            )
            if set(arrays) != set(expected) or any(
                not np.array_equal(arrays[k], expected[k]) for k in arrays
            ):
                raise ValueError(
                    f"Legacy pose artifact differs from approved raw evidence: {clip.clip_id}"
                )
            entry["action"] = "reuse"
            entry["reuse"] = {
                "dataset": str(source.directory),
                "manifest_sha256": config["reuse_manifest_sha256"],
                "source_store": source.manifest["ball_store"],
                "source_index": old.index,
                "file": old_entry["file"],
                "sha256": old_entry["sha256"],
                "review_file": old_entry["review_file"],
                "review_sha256": old_entry["review_sha256"],
                "raw_tracks_sha256": old_entry["raw_tracks_sha256"],
                "shard_sha256": image_hash,
                "generation_mode": "legacy_pose_before_review.v1",
            }
        counts[entry["action"]] += 1
    return counts


def reuse_approved(campaign: Path) -> None:
    from .selection import load_campaign

    config, plan, store = load_campaign(campaign)
    dataset = Path(config["dataset"])
    for entry in plan["clips"]:
        if entry["action"] != "reuse":
            continue
        index, source = entry["index"], entry["reuse"]
        root = clip_root(campaign, index)
        if (root / "publication.json").exists():
            verify_record(root / "publication.json")
            continue
        old_dataset = Path(source["dataset"])
        if digest(old_dataset / "manifest.json") != source["manifest_sha256"]:
            raise ValueError("Frozen legacy pose manifest changed")
        for field, checksum in (("file", "sha256"), ("review_file", "review_sha256")):
            if digest(old_dataset / source[field]) != source[checksum]:
                raise ValueError("Frozen legacy pose/review bytes changed")
        shard = store.directory / SHARDS_DIR / shard_name(index)
        if digest(shard) != source["shard_sha256"]:
            raise ValueError("Target JPEG bytes changed after reuse planning")
        with np.load(old_dataset / source["file"], allow_pickle=False) as data:
            arrays = dict(data)
        validate_pose_arrays(arrays, store, store.clips[index])
        target = (
            dataset / "clips" / f"clip-{index:05d}-reused-{source['sha256'][:16]}.npz"
        )
        decision = (
            dataset
            / "reviews"
            / f"clip-{index:05d}-reused-{source['review_sha256'][:16]}.json"
        )
        write_npz(target, **arrays)
        write_json(decision, read_json(old_dataset / source["review_file"]))
        update_entry(
            dataset,
            index,
            pose_status="approved",
            file=str(target.relative_to(dataset)),
            sha256=digest(target),
            review_file=str(decision.relative_to(dataset)),
            review_sha256=digest(decision),
            raw_tracks_sha256=source["raw_tracks_sha256"],
            players=len(arrays["player_ids"]),
            provenance=source,
        )
        write_json(
            root / "publication.json",
            {
                "status": "published",
                "reused": True,
                "provenance": source,
                "files": {str(target): digest(target), str(decision): digest(decision)},
            },
        )
