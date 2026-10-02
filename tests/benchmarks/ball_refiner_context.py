"""Read every generated context clip in a separate CPU process and record checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tasks.ball_detection.data.store import BallFrameStore, shard_name
from src.tasks.ball_refiner.data.context_cache import ContextCache
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.utils.checksum import dual_sha256


def verify_context(cache: ContextCache, selected: tuple[str, ...], *, pose_threshold: float) -> dict[str, Any]:
    if not selected or len(set(selected)) != len(selected) or set(cache.clip_ids) != set(selected):
        raise ValueError("Context cache must contain exactly the unique requested clips")
    cache.require_clips(selected)
    manifest_hash = dual_sha256(cache.directory / "manifest.json")
    rows = []
    for record in cache.manifest["clips"]:
        clip_id = record["clip"]["clip_id"]
        generated = cache.load(clip_id)
        clip = cache.evidence.store.clip_by_id(clip_id)
        shard = cache.evidence.store.directory / "shards" / shard_name(clip.index)
        if dual_sha256(shard) != record["jpeg_shard_sha256"]:
            raise ValueError(f"JPEG shard changed after context generation: {clip_id}")
        context = generated.arrays.model_context(clip, pose_threshold=pose_threshold, provenance={})
        pose, court = context.require_complete()
        rows.append({
            "clip_id": clip_id, "frames": clip.frame_count, "tracks": len(generated.arrays.track_ids),
            "pose_valid_frames": int(pose.valid.any(axis=(1, 2)).sum()),
            "pose_valid_slots": int(pose.valid.sum()), "court_valid_points": int(court.valid.sum()),
            "model_context_provenance": context.provenance, "execution": generated.execution,
        })
    if dual_sha256(cache.directory / "manifest.json") != manifest_hash:
        raise ValueError("Context manifest changed during verification")
    return {"status": "verified_context_load", "context_manifest_sha256": manifest_hash,
            "pose_threshold": pose_threshold, "clips": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("store", "evidence", "context", "report"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--clip-id", action="append", required=True)
    parser.add_argument("--pose-threshold", type=float, required=True)
    args = parser.parse_args()
    if not all(getattr(args, name).is_absolute() for name in ("store", "evidence", "context", "report")):
        parser.error("All paths must be absolute")
    if args.report.exists():
        raise FileExistsError(args.report)
    cache = ContextCache(args.context, EvidenceCache(args.evidence, BallFrameStore(args.store)))
    result = verify_context(cache, tuple(args.clip_id), pose_threshold=args.pose_threshold)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
