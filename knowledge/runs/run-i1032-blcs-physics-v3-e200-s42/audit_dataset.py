"""Audit the frozen PR1038 dataset without changing its contents (CPU only)."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from src.utils.physics.ball.record import BallPhysicsRecord, record_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-physics", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    root = args.dataset.resolve()
    manifest_bytes = gzip.decompress(args.manifest.read_bytes())
    entries = [json.loads(line) for line in manifest_bytes.splitlines()]
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != {entry["path"] for entry in entries}:
        raise ValueError("Dataset file inventory changed")
    for entry in entries:
        path = root / entry["path"]
        if path.stat().st_size != entry["size"]:
            raise ValueError(f"Dataset size changed: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Dataset hash changed: {path}")

    splits = {
        split: (root / f"{split}.txt").read_text().splitlines()
        for split in ("train", "val", "test")
    }
    all_ids = [scene for ids in splits.values() for scene in ids]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate scene or overlapping splits")
    if set(all_ids) != {path.name for path in (root / "scenes").iterdir()}:
        raise ValueError("Split inventory does not cover the dataset")
    expected_schema = "blcs_generated_dataset_v3"
    root_meta = json.loads((root / "meta.json").read_text())
    if root_meta["court_keypoints"]["dataset_schema_id"] != expected_schema:
        raise ValueError("Unexpected root schema")
    surfaces: Counter[str] = Counter()
    frame_counts = {}
    max_error = 0.0
    for index, scene_id in enumerate(all_ids):
        scene = root / "scenes" / scene_id
        meta = json.loads((scene / "meta.json").read_text())
        if meta["court_keypoints"]["dataset_schema_id"] != expected_schema:
            raise ValueError(f"Unexpected scene schema: {scene_id}")
        record = BallPhysicsRecord.from_arrays(
            {key: np.load(scene / f"{key}.npy", allow_pickle=False) for key in record_keys()}
        )
        positions = np.load(scene / "ball_pos_world.npy", allow_pickle=False)
        if positions.shape != (record.frames, 3) or not np.isfinite(positions).all():
            raise ValueError(f"Invalid trajectory: {scene_id}")
        if record.frames != meta["num_frames"] or record.output_fps != 30:
            raise ValueError(f"Unexpected frames/FPS: {scene_id}")
        if record.gravity != 9.8 or meta["num_cameras"] != 6:
            raise ValueError(f"Unexpected physics/cameras: {scene_id}")
        if args.verify_physics:
            max_error = max(max_error, record.verify(positions, 1e-6))
        surfaces[record.surface] += 1
        frame_counts[scene_id] = record.frames
        if (index + 1) % 100 == 0:
            print(f"Audited {index + 1}/{len(all_ids)} scenes", flush=True)
    result = {
        "dataset": str(root),
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(entries),
        "all_file_hashes_match": True,
        "schema": expected_schema,
        "physics_schema": "ball_physics.v1",
        "surfaces": dict(surfaces),
        "splits": {
            split: {
                "raw_count": len(ids),
                "eligible_t128": [scene for scene in ids if frame_counts[scene] >= 128],
                "excluded_short": [scene for scene in ids if frame_counts[scene] < 128],
                "total_frames": sum(frame_counts[scene] for scene in ids),
            }
            for split, ids in splits.items()
        },
        "physics_reintegrated": bool(args.verify_physics),
        "max_physics_error_m": max_error if args.verify_physics else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "splits"}, indent=2))


if __name__ == "__main__":
    main()
