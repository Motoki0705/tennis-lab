"""Describe court framing from immutable original-image KP geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

GROUPS = ("full", "near_full", "partial")
RULE = {
    "version": "in_frame_kp14_v1",
    "full": "14 KP in frame",
    "near_full": "12 or 13 KP in frame",
    "partial": "0 through 11 KP in frame",
    "in_frame": "0 <= u < width and 0 <= v < height; positive camera depth for synthetic",
    "scope": "original-image target-court geometry; renderer occlusion is not framing",
}


def group_for_count(count: int) -> str:
    if not 0 <= count <= 14:
        raise ValueError(f"Expected 0..14 KP, got {count}")
    return "full" if count == 14 else "near_full" if count >= 12 else "partial"


def in_frame_count(points: Any, width: int, height: int, front: Any = None) -> int:
    xy = np.asarray(points, dtype=np.float64)
    if xy.shape != (14, 2) or not np.isfinite(xy).all():
        raise ValueError("Framing requires finite original-pixel KP14")
    accepted = (
        (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    )
    if front is not None:
        mask = np.asarray(front, dtype=bool)
        if mask.shape != (14,):
            raise ValueError("Expected one depth sign per KP")
        accepted &= mask
    return int(accepted.sum())


def real_rows(root: Path, registry: Path) -> list[dict[str, Any]]:
    from src.tasks.court_detection.scaling.real_split import make_registry, real_source

    frozen = json.loads(registry.read_text())
    source = real_source(root)
    assert frozen == make_registry(
        source,
        version=frozen["version"],
        seed=frozen["seed"],
        val_fraction=frozen["val_fraction"],
    )
    split_by_id = {key: split for split, ids in frozen["splits"].items() for key in ids}
    rows: list[dict[str, Any]] = []
    for original_split in ("train", "val"):
        for record in source.records(original_split):
            count = in_frame_count(
                record.payload["keypoints"],
                record.payload["width"],
                record.payload["height"],
            )
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "split": split_by_id[record.sample_id],
                    "video_id": record.sample_id.rsplit("_", 1)[0],
                    "in_frame_kp": count,
                    "framing": group_for_count(count),
                }
            )
    assert len(rows) == len(split_by_id) == len({row["sample_id"] for row in rows})
    return sorted(rows, key=lambda row: row["sample_id"])


def synthetic_rows(version_path: Path) -> list[dict[str, Any]]:
    from src.synthetic_data_generation.dataset.court.sample_store import (
        open_court_store,
        read_court_manifest,
    )
    from src.synthetic_data_generation.dataset.court.scale.publication import (
        load_dataset_version,
    )

    version = load_dataset_version(version_path, verify_images=True)
    rows: list[dict[str, Any]] = []
    for scene in version["scenes"]:
        start = len(rows)
        for entry in scene["stores"]:
            root = version_path.parent / entry["path"]
            manifest = read_court_manifest(root, store=open_court_store(root))
            for sample in manifest["samples"]:
                target = sample["target_court"]["binding"]["court_instance_id"]
                courts = [
                    court
                    for court in sample["projection"]["courts"]
                    if court["court_instance_id"] == target
                ]
                assert len(courts) == 1
                classes = sorted(
                    courts[0]["classes"], key=lambda item: item["class_id"]
                )
                assert [item["class_id"] for item in classes] == list(range(14))
                points = [item["points"][0] for item in classes]
                width, height = sample["projection"]["resolution"]
                count = in_frame_count(
                    [point["uv"] for point in points],
                    width,
                    height,
                    [point["in_front"] for point in points],
                )
                rows.append(
                    {
                        "sample_id": f"{scene['scene_id']}:{sample['sample_id']}",
                        "split": "train",
                        "scene_id": scene["scene_id"],
                        "in_frame_kp": count,
                        "renderer_visible_kp": sample["qa"]["target_visible_kp"],
                        "framing": group_for_count(count),
                    }
                )
        assert len(rows) - start == scene["counts"]["qa_accepted"]
    assert len(rows) == len({row["sample_id"] for row in rows})
    return sorted(rows, key=lambda row: row["sample_id"])


def summarize(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    partitions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        partitions[row[field]].append(row)
    result = {}
    for key, selected in sorted(partitions.items()):
        counts = Counter(row["framing"] for row in selected)
        result[key] = {
            "images": len(selected),
            "groups": {group: counts[group] for group in GROUPS},
            "in_frame_kp_histogram": dict(
                sorted(Counter(row["in_frame_kp"] for row in selected).items())
            ),
        }
        if field == "split":
            result[key]["videos_per_group"] = {
                group: len(
                    {row["video_id"] for row in selected if row["framing"] == group}
                )
                for group in GROUPS
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--synthetic-version", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    real = real_rows(args.real_root, args.registry)
    summary = {
        "schema": "court_pilot_framing_v1",
        "rule": RULE,
        "real": summarize(real, "split"),
    }
    all_rows = [{"source": "real", **row} for row in real]
    if args.synthetic_version is not None:
        synthetic = synthetic_rows(args.synthetic_version)
        summary["synthetic"] = summarize(synthetic, "scene_id")
        all_rows.extend({"source": "synthetic", **row} for row in synthetic)
    assignments = args.output / "assignments.jsonl"
    assignments.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows)
    )
    summary["assignments"] = {
        "path": assignments.name,
        "count": len(all_rows),
        "bytes": assignments.stat().st_size,
        "sha256": hashlib.sha256(assignments.read_bytes()).hexdigest(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
