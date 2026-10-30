"""Audit the completed fixed-size subsets and their selected QA distributions.

The generation job already performs full strict JPEG/teacher validation twice.
This adds the independent subset, original-input and media acceptance checks;
it never changes the pool, selection manifests, QA, or trajectory partition.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.dataset.court.scale.core import (
    canonical_bytes,
    digest,
)
from src.synthetic_data_generation.dataset.court.scale.curves import Curve
from src.synthetic_data_generation.dataset.court.scale.publication import read_json
from src.utils.data.image_record_store import file_sha256

EXPECTED = (
    {"train": {"B00": 5500}, "val": {"B00": 500}},
    {"train": {"B00": 2750, "B01": 2750}, "val": {"B00": 250, "B01": 250}},
    {
        "train": {"B00": 1833, "B01": 1833, "B02": 1834},
        "val": {"B00": 167, "B01": 167, "B02": 166},
    },
    {
        "train": dict.fromkeys(("B00", "B01", "B02", "B03"), 1375),
        "val": dict.fromkeys(("B00", "B01", "B02", "B03"), 125),
    },
)


def qa_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "target_visible_kp_zero": sum(
            r["qa"]["target_visible_kp"] == 0 for r in records
        ),
        "target_visible_kp_lt4": sum(r["qa"]["target_visible_kp"] < 4 for r in records),
        "target_visible_kp_histogram": dict(
            sorted(
                Counter(str(r["qa"]["target_visible_kp"]) for r in records).items(),
                key=lambda item: int(item[0]),
            )
        ),
        "coverage": dict(
            sorted(Counter(r["qa"]["target_coverage"] for r in records).items())
        ),
        "trajectory_count": len({r["trajectory_id"] for r in records}),
        "roi_absent": sum(r["qa"]["roi_hole_fraction"] is None for r in records),
        "roi_hole_max": max(
            (
                r["qa"]["roi_hole_fraction"]
                for r in records
                if r["qa"]["roi_hole_fraction"] is not None
            ),
            default=None,
        ),
    }


def verify(root: Path, repo: Path) -> dict[str, Any]:
    completion = read_json(root / "generation-completion.json")
    manifest = root / completion["source_manifest"]
    manifest_hash = file_sha256(manifest)
    assert manifest_hash == completion["source_manifest_sha256"]
    version = read_json(manifest)
    assert version["dataset_version"] == digest(
        {k: v for k, v in version.items() if k != "dataset_version"}
    )
    partition = read_json(root / "trajectory-partition.json")["scenes"]
    audit = read_json(root / "plan-audit.json")
    assert audit["request_sha256"] == version["request_sha256"]
    records: dict[tuple[str, str], dict[str, Any]] = {}
    source_groups: dict[tuple[str, str], str] = {}
    pool_scenes = {}
    source_files_checked = []
    prior = read_json(
        repo / "knowledge/runs/run-i983-court-l512-profile-s42/inputs.json"
    )
    for scene in version["scenes"]:
        sid = scene["scene_id"]
        plan = read_json(root / scene["plan_path"])
        asset = read_json(root / scene["asset_manifest_path"])
        for item in asset["files"]:
            assert file_sha256(Path(item["path"])) == item["sha256"]
            source_files_checked.append(item["path"])
        old_root = (
            repo / "data/synthetic_data_generation/scenes" / sid / "datasets/court"
        )
        original = open_court_store(old_root)
        assert (
            plan["saved_source"]["trajectory_groups"]
            == original.metadata["trajectory_groups"]
        )
        pinned = next(
            a
            for a in prior["assets"]
            if a["destination"]
            == f"data/synthetic_data_generation/scenes/{sid}/datasets/court"
        )
        assert file_sha256(old_root / "dataset.json") == pinned["manifest_sha256"]
        for relative, sha in pinned["files"]:
            assert file_sha256(old_root / "samples" / relative) == sha
            source_files_checked.append(str(old_root / "samples" / relative))
        original_train = {
            g["trajectory"]["trajectory_group_id"]
            for g in original.metadata["trajectory_groups"]
            if g["split"] == "train"
        }
        physical_curves = set()
        single_court = len(asset["alignment"]["layout"]["courts"]) == 1
        for reference in plan["trajectory_group_refs"]:
            group = read_json(root / reference["path"])
            assert file_sha256(root / reference["path"]) == reference["sha256"]
            raw = group["provenance"]["saved_group"]
            gid = raw["trajectory"]["trajectory_group_id"]
            assert raw["split"] == "train" and gid in original_train
            source_groups[sid, group["trajectory_id"]] = gid
            physical = Curve.from_dict(group["curve"]).physical_key()
            if single_court:
                physical["center_court_id"] = asset["alignment"]["layout"]["courts"][0][
                    "court_instance_id"
                ]
            fingerprint = digest(physical)
            assert fingerprint not in physical_curves
            physical_curves.add(fingerprint)
        assert {v for (s, _), v in source_groups.items() if s == sid} == original_train
        images, cameras, scene_records = set(), set(), []
        for entry in scene["stores"]:
            store_root = root / entry["path"]
            assert file_sha256(store_root / "dataset.json") == entry["dataset_sha256"]
            assert (
                file_sha256(store_root / "samples/index.npz") == entry["index_sha256"]
            )
            store = open_court_store(store_root)
            for row in range(len(store)):
                record = store.record(row)
                key = sid, record["sample_id"]
                assert key not in records and record["split"] == "train"
                assert (sid, record["trajectory_id"]) in source_groups
                image_sha = str(store.arrays["sha256"][row])
                camera_sha = digest(
                    [
                        record["camera"]["camera_to_scene"],
                        record["camera"]["intrinsics"],
                    ]
                )
                assert image_sha not in images and camera_sha not in cameras
                images.add(image_sha)
                cameras.add(camera_sha)
                records[key] = record
                scene_records.append(record)
        assert len(scene_records) == scene["counts"]["qa_accepted"]
        assert (
            scene["counts"]["qa_rejected"]
            == scene["counts"]["pending"]
            == scene["counts"]["failed"]
            == 0
        )
        assert set(partition[sid]["train"]).isdisjoint(partition[sid]["val"])
        pool_scenes[sid] = {
            **qa_summary(scene_records),
            "unique_jpeg": len(images),
            "unique_camera": len(cameras),
            "source_train_groups": len(original_train),
            "physical_curve_duplicates": 0,
        }
    subsets, prior_by_scene = {}, {}
    train_groups, val_groups = set(), set()
    all_selected = set()
    for condition, expected in enumerate(EXPECTED, 1):
        condition_ids = set()
        for split in ("train", "val"):
            name = f"scene-{condition}.{split}.subset.json"
            value = read_json(root / name)
            assert set(value) == {
                "schema",
                "source_manifest",
                "source_manifest_sha256",
                "selection_seed",
                "split",
                "samples",
            }
            assert (
                value["schema"] == "court_dataset_subset_v1"
                and value["split"] == split
                and value["selection_seed"] == 42
            )
            assert (
                value["source_manifest"] == manifest.name
                and value["source_manifest_sha256"] == manifest_hash
            )
            ids = [(r["scene_id"], r["sample_id"]) for r in value["samples"]]
            assert len(ids) == len(set(ids)) == (5500 if split == "train" else 500)
            assert dict(Counter(s for s, _ in ids)) == expected[split]
            assert not condition_ids.intersection(ids)
            condition_ids.update(ids)
            all_selected.update(ids)
            selected = [records[key] for key in ids]
            per_scene = {}
            for sid in expected[split]:
                scene_ids = [sample for s, sample in ids if s == sid]
                previous = prior_by_scene.get((sid, split))
                if previous is not None:
                    assert scene_ids == previous[: len(scene_ids)]
                prior_by_scene[sid, split] = scene_ids
                scene_rows = [records[sid, sample] for sample in scene_ids]
                assert {r["trajectory_id"] for r in scene_rows} <= set(
                    partition[sid][split]
                )
                for record in scene_rows:
                    (train_groups if split == "train" else val_groups).add(
                        (sid, record["trajectory_id"])
                    )
                per_scene[sid] = {
                    **qa_summary(scene_rows),
                    "arc_octile_counts": dict(
                        sorted(
                            Counter(
                                min(7, int(Fraction(*r["scale"]["arc_fraction"]) * 8))
                                for r in scene_rows
                            ).items()
                        )
                    ),
                }
            subsets[name] = {
                "sha256": file_sha256(root / name),
                **qa_summary(selected),
                "scenes": per_scene,
            }
        assert len(condition_ids) == 6000
    assert train_groups.isdisjoint(val_groups)
    media = {}
    for path in sorted((root / "review").glob("*.mp4")):
        capture = cv2.VideoCapture(str(path))
        assert capture.isOpened()
        frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (270, 960, 3)
            frames += 1
        capture.release()
        assert frames == 48
        media[path.name] = {
            "frames_decoded": frames,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    assert len(media) == 4
    result = {
        "schema": "court_scene_diversity_subset_audit_v1",
        "source_manifest": manifest.name,
        "source_manifest_sha256": manifest_hash,
        "source_files_checked": sorted(set(source_files_checked)),
        "strict_pool_validation_evidence": "Completed generation job executes load_dataset_version(verify_images=True) in both execute_request and finalize before its terminal success.",
        "pool_scenes": pool_scenes,
        "subsets": subsets,
        "unique_selected_images_across_conditions": len(all_selected),
        "old_train_only": True,
        "train_val_trajectory_disjoint": True,
        "nested_prefixes": True,
        "videos": media,
    }
    output = root / "subset-quality-audit.json"
    payload = canonical_bytes(result)
    if output.exists():
        assert output.read_bytes() == payload
    else:
        output.write_bytes(payload)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--repo-root", default=Path("/home/kamimura/projects/tennis-lab"), type=Path
    )
    args = parser.parse_args()
    result = verify(args.output_root.resolve(), args.repo_root.resolve())
    print(
        json.dumps(
            {
                "output": str(args.output_root / "subset-quality-audit.json"),
                "unique_selected_images": result[
                    "unique_selected_images_across_conditions"
                ],
                "subsets": {
                    name: {
                        key: item[key]
                        for key in (
                            "count",
                            "target_visible_kp_zero",
                            "target_visible_kp_lt4",
                            "coverage",
                        )
                    }
                    for name, item in result["subsets"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
