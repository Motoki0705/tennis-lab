"""Verify the existing B00–B03 publication and freeze the two train subsets."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from protocol import (
    CAPTURE_LIMITS,
    CONDITIONS,
    REAL_REGISTRY_SHA256,
    REAL_SPLIT_COUNTS,
    REAL_VERSION,
    SELECTION_SEED,
    annotate_view,
    capture_match,
    stratum_counts,
)

from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.dataset.court.scale.publication import (
    load_dataset_version,
    read_json,
)
from src.tasks.court_detection.scaling.real_split import make_registry, real_source
from src.utils.data.image_record_store import file_sha256
from src.utils.io import write_json_atomic

POOL_SHA256 = "3b685927e56aac0ab6bd8335d6d7e9ab654f461ed68f57e29db84648f86fbc4c"
PARTITION_SHA256 = "4355b16613b7e53a5ca76e1db87c55f1a18ee6ce04d1209ea4447bbf38524d43"
QUOTAS = {"B00": (64, 64), "B01": (43, 43, 42), "B02": (128,), "B03": (128,)}
MIN_VISIBLE_POINTS = 4
FIT_VAL_PER_SCENE = 16
DINO_CHECKPOINT = "dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


@dataclass(frozen=True, slots=True)
class View:
    sample_id: str
    group: str
    court: str
    jpeg_sha256: str
    azimuth_deg: float
    height_m: float
    hfov_deg: float
    capture_index: int
    capture_distance: float
    capture_offsets: tuple[float, ...]
    visible_kp: int

    @property
    def bins(self) -> tuple[int, int, int]:
        return (
            math.floor((self.azimuth_deg + 180) % 360 / 45),
            math.floor(self.height_m / 0.5),
            math.floor(round(self.hfov_deg, 8) / 10),
        )


def camera_features(
    camera: dict[str, Any], metric: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    camera_to_scene = np.asarray(camera["camera_to_scene"]).reshape(4, 4)
    intrinsics = camera["intrinsics"]
    if isinstance(intrinsics, dict):
        intrinsics = intrinsics["matrix"]
    focal_x = np.asarray(intrinsics).reshape(3, 3)[0, 0]
    hfov = math.degrees(2 * math.atan(camera["width"] / (2 * focal_x)))
    center = (metric @ camera_to_scene[:, 3])[:3]
    forward = metric[:3, :3] @ camera_to_scene[:3, 2]
    return center, forward / np.linalg.norm(forward), hfov


def candidate_views(
    pool: Path, scene: dict[str, Any], scenes_root: Path
) -> tuple[list[View], list[str], dict[str, str]]:
    asset = read_json(pool / scene["asset_manifest_path"])
    alignment = asset["alignment"]
    metric = np.asarray(
        alignment["metric_scene_adapter"]["metric_scene_from_nht_scene"]
    ).reshape(4, 4)
    courts = {
        row["court_instance_id"]: np.asarray(row["court_from_scene"]).reshape(4, 4)
        for row in alignment["layout"]["courts"]
    }
    capture_path = (
        scenes_root / scene["scene_id"] / "reconstruction/export/cameras.json"
    )
    capture_sha = file_sha256(capture_path)
    if [
        row["sha256"]
        for row in asset["files"]
        if Path(row["path"]).name == "cameras.json"
    ] != [capture_sha]:
        raise ValueError(
            f"Captured camera file disagrees with the frozen asset: {capture_path}"
        )
    cameras = read_json(capture_path)["cameras"]
    ids = [camera["camera_id"] for camera in cameras]
    centers, forwards, hfovs = map(
        np.asarray,
        zip(*(camera_features(camera, metric) for camera in cameras), strict=True),
    )
    court_centers = {
        court: centers @ transform[:3, :3].T + transform[:3, 3]
        for court, transform in courts.items()
    }
    identity = np.eye(4)
    views = []
    for entry in scene["stores"]:
        store = open_court_store(pool / entry["path"])
        for index in range(len(store)):
            record = store.record(index)
            court = record["target_court"]["binding"]["court_instance_id"]
            projection = next(
                p
                for p in record["projection"]["courts"]
                if p["court_instance_id"] == court
            )
            visible = sum(p["renderer_visible"] is True for p in projection["classes"])
            center, forward, hfov = camera_features(record["camera"], identity)
            transform = courts[court]
            x, y, z = transform[:3, :3] @ center + transform[:3, 3]
            distance, closest, offsets = capture_match(
                np.array([x, y, z]),
                forward,
                hfov,
                court_centers[court],
                forwards,
                hfovs,
            )
            views.append(
                View(
                    record["sample_id"],
                    record["trajectory_group_id"],
                    court,
                    bytes(store.arrays["sha256"][index]).decode("ascii"),
                    math.degrees(math.atan2(x, y)),
                    float(z),
                    hfov,
                    closest,
                    distance,
                    tuple(float(offset) for offset in offsets),
                    visible,
                )
            )
    if len(views) != scene["counts"]["qa_accepted"] or len(
        {view.sample_id for view in views}
    ) != len(views):
        raise ValueError("Selection pool differs from the verified membership.")
    return (
        views,
        ids,
        {"cameras_sha256": capture_sha, "asset_sha256": scene["asset_manifest_sha256"]},
    )


def select_cells(candidates: Iterable[View], count: int) -> list[View]:
    """Visit occupied azimuth × height × HFOV cells in stable seeded order."""
    ordered = sorted(
        candidates,
        key=lambda view: hashlib.sha256(
            f"{SELECTION_SEED}:{view.sample_id}".encode()
        ).digest(),
    )
    if len(ordered) < count:
        raise ValueError(f"Needed {count} eligible views, found {len(ordered)}.")
    cells: dict[tuple[int, int, int], list[View]] = defaultdict(list)
    for view in ordered:
        cells[view.bins].append(view)
    selected = []
    for depth in range(max(map(len, cells.values()))):
        for members in cells.values():
            if depth < len(members):
                selected.append(members[depth])
                if len(selected) == count:
                    return selected
    raise RuntimeError("Cell traversal failed to cover its candidates.")


def matched_selection(train: list[View], count: int) -> tuple[list[View], list[View]]:
    narrow = select_cells((view for view in train if view.capture_distance <= 1), count)
    anchors = narrow[: count // 4]
    broad = anchors + select_cells(
        (view for view in train if view.capture_distance > 1), count - len(anchors)
    )
    if len(
        {view.sample_id for view in narrow} & {view.sample_id for view in broad}
    ) != len(anchors):
        raise ValueError("The paired subsets overlap beyond their fixed anchors.")
    return narrow, broad


def summarize_views(views: list[View]) -> dict[str, Any]:
    values = {
        "azimuth_deg": [view.azimuth_deg for view in views],
        "height_m": [view.height_m for view in views],
        "hfov_deg": [view.hfov_deg for view in views],
        "capture_distance": [view.capture_distance for view in views],
        "visible_kp": [view.visible_kp for view in views],
    }
    bins_by_axis = {}
    for axis, name in enumerate(("azimuth_deg", "height_m", "hfov_deg")):
        bins = dict(sorted(Counter(view.bins[axis] for view in views).items()))
        effective = math.exp(
            -sum((n / len(views)) * math.log(n / len(views)) for n in bins.values())
        )
        bins_by_axis[name] = {"counts": bins, "effective_bins": effective}
    return {
        "images": len(views),
        "trajectories": len({view.group for view in views}),
        "near_capture": sum(view.capture_distance <= 1 for view in views),
        "quantiles_min_p05_p50_p95_max": {
            key: np.quantile(items, [0, 0.05, 0.5, 0.95, 1])
            for key, items in values.items()
        },
        "bins_by_axis": bins_by_axis,
    }


def copy_inputs(source_pool: Path, tcd: Path, inputs: Path) -> None:
    """Hard-link immutable shared payloads into a fresh experiment owner."""
    pool = inputs / "pool"
    pool.mkdir(parents=True)
    for name in ("assets", "curves", "memberships", "plans", "requests", "stores"):
        shutil.copytree(source_pool.parent / name, pool / name, copy_function=os.link)
    os.link(source_pool, pool / source_pool.name)
    os.link(
        source_pool.parent / "trajectory-partition.json",
        pool / "trajectory-partition.json",
    )
    shutil.copytree(tcd, inputs / "tcd", copy_function=os.link)


def prepare(source_pool: Path, scenes_root: Path, tcd: Path, output: Path) -> None:
    started = time.monotonic()
    if output.exists():
        raise FileExistsError(output)
    partition_path = source_pool.parent / "trajectory-partition.json"
    for path, expected in (
        (source_pool, POOL_SHA256),
        (partition_path, PARTITION_SHA256),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"SHA-256 mismatch for {path}: expected {expected}, found {actual}"
            )
    print("Verifying the frozen synthetic publication", flush=True)
    publication = load_dataset_version(source_pool, verify_images=True)
    partition = read_json(partition_path)["scenes"]
    if [scene["scene_id"] for scene in publication["scenes"]] != list(QUOTAS):
        raise ValueError("Expected exactly B00–B03 in the source publication.")

    source = real_source(tcd)
    source.image_store.validate(decode=False)
    registry = make_registry(source, version=REAL_VERSION, seed=SELECTION_SEED)
    if registry["sha256"] != REAL_REGISTRY_SHA256:
        raise ValueError("Real-video registry differs from the pinned population.")
    records = {
        record.sample_id: record
        for split in source.available_splits
        for record in source.records(split)
    }
    strata = {
        split: [annotate_view(key, records[key].payload) for key in ids]
        for split, ids in registry["splits"].items()
    }
    counts = {split: stratum_counts(views) for split, views in strata.items()}
    for split, (images, videos) in REAL_SPLIT_COUNTS.items():
        if counts[split]["all"] != {"images": images, "videos": videos} or any(
            view["visible_kp"] < MIN_VISIBLE_POINTS for view in strata[split]
        ):
            raise ValueError("The real split or the coverage for H scoring changed.")
    print(f"Real strata: {counts}", flush=True)

    train_rows: dict[str, list[dict[str, str]]] = {
        condition: [] for condition in CONDITIONS
    }
    val_rows = []
    selection: dict[str, Any] = {}
    selected_records = []
    jpeg_ids = {condition: set() for condition in CONDITIONS}
    for scene in publication["scenes"]:
        scene_id = scene["scene_id"]
        candidates, capture_ids, provenance = candidate_views(
            source_pool.parent, scene, scenes_root
        )
        groups = {
            item["trajectory_id"]: item["assigned_split"]
            for item in partition[scene_id]["groups"]
        }
        eligible = [
            view for view in candidates if view.visible_kp >= MIN_VISIBLE_POINTS
        ]
        train = [view for view in eligible if groups[view.group] == "train"]
        validation = select_cells(
            (view for view in eligible if groups[view.group] == "val"),
            FIT_VAL_PER_SCENE,
        )
        val_rows.extend(
            {"scene_id": scene_id, "sample_id": view.sample_id} for view in validation
        )
        scene_selection = {condition: [] for condition in CONDITIONS}
        court_selection = {}
        for court, quota in zip(
            sorted({view.court for view in candidates}), QUOTAS[scene_id], strict=True
        ):
            same_court = [view for view in train if view.court == court]
            subsets = matched_selection(same_court, quota)
            court_selection[court] = {
                "eligible_near": sum(view.capture_distance <= 1 for view in same_court),
                "eligible_far": sum(view.capture_distance > 1 for view in same_court),
                **{
                    condition: summarize_views(views)
                    for condition, views in zip(CONDITIONS, subsets, strict=True)
                },
            }
            for condition, views in zip(CONDITIONS, subsets, strict=True):
                scene_selection[condition].extend(views)
        for condition, views in scene_selection.items():
            for view in views:
                if view.jpeg_sha256 in jpeg_ids[condition]:
                    raise ValueError("Duplicate JPEG bytes in a training subset.")
                jpeg_ids[condition].add(view.jpeg_sha256)
                train_rows[condition].append(
                    {"scene_id": scene_id, "sample_id": view.sample_id}
                )
                selected_records.append(
                    {
                        "condition": condition,
                        "scene_id": scene_id,
                        "sample_id": view.sample_id,
                        "court": view.court,
                        "trajectory": view.group,
                        "azimuth_deg": view.azimuth_deg,
                        "height_m": view.height_m,
                        "hfov_deg": view.hfov_deg,
                        "nearest_capture_id": capture_ids[view.capture_index],
                        "capture_distance": view.capture_distance,
                        "capture_offsets": view.capture_offsets,
                        "visible_kp": view.visible_kp,
                        "jpeg_sha256": view.jpeg_sha256,
                    }
                )
        selection[scene_id] = {
            **provenance,
            "pool_images": len(candidates),
            "capture_cameras": len(capture_ids),
            "courts": court_selection,
            **{
                condition: summarize_views(views)
                for condition, views in scene_selection.items()
            },
        }
        print(
            f"{scene_id} selections: {[(key, len(views)) for key, views in scene_selection.items()]}",
            flush=True,
        )

    axes = ("azimuth_deg", "height_m", "hfov_deg")
    coverage = {
        condition: {
            axis: float(
                np.mean(
                    [
                        scene[condition]["bins_by_axis"][axis]["effective_bins"]
                        for scene in selection.values()
                    ]
                )
            )
            for axis in axes
        }
        for condition in CONDITIONS
    }
    for axis in axes:
        if coverage["broad"][axis] <= coverage["narrow"][axis]:
            raise ValueError(
                f"The selected broad condition has not expanded {axis}: {coverage}"
            )
    print(f"Effective bins: {coverage}", flush=True)
    if sum(scene["pool_images"] for scene in selection.values()) != 19934:
        raise ValueError("Unexpected pool cardinality.")
    intersection = jpeg_ids["narrow"] & jpeg_ids["broad"]
    if len(intersection) != sum(n // 4 for quotas in QUOTAS.values() for n in quotas):
        raise ValueError("Unexpected train JPEG overlap between the two conditions.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".viewpoint-prep-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "campaign"
        copy_inputs(source_pool, tcd, stage / "inputs")
        pool = stage / "inputs/pool"
        manifests = {}
        for label, split, rows in [
            *((condition, "train", train_rows[condition]) for condition in CONDITIONS),
            ("common", "val", val_rows),
        ]:
            path = pool / f"{label}.{split}.subset.json"
            write_json_atomic(
                path,
                {
                    "schema": "court_dataset_subset_v1",
                    "source_manifest": source_pool.name,
                    "source_manifest_sha256": POOL_SHA256,
                    "selection_seed": SELECTION_SEED,
                    "split": split,
                    "samples": rows,
                },
            )
            manifests[path.name] = {"count": len(rows), "sha256": file_sha256(path)}
        write_json_atomic(stage / "registry.json", registry)
        write_json_atomic(stage / "strata.json", strata)
        write_json_atomic(stage / "selected-views.json", selected_records)
        write_json_atomic(
            stage / "preparation.json",
            {
                "schema": "court_viewpoint_comparison_preparation_v1",
                "pool_sha256": POOL_SHA256,
                "partition_sha256": PARTITION_SHA256,
                "capture_limits_xy_m_z_m_axis_deg_hfov_deg": CAPTURE_LIMITS,
                "manifests": manifests,
                "real_registry_sha256": registry["sha256"],
                "source_tcd_hashes": registry["source_sha256"],
                "real_strata_sha256": file_sha256(stage / "strata.json"),
                "real_strata_counts": counts,
                "synthetic_scenes": selection,
                "effective_bins_scene_mean": coverage,
                "shared_training_jpegs": len(intersection),
                "selected_views_sha256": file_sha256(stage / "selected-views.json"),
                "seconds": time.monotonic() - started,
            },
        )
        stage.rename(output)
    print(f"Prepared {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-pool", "scenes-root", "tcd", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(
        args.source_pool.resolve(strict=True),
        args.scenes_root.resolve(strict=True),
        args.tcd.resolve(strict=True),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
