"""Reproduce fixed-size scene diversity data from the pinned #983 train curves."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.synthetic_data_generation.alignment.contracts import AlignmentResult
from src.synthetic_data_generation.dataset.court.sample_store import open_court_store
from src.synthetic_data_generation.dataset.court.scale.core import (
    QualityPolicy,
    canonical_bytes,
    stable_id,
)
from src.synthetic_data_generation.dataset.court.scale.publication import (
    execute_request,
    freeze_request,
    iter_plan_groups,
    load_dataset_version,
    read_json,
    write_content,
)
from src.synthetic_data_generation.dataset.court.scale.quality import evaluate_render
from src.synthetic_data_generation.dataset.court.scale.runtime import (
    LocalNHTRenderer,
    load_scene_inputs,
    renderer_fingerprint,
)
from src.synthetic_data_generation.dataset.court.scale.saved_train import (
    iter_saved_train_groups,
    pin_saved_source,
    saved_train_header,
)
from src.synthetic_data_generation.dataset.court.scale.validation import (
    validate_plan_geometry,
)
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.visualization.overlays import render_court_overlay
from src.synthetic_data_generation.visualization.sources import CourtSourceFrame
from src.utils.data.image_record_store import file_sha256

SCENES = ("B00", "B01", "B02", "B03")
DENSITY = {"B00": 2, "B01": 2, "B02": 1, "B03": 1}
REQUIRED = {"B00": 6000, "B01": 3000, "B02": 2000, "B03": 1500}
TRAIN_COUNTS = (
    {"B00": 5500},
    {"B00": 2750, "B01": 2750},
    {"B00": 1833, "B01": 1833, "B02": 1834},
    dict.fromkeys(SCENES, 1375),
)
VAL_COUNTS = (
    {"B00": 500},
    {"B00": 250, "B01": 250},
    {"B00": 167, "B01": 167, "B02": 166},
    dict.fromkeys(SCENES, 125),
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(
                f"Refusing to overwrite conflicting immutable output: {path}"
            )
        return
    path.write_bytes(payload)


def plan(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prior = read_json(
        args.repo_root / "knowledge/runs/run-i983-court-l512-profile-s42/inputs.json"
    )
    plans, assets, audit, partitions = [], [], {}, {}
    for scene_id in SCENES:
        scene_root = args.repo_root / "data/synthetic_data_generation/scenes" / scene_id
        source_root = scene_root / "datasets/court"
        pinned = next(
            a
            for a in prior["assets"]
            if a["destination"]
            == f"data/synthetic_data_generation/scenes/{scene_id}/datasets/court"
        )
        index_hash = dict(pinned["files"])["index.npz"]
        source = pin_saved_source(
            source_root,
            dataset_sha256=pinned["manifest_sha256"],
            index_sha256=index_hash,
        )
        inputs = load_scene_inputs(scene_root, replay=False)
        candidate = saved_train_header(
            source=source,
            seed=42,
            density_level=DENSITY[scene_id],
            cameras=inputs["cameras"],
            layout=inputs["layout"],
        )
        references, totals, group_audit = [], Counter(), []
        coarse_by_source = {}
        for group in iter_saved_train_groups(candidate, inputs["layout"]):
            validate_plan_geometry(
                {**candidate, "trajectory_groups": [group]}, inputs["layout"]
            )
            group_path, group_hash = write_content(root, "curves", group)
            references.append(
                {
                    "trajectory_id": group["trajectory_id"],
                    "path": group_path,
                    "sha256": group_hash,
                }
            )
            raw = group["provenance"]["saved_group"]
            base_count = raw["sample_count"]
            coarse_by_source[raw["trajectory"]["trajectory_group_id"]] = {
                int(Fraction(*r["scale"]["arc_fraction"]) * base_count): r
                for r in group["samples"]
                if r["scale"]["birth_level"] == 0
            }
            totals.update(
                candidates=len(group["samples"]),
                geometry_pass=sum(not r["reasons"] for r in group["samples"]),
                duplicate_groups=int(group["duplicate_of"] is not None),
            )
            group_audit.append(
                {
                    "trajectory_id": group["trajectory_id"],
                    "source_group_id": raw["trajectory"]["trajectory_group_id"],
                    "geometry_pass": sum(not r["reasons"] for r in group["samples"]),
                }
            )
        original = open_court_store(source_root)
        old_train_rows = 0
        for row in range(len(original)):
            record = original.record(row)
            if record["split"] != "train":
                continue
            expected = coarse_by_source[record["trajectory_group_id"]][
                record["trajectory_frame_index"]
            ]
            old_pose = np.asarray(record["camera"]["camera_to_scene"]).reshape(4, 4)
            new_pose = np.asarray(expected["camera"]["camera_to_scene"]).reshape(4, 4)
            if (
                not np.allclose(old_pose[:3, 3], new_pose[:3, 3], rtol=0, atol=1e-9)
                or record["camera"]["intrinsics"] != expected["camera"]["intrinsics"]
            ):
                raise ValueError(
                    f"Old train camera position/intrinsics changed: {scene_id}/{record['sample_id']}"
                )
            old_train_rows += 1
        plans.append(
            {
                **candidate,
                "trajectory_storage": "curve_files_v1",
                "trajectory_group_refs": references,
            }
        )
        asset = inputs["asset"]
        asset["files"].extend(
            {"path": str(source_root / rel), "sha256": sha}
            for rel, sha in (
                ("dataset.json", pinned["manifest_sha256"]),
                ("samples/index.npz", index_hash),
            )
        )
        assets.append(asset)
        held = sorted(
            group_audit,
            key=lambda g: stable_id(
                "synthetic_validation_group", 42, scene_id, g["source_group_id"]
            ),
        )[: math.ceil(len(group_audit) * 0.1)]
        val_ids = {g["trajectory_id"] for g in held}
        train_geometry = sum(
            g["geometry_pass"] for g in group_audit if g["trajectory_id"] not in val_ids
        )
        val_geometry = sum(g["geometry_pass"] for g in held)
        needed_train = max(c.get(scene_id, 0) for c in TRAIN_COUNTS)
        needed_val = max(c.get(scene_id, 0) for c in VAL_COUNTS)
        if train_geometry < needed_train or val_geometry < needed_val:
            raise ValueError(
                f"Increase fixed-curve density before freezing: {scene_id} train {train_geometry}/{needed_train} val {val_geometry}/{needed_val}"
            )
        partitions[scene_id] = {
            "train": [
                g["trajectory_id"]
                for g in group_audit
                if g["trajectory_id"] not in val_ids
            ],
            "val": [g["trajectory_id"] for g in held],
            "groups": [
                {
                    **g,
                    "assigned_split": "val"
                    if g["trajectory_id"] in val_ids
                    else "train",
                }
                for g in group_audit
            ],
        }
        audit[scene_id] = {
            **dict(totals),
            "original_train_rows_checked": old_train_rows,
            "source_train_groups": len(references),
            "density_level": DENSITY[scene_id],
            "source_index_sha256": index_hash,
            "source_dataset_sha256": pinned["manifest_sha256"],
            "new_jitter": "PCG64 stable point ID uniform-area disk; seed=42; legacy index stream is not reused",
        }
        audit[scene_id].update(
            train_geometry_pass=train_geometry,
            val_geometry_pass=val_geometry,
            val_trajectory_count=len(held),
        )
        print(
            json.dumps(
                {"stage": "planned_scene", "scene_id": scene_id, **audit[scene_id]}
            ),
            flush=True,
        )
    request = freeze_request(
        root,
        plans=plans,
        assets=assets,
        quality=QualityPolicy(),
        renderer=renderer_fingerprint(),
    )
    write_json(
        root / "trajectory-partition.json",
        {
            "schema": "court_scene_diversity_trajectory_partition_v1",
            "rule": "ceil(10% source train groups), smallest stable_id(synthetic_validation_group,42,scene_id,source_group_id)",
            "seed": 42,
            "scenes": partitions,
        },
    )
    write_json(
        root / "plan-audit.json",
        {
            "schema": "court_scene_diversity_plan_audit_v1",
            "prior_inputs_sha256": file_sha256(
                args.repo_root
                / "knowledge/runs/run-i983-court-l512-profile-s42/inputs.json"
            ),
            "request_path": request.relative_to(root).as_posix(),
            "request_sha256": file_sha256(request),
            "scenes": audit,
        },
    )
    print(json.dumps({"request": str(request)}), flush=True)


def request_path(root: Path) -> Path:
    return root / read_json(root / "plan-audit.json")["request_path"]


def overlay(rgb: np.ndarray, record: dict[str, Any]) -> np.ndarray:
    frame = CourtSourceFrame(
        rgb=rgb.astype(np.float32) / 255,
        sample_id=record["sample_id"],
        view_id=record["view_id"],
        trajectory_frame_index=record["trajectory_frame_index"],
        projection=record["projection"],
        schema_version=CourtDatasetSchemaVersion.V3,
    )
    return render_court_overlay(
        frame, trajectory_id=record["trajectory_id"], show_metadata=False
    )


def smoke(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    request = read_json(request_path(root))
    renderer = LocalNHTRenderer(root, fingerprint=request["renderer"], batch_size=1)
    qa = {}
    media = root / "smoke"
    media.mkdir(exist_ok=False)
    for scene in request["scenes"]:
        candidate = read_json(root / scene["plan_path"])
        layout = AlignmentResult.from_dict(
            read_json(root / scene["asset_manifest_path"])["alignment"]
        ).layout
        groups = iter_plan_groups(root, candidate)
        group = next(g for g in groups if any(not r["reasons"] for r in g["samples"]))
        eligible = [r for r in group["samples"] if not r["reasons"]]
        selected = [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, 8)]
        by_id = {r["sample_id"]: r for r in selected}
        frames = []
        records = []
        for sample_id, arrays in renderer(scene, selected):
            record = evaluate_render(
                by_id[sample_id],
                arrays,
                layout=layout,
                policy=QualityPolicy(**request["quality"]),
                near_plane_m=candidate["recipe"]["near_plane_m"],
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "qa": record["qa"],
                    "reasons": record["reasons"],
                }
            )
            rgb = np.rint(np.clip(arrays.rgb, 0, 1) * 255).astype(np.uint8)
            combined = np.hstack([rgb, overlay(rgb, record)])
            frames.append(cv2.resize(combined, (960, 270)))
        sheet = np.vstack(
            [np.hstack(frames[i : i + 2]) for i in range(0, len(frames), 2)]
        )
        cv2.imwrite(
            str(media / f"{scene['scene_id']}-rgb-overlay.jpg"),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
        qa[scene["scene_id"]] = records
        print(
            json.dumps(
                {
                    "stage": "smoke_scene",
                    "scene_id": scene["scene_id"],
                    "rendered": len(records),
                    "accepted": sum(not r["reasons"] for r in records),
                }
            ),
            flush=True,
        )
    write_json(media / "qa.json", qa)


def render(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    request = read_json(request_path(root))
    renderer = LocalNHTRenderer(root, fingerprint=request["renderer"], batch_size=1)
    began = time.monotonic()
    caches = 0

    def completed(context: Any, completion: dict[str, Any], reused: bool) -> None:
        nonlocal caches
        caches += 1
        print(
            json.dumps(
                {
                    "stage": "cache_complete",
                    "scene_id": context.scene_id,
                    "cache_key": context.cache_key,
                    "reused": reused,
                    "states": dict(Counter(r["state"] for r in completion["records"])),
                    "caches_complete": caches,
                    "elapsed_seconds": round(time.monotonic() - began, 2),
                }
            ),
            flush=True,
        )

    version = execute_request(
        root, request_path(root), renderer, complete_cache=completed
    )
    write_json(
        root / "generation-completion.json",
        {
            "schema": "court_scene_diversity_generation_completion_v1",
            "source_manifest": version.name,
            "source_manifest_sha256": file_sha256(version),
            "elapsed_seconds": time.monotonic() - began,
        },
    )
    print(
        json.dumps({"stage": "generation_complete", "dataset_version": str(version)}),
        flush=True,
    )


def nested_order(records: list[dict[str, Any]], seed: int = 42) -> list[str]:
    """Proportional fair trajectory scheduling + progressive circular maximin.

    Group weights are the accepted pool counts. At every prefix the next group
    has the earliest virtual finish time. Within a trajectory, each next point
    maximizes distance in normalized arc length from already selected points.
    Stable hashes break ties. Input row order never affects this ranking.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["trajectory_id"]].append(record)
    ordered = {}
    schedule = []
    for gid, members in sorted(groups.items()):
        members.sort(key=lambda r: Fraction(*r["scale"]["arc_fraction"]))
        if len({r["sample_id"] for r in members}) != len(members):
            raise ValueError("Duplicate sample in the accepted pool.")
        q = np.array([float(Fraction(*r["scale"]["arc_fraction"])) for r in members])
        tie_order = np.array(
            [
                int(stable_id("subset_point_tie", seed, r["sample_id"])[:12], 16)
                for r in members
            ]
        )
        distance = np.full(len(q), np.inf)
        chosen = np.zeros(len(q), dtype=bool)
        order = []
        for _ in members:
            candidates = np.flatnonzero(~chosen)
            best = distance[candidates].max()
            tied = candidates[
                np.isclose(distance[candidates], best, rtol=0, atol=1e-14)
            ]
            index = int(tied[np.argmin(tie_order[tied])])
            order.append(members[index]["sample_id"])
            chosen[index] = True
            delta = np.abs(q - q[index])
            distance = np.minimum(distance, np.minimum(delta, 1 - delta))
        ordered[gid] = order
        heapq.heappush(
            schedule,
            (Fraction(1, len(order)), stable_id("subset_group_tie", seed, gid), gid, 0),
        )
    result = []
    while schedule:
        _, tie, gid, offset = heapq.heappop(schedule)
        result.append(ordered[gid][offset])
        if offset + 1 < len(ordered[gid]):
            heapq.heappush(
                schedule,
                (Fraction(offset + 2, len(ordered[gid])), tie, gid, offset + 1),
            )
    return result


def finalize(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    receipt = read_json(root / "generation-completion.json")
    version_path = root / receipt["source_manifest"]
    version = load_dataset_version(version_path, verify_images=True)
    rankings, diagnostics = {}, {}
    partitions = read_json(root / "trajectory-partition.json")["scenes"]
    lookup, video_groups = {}, {}
    for scene in version["scenes"]:
        sid = scene["scene_id"]
        rows, camera_hashes, image_hashes = [], set(), set()
        by_id = {}
        for entry in scene["stores"]:
            store = open_court_store(root / entry["path"])
            for row in range(len(store)):
                record = store.record(row)
                rows.append(record)
                camera_hashes.add(
                    hashlib.sha256(
                        canonical_bytes(
                            [
                                record["camera"]["camera_to_scene"],
                                record["camera"]["intrinsics"],
                            ]
                        )
                    ).hexdigest()
                )
                image_hashes.add(str(store.arrays["sha256"][row]))
                by_id[record["sample_id"]] = (entry["path"], row)
        if len(rows) < REQUIRED[sid]:
            raise ValueError(
                f"QA accepted pool insufficient: {sid}: {len(rows)} < {REQUIRED[sid]}"
            )
        if len(camera_hashes) != len(rows) or len(image_hashes) != len(rows):
            raise ValueError(f"Duplicate camera/image in accepted {sid} pool.")
        rankings[sid] = {}
        accepted_splits = {}
        train_groups, val_groups = (
            set(partitions[sid]["train"]),
            set(partitions[sid]["val"]),
        )
        if train_groups & val_groups or {r["trajectory_id"] for r in rows} - (
            train_groups | val_groups
        ):
            raise ValueError("Train/val trajectory leakage or unknown source group.")
        for split, groups in (("train", train_groups), ("val", val_groups)):
            selected_rows = [r for r in rows if r["trajectory_id"] in groups]
            required = max(
                c.get(sid, 0)
                for c in (TRAIN_COUNTS if split == "train" else VAL_COUNTS)
            )
            if len(selected_rows) < required:
                raise ValueError(
                    f"QA accepted {split} pool insufficient: {sid}: {len(selected_rows)} < {required}"
                )
            rankings[sid][split] = nested_order(selected_rows)
            accepted_splits[split] = {
                "count": len(selected_rows),
                "trajectory_count": len({r["trajectory_id"] for r in selected_rows}),
            }
        lookup[sid] = by_id
        grouped = defaultdict(list)
        for record in rows:
            grouped[record["trajectory_id"]].append(
                (Fraction(*record["scale"]["arc_fraction"]), record["sample_id"])
            )
        video_groups[sid] = sorted(max(grouped.values(), key=len))
        qa = [r["qa"] for r in rows]
        diagnostics[sid] = {
            "counts": scene["counts"],
            "unique_ids": len(by_id),
            "unique_cameras": len(camera_hashes),
            "unique_jpeg_sha256": len(image_hashes),
            "target_visible_kp_counts": dict(
                Counter(str(q["target_visible_kp"]) for q in qa)
            ),
            "target_coverage_counts": dict(Counter(q["target_coverage"] for q in qa)),
            "roi_absent_count": sum(q["roi_hole_fraction"] is None for q in qa),
            "roi_hole_max": max(
                (
                    q["roi_hole_fraction"]
                    for q in qa
                    if q["roi_hole_fraction"] is not None
                ),
                default=None,
            ),
            "all_visible_kp_min": min(q["all_visible_kp"] for q in qa),
            "accepted_trajectory_count": len({r["trajectory_id"] for r in rows}),
            "partition_accepted": accepted_splits,
        }
    conditions = {}
    for count in range(1, 5):
        combined = set()
        for split, budgets, total in (
            ("train", TRAIN_COUNTS, 5500),
            ("val", VAL_COUNTS, 500),
        ):
            samples = [
                {"scene_id": sid, "sample_id": sample_id}
                for sid, budget in budgets[count - 1].items()
                for sample_id in rankings[sid][split][:budget]
            ]
            unique = {(r["scene_id"], r["sample_id"]) for r in samples}
            if len(samples) != total or len(unique) != total or unique & combined:
                raise ValueError(
                    "Fixed-size train/val condition has duplicate or missing samples."
                )
            combined.update(unique)
            name = f"scene-{count}.{split}.subset.json"
            write_json(
                root / name,
                {
                    "schema": "court_dataset_subset_v1",
                    "source_manifest": version_path.name,
                    "source_manifest_sha256": file_sha256(version_path),
                    "selection_seed": 42,
                    "split": split,
                    "samples": samples,
                },
            )
            conditions[name] = {
                "scene_counts": dict(Counter(r["scene_id"] for r in samples)),
                "sha256": file_sha256(root / name),
            }
    media = root / "review"
    media.mkdir(exist_ok=True)
    for sid in SCENES:
        frames = []
        for sample_id in rankings[sid]["train"][:10] + rankings[sid]["val"][:2]:
            store_path, row = lookup[sid][sample_id]
            store = open_court_store(root / store_path)
            record, rgb = store.record(row), store.rgb(row)
            frames.append(
                cv2.resize(np.hstack([rgb, overlay(rgb, record)]), (960, 270))
            )
        sheet = np.vstack(
            [np.hstack(frames[i : i + 2]) for i in range(0, len(frames), 2)]
        )
        cv2.imwrite(
            str(media / f"{sid}-rgb-overlay.jpg"),
            cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR),
        )
        members = video_groups[sid]
        indices = np.linspace(0, len(members) - 1, min(48, len(members)), dtype=int)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "960x270",
            "-framerate",
            "12",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(media / f"{sid}-rgb-overlay.mp4"),
        ]
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            assert process.stdin is not None
            for index in indices:
                store_path, row = lookup[sid][members[index][1]]
                store = open_court_store(root / store_path)
                record, rgb = store.record(row), store.rgb(row)
                process.stdin.write(
                    cv2.resize(
                        np.hstack([rgb, overlay(rgb, record)]), (960, 270)
                    ).tobytes()
                )
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Representative movie encoding failed.")
    write_json(
        root / "selection-report.json",
        {
            "schema": "court_scene_diversity_selection_report_v1",
            "source_manifest": version_path.name,
            "conditions": conditions,
            "scenes": diagnostics,
            "selection_rule": "separate fixed train/val trajectory partitions; accepted-count-proportional virtual finish time; circular arc maximin; stable hash ties; nested prefixes",
            "seed": 42,
            "nested_subsets_verified": True,
            "trajectory_partition_sha256": file_sha256(
                root / "trajectory-partition.json"
            ),
            "strict_jpeg_and_teacher_validation": True,
        },
    )
    print(
        json.dumps(
            {"stage": "finalized", "conditions": conditions, "diagnostics": diagnostics}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("plan", "smoke", "render", "finalize"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path("/home/kamimura/projects/tennis-lab")
    )
    options = parser.parse_args()
    {"plan": plan, "smoke": smoke, "render": render, "finalize": finalize}[
        options.stage
    ](options)
