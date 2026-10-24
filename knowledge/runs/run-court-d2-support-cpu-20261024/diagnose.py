"""Reproduce the original 24-trajectory geometry and val-only framing diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from support import (
    GROUPS,
    PREVIOUS,
    SCENES,
    counterfactual,
    describe,
    framing,
    hull_for,
    load_inputs,
    partitions,
    recipe,
    row_digest,
    stats,
    summary,
    write,
)

from src.synthetic_data_generation.dataset.court.scale.core import TrajectoryMethod
from src.synthetic_data_generation.dataset.court.scale.planning import (
    iter_trajectory_plans,
)
from src.tasks.court_detection.scaling.real_split import canonical_sha256, real_source
from src.utils.data.image_record_store import file_sha256
from src.utils.geometry.planar_camera import PlanarCameraFitError, fit_planar_camera
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def val_analysis(root: Path, registry_path: Path, output: Path) -> None:
    registry = json.loads(registry_path.read_text())
    payload = {k: v for k, v in registry.items() if k != "sha256"}
    assert canonical_sha256(payload) == registry["sha256"]
    assert (
        registry["sha256"]
        == "d943d569c87e0974590eeaf01cdc1b526da4f6eeb0ca65bbd54c05a12c32d60a"
    )
    for name, sha in registry["source_sha256"].items():
        assert file_sha256(root / name) == sha
    # The packed source contains the original splits; select IDs BEFORE any geometry.
    val_ids = set(registry["splits"]["val"])
    source = real_source(root)
    records = sorted(
        (
            r
            for split in source.available_splits
            for r in source.records(split)
            if r.sample_id in val_ids
        ),
        key=lambda r: r.sample_id,
    )
    assert {r.sample_id for r in records} == val_ids
    rows: list[dict[str, Any]] = []
    video_records: dict[str, Any] = {}
    for record in records:
        uv = np.array(record.payload["keypoints"])
        w, h = record.payload["width"], record.payload["height"]
        count = int(
            ((uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)).sum()
        )
        video = record.sample_id.rsplit("_", 1)[0]
        rows.append(
            {
                "sample_id": record.sample_id,
                "video": video,
                "framing": framing(count),
                "in_frame_kp": count,
            }
        )
        video_records.setdefault(video, record)
    previous = json.loads((PREVIOUS / "evidence/val-framing.json").read_text())
    # Only the old val-specific evaluation report is used; no test report is opened.
    counts = {g: sum(r["framing"] == g for r in rows) for g in GROUPS}
    assert counts == {"full": 4400, "near_full": 55, "partial": 13}
    assert previous["groups"]["all"]["images"] == len(rows) == 4468
    fits: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    world = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(np.float64)
    for video, record in sorted(video_records.items()):
        w, h = record.payload["width"], record.payload["height"]
        try:
            fit = fit_planar_camera(
                world,
                np.array(record.payload["keypoints"]),
                np.ones(14),
                (w, h),
                require_in_image=True,
            )
        except PlanarCameraFitError as error:
            failures[error.reason.value] += 1
            continue
        fits.append(
            {
                "video": video,
                "sample_id": record.sample_id,
                "height_m": float(fit.center[2]),
                "distance_xy_m": float(np.linalg.norm(fit.center[:2])),
                "hfov_deg": math.degrees(2 * math.atan(w / (2 * fit.K[0, 0]))),
                "x_m": float(fit.center[0]),
                "y_m": float(fit.center[1]),
                "rmse_px": fit.rmse_px,
                "rmse_over_diagonal": fit.rmse_px / math.hypot(w, h),
            }
        )
    write(output / "val-rows.json", rows)
    write(output / "val-camera-rows.json", fits)
    write(
        output / "val-summary.json",
        {
            "schema": "court_d2_val_support_v1",
            "split": "val",
            "rule": "in_frame_kp14_v1",
            "registry_sha256": registry["sha256"],
            "images": len(rows),
            "videos": len(video_records),
            "framing": counts,
            "in_frame_kp": dict(
                sorted(Counter(r["in_frame_kp"] for r in rows).items())
            ),
            "videos_per_group": {
                g: len({r["video"] for r in rows if r["framing"] == g}) for g in GROUPS
            },
            "rows_sha256": row_digest(rows),
            "camera_estimation": {
                "rule": "lexicographically first val frame per video; existing fit_planar_camera, fx=fy, centred principal point, zero distortion; positive-confidence in-image KP only",
                "attempts": len(video_records),
                "successes": len(fits),
                "failure_reasons": dict(failures),
                "all_successful": {
                    f: stats([r[f] for r in fits])
                    for f in (
                        "height_m",
                        "distance_xy_m",
                        "hfov_deg",
                        "rmse_over_diagonal",
                    )
                },
                "rmse_ratio_le_0_005": {
                    f: stats([r[f] for r in fits if r["rmse_over_diagonal"] <= 0.005])
                    for f in ("height_m", "distance_xy_m", "hfov_deg")
                },
                "limitations": "Assumption-dependent diagnostics, not camera ground truth or a new data-selection gate. Video-balanced subset, not a 4468-camera distribution.",
            },
            "test_geometry_processed": False,
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.7))
    for ax, category in zip(axes, GROUPS, strict=True):
        chosen = next(r for r in rows if r["framing"] == category)
        record = next(r for r in records if r.sample_id == chosen["sample_id"])
        raw = source.load(record)
        ax.imshow(raw.image)
        uv = np.array(record.payload["keypoints"])
        ax.scatter(uv[:, 0], uv[:, 1], s=5, c="yellow")
        ax.set_xlim(0, raw.image.width)
        ax.set_ylim(raw.image.height, 0)
        ax.set_title(
            f"val {category}: {chosen['in_frame_kp']} KP\n{record.sample_id}",
            fontsize=8,
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "val-examples.png", dpi=110)
    plt.close(fig)


def geometry_analysis(scenes_root: Path, output: Path) -> None:
    inputs, verification = load_inputs(scenes_root)
    rows: list[dict[str, Any]] = []
    cf: dict[str, Counter[str]] = defaultdict(Counter)
    captures = []
    for scene, data in inputs.items():
        hull = hull_for(data)
        for group in iter_trajectory_plans(
            scene_id=scene,
            method=TrajectoryMethod.STRATIFIED_COUNTER,
            recipe=recipe(),
            seed=42,
            trajectory_count=24,
            density_level=0,
            cameras=data["cameras"],
            layout=data["layout"],
        ):
            assert group["curve"] is not None
            for record in group["samples"]:
                rows.append(describe(record, group, scene, data["layout"], hull))
                # Fixed camera/target XY probes deliberately keep the original jitter.
                for hfov in (110, 125, 140, 155):
                    cf[f"fixed_pose_hfov_{hfov}"][
                        counterfactual(record, data, hfov=hfov)
                    ] += 1
                for height in (0, 0.75, 1.5):
                    cf[f"fixed_camera_look_height_{height}"][
                        counterfactual(record, data, target_height=height)
                    ] += 1
        for camera in data["cameras"]:
            pos = camera.camera_to_scene.matrix()[:3, 3]
            court = min(
                data["layout"].courts,
                key=lambda c: np.linalg.norm(c.court_from_scene.apply(pos[None])[0]),
            )
            local = court.court_from_scene.apply(pos[None])[0]
            captures.append(
                {
                    "scene": scene,
                    "height_m": float(local[2]),
                    "distance_xy_m": float(np.linalg.norm(local[:2])),
                }
            )
    assert len(rows) == 1209
    assert {
        s: sum(r["geometry_pass"] for r in rows if r["scene"] == s) for s in SCENES
    } == {"B00": 289, "B01": 353, "B02": 243, "B03": 238}
    write(output / "baseline-geometry-rows.json", rows)
    write(output / "capture-rows.json", captures)
    write(
        output / "baseline-geometry.json",
        {
            "schema": "court_d2_baseline_diagnosis_v1",
            "rule": "in_frame_kp14_v1",
            "generation_seed": 42,
            "trajectories_per_scene": 24,
            "density_level": 0,
            "asset_verification": verification,
            "all_candidates": summary(rows),
            "geometry_pass": summary([r for r in rows if r["geometry_pass"]]),
            "geometry_rejected": summary([r for r in rows if not r["geometry_pass"]]),
            "by_scene": partitions(rows, "scene"),
            "by_hfov": partitions(rows, "hfov_deg"),
            "by_look_height": partitions(rows, "look_at_height_m"),
            "by_region": partitions(rows, "region"),
            "by_framing": partitions(rows, "framing"),
            "fixed_pose_counterfactual_all_candidates": {
                name: {g: counts[g] for g in GROUPS} for name, counts in cf.items()
            },
            "captured_camera_counts": dict(Counter(r["scene"] for r in captures)),
        },
    )
    print(json.dumps({"baseline": summary(rows), "counterfactual": dict(cf)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    val_analysis(args.real_root, args.registry, args.output)
    geometry_analysis(args.scenes_root, args.output)


if __name__ == "__main__":
    main()
