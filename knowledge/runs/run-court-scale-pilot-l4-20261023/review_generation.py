"""Check published teachers and retain deterministic quality examples."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_manifest,
)
from src.synthetic_data_generation.dataset.court.scale.publication import (
    load_dataset_version,
)
from src.tasks.court_detection.configuration import (
    SyntheticCourtDatasetVersionSourceConfig,
)
from src.tasks.court_detection.data.inputs.synthetic_court_dataset_version import (
    SyntheticCourtDatasetVersionInput,
)
from src.tasks.court_detection.geometry.pose import (
    build_pose_target,
    semantic_in_front_mask,
    validate_projection_round_trip,
)
from src.utils.schema.court import COURT_SKELETON
from src.utils.video import VideoWriter


def overlay(
    store: Any, sample: dict[str, Any], caption: str, width: int
) -> Image.Image:
    image = Image.fromarray(store.rgb(sample["image_index"]))
    height = round(image.height * width / image.width)
    assert height % 2 == 0 and width % 2 == 0
    scale = width / image.width
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 32), "black")
    canvas.paste(image, (0, 32))
    target = sample["target_court"]["binding"]["court_instance_id"]
    court = next(
        c for c in sample["projection"]["courts"] if c["court_instance_id"] == target
    )
    points = {item["class_id"]: item["points"][0] for item in court["classes"]}
    draw = ImageDraw.Draw(canvas)
    for left, right in COURT_SKELETON:
        if (
            left >= 14
            or right >= 14
            or not all(points[i]["in_front"] for i in (left, right))
        ):
            continue
        draw.line(
            [
                (points[i]["uv"][0] * scale, points[i]["uv"][1] * scale + 32)
                for i in (left, right)
            ],
            fill="#ffff00",
            width=1,
        )
    for index, point in points.items():
        if not point["in_front"] or not point["in_frame"]:
            continue
        x, y = point["uv"][0] * scale, point["uv"][1] * scale + 32
        color = "lime" if point["renderer_visible"] else "orange"
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.text((x + 2, y + 1), str(index), fill=color)
    qa = sample["qa"]
    draw.text((4, 2), caption, fill="white")
    draw.text(
        (4, 15),
        f"KP geom/visible {qa['target_geometry_kp']}/{qa['target_visible_kp']}, holes {qa['roi_hole_fraction']:.4f}",
        fill="white",
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    version = load_dataset_version(args.version, verify_images=True)
    args.output.mkdir(parents=True, exist_ok=False)
    source = SyntheticCourtDatasetVersionInput(
        SyntheticCourtDatasetVersionSourceConfig(
            "synthetic_court_dataset_version", "v3", "target_court", args.version
        )
    )
    verified = 0
    for record in source.records("train"):
        raw = source.load(record)
        assert raw.pose_authority is not None and raw.keypoint_channels is not None
        assert len(raw.court_instances) == 1
        pose = build_pose_target(raw.pose_authority)
        assert torch.equal(
            raw.keypoint_channels.physical_indices[:, 0], pose.semantic_to_physical
        )
        validate_projection_round_trip(
            pose,
            raw.keypoint_channels.points_xy[:, 0],
            semantic_in_front=semantic_in_front_mask(pose, raw.court_instances[0]),
        )
        verified += 1
    result: dict[str, Any] = {
        "schema": "court_pilot_generation_review_v1",
        "pose_kp_roundtrip_verified_images": verified,
        "selection_rule": "max target geometry KP; min target renderer-visible KP; max ROI alpha-below-0.9 (diagnostic only). Stable sample ID breaks ties.",
        "scenes": [],
    }
    for scene in version["scenes"]:
        samples: list[tuple[dict[str, Any], Any]] = []
        for entry in scene["stores"]:
            root = args.version.parent / entry["path"]
            store = open_court_store(root)
            manifest = read_court_manifest(root, store=store)
            samples.extend((sample, store) for sample in manifest["samples"])
        assert len(samples) == scene["counts"]["qa_accepted"]
        selected = [
            (
                "max geometry KP",
                min(
                    samples,
                    key=lambda pair: (
                        -pair[0]["qa"]["target_geometry_kp"],
                        pair[0]["sample_id"],
                    ),
                ),
            ),
            (
                "min visible KP",
                min(
                    samples,
                    key=lambda pair: (
                        pair[0]["qa"]["target_visible_kp"],
                        pair[0]["sample_id"],
                    ),
                ),
            ),
            (
                "max low-alpha area",
                min(
                    samples,
                    key=lambda pair: (
                        -pair[0]["qa"]["roi_alpha_lt_0_9"],
                        pair[0]["sample_id"],
                    ),
                ),
            ),
        ]
        sheet = Image.new("RGB", (768, 176), "black")
        cases = []
        for index, (criterion, (sample, store)) in enumerate(selected):
            sheet.paste(
                overlay(store, sample, f"{scene['scene_id']}: {criterion}", 256),
                (index * 256, 0),
            )
            cases.append(
                {
                    "criterion": criterion,
                    "sample_id": sample["sample_id"],
                    "trajectory_id": sample["trajectory_id"],
                    "qa": sample["qa"],
                }
            )
        sheet_path = args.output / f"{scene['scene_id']}-cases.png"
        sheet.save(sheet_path, optimize=True)
        trajectory = samples[0][0]["trajectory_id"]
        frames = sorted(
            (pair for pair in samples if pair[0]["trajectory_id"] == trajectory),
            key=lambda pair: Fraction(*pair[0]["scale"]["arc_fraction"]),
        )
        video = args.output / f"{scene['scene_id']}-trajectory.mp4"
        with VideoWriter(video, fps=6, crf=20) as writer:
            for sample, store in frames:
                writer.write_frame(
                    np.asarray(overlay(store, sample, scene["scene_id"], 640))
                )
        diagnostics = {}
        for field in (
            "target_geometry_kp",
            "target_visible_kp",
            "roi_hole_fraction",
            "roi_alpha_lt_0_9",
        ):
            values = np.asarray(
                [sample["qa"][field] for sample, _ in samples], dtype=np.float64
            )
            assert np.isfinite(values).all()
            diagnostics[field] = {
                "min": float(values.min()),
                "median": float(np.median(values)),
                "max": float(values.max()),
            }
        result["scenes"].append(
            {
                "scene_id": scene["scene_id"],
                "counts": scene["counts"],
                "rejection_reasons": scene["rejection_reasons"],
                "empty_trajectory_count": scene["empty_trajectory_count"],
                "accepted_diagnostics": diagnostics,
                "cases": cases,
                "contact_sheet": sheet_path.name,
                "video": {
                    "path": video.name,
                    "trajectory_id": trajectory,
                    "accepted_frames": len(frames),
                    "order": "increasing exact arc fraction; rejected frames remain gaps",
                },
            }
        )
    assert verified == sum(
        scene["counts"]["qa_accepted"] for scene in version["scenes"]
    )
    (args.output / "review.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "pose_kp_roundtrip_verified_images": verified,
                "scenes": len(result["scenes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
