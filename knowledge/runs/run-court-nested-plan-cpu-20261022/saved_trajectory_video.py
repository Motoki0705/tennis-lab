"""Review existing RGB trajectories on CPU; never render or mutate their stores."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.synthetic_data_generation.dataset.court.sample_store import (
    open_court_store,
    read_court_manifest,
    read_court_rgb,
)
from src.utils.schema.court import COURT_SKELETON


def visible_target(record: dict[str, Any]) -> int:
    target_id = record["target_court"]["binding"]["court_instance_id"]
    court = next(
        court
        for court in record["projection"]["courts"]
        if court["court_instance_id"] == target_id
    )
    return sum(
        point["renderer_visible"] for cls in court["classes"] for point in cls["points"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.scenes_root.resolve()):
        raise ValueError("output must be outside the read-only scenes root")
    args.output.mkdir(parents=True, exist_ok=True)
    reviews = []
    for scene in ["B00", "B01", "B02", "B03"]:
        owner = args.scenes_root / scene / "datasets/court"
        source_hash = hashlib.sha256(
            (owner / "samples/index.npz").read_bytes()
        ).hexdigest()
        manifest = read_court_manifest(owner)
        records = manifest["samples"]
        anchor = min(
            records, key=lambda record: (visible_target(record), record["sample_id"])
        )
        group = sorted(
            [
                record
                for record in records
                if record["trajectory_group_id"] == anchor["trajectory_group_id"]
            ],
            key=lambda record: record["trajectory_frame_index"],
        )
        indices = np.linspace(0, len(group) - 1, min(32, len(group))).astype(int)
        selected = [group[index] for index in indices]
        frames = args.output / f"saved-{scene}-frames"
        frames.mkdir(exist_ok=False)
        store = open_court_store(owner)
        for index, record in enumerate(selected):
            raw = Image.fromarray(read_court_rgb(owner, record, store=store))
            overlay = raw.copy()
            draw = ImageDraw.Draw(overlay)
            target_id = record["target_court"]["binding"]["court_instance_id"]
            for court in record["projection"]["courts"]:
                color = (
                    (255, 150, 20)
                    if court["court_instance_id"] == target_id
                    else (30, 230, 180)
                )
                points = {
                    point["physical_index"]: point
                    for cls in court["classes"]
                    for point in cls["points"]
                }
                for a, b in COURT_SKELETON:
                    if (
                        a in points
                        and b in points
                        and points[a]["in_front"]
                        and points[b]["in_front"]
                    ):
                        draw.line(
                            [tuple(points[a]["uv"]), tuple(points[b]["uv"])],
                            fill=color,
                            width=2,
                        )
                for point in points.values():
                    if point["in_frame"]:
                        x, y = point["uv"]
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            canvas = Image.new("RGB", (1920, 570), "black")
            canvas.paste(raw.resize((960, 540)), (0, 30))
            canvas.paste(overlay.resize((960, 540)), (960, 30))
            ImageDraw.Draw(canvas).text(
                (6, 8),
                f"{scene} SAVED RGB {record['sample_id']} - sampled trajectory playback, no new render",
                fill="white",
            )
            ImageDraw.Draw(canvas).text(
                (966, 8),
                f"orange: target court ({visible_target(record)}/14 visible); green: other courts",
                fill="white",
            )
            canvas.save(frames / f"{index:03d}.png")
        video = args.output / f"saved-{scene}-trajectory.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-n",
                "-framerate",
                "8",
                "-i",
                str(frames / "%03d.png"),
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
        )
        if (
            hashlib.sha256((owner / "samples/index.npz").read_bytes()).hexdigest()
            != source_hash
        ):
            raise ValueError("read-only sparse index changed during review")
        reviews.append(
            {
                "scene": scene,
                "source_index_sha256": source_hash,
                "group": anchor["trajectory_group_id"],
                "sample_ids": [record["sample_id"] for record in selected],
                "visible_target": [visible_target(record) for record in selected],
                "video": str(video),
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            }
        )
    (args.output / "saved-video-review.json").write_text(
        json.dumps(
            {
                "script_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "fps": 8,
                "reviews": reviews,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
