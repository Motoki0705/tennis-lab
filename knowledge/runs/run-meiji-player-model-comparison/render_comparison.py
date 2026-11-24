"""Render a four-panel person comparison at each preselected #2495 frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from compare_conditions import POSE_MODELS, read_json
from prediction_rows import PredictionHeader, read_predictions

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.reader import read_video_frame

COLOR = (79, 234, 134)


def draw_panel(image, title, boxes, xy, valid):
    panel = image.copy()
    height, width = panel.shape[:2]
    bounds = np.array([width - 1, height - 1])
    for box in np.rint(boxes).clip(0, np.tile(bounds, 2)).astype(int):
        x0, y0, x1, y1 = box
        cv2.rectangle(panel, (x0, y0), (x1, y1), COLOR, 2, cv2.LINE_AA)
    if xy is not None:
        for pose, mask in zip(xy, valid, strict=True):
            inside = mask & np.isfinite(pose).all(axis=-1)
            inside &= ((pose >= 0) & (pose <= bounds)).all(axis=-1)
            points = np.rint(np.where(inside[:, None], pose, 0)).astype(int)
            for a, b in COCO17_SKELETON:
                if inside[a] and inside[b]:
                    cv2.line(panel, points[a], points[b], COLOR, 2, cv2.LINE_AA)
            for point in points[inside]:
                cv2.circle(panel, point, 3, COLOR, -1, cv2.LINE_AA)
    panel = cv2.resize(panel, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(
        panel, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24)
    )
    cv2.putText(
        panel,
        title,
        (11, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="cam0")
    parser.add_argument(
        "--model", action="append", required=True, help="model_id/backend, three panels"
    )
    parser.add_argument("--display-score", type=float, default=0.30)
    args = parser.parse_args()
    if (
        len(set(args.model)) != 3
        or len(args.model) != 3
        or not 0 <= args.display_score <= 1
    ):
        parser.error(
            "Choose three distinct model/backend conditions and a display score in [0, 1]"
        )

    data = read_json(args.evidence)
    cohort = data["evaluation"]
    manifest = load_manifest(args.reference, expected_sha256=cohort["sha256"])
    references = load_reference_frames(args.reference, expected_sha256=cohort["sha256"])
    pinned_file(
        args.source,
        {
            "path": "manifest.json",
            "sha256": manifest["source_benchmark"]["manifest_sha256"],
        },
    )
    cases = {f"{row['model_id']}/{row['backend']}": row for row in data["conditions"]}
    predictions, pins = {}, {}
    for condition in args.model:
        model, backend = condition.split("/", 1)
        pin = cases[condition]["predictions"]
        path = pinned_file(args.artifacts, pin)
        header = PredictionHeader(
            cohort["id"], cohort["sha256"], model, backend, model in POSE_MODELS
        )
        predictions[condition] = read_predictions(
            path, header, expected_frames=references
        )
        pins[condition] = pin

    args.output.mkdir(parents=True, exist_ok=True)
    generated = []
    for clip in manifest["clips"]:
        overlay = next(
            row for row in clip["overlays"] if row["camera_id"] == args.camera
        )
        source = next(
            row for row in clip["source_videos"] if row["camera_id"] == args.camera
        )
        key = FrameKey(clip["clip_id"], args.camera, overlay["source_frame"])
        image = read_video_frame(
            pinned_file(args.source, source), key.source_frame
        ).frame
        if image.shape != (clip["height"], clip["width"], 3):
            raise ValueError(f"Video shape differs from fixed clip {key}")
        reference = references[key]
        label = f"{key.clip_id} {key.camera_id} F{key.source_frame}"
        panels = [
            draw_panel(
                image,
                f"SMPL reference | {label}",
                reference.boxes_xyxy,
                reference.keypoints_xy,
                reference.joint_valid,
            )
        ]
        counts = {"reference": len(reference.boxes_xyxy)}
        for condition, rows in predictions.items():
            row = rows[key]
            keep = row.scores >= args.display_score
            pose = None if row.keypoints_xyc is None else row.keypoints_xyc[keep]
            panels.append(
                draw_panel(
                    image,
                    f"{condition} | score >= {args.display_score:.2f}",
                    row.boxes_xyxy[keep],
                    None if pose is None else pose[..., :2],
                    None if pose is None else pose[..., 2] > 0,
                )
            )
            counts[condition] = int(keep.sum())
        grid = np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:])))
        path = args.output / f"{Path(key.clip_id).name}-{key.camera_id}.png"
        if not cv2.imwrite(str(path), grid):
            raise OSError(f"Could not write {path}")
        generated.append(
            {
                **key._asdict(),
                "source_video_sha256": source["sha256"],
                "boxes_displayed": counts,
                "path": path.name,
                "sha256": dual_sha256(path),
                "width": grid.shape[1],
                "height": grid.shape[0],
            }
        )
    write_json_atomic(
        args.output / "index.json",
        {
            "evaluation": cohort,
            "display_score_inclusive": args.display_score,
            "predictions": pins,
            "images": generated,
        },
    )


if __name__ == "__main__":
    main()
