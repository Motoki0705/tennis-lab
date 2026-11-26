"""Render a four-panel person comparison at each preselected #2495 frame."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from compare_conditions import POSE_MODELS, read_json
from numpy.typing import NDArray
from prediction_rows import PredictionHeader, read_predictions

from src.tennis_scene.evaluation.player_pose.bundle import (
    load_manifest,
    load_reference_frames,
    pinned_file,
)
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.utils.checksum import dual_sha256
from src.utils.io import write_json_atomic
from src.utils.schema.player import COCO17_SKELETON
from src.utils.video.reader import read_video_frame

COLOR = (79, 234, 134)
PREVIEW_MODEL = "ecpose-l-o365"
PREVIEW_LABEL = f"{PREVIEW_MODEL}/pt32 (CPU preview)"


def draw_panel(
    image: NDArray[np.uint8],
    title: str,
    boxes: NDArray[np.float32],
    xy: NDArray[np.float32] | None,
    valid: NDArray[np.bool_] | None,
) -> NDArray[np.uint8]:
    panel = image.copy()
    height, width = panel.shape[:2]
    bounds = np.array([width - 1, height - 1])
    for box in np.rint(boxes).clip(0, np.tile(bounds, 2)).astype(int):
        x0, y0, x1, y1 = box
        cv2.rectangle(panel, (x0, y0), (x1, y1), COLOR, 2, cv2.LINE_AA)
    if xy is not None:
        assert valid is not None
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


def cpu_pose_preview(
    assets: Path,
) -> tuple[Callable[[NDArray[np.uint8]], Predictions], dict[str, Any]]:
    import torch

    from src.submodules.models.ecpose.inference import (
        decode_predictions,
        preprocess_images,
    )
    from src.submodules.models.ecpose.releases import RELEASES, SOURCE_COMMIT
    from src.submodules.models.ecpose.runtime import load_backend

    release = RELEASES[PREVIEW_MODEL]
    margin = 0.10
    infer = load_backend(
        assets / "EdgeCrafter",
        assets / release.filename,
        PREVIEW_MODEL,
        "pt32",
        torch.device("cpu"),
        None,
    )

    def predict(image: NDArray[np.uint8]) -> Predictions:
        pixels, sizes = preprocess_images([image])
        with torch.inference_mode():
            (row,) = decode_predictions(
                release, infer(pixels), sizes, confidence=0.0, pose_box_margin=margin
            )
        assert row.keypoints_xyc is not None
        return Predictions(
            row.boxes_xyxy.numpy(), row.scores.numpy(), row.keypoints_xyc.numpy()
        )

    return predict, {
        "model_id": PREVIEW_MODEL,
        "backend": "pt32",
        "device": "cpu",
        "batch_size": 1,
        "source_commit": SOURCE_COMMIT,
        "weight_sha256": release.sha256,
        "box_margin": margin,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="cam0")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="model_id/backend, stored predictions",
    )
    parser.add_argument(
        "--preview-ecpose-assets",
        type=Path,
        help="Use a fresh ECPose-L CPU prediction for the fourth panel",
    )
    parser.add_argument("--display-score", type=float, default=0.30)
    args = parser.parse_args()
    if (
        len(set(args.model)) != len(args.model)
        or len(args.model) + int(args.preview_ecpose_assets is not None) != 3
        or not 0 <= args.display_score <= 1
    ):
        parser.error(
            "Choose three distinct model panels in total and a display score in [0, 1]"
        )
    if args.preview_ecpose_assets is not None and f"{PREVIEW_MODEL}/pt32" in args.model:
        parser.error("ECPose-L PT32 is already selected as a stored panel")

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
    preview = (
        None
        if args.preview_ecpose_assets is None
        else cpu_pose_preview(args.preview_ecpose_assets)
    )

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
        overlays = {condition: rows[key] for condition, rows in predictions.items()}
        if preview is not None:
            overlays[PREVIEW_LABEL] = preview[0](image)
        for condition, row in overlays.items():
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
    index = {
        "evaluation": cohort,
        "display_score_inclusive": args.display_score,
        "predictions": pins,
        "images": generated,
    }
    if preview is not None:
        index["cpu_preview"] = preview[1]
    write_json_atomic(args.output / "index.json", index)


if __name__ == "__main__":
    main()
