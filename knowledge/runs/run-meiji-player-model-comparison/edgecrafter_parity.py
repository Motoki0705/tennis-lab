"""Compare EdgeCrafter backend outputs for the same pinned images."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from edgecrafter import PREDICTION_SCHEMA

from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_parity import Coco17Output, DetectionOutput, compare_detections
from src.utils.io import write_json_atomic
from src.utils.schema.player import NUM_HUMAN_KP


def detection_output(row: Predictions) -> DetectionOutput:
    """Keep every predicted box and use ECPose joint confidence as a validity mask."""
    joints = row.keypoints_xyc
    return DetectionOutput(
        row.boxes_xyxy,
        row.scores,
        None if joints is None else Coco17Output(joints, joints[..., 2] > 0),
    )


def _prediction(record: dict[str, Any], has_pose: bool) -> Predictions:
    boxes = np.asarray(record["boxes_xyxy"], dtype=np.float32)
    if boxes.shape == (0,):
        boxes = boxes.reshape((0, 4))
    values = record["keypoints_xyc"]
    if has_pose != (values is not None):
        raise ValueError("Prediction row disagrees with the pose contract")
    joints = None if values is None else np.asarray(values, dtype=np.float32)
    if joints is not None:
        if joints.shape == (0,):
            joints = joints.reshape((0, NUM_HUMAN_KP, 3))
        if not np.isfinite(joints).all():
            raise ValueError("Pose parity requires finite COCO17 rows")
    return Predictions(boxes, np.asarray(record["scores"], dtype=np.float32), joints)


def compare_files(reference: Path, candidate: Path) -> dict[str, object]:
    """Verify both prediction files and compare every image in their fixed order."""
    metadata = [
        json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for path in (reference, candidate)
    ]
    first, second = metadata
    header_keys = ("schema", "benchmark_id", "manifest_sha256", "model_id", "has_pose")
    settings_keys = (
        "weight_sha256",
        "source_commit",
        "input_size_hw",
        "image_batch_size",
        "person_confidence_exclusive",
        "pose_bbox_joint_margin",
    )
    if (
        first["backend"] != "pt32"
        or second["backend"] not in {"pt16", "trt16"}
        or first["schema"] != PREDICTION_SCHEMA
        or any(first[key] != second[key] for key in (*header_keys, *settings_keys))
    ):
        raise ValueError("PT32 and candidate settings refer to different conditions")
    expected_images = first["score"]["images"]
    if expected_images != second["score"]["images"]:
        raise ValueError("PT32 and candidate scores cover different image counts")
    paths = [reference / "predictions.jsonl", candidate / "predictions.jsonl"]
    for path, receipt in zip(paths, metadata, strict=True):
        if dual_sha256(path) != receipt["predictions_sha256"]:
            raise ValueError(
                f"Predictions SHA-256 differs from the metrics receipt for {path}"
            )
    has_pose = first["has_pose"]
    expected: dict[str, DetectionOutput] = {}
    observed: dict[str, DetectionOutput] = {}
    with (
        paths[0].open(encoding="utf-8") as pt32,
        paths[1].open(encoding="utf-8") as other,
    ):
        for handle, receipt in zip((pt32, other), metadata, strict=True):
            if json.loads(next(handle)) != {
                key: receipt[key] for key in (*header_keys, "backend")
            }:
                raise ValueError("Predictions header differs from the metrics receipt")
        for ref_line, cand_line in zip(pt32, other, strict=True):
            ref, cand = json.loads(ref_line), json.loads(cand_line)
            key = FrameKey(*(ref[name] for name in FrameKey._fields))
            if type(key.source_frame) is not int:
                raise ValueError(f"Expected an integer source frame: {key}")
            image_id = f"{key.clip_id}/{key.camera_id}/frame_{key.source_frame:06d}"
            if image_id in expected or any(
                ref[name] != cand[name] for name in FrameKey._fields
            ):
                raise ValueError(
                    f"Prediction images are duplicated or misaligned at {key}"
                )
            expected[image_id] = detection_output(_prediction(ref, has_pose))
            observed[image_id] = detection_output(_prediction(cand, has_pose))
    if len(expected) != expected_images:
        raise ValueError(
            f"Expected {expected_images} prediction images, found {len(expected)}"
        )
    result = compare_detections(expected, observed, with_pose=has_pose)
    return {
        **{key: first[key] for key in (*header_keys, *settings_keys)},
        "candidate_backend": second["backend"],
        "status": "ok" if result.passed else "parity_failed",
        "pt32_predictions_sha256": first["predictions_sha256"],
        "candidate_predictions_sha256": second["predictions_sha256"],
        "parity_source_sha256": dual_sha256(Path("src/utils/inference_parity.py")),
        **asdict(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt32", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    report = compare_files(args.pt32, args.candidate)
    path = args.candidate / "parity-vs-pt32-shared.json"
    write_json_atomic(path, report)
    print(
        f"{report['model_id']} {report['candidate_backend']} parity {report['status']}. Receipt {path}"
    )


if __name__ == "__main__":
    main()
