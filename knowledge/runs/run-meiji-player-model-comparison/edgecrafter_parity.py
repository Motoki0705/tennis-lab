"""Compare EdgeCrafter backend outputs for the same pinned images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from edgecrafter import PREDICTION_SCHEMA

from src.tennis_scene.evaluation.player_pose.scoring import (
    FrameKey,
    Predictions,
    describe,
)
from src.utils.checksum import dual_sha256
from src.utils.geometry.bbox import match_iou, pairwise_iou
from src.utils.inference_benchmark import ParityResult
from src.utils.io import write_json_atomic
from src.utils.schema.player import NUM_HUMAN_KP

MIN_SCORE = 0.25
MATCH_IOU = 0.50
GATE = {
    "box_matched_fraction_min": 0.98,
    "box_iou_p05_min": 0.90,
    "score_error_p95_max": 0.025,
    "joint_availability_agreement_min": 0.98,
    "joint_error_p95_px_max": 8.0,
}


def prediction_parity(
    pairs: Iterable[tuple[Predictions, Predictions]],
    *,
    has_pose: bool,
) -> ParityResult:
    """Match both directions by IoU, then compare the corresponding outputs."""
    images = pt32_rows = candidate_rows = 0
    joint_agreement = joint_total = 0
    ious, edges, scores, joint_errors = [], [], [], []
    for reference, candidate in pairs:
        images += 1
        ref_keep = reference.scores >= MIN_SCORE
        cand_keep = candidate.scores >= MIN_SCORE
        ref_boxes = reference.boxes_xyxy[ref_keep]
        cand_boxes = candidate.boxes_xyxy[cand_keep]
        pt32_rows += len(ref_boxes)
        candidate_rows += len(cand_boxes)
        matrix = pairwise_iou(ref_boxes, cand_boxes)
        i, j = match_iou(matrix, MATCH_IOU)
        ious.extend(matrix[i, j])
        edges.extend(np.abs(ref_boxes[i] - cand_boxes[j]).ravel())
        scores.extend(
            np.abs(reference.scores[ref_keep][i] - candidate.scores[cand_keep][j])
        )
        if has_pose:
            if reference.keypoints_xyc is None or candidate.keypoints_xyc is None:
                raise ValueError("Pose parity requires COCO17 rows from both backends")
            ref_pose = reference.keypoints_xyc[ref_keep][i]
            cand_pose = candidate.keypoints_xyc[cand_keep][j]
            ref_present, cand_present = ref_pose[..., 2] > 0, cand_pose[..., 2] > 0
            joint_total += ref_present.size
            joint_agreement += int(np.count_nonzero(ref_present == cand_present))
            both = ref_present & cand_present
            joint_errors.extend(
                np.linalg.norm((ref_pose[..., :2] - cand_pose[..., :2])[both], axis=-1)
            )

    matched = len(ious)
    iou_p05 = float(np.percentile(ious, 5)) if matched else None
    matched_fraction = (
        matched / max(pt32_rows, candidate_rows)
        if pt32_rows or candidate_rows
        else None
    )
    score_error = describe(np.asarray(scores))
    score_p95 = score_error["p95"]
    passed = (
        matched_fraction is not None
        and matched_fraction >= GATE["box_matched_fraction_min"]
        and iou_p05 is not None
        and iou_p05 >= GATE["box_iou_p05_min"]
        and score_p95 is not None
        and score_p95 <= GATE["score_error_p95_max"]
    )
    metrics: dict[str, object] = {
        "images": images,
        "score_cutoff_inclusive": MIN_SCORE,
        "matching_min_iou": MATCH_IOU,
        "gate": GATE,
        "pt32_boxes": pt32_rows,
        "candidate_boxes": candidate_rows,
        "matched": matched,
        "matched_fraction_both_directions": matched_fraction,
        "matched_iou_p05": iou_p05,
        "abs_xyxy_error_px": describe(np.asarray(edges)),
        "abs_score_error": score_error,
    }
    if has_pose:
        availability = joint_agreement / joint_total if joint_total else None
        error = describe(np.asarray(joint_errors))
        joint_p95 = error["p95"]
        passed = (
            passed
            and availability is not None
            and availability >= GATE["joint_availability_agreement_min"]
            and joint_p95 is not None
            and joint_p95 <= GATE["joint_error_p95_px_max"]
        )
        metrics.update(
            joint_availability_agreement=availability,
            joint_pairs_total=joint_total,
            joint_error_px=error,
        )
    return ParityResult(
        passed, metrics, None if passed else "PT32 output parity gate failed"
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
        json.loads((path / "metrics.json").read_text())
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
    seen: set[FrameKey] = set()

    def pairs() -> Iterator[tuple[Predictions, Predictions]]:
        with (
            paths[0].open(encoding="utf-8") as pt32,
            paths[1].open(encoding="utf-8") as other,
        ):
            for handle, receipt in zip((pt32, other), metadata, strict=True):
                if json.loads(next(handle)) != {
                    key: receipt[key] for key in (*header_keys, "backend")
                }:
                    raise ValueError(
                        "Predictions header differs from the metrics receipt"
                    )
            for ref_line, cand_line in zip(pt32, other, strict=True):
                ref, cand = json.loads(ref_line), json.loads(cand_line)
                key = FrameKey(*(ref[name] for name in FrameKey._fields))
                if key in seen or any(
                    ref[name] != cand[name] for name in FrameKey._fields
                ):
                    raise ValueError(
                        f"Prediction images are duplicated or misaligned at {key}"
                    )
                seen.add(key)
                yield _prediction(ref, has_pose), _prediction(cand, has_pose)

    result = prediction_parity(pairs(), has_pose=has_pose)
    if len(seen) != expected_images:
        raise ValueError(
            f"Expected {expected_images} prediction images, found {len(seen)}"
        )
    return {
        **{key: first[key] for key in (*header_keys, *settings_keys)},
        "candidate_backend": second["backend"],
        "status": "ok" if result.passed else "parity_failed",
        "pt32_predictions_sha256": first["predictions_sha256"],
        "candidate_predictions_sha256": second["predictions_sha256"],
        **asdict(result),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt32", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    report = compare_files(args.pt32, args.candidate)
    path = args.candidate / "parity-vs-pt32.json"
    write_json_atomic(path, report)
    print(
        f"{report['model_id']} {report['candidate_backend']} parity {report['status']}. Receipt {path}"
    )


if __name__ == "__main__":
    main()
