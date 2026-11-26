"""Audit PMPose auxiliary scores and backend outliers on the fixed accuracy rows."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from evaluate import POLICY
from fixed_inputs import RowKey, row_keys
from numpy.typing import NDArray
from pmpose_results import fixed_prediction_rows, validate_predictions
from pose_contract import PoseBatch
from scipy.stats import rankdata

from src.tennis_scene.evaluation.player_pose.bundle import load_reference_frames
from src.tennis_scene.evaluation.player_pose.scoring import COCO_SIGMAS
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import Backend
from src.utils.io import write_json_atomic
from src.utils.schema.player import COCO_KP_NAMES, NUM_HUMAN_KP


def pck_auc(scores: NDArray[np.floating], success: NDArray[np.bool_]) -> float:
    """Probability that a PCK hit has a higher score than a miss, with half for ties."""
    if scores.shape != success.shape or not np.isfinite(scores).all():
        raise ValueError("Expected one finite score per PCK result")
    hits = int(success.sum())
    misses = len(success) - hits
    if not hits or not misses:
        raise ValueError("PCK AUC requires both hits and misses")
    ranks = rankdata(scores, method="average")
    return float((ranks[success].sum() - hits * (hits + 1) / 2) / (hits * misses))


def _load(
    artifact: dict[str, str], expected: Mapping[str, NDArray[np.generic]]
) -> PoseBatch:
    path = Path(artifact["path"])
    if dual_sha256(path) != artifact["sha256"]:
        raise ValueError(f"Predictions differ from the verified accuracy run: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return validate_predictions(dict(arrays.items()), expected)


def _auxiliary(
    poses: PoseBatch,
    xy: NDArray[np.floating],
    valid: NDArray[np.bool_],
    box_area: NDArray[np.float64],
    cutoff: float,
) -> dict[str, Any]:
    delta = poses.keypoints_xyc[..., :2].astype(np.float64) - xy
    error = np.linalg.norm(delta, axis=-1)
    correct = (error <= cutoff * np.sqrt(box_area[:, None]))[valid]
    observed_oks = np.exp(
        -(error**2) / (2 * box_area[:, None] * (2 * COCO_SIGMAS) ** 2)
    )
    signals = {
        name: poses.extras[name][:, :NUM_HUMAN_KP][valid].astype(np.float64)
        for name in ("presence", "visibility", "expected_oks", "heatmap_peak")
    }
    return {
        "eligible_joints": int(correct.size),
        "pck_success_fraction": float(correct.mean()),
        "mean_joint_expected_oks": float(signals["expected_oks"].mean()),
        "mean_joint_pseudo_gt_oks": float(observed_oks[valid].mean()),
        "pck_auc_by_signal": {
            name: pck_auc(values, correct) for name, values in signals.items()
        },
    }


def _outlier(
    reference: PoseBatch,
    candidate: PoseBatch,
    valid: NDArray[np.bool_],
    keys: tuple[RowKey, ...],
    boxes: NDArray[np.floating],
) -> dict[str, Any]:
    a, b = (
        reference.keypoints_xyc.astype(np.float64),
        candidate.keypoints_xyc.astype(np.float64),
    )
    d = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
    row, joint = np.unravel_index(np.nanargmax(d), d.shape)
    address = keys[row]
    return {
        **address.frame._asdict(),
        "player_id": address.player_id,
        "joint": COCO_KP_NAMES[joint],
        "scoreable_by_pseudo_gt": bool(valid[row, joint]),
        "bbox_xyxy": boxes[row].tolist(),
        "pt32_xyc": a[row, joint].tolist(),
        "candidate_xyc": b[row, joint].tolist(),
        "coordinate_l2_px": float(d[row, joint]),
        "max_scoreable_coordinate_l2_px": float(np.nanmax(d[valid])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accuracy-record", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    accuracy = json.loads(args.accuracy_record.read_bytes())
    references = load_reference_frames(
        args.bundle, expected_sha256=accuracy["bundle_sha256"]
    )
    keys = row_keys(references)
    expected = fixed_prediction_rows(references, keys)
    boxes = expected["bbox_xyxy"].astype(np.float64)
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=-1)
    frames = [references[key] for key in sorted(references)]
    xy = np.concatenate([frame.keypoints_xy for frame in frames])
    valid = np.concatenate([frame.joint_valid for frame in frames])
    cutoff = POLICY.pck_cutoffs[0]
    models = {}
    for model, modes in accuracy["conditions"].items():
        pt32 = _load(modes[Backend.PT32]["files"]["predictions"], expected)
        auxiliary = _auxiliary(pt32, xy, valid, area, cutoff)
        expected_pck = modes[Backend.PT32]["gt_fixed_denominator"]["pck"][str(cutoff)]
        if auxiliary["pck_success_fraction"] != expected_pck:
            raise ValueError(
                f"Diagnostic PCK differs from the canonical scorer for {model}"
            )
        outliers = {}
        for backend in (Backend.PT16, Backend.TRT16):
            candidate = _load(modes[backend]["files"]["predictions"], expected)
            outliers[backend] = _outlier(pt32, candidate, valid, keys, boxes)
        models[model] = {
            "pt32_auxiliary": auxiliary,
            "largest_backend_differences": outliers,
        }
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json_atomic(
        args.output,
        {
            "accuracy_record_sha256": dual_sha256(args.accuracy_record),
            "bundle_sha256": accuracy["bundle_sha256"],
            "pck_cutoff": cutoff,
            "models": models,
        },
    )


if __name__ == "__main__":
    main()
