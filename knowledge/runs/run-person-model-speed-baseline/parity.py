"""Compare current stage outputs with the same pinned PT32 B=1 outputs."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from src.utils.geometry.bbox import pairwise_iou
from src.utils.geometry.rotation_conversions import axis_angle_to_matrix
from src.utils.inference_benchmark import ParityResult

_source = (
    Path(__file__).resolve().parents[1] / "run-i2297-clipreid-tensorrt/dino_accuracy.py"
)
compare_dino_frame = runpy.run_path(str(_source))["compare_dino_frame"]


def serialize(workload, value, body_model=None):
    if workload == "dino":
        return {"boxes": value.boxes_xyxy.tolist(), "scores": value.scores.tolist()}
    if workload == "dino-vitpose":
        detection, poses = value
        return {**serialize("dino", detection), "poses": poses.tolist()}
    if workload == "body":
        return {
            "params": {k: array.tolist() for k, array in value["params"].items()},
            "features": value["features"].tolist(),
            "joints": body_model.joints(value["params"]).tolist(),
        }
    return value.tolist()


def percentile(values, q):
    return float(np.percentile(values, q)) if np.size(values) else None


def box_errors(reference, candidate):
    counts = {"missing": 0, "extra": 0, "reference_boxes": 0, "candidate_boxes": 0}
    iou, coords, scores, pose_samples = [], [], [], []
    for key, expected in reference.items():
        actual = candidate[key]
        a = np.asarray(expected["boxes"]).reshape(-1, 4)
        b = np.asarray(actual["boxes"]).reshape(-1, 4)
        result = compare_dino_frame(a, expected["scores"], b, actual["scores"])
        iou.extend(result["iou"])
        coords.extend(result["xyxy_abs_error_px"].ravel())
        scores.extend(result["score_abs_error"])
        counts["missing"] += result["missing"]
        counts["extra"] += result["extra"]
        counts["reference_boxes"] += len(a)
        counts["candidate_boxes"] += len(b)
        if "poses" in expected:
            overlaps = pairwise_iou(a, b)
            eligible = overlaps >= 0.5
            rows, cols = linear_sum_assignment(
                np.where(eligible, min(overlaps.shape) + 1 + overlaps, 0), maximize=True
            )
            keep = eligible[rows, cols]
            pose_samples.append(
                (
                    np.asarray(expected["poses"]).reshape(-1, 17, 3)[rows[keep]],
                    np.asarray(actual["poses"]).reshape(-1, 17, 3)[cols[keep]],
                )
            )
    metrics = {
        **counts,
        "iou_p05": percentile(iou, 5),
        "xyxy_abs_px_p95": percentile(coords, 95),
        "score_abs_p95": percentile(scores, 95),
    }
    passed = bool(
        scores
        and metrics["iou_p05"] >= 0.98
        and metrics["score_abs_p95"] <= 0.01
        and counts["missing"] <= counts["reference_boxes"] * 0.01
        and counts["extra"] <= counts["candidate_boxes"] * 0.01
    )
    if pose_samples:
        pose_ok, pose_metrics = pose_errors(pose_samples)
        metrics.update(pose_metrics)
        passed &= pose_ok
    return passed, metrics


def pose_errors(samples):
    deltas = np.concatenate([np.asarray(a) - np.asarray(b) for a, b in samples], axis=0)
    if not deltas.size or not np.isfinite(deltas).all():
        return False, {"valid_pose_samples": False}
    xy = np.linalg.norm(deltas[..., :2], axis=-1)
    metrics = {
        "pose_xy_px_p95": percentile(xy, 95),
        "pose_xy_px_max": float(xy.max()),
        "pose_confidence_abs_p95": percentile(np.abs(deltas[..., 2]), 95),
    }
    return metrics["pose_xy_px_p95"] <= 1 and metrics[
        "pose_xy_px_max"
    ] <= 5 and metrics["pose_confidence_abs_p95"] <= 0.02, metrics


def rotations(params):
    root = np.asarray(params["global_orient"], dtype=np.float64)
    body = np.asarray(params["body_pose"], dtype=np.float64).reshape(len(root), 21, 3)
    return axis_angle_to_matrix(
        torch.from_numpy(np.concatenate((root[:, None, :], body), axis=1))
    )


def body_errors(reference, candidate):
    cosine, relative, angles, joints, transl, betas = [], [], [], [], [], []
    for key, expected in reference.items():
        actual = candidate[key]
        a, b = np.asarray(expected["features"]), np.asarray(actual["features"])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            return False, {"valid_body_features": False}
        norm_a, norm_b = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
        cosine.extend(np.sum(a * b, axis=-1) / np.maximum(norm_a * norm_b, 1e-12))
        relative.extend(np.linalg.norm(a - b, axis=-1) / np.maximum(norm_a, 1e-12))
        a, b = rotations(expected["params"]), rotations(actual["params"])
        trace = (a * b).sum(dim=(-1, -2))
        angles.extend(
            torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1, 1))).numpy().ravel()
        )
        joints.extend(
            np.linalg.norm(
                np.asarray(expected["joints"]) - actual["joints"], axis=-1
            ).ravel()
            * 1000
        )
        transl.extend(
            np.linalg.norm(
                np.asarray(expected["params"]["transl"]) - actual["params"]["transl"],
                axis=-1,
            )
            * 1000
        )
        betas.extend(
            np.abs(
                np.asarray(expected["params"]["betas"]) - actual["params"]["betas"]
            ).ravel()
        )
    if any(
        not np.isfinite(metric).all()
        for metric in (cosine, relative, angles, joints, transl, betas)
    ):
        return False, {"valid_body_parameters": False}
    metrics = {
        "feature_cosine_p05": percentile(cosine, 5),
        "feature_relative_l2_p95": percentile(relative, 95),
        "rotation_deg_p95": percentile(angles, 95),
        "joints_mm_p95": percentile(joints, 95),
        "translation_mm_p95": percentile(transl, 95),
        "betas_abs_p95": percentile(betas, 95),
    }
    return (
        metrics["feature_cosine_p05"] >= 0.995
        and metrics["feature_relative_l2_p95"] <= 0.05
        and metrics["rotation_deg_p95"] <= 1
        and metrics["joints_mm_p95"] <= 10
    ), metrics


def compare(workload, reference, candidate):
    if not candidate or candidate.keys() != reference.keys():
        return ParityResult(
            False,
            {"reference_items": len(reference), "candidate_items": len(candidate)},
            "Input ID coverage differs",
        )
    if workload in ("dino", "dino-vitpose"):
        passed, metrics = box_errors(reference, candidate)
    elif workload == "body":
        passed, metrics = body_errors(reference, candidate)
    else:
        passed, metrics = pose_errors(
            [(reference[key], value) for key, value in candidate.items()]
        )
    return ParityResult(
        bool(passed),
        {"checked_ids": len(candidate), **metrics},
        None if passed else "PT32 output parity thresholds exceeded",
    )
