from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.utils.inference_parity import (
    BodyOutput,
    Coco17Output,
    DetectionOutput,
    compare_bodies,
    compare_coco17,
    compare_detections,
)


def test_detection_matches_boxes_before_applying_shared_limits() -> None:
    expected = DetectionOutput([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8])
    reordered = DetectionOutput([[20, 20, 30, 30], [0, 0, 10, 10]], [0.8, 0.9])
    reference = {"frame": expected}
    assert compare_detections(reference, {"frame": reordered}).passed

    shifted = replace(
        reordered, boxes_xyxy=[[20, 20, 30, 30], [0.7, 0, 10, 10]], scores=[0.8, 0.9135]
    )
    result = compare_detections(reference, {"frame": shifted})
    assert not result.passed
    assert result.metrics["iou_p05"] == pytest.approx(0.9335)
    assert result.metrics["score_abs_p95"] == pytest.approx(0.012825)
    assert result.criteria["reference"] == "same model PT32"
    rules = result.criteria["rules"]
    assert isinstance(rules, dict)
    assert rules["iou_p05"] == {"statistic": "p05", "limit": 0.98, "comparison": ">="}
    assert rules["score_abs_p95"]["limit"] == 0.01
    assert rules["missing_ratio"]["limit"] == rules["extra_ratio"]["limit"] == 0.01


def test_detection_keeps_missing_counts_and_rejects_misaligned_scores() -> None:
    expected = DetectionOutput([[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8])
    missing = DetectionOutput([[0, 0, 10, 10]], [0.9])
    result = compare_detections({"frame": expected}, {"frame": missing})
    assert not result.passed
    assert result.metrics["missing"] == 1
    assert result.metrics["missing_ratio"] == 0.5
    assert result.metrics["extra_ratio"] == 0.0

    invalid = compare_detections(
        {"frame": expected}, {"frame": replace(expected, scores=[0.9])}
    )
    assert not invalid.passed and invalid.metrics["invalid"]
    assert invalid.reason is not None and "Detection scores" in invalid.reason


def test_coco17_rejects_broadcast_and_valid_mask_changes() -> None:
    ref = np.zeros((17, 3))
    valid = np.arange(17) != 2
    ref[~valid] = np.nan
    reference = {"crop": Coco17Output(ref, valid)}
    fail_outputs = (
        Coco17Output(ref[:1], valid[:1]),
        Coco17Output(ref, np.roll(valid, 1)),
        Coco17Output(ref, valid.astype(np.int64)),
    )
    for output in fail_outputs:
        result = compare_coco17(reference, {"crop": output})
        assert not result.passed and result.metrics["invalid"]

    observed = ref.copy()
    observed[0, 0] = np.inf
    invalid = compare_coco17(reference, {"crop": Coco17Output(observed, valid)})
    assert not invalid.passed and invalid.metrics["invalid"]

    observed[0, 0] = 0.8
    result = compare_coco17(reference, {"crop": Coco17Output(observed, valid)})
    assert result.passed and result.metrics["compared_joints"] == 16
    assert result.metrics["pose_xy_px_p95"] == pytest.approx(0.2)
    assert result.metrics["pose_xy_px_max"] == pytest.approx(0.8)


def test_composed_pose_parity_uses_shape_and_mask_at_matched_boxes() -> None:
    poses = np.zeros((2, 17, 3))
    poses[0] = 1
    reference = DetectionOutput(
        [[0, 0, 10, 10], [20, 20, 30, 30]], [0.9, 0.8], Coco17Output(poses)
    )
    candidate = DetectionOutput(
        [[20, 20, 30, 30], [0, 0, 10, 10]], [0.8, 0.9], Coco17Output(poses[::-1])
    )
    assert compare_detections(
        {"frame": reference}, {"frame": candidate}, with_pose=True
    ).passed

    invalid = replace(candidate, poses=Coco17Output(poses[:1]))
    result = compare_detections(
        {"frame": reference}, {"frame": invalid}, with_pose=True
    )
    assert not result.passed and result.metrics["invalid"]


def test_body_parity_checks_time_joints_and_camera_space_errors() -> None:
    joints = np.zeros((2, 3, 3))
    reference = BodyOutput(
        joints, np.zeros((2, 1, 3)), np.zeros((2, 3)), frame_ids=(12, 14)
    )
    refs = {"sequence": reference}
    assert compare_bodies(refs, refs).passed
    for output in (
        replace(reference, joints=joints[:, :1]),
        replace(reference, frame_ids=(14, 12)),
    ):
        invalid = compare_bodies(refs, {"sequence": output})
        assert not invalid.passed and invalid.metrics["invalid"]

    result = compare_bodies(
        refs, {"sequence": replace(reference, joints=joints + [0.012, 0, 0])}
    )
    assert not result.passed and result.metrics["joints_mm_p95"] == pytest.approx(12)
    assert result.metrics["compared_joints"] == 6
    assert result.criteria["joint_coordinates"] == "camera"
