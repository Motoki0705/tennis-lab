"""Comparison boundaries for fixed timing reports and representative overlays."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models.dino.person_detector import PersonDetectionResult
from src.submodules.models.vitpose.pose2d import Pose2DResult
from src.utils.inference_benchmark import BenchmarkProtocol
from src.utils.inference_benchmark_inputs import TimingSchedule

RUN = (
    Path(__file__).resolve().parents[3]
    / "knowledge/runs/run-meiji-player-model-comparison"
)


def test_frontier_keeps_equal_points_and_excludes_unqualified_conditions(
    tmp_path: Path,
) -> None:
    compare = runpy.run_path(str(RUN / "compare_conditions.py"))

    def row(model, latency, quality, *, status="measured", speed="ok", pose=False):
        return {
            "model_id": model,
            "backend": "pt32",
            "stage_p50_ms": latency,
            "workload": "detection_and_pose" if pose else "detection",
            "coverage_oks" if pose else "recall_50": quality,
            "accuracy_status": status,
            "speed_status": speed,
        }

    rows = [
        row("fast", 10, 0.90),
        row("slower-same-recall", 11, 0.90),
        row("same-speed-lower-recall", 10, 0.86),
        row("dino", 20, 0.98),
        row("equal-to-dino", 20, 0.98),
        row("failed", 1, 1, status="parity_failed"),
        row("historical", 1, 1, speed="historical"),
        row("no-speed", 1, 1, speed="pending"),
        row("joint-fast", 2, 0.80, pose=True),
        row("joint-accurate", 12, 0.92, pose=True),
    ]
    pareto = compare["frontiers"](rows)
    assert pareto == {
        "detection": ["fast/pt32", "dino/pt32", "equal-to-dino/pt32"],
        "detection_and_pose": ["joint-fast/pt32", "joint-accurate/pt32"],
    }
    compare["plot_pareto"](tmp_path, {"rows": rows, "pareto": pareto})
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "pareto-detection.png",
        "pareto-detection_and_pose.png",
    }


@pytest.mark.parametrize(
    "outcome", ["ok", "wrong_order", "legacy", "infeasible", "implicit_parity"]
)
def test_timing_uses_the_pinned_frame_order_and_wall_boundary(
    outcome: str,
) -> None:
    compare = runpy.run_path(str(RUN / "compare_conditions.py"))
    ids = ["clip_003/cam0/074", "clip_013/cam2/162", "clip_028/cam3/736"]
    pin = "a" * 64
    timing_pin = "b" * 64
    schedule = TimingSchedule(
        BenchmarkProtocol(batch_sizes=(1, 2)),
        tuple((key,) for key in ids),
        "c" * 64,
    )
    samples = [{"block": 1 + i // 100, "ids": [ids[i % 3]]} for i in range(300)]
    if outcome == "wrong_order":
        samples[0], samples[1] = samples[1], samples[0]
    report: dict[str, object] = {
        "schema_version": 2 if outcome == "legacy" else 3,
        "case": {
            "name": "dino",
            "backend": "PT16",
            "batch_size": 1,
            "item_unit": "image",
            "input_manifest_sha256": pin,
            "settings": {
                "timing_protocol_sha256": timing_pin,
                "timed_input_ids_sha256": schedule.timed_ids_sha256,
            },
        },
        "protocol": {
            "batch_sizes": [1, 2],
            "warmup": 30,
            "iterations_per_block": 100,
            "blocks": 3,
        },
        "environment": {
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "fp32_precision": {"matmul": "ieee"},
        },
        "status": "ok",
        "parity_status": "not_evaluated",
        "parity_not_evaluated_reason": None
        if outcome == "implicit_parity"
        else "PT32 excluded by scope",
        "samples": samples,
        "summary": {
            "stage_wall": {
                "p50_ms_per_item": 12.0,
                "p95_ms_per_item": 20.0,
                "items_per_second": 75.0,
            },
            "model_gpu": {"p50_ms_per_item": 8.0, "p95_ms_per_item": 14.0},
        },
        "nvml_peak_minus_before_load_bytes": 2 * 1024**2,
        "torch_peak_allocated_bytes": 1024**2,
    }
    condition = {"model_id": "dino-swin-l", "backend": "pt16"}
    if outcome == "infeasible":
        report.update(
            status="infeasible",
            summary=None,
            failure={"message": "Unsupported operator"},
        )
    errors = {
        "wrong_order": "sample order differs",
        "legacy": "Timing protocol or identity mismatch",
        "implicit_parity": "explicit PT32 parity exclusion",
    }
    if outcome in errors:
        with pytest.raises(ValueError, match=errors[outcome]):
            compare["summarize_timing"](report, condition, pin, timing_pin, schedule)
    else:
        summary = compare["summarize_timing"](
            report, condition, pin, timing_pin, schedule
        )
        assert summary["parity_status"] == "PT32未計測"
        assert summary["speed_status"] == outcome
        if outcome == "infeasible":
            assert summary["speed_reason"] == "Unsupported operator"
            assert "stage_p50_ms" not in summary
        else:
            assert (
                summary["stage_p50_ms"],
                summary["model_p50_ms"],
                summary["images_per_s"],
            ) == (12, 8, 75)
            assert (
                summary["nvml_peak_delta_mib"],
                summary["torch_peak_allocated_mib"],
            ) == (2, 1)


def test_panel_keeps_source_pixels_and_does_not_draw_invalid_joints(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    draw = runpy.run_path(str(RUN / "render_comparison.py"))["draw_panel"]
    image: NDArray[np.uint8] = np.zeros((100, 200, 3), np.uint8)
    xy: NDArray[np.float32] = np.zeros((1, 17, 2), np.float32)
    xy[0, 0], xy[0, 1] = [48, 40], [160, 70]
    valid: NDArray[np.bool_] = np.zeros((1, 17), bool)
    valid[0, 0] = True
    panel = draw(image, "An image", np.array([[22, 20, 78, 64]]), xy, valid)
    assert panel.shape == (84, 100, 3)
    assert panel[34 + 20, 24, 1] > 100
    assert panel[34 + 35, 80].tolist() == [0, 0, 0]
    assert panel[34 + 10, 11, 1] > 100
    assert image.sum() == 0


def test_baseline_crops_only_detected_people_in_source_order(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    infer_frame = runpy.run_path(str(RUN / "baseline_pytorch.py"))["infer_frame"]
    image: NDArray[np.uint8] = np.full((40, 60, 3), 16, np.uint8)
    boxes = np.array([[3, 7, 13, 27], [20, 2, 50, 12]], np.float32)
    scores = np.array([0.81, 0.65], np.float32)
    poses = torch.zeros(2, 17, 3)
    poses[:, 0] = torch.tensor([[9.0, 15.0, 1.4], [33.0, 6.0, 0.6]])
    poses[0, 1] = torch.tensor([8.0, 8.0, -0.1])

    class Detector:
        def predict(self, request):
            assert request.frame_bgr is image
            return PersonDetectionResult(boxes, scores)

    class Pose:
        def predict(self, request):
            (frame,) = request.frames_bgr
            assert frame[0] == 142 and frame[1] is image
            assert request.frame_indices.tolist() == [142, 142]
            torch.testing.assert_close(
                request.bbx_xys, torch.tensor([[8.0, 17.0, 24.0], [35.0, 7.0, 36.0]])
            )
            return Pose2DResult(poses)

    row = infer_frame(142, image, Detector(), Pose())
    np.testing.assert_array_equal(row.boxes_xyxy, boxes)
    np.testing.assert_array_equal(row.scores, scores)
    assert row.keypoints_xyc is not None
    np.testing.assert_array_equal(row.keypoints_xyc[:, 0], poses[:, 0])
    assert row.keypoints_xyc[0, 1].tolist() == [0.0, 0.0, 0.0]

    boxes, scores = boxes[:0], scores[:0]
    row = infer_frame(142, image, Detector(), Pose())
    assert row.boxes_xyxy.shape == (0, 4)
    assert row.keypoints_xyc is not None and row.keypoints_xyc.shape == (0, 17, 3)
