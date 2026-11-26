"""Comparison boundaries for fixed timing reports and representative overlays."""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models.dino.person_detector import PersonDetectionResult
from src.submodules.models.vitpose.pose2d import Pose2DResult
from src.utils.checksum import FileIntegrityError
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
            "backend": "pt16",
            "accuracy_batch_size": 1,
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
        row("unmatched", None, 1, speed="unmatched_host"),
        {**row("trt-diagnostic", 1, 1), "pareto_exclusion_reason": "Unstable plan"},
        row("joint-fast", 2, 0.80, pose=True),
        row("joint-accurate", 12, 0.92, pose=True),
    ]
    pareto = compare["frontiers"](rows)
    assert pareto == {
        "detection": ["fast/pt16", "dino/pt16", "equal-to-dino/pt16"],
        "detection_and_pose": ["joint-fast/pt16", "joint-accurate/pt16"],
    }
    compare["plot_pareto"](tmp_path, {"rows": rows, "pareto": pareto})
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "pareto-detection.png",
        "pareto-detection_and_pose.png",
    }


def _timing_report(
    schedule: TimingSchedule, input_sha: str, protocol_sha: str
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "case": {
            "name": "dino",
            "backend": "PT16",
            "batch_size": 1,
            "item_unit": "image",
            "input_manifest_sha256": input_sha,
            "settings": {
                "timing_protocol_sha256": protocol_sha,
                "timed_input_ids_sha256": schedule.timed_ids_sha256,
                "host_threads": 4,
                "opencv_threads": 1,
            },
        },
        "protocol": asdict(schedule.protocol),
        "environment": {
            "gpu": "NVIDIA GeForce RTX 5060 Ti",
            "fp32_precision": {"matmul": "ieee"},
        },
        "status": "ok",
        "parity_status": "not_evaluated",
        "parity_not_evaluated_reason": "PT32 excluded by scope",
        "samples": [
            {
                "block": 1 + i // 100,
                "ids": list(schedule.batch_ids[i % len(schedule.batch_ids)]),
                "items": 1,
            }
            for i in range(300)
        ],
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


@pytest.mark.parametrize(
    "outcome",
    [
        "ok",
        "no_order_digest",
        "wrong_order",
        "wrong_order_digest",
        "legacy",
        "infeasible_without_log",
        "implicit_parity",
        "unmatched_threads",
        "baseline_opencv_pin",
        "fp32_attention",
        "wrong_attention_identity",
    ],
)
def test_timing_uses_the_pinned_frame_order_and_wall_boundary(outcome: str) -> None:
    summarize = runpy.run_path(str(RUN / "compare_conditions.py"))["summarize_timing"]
    pin, timing_pin = "a" * 64, "b" * 64
    schedule = TimingSchedule(
        BenchmarkProtocol(batch_sizes=(1, 2)),
        (("clip_003/cam0/074",), ("clip_013/cam2/162",), ("clip_028/cam3/736",)),
        "c" * 64,
    )
    report = _timing_report(schedule, pin, timing_pin)
    settings = report["case"]["settings"]
    row = {"model_id": "dino-swin-l", "backend": "pt16"}
    opencv_pin = None
    if outcome == "wrong_order":
        report["samples"][0], report["samples"][1] = (
            report["samples"][1],
            report["samples"][0],
        )
    if outcome == "no_order_digest":
        del settings["timed_input_ids_sha256"]
    if outcome == "wrong_order_digest":
        settings["timed_input_ids_sha256"] = "d" * 64
    if outcome == "legacy":
        report["schema_version"] = 2
    if outcome == "implicit_parity":
        report["parity_not_evaluated_reason"] = None
    if outcome == "infeasible_without_log":
        report.update(
            status="infeasible",
            summary=None,
            failure={"message": "Unsupported operator"},
        )
    if outcome == "unmatched_threads":
        settings["host_threads"] = 2
    if outcome == "baseline_opencv_pin":
        del settings["opencv_threads"]
        opencv_pin = 1
    if outcome in {"fp32_attention", "wrong_attention_identity"}:
        row = {"model_id": "ecdet-l-coco", "backend": "trt16-fp32attn"}
        report["case"].update(name="ecdet-l-coco/trt16-fp32attn", backend="TRT16")
        settings["engine_condition"] = (
            "trt16" if outcome == "wrong_attention_identity" else "trt16-fp32attn"
        )

    errors = {
        "wrong_order": "sample order differs",
        "wrong_order_digest": "Timing protocol or identity mismatch",
        "legacy": "Timing protocol or identity mismatch",
        "implicit_parity": "explicit PT32 parity exclusion",
        "infeasible_without_log": "pinned terminal failure evidence",
        "unmatched_threads": "host thread settings differ",
        "wrong_attention_identity": "Timing protocol or identity mismatch",
    }
    if outcome in errors:
        with pytest.raises(ValueError, match=errors[outcome]):
            summarize(report, row, pin, timing_pin, schedule)
        return
    result = summarize(
        report, row, pin, timing_pin, schedule, verified_opencv_threads=opencv_pin
    )
    assert result["parity_status"] == "PT32未計測"
    assert result["speed_status"] == "ok"
    assert (
        result["stage_p50_ms"],
        result["model_p50_ms"],
        result["images_per_s"],
        result["nvml_peak_delta_mib"],
        result["torch_peak_allocated_mib"],
    ) == (12, 8, 75, 2, 1)
    assert (
        result["speed_batch_size"],
        result["host_threads"],
        result["opencv_threads"],
    ) == (1, 4, 1)


@pytest.mark.parametrize(
    ("backend", "expected_iou", "expected_status"),
    [("pt16", 0.9963, "passed"), ("trt16", 0.9620, "failed")],
)
def test_rfdetr_saved_pt32_comparison_preserves_measured_frontier_point(
    backend: str,
    expected_iou: float,
    expected_status: str,
) -> None:
    comparison = runpy.run_path(str(RUN / "compare_conditions.py"))
    saved = json.loads(
        (RUN.parent / "run-meiji-rfdetr-large-accuracy/results.json").read_bytes()
    )
    reference = saved["conditions"]["pt32"]["artifacts"]
    candidate = saved["conditions"][backend]["artifacts"]
    case = {"model_id": "rfdetr-l", "backend": backend, **candidate}
    cohort = {
        "id": saved["benchmark_id"],
        "sha256": saved["manifest_sha256"],
        "images": 6844,
    }
    plan = {
        "sha256": "4e3ae2a8206caa607565a8fe5726509201a11c9f11b3134b13c7aa2ae3c40e35"
    }
    timing = {
        "weights": {"model": saved["weights"]},
        "engines": {"plan": plan} if backend == "trt16" else {},
    }
    fields = comparison["rfdetr_saved_parity"](saved, case, reference, timing, cohort)
    row = {
        **case,
        **fields,
        "workload": "detection",
        "accuracy_status": "measured",
        "speed_status": "ok",
        "parity_status": "PT32未計測",
        "stage_p50_ms": 26.43,
        "recall_50": 0.9681,
    }
    assert row["historical_parity_status"] == "historical_passed"
    assert row["common_parity_status"] == expected_status
    assert row["common_parity_iou_p05"] == pytest.approx(expected_iou, abs=0.00005)
    assert row["parity_status"] == "PT32未計測"
    assert comparison["frontiers"]([row])["detection"] == [f"rfdetr-l/{backend}"]
    if backend == "trt16":
        assert "IoU P05 0.9620 < 0.98" in comparison["saved_pt32_note"](row)
        different = {**timing, "engines": {"plan": {"sha256": "0" * 64}}}
        with pytest.raises(
            ValueError,
            match="Saved RF-DETR PT32 comparison disagrees with accuracy or timing",
        ):
            comparison["rfdetr_saved_parity"](saved, case, reference, different, cohort)


def _pin(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def comparison_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    main = runpy.run_path(str(RUN / "compare_conditions.py"))["main"]
    schedule = TimingSchedule(
        BenchmarkProtocol(batch_sizes=(1, 2)),
        tuple((f"image-{index}",) for index in range(96)),
        "c" * 64,
    )
    fixed = SimpleNamespace(
        manifest={"id": "fixed-speed"}, schedule=lambda *_: schedule
    )
    monkeypatch.setitem(
        main.__globals__, "PinnedBenchmarkInputs", lambda *_args, **_kwargs: fixed
    )
    monkeypatch.setitem(main.__globals__, "REPO_ROOT", tmp_path)
    speed_manifest = tmp_path / "manifest.json"
    timing_protocol = tmp_path / "timing-protocol.json"
    speed_manifest.write_text('{"id": "fixed-speed"}')
    timing_protocol.write_text(json.dumps(asdict(schedule.protocol)))
    evidence: dict[str, Any] = {
        "speed_input": {
            **_pin(speed_manifest),
            "timing_protocol": _pin(timing_protocol),
            "id": "fixed-speed",
        },
        "evaluation": {
            "id": "fixed-cohort",
            "sha256": "a" * 64,
            "images": 5,
            "eligible_boxes": 10,
            "scorer_commit": "pinned",
        },
    }
    index = tmp_path / "comparison-evidence.json"
    output = tmp_path / "output"

    def run(case: dict[str, Any]) -> dict[str, str]:
        index.write_text(json.dumps({**evidence, "conditions": [case]}))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_conditions.py",
                "--artifacts",
                str(tmp_path),
                "--evidence",
                str(index),
                "--output",
                str(output),
                "--tables-only",
            ],
        )
        main()
        with (output / "conditions.csv").open(newline="") as handle:
            return next(
                row
                for row in csv.DictReader(handle)
                if (row["model_id"], row["backend"])
                == (case["model_id"], case["backend"])
            )

    return run, evidence, schedule


def test_cli_pins_the_speed_report_and_keeps_separate_batches(
    tmp_path: Path, comparison_cli
) -> None:
    run, evidence, schedule = comparison_cli
    speed, cohort = evidence["speed_input"], evidence["evaluation"]
    raw = _timing_report(schedule, speed["sha256"], speed["timing_protocol"]["sha256"])
    timing = tmp_path / "timing.json"
    timing.write_text(json.dumps(raw))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("fixture")
    accuracy = tmp_path / "accuracy.json"
    accuracy.write_text(
        json.dumps(
            {
                "condition": {
                    "model_id": "dino-swin-l",
                    "backend": "pt16",
                    "has_pose": False,
                    "benchmark_id": cohort["id"],
                    "manifest_sha256": cohort["sha256"],
                },
                "scorer_commit": cohort["scorer_commit"],
                "image_batch_size": 2,
                "predictions_sha256": _pin(predictions)["sha256"],
                "score": {
                    "images": cohort["images"],
                    "overall": {
                        "detection": {
                            "references": cohort["eligible_boxes"],
                            "recall_by_iou": {"0.50": 0.9, "0.75": 0.7},
                            "recall_by_min_score": {"0.5": {"0.50": 0.8}},
                        },
                        "pose": {"with_detection_coverage": {}},
                    },
                    "by_size": {
                        "small": {"detection": {"recall_by_iou": {"0.50": 0.6}}}
                    },
                },
            }
        )
    )
    case = {
        "model_id": "dino-swin-l",
        "backend": "pt16",
        "accuracy": {**_pin(accuracy), "batch_size": 2},
        "predictions": _pin(predictions),
        "timing": _pin(timing),
    }
    row = run(case)
    assert (
        row["accuracy_batch_size"],
        row["speed_batch_size"],
        row["stage_p50_ms"],
    ) == ("2", "1", "12.0")
    raw["summary"]["stage_wall"]["p50_ms_per_item"] = 1.0
    timing.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="SHA-256 differs from the pin"):
        run(case)


@pytest.mark.parametrize(
    "log_state",
    ["terminal", "missing", "nonterminal", "wrong_condition", "success_exit"],
)
def test_cli_requires_the_pinned_terminal_log_for_infeasible_condition(
    tmp_path: Path, comparison_cli, log_state: str
) -> None:
    run, evidence, _ = comparison_cli
    model, backend = "rtdetrv4-l", "trt16-fp32attn"
    log = tmp_path / "diagnosis.log"
    log_model = "rtdetrv4-x" if log_state == "wrong_condition" else model
    line = f"{log_model}/{backend} infeasible (8 cases) at outputs/diagnosis.json\n"
    exit_code = 0 if log_state == "success_exit" else 1
    log.write_text(
        line + ("" if log_state == "nonterminal" else f"exit_code={exit_code}\n")
    )
    report = tmp_path / "diagnosis.json"
    report.write_text(
        json.dumps(
            {
                "model_id": model,
                "condition": backend,
                "status": "infeasible",
                "accuracy_manifest_sha256": evidence["evaluation"]["sha256"],
                "speed_manifest_sha256": evidence["speed_input"]["sha256"],
            }
        )
    )
    condition = {
        "model_id": model,
        "backend": backend,
        "failure": {
            "reason": "NaN before postprocessing",
            "report": _pin(report),
            "log": _pin(log),
        },
    }
    if log_state == "missing":
        log.unlink()
        with pytest.raises(FileIntegrityError, match="diagnosis.log"):
            run(condition)
    elif log_state != "terminal":
        with pytest.raises(ValueError, match="Terminal log does not record a failure"):
            run(condition)
    else:
        row = run(condition)
        assert (row["accuracy_status"], row["speed_status"]) == (
            "infeasible",
            "infeasible",
        )
        assert (
            row["accuracy_batch_size"],
            row["speed_batch_size"],
            row.get("stage_p50_ms", ""),
        ) == ("", "", "")


@pytest.mark.parametrize(
    ("reported_commit", "verified_commit", "accepted"),
    [(None, "pinned", True), (None, None, False), ("different", "pinned", False)],
)
def test_accuracy_requires_the_pinned_scorer(
    reported_commit, verified_commit, accepted
) -> None:
    summarize = runpy.run_path(str(RUN / "compare_conditions.py"))["summarize_accuracy"]
    detection = {
        "references": 10,
        "recall_by_iou": {"0.50": 0.9, "0.75": 0.7},
        "recall_by_min_score": {"0.5": {"0.50": 0.8}},
    }
    case = {"model_id": "rtdetrv4-l", "backend": "pt16", "workload": "detection"}
    cohort = {
        "id": "fixed",
        "sha256": "digest",
        "images": 5,
        "eligible_boxes": 10,
        "scorer_commit": "pinned",
    }
    report = {
        "condition": {
            "model_id": "rtdetrv4-l",
            "backend": "pt16",
            "has_pose": False,
            "benchmark_id": "fixed",
            "manifest_sha256": "digest",
        },
        "score": {
            "images": 5,
            "overall": {
                "detection": detection,
                "pose": {"with_detection_coverage": {}},
            },
            "by_size": {"small": {"detection": {"recall_by_iou": {"0.50": 0.6}}}},
        },
    }
    if reported_commit is not None:
        report["scorer_commit"] = reported_commit
    if accepted:
        assert (
            summarize(report, case, cohort, verified_scorer_commit=verified_commit)[
                "recall_50"
            ]
            == 0.9
        )
    else:
        with pytest.raises(ValueError, match="frozen scorer or cohort"):
            summarize(report, case, cohort, verified_scorer_commit=verified_commit)


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
