from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import record as publish_run
import speed_inputs
import torch
from parity import compare
from run import reference_predictions, start_suite
from speed_inputs import ClipRows, publish, require_sha, write_crops
from speed_models import classify_startup, dino_decode_gpu

from src.submodules.models.dino.architecture import COCO_PERSON_CLASS_ID
from src.submodules.models.dino.person_detector import (
    PersonDetectionResult,
    decode_person_detections,
)
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    BatchSample,
    BenchmarkCase,
    BenchmarkProtocol,
    BenchmarkReport,
    ConditionUnavailable,
    Conversion,
    ConversionResult,
    Failure,
    Stage,
    StageSample,
    Status,
    TimingStats,
)
from src.utils.inference_benchmark_inputs import read_pinned_json
from src.utils.io import write_json_atomic


def test_detector_postprocess_retains_production_order_and_coordinates() -> None:
    logits = torch.zeros((2, 3, COCO_PERSON_CLASS_ID + 1))
    logits[0, :, COCO_PERSON_CLASS_ID] = torch.tensor([0.7, 2.0, -2.0])
    boxes = torch.tensor(
        [[[0.2, 0.4, 0.3, 0.6], [0.85, 0.7, 0.4, 0.2], [0.45, 0.55, 0.1, 0.1]]] * 2
    )
    raw = {"pred_logits": logits, "pred_boxes": boxes}
    sizes = [(80, 120), (160, 40)]
    expected = decode_person_detections(raw, image_sizes=sizes, confidence=0.6)
    actual = dino_decode_gpu(raw, sizes, confidence=0.6)
    for (xyxy, score), result in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(xyxy.numpy(), result.boxes_xyxy)
        np.testing.assert_array_equal(score.numpy(), result.scores)
    assert actual[0][0].shape == (2, 4)
    assert actual[1][0].shape == (0, 4)


def test_fixed_crops_prioritize_observations_and_pin_exact_bytes(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(2):
        path = tmp_path / f"frame{index}.npy"
        image = np.full((80, 120, 3), 22 + index, dtype=np.uint8)
        np.save(path, image)
        rows.append(
            {
                "id": f"frame{index}",
                "frame": index,
                "clip_id": "clip",
                "camera": "cam0",
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": [80, 120, 3],
            }
        )
    tracked = ClipRows(
        "clip",
        {},
        {"cam0": np.ones((2, 1), bool)},
        {"cam0": np.array([[[44, 48, 38.4]]] * 2)},
        (0, 1),
        [],
    )
    boxes = np.array(
        [
            [8, 8, 24, 24],
            [34, 8, 50, 24],
            [58, 8, 74, 24],
            [78, 8, 94, 24],
            [30, 32, 58, 64],
        ],
        np.float32,
    )
    scores = np.array([0.91, 0.82, 0.7, 0.61, 0.33], np.float32)
    crops = write_crops(
        tmp_path, rows, [tracked], lambda _: PersonDetectionResult(boxes, scores)
    )
    manifest = publish(tmp_path, rows, [], [], crops, {"fixture": True})
    data = json.loads(manifest.read_bytes())
    assert data["pose_sweep_crop_count"] == 8
    spec = data["indexes"]["crops"]
    pinned_crops = read_pinned_json(tmp_path / spec["path"], spec["sha256"])
    assert [row["id"] for row in pinned_crops[:4]] == [
        "frame0/dino_000",
        "frame0/dino_001",
        "frame0/dino_002",
        "frame0/dino_004",
    ]
    crop = pinned_crops[3]
    assert crop["scene_observation_iou"] == pytest.approx(1.0)
    path = require_sha(tmp_path / crop["path"], crop["sha256"])
    assert np.load(path).shape == (3, 256, 256)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        require_sha(path, crop["sha256"])


def test_pose_parity_preserves_pixel_units_and_fixed_ids() -> None:
    reference = np.tile([800.0, 500.0, 0.8], (17, 1))
    candidate = reference.copy()
    candidate[:, 1] += 1.2
    result = compare("vitpose-flip", {"crop": reference}, {"crop": candidate})
    assert not result.passed
    assert result.metrics["pose_xy_px_p95"] == pytest.approx(1.2)
    assert (
        compare("vitpose-flip", {"crop": reference}, {"other": reference}).reason
        == "Input ID coverage differs or is empty"
    )


def test_suite_rejects_existing_results_before_running_children(tmp_path: Path) -> None:
    (tmp_path / "results/dino").mkdir(parents=True)
    args = SimpleNamespace(
        group="detection", output=tmp_path, trt_root=tmp_path / "missing"
    )
    with pytest.raises(FileExistsError, match="Use a fresh output for detection"):
        start_suite(args)
    assert not (tmp_path / "invocation-detection.json").exists()


def test_pt32_reference_requires_same_invocation_and_current_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"reference model")
    predictions = tmp_path / "predictions.json"
    write_json_atomic(predictions, {"a": 1})
    identity = {
        "commit": "ref-commit",
        "suite_id": "fresh-suite",
        "contract": {"flip": True},
    }
    case = BenchmarkCase(
        "dino", Backend.PT16, 2, "image", "a" * 64, {"reference_identity": identity}
    )
    write_json_atomic(
        tmp_path / "report.json",
        {
            "case": {
                "name": "dino",
                "backend": "PT32",
                "batch_size": 1,
                "input_manifest_sha256": case.input_manifest_sha256,
                "settings": case.settings,
            },
            "status": "ok",
            "predictions_sha256": dual_sha256(predictions),
            "weights": {
                "dino": {"path": str(checkpoint), "sha256": dual_sha256(checkpoint)}
            },
        },
    )
    weights = {"dino": checkpoint}
    assert reference_predictions(tmp_path, case, weights) == {"a": 1}
    older = replace(
        case, settings={"reference_identity": {**identity, "commit": "older"}}
    )
    with pytest.raises(ConditionUnavailable, match="reference identity"):
        reference_predictions(tmp_path, older, weights)
    with pytest.raises(ConditionUnavailable, match="asset path differs"):
        reference_predictions(tmp_path, case, {"dino": tmp_path / "alternate.pt"})
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ConditionUnavailable, match="SHA-256 mismatch"):
        reference_predictions(tmp_path, case, weights)
    checkpoint.write_bytes(b"reference model")
    write_json_atomic(predictions, {"a": 2})
    with pytest.raises(ConditionUnavailable, match="SHA-256 mismatch"):
        reference_predictions(tmp_path, case, weights)


def test_trt_context_creation_failure_can_be_recorded_as_unavailable() -> None:
    with (
        pytest.raises(
            ConditionUnavailable, match="Could not create TensorRT context"
        ) as failure,
        classify_startup(Backend.TRT16),
    ):
        raise RuntimeError("Could not create TensorRT context")
    assert failure.value.status is Status.UNAVAILABLE
    with pytest.raises(torch.OutOfMemoryError), classify_startup(Backend.TRT16):
        raise torch.OutOfMemoryError("allocator failed")


def _record_fixture(
    directory: Path, monkeypatch: pytest.MonkeyPatch
) -> speed_inputs.InputSet:
    directory.mkdir()
    path = directory / "frame.npy"
    path.write_bytes(b"fixed payload")
    rows = [{"id": "video/clip/frame", "path": path.name, "sha256": dual_sha256(path)}]
    indexes = {}
    for name, data in (("frames", rows), ("body_frames", [])):
        index = directory / f"{name}.json"
        write_json_atomic(index, data)
        indexes[name] = {
            "path": index.name,
            "records": len(data),
            "sha256": dual_sha256(index),
        }
    manifest = directory / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "id": speed_inputs.INPUT_ID,
            "source_benchmark_sha256": speed_inputs.SOURCE_SHA256,
            "indexes": indexes,
        },
    )
    timing = directory / "timing-protocol.json"
    sha = dual_sha256(manifest)
    write_json_atomic(
        timing,
        {
            "schema_version": 1,
            "input_manifest_sha256": sha,
            "warmup": 1,
            "iterations_per_block": 2,
            "blocks": 1,
            "schedules": {
                "frames": {
                    "records": 1,
                    "batch_sizes": {
                        "1": hashlib.sha256(
                            b'[["video/clip/frame"],["video/clip/frame"]]'
                        ).hexdigest(),
                    },
                }
            },
        },
    )
    monkeypatch.setattr(speed_inputs, "INPUT_MANIFEST_SHA256", sha)
    monkeypatch.setattr(speed_inputs, "TIMING_PROTOCOL_SHA256", dual_sha256(timing))
    monkeypatch.setattr(speed_inputs, "TIMING_PROTOCOL", timing)
    return speed_inputs.InputSet(directory)


@pytest.mark.parametrize("oom_samples", [0, 1])
def test_failures_reach_public_json_and_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oom_samples: int
) -> None:
    fixed = _record_fixture(tmp_path / "inputs", monkeypatch)
    schedule = fixed.pins.schedule("frames", 1)
    monkeypatch.setattr(publish_run, "SWEEPS", {"dino": (1,)})
    monkeypatch.setattr(publish_run, "GROUPS", {"detection": ("dino",)})
    speed = TimingStats(2, 2, 4, 2, 2, 2, 2, 500)
    sample = BatchSample(
        1, ("video/clip/frame",), 1, 1, 2, (StageSample(Stage.FORWARD, "dino", 2, 2),)
    )
    statuses = {
        Backend.PT32: (
            Failure(
                Status.OOM, "measure" if oom_samples else "load", "allocator failed"
            ),
            [sample] * oom_samples,
        ),
        Backend.PT16: (
            Failure(Status.PARITY_FAILED, "parity", "score limit exceeded"),
            [sample, sample],
        ),
        Backend.TRT16: (
            Failure(Status.CONVERSION_FAILED, "engine_build", "unsupported TRT op"),
            [],
        ),
    }
    attempts = []
    for backend, (failure, samples) in statuses.items():
        case = BenchmarkCase(
            "dino",
            backend,
            1,
            "image",
            fixed.sha256,
            {
                "source_commit": "a" * 40,
                "timing_protocol_sha256": fixed.pins.timing_sha256,
                "timed_input_ids_sha256": schedule.timed_ids_sha256,
            },
        )
        report = BenchmarkReport(
            case, BenchmarkProtocol((1,), 1, 2, 1), (), samples=samples, failure=failure
        )
        if backend is Backend.PT16:
            report.parity = compare(
                "dino",
                {"frame": {"boxes": [[0, 0, 10, 10]], "scores": [0.9]}},
                {"frame": {"boxes": [[0, 0, 10, 10]], "scores": [0.8]}},
            )
            report.summary = dict.fromkeys(
                ("stage_wall", "model_gpu", "forward"), speed
            )
        if backend is Backend.TRT16:
            report.conversion = (
                ConversionResult(Conversion.ENGINE_BUILD, None, failure),
            )
        relative = Path("results") / "dino" / f"{backend}-b1"
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        write_json_atomic(directory / "report.json", report.to_dict())
        (directory / "case.log").write_text(failure.message)
        attempts.append(
            {
                "workload": "dino",
                "backend": backend,
                "batch": 1,
                "path": str(relative),
                "status": report.status,
                "exit_code": 0,
            }
        )
    write_json_atomic(tmp_path / "attempts-detection.json", attempts)

    publication = tmp_path / "published"
    publication.mkdir()
    publish_run.record(tmp_path, publication)
    conditions = json.loads((publication / "cases.json").read_bytes())["conditions"]
    with (publication / "cases.csv").open() as file:
        rows = {row["backend"]: row for row in csv.DictReader(file)}
    for backend, code, count in (
        ("PT32", "oom", oom_samples),
        ("TRT16", "conversion_failed", 0),
    ):
        row = rows[backend]
        assert row["status"] == "infeasible" and row["failure_code"] == code
        assert (
            row["samples"] == str(count)
            and row["wall_p50_ms"] == ""
            and row["failure_reason"]
        )
        case = next(
            condition
            for condition in conditions
            if condition["case"]["backend"] == backend
        )
        assert case["summary"] is None and case["parity"] is None
        assert case["failure"]["status"] == code and case["samples"] == count
    with (publication / "measurements/dino-pt32.csv").open() as file:
        assert len(list(csv.DictReader(file))) == oom_samples
    assert (
        rows["PT16"]["status"] == "parity_failed"
        and float(rows["PT16"]["wall_p50_ms"]) == 2
    )
    parity_case = next(
        condition for condition in conditions if condition["status"] == "parity_failed"
    )
    assert parity_case["summary"]["stage_wall"]["p95_ms"] == 2
    assert not parity_case["parity"]["passed"]
