from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from subprocess import STDOUT, CompletedProcess
from types import SimpleNamespace

import numpy as np
import pytest
import record as publish_run
import run as baseline
import speed_inputs
import torch
from parity import compare
from run import PARITY_NOT_EVALUATED_REASON, start_suite
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


def test_baseline_suites_only_schedule_the_authorized_conditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = []

    def child_process(command, **kwargs):
        action, workload, backend, size, _ = command[-5:]
        assert action == "case"
        launched.append((workload, backend, int(size)))
        result = tmp_path / "results" / workload / f"{backend}-b{size}" / "report.json"
        write_json_atomic(result, {"status": "ok"})
        return CompletedProcess(command, 0)

    monkeypatch.setattr(
        baseline,
        "subprocess",
        SimpleNamespace(
            run=child_process,
            check_output=lambda *a, **kw: "a" * 40,
            STDOUT=STDOUT,
        ),
    )
    monkeypatch.setattr(
        baseline,
        "InputSet",
        lambda _: SimpleNamespace(
            pins=SimpleNamespace(schedule=lambda *a: None),
        ),
    )
    for group in ("detection", "pose", "body"):
        args = SimpleNamespace(
            group=group,
            output=tmp_path,
            input_dir=tmp_path / "fixed",
            benchmark=baseline.BENCHMARK,
            external_assets=tmp_path / "external",
            trt_root=tmp_path / "trt",
            pose_engine_run=tmp_path / "prior",
        )
        assert baseline.suite(args) == 0
        invocation = json.loads((tmp_path / f"invocation-{group}.json").read_bytes())
        assert (
            invocation["settings"]["parity_not_evaluated_reason"]
            == PARITY_NOT_EVALUATED_REASON
        )
    assert len(launched) == len(set(launched)) == 22
    assert {backend for _, backend, _ in launched} == {"PT16", "TRT16"}
    assert {(workload, size) for workload, _, size in launched} == {
        ("dino", 1),
        ("dino", 2),
        ("vitpose", 1),
        ("vitpose", 4),
        ("vitpose", 8),
        ("vitpose-flip", 1),
        ("vitpose-flip", 4),
        ("vitpose-flip", 8),
        ("body", 1),
        ("body", 2),
        ("body", 4),
    }


def test_direct_case_rejects_pt32_before_accessing_assets() -> None:
    with pytest.raises(
        ValueError, match="PT32 is outside the baseline measurement scope"
    ):
        baseline.one_case(SimpleNamespace(workload="dino", backend="PT32", batch=1))


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


@pytest.mark.parametrize(
    ("code", "phase", "completed"),
    [
        (Status.OOM, "load", 0),
        (Status.OOM, "measure", 1),
        (Status.CONVERSION_FAILED, "engine_build", 0),
        (Status.PARITY_FAILED, "parity", 2),
    ],
)
def test_outcomes_reach_public_json_and_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: Status,
    phase: str,
    completed: int,
) -> None:
    fixed = _record_fixture(tmp_path / "inputs", monkeypatch)
    schedule = fixed.pins.schedule("frames", 1)
    monkeypatch.setattr(publish_run, "SWEEPS", {"dino": (1,)})
    monkeypatch.setattr(publish_run, "GROUPS", {"detection": ("dino",)})
    speed = TimingStats(2, 2, 4, 2, 2, 2, 2, 500)
    sample = BatchSample(
        1, ("video/clip/frame",), 1, 1, 2, (StageSample(Stage.FORWARD, "dino", 2, 2),)
    )
    attempts = []
    for backend in (Backend.PT16, Backend.TRT16):
        reference_checked = backend is Backend.TRT16 and code is Status.PARITY_FAILED
        failure = (
            None if backend is Backend.PT16 else Failure(code, phase, "fixture failure")
        )
        samples = [sample] * (2 if backend is Backend.PT16 else completed)
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
            case,
            BenchmarkProtocol((1,), 1, 2, 1),
            (),
            samples=samples,
            failure=failure,
            parity_not_evaluated_reason=None
            if reference_checked
            else PARITY_NOT_EVALUATED_REASON,
        )
        if reference_checked:
            report.parity = compare(
                "dino",
                {"frame": {"boxes": [[0, 0, 10, 10]], "scores": [0.9]}},
                {"frame": {"boxes": [[0, 0, 10, 10]], "scores": [0.8]}},
            )
        if backend is Backend.PT16 or reference_checked:
            report.summary = dict.fromkeys(
                ("stage_wall", "model_gpu", "forward"), speed
            )
        if failure and failure.status is Status.CONVERSION_FAILED:
            report.conversion = (
                ConversionResult(Conversion.ENGINE_BUILD, None, failure),
            )
        relative = Path("results") / "dino" / f"{backend}-b1"
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        write_json_atomic(directory / "report.json", report.to_dict())
        (directory / "case.log").write_text(
            failure.message if failure else "case completed"
        )
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
    publish_run.record(tmp_path, publication, input_dir=tmp_path / "inputs")
    conditions = json.loads((publication / "cases.json").read_bytes())["conditions"]
    with (publication / "cases.csv").open() as file:
        rows = {row["backend"]: row for row in csv.DictReader(file)}
    assert set(rows) == {"PT16", "TRT16"}
    measured = rows["PT16"]
    assert measured["status"] == "ok" and measured["parity_status"] == "not_evaluated"
    assert measured["parity_not_evaluated_reason"] == PARITY_NOT_EVALUATED_REASON
    assert float(measured["wall_p50_ms"]) == 2
    measured_case, failed_case = conditions
    assert measured_case["parity"] is None and measured_case["failure"] is None
    assert measured_case["parity_not_evaluated_reason"] == PARITY_NOT_EVALUATED_REASON

    row = rows["TRT16"]
    assert (
        row["failure_code"] == code
        and row["failure_phase"] == phase
        and row["failure_reason"]
    )
    assert row["samples"] == str(completed)
    assert (
        failed_case["failure"]["status"] == code and failed_case["samples"] == completed
    )
    with (publication / "measurements/dino-trt16.csv").open() as file:
        assert len(list(csv.DictReader(file))) == completed
    if code is Status.PARITY_FAILED:
        assert row["status"] == "parity_failed" and row["parity_status"] == "failed"
        assert float(row["wall_p50_ms"]) == 2 and not failed_case["parity"]["passed"]
    else:
        assert row["status"] == "infeasible" and row["parity_status"] == "not_evaluated"
        assert (
            row["wall_p50_ms"] == ""
            and failed_case["summary"] is None
            and failed_case["parity"] is None
        )
