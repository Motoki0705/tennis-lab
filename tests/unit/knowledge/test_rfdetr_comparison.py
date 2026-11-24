"""Test the RF-DETR experiment adapter without upstream models or weights."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.utils.checksum import dual_sha256
from src.utils.inference_benchmark import (
    Backend,
    ConditionUnavailable,
    Conversion,
    Status,
)
from src.utils.io import write_json_atomic
from src.utils.models.tensorrt_artifacts import (
    ONNX_FILE,
    EngineBundle,
    EngineEnvironment,
    EngineManifest,
    ExportManifest,
    InputProfile,
    TensorSpec,
)

_SCRIPT = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "knowledge/runs/run-meiji-player-model-comparison/rfdetr.py"
    )
)


def test_preprocess_keeps_native_height_width_and_converts_bgr_to_rgb() -> None:
    image: NDArray[np.uint8] = np.full((3, 5, 3), [51, 102, 204], dtype=np.uint8)
    images, shape = _SCRIPT["prepare_images"]((image,), 2)

    assert shape.tolist() == [[3, 5]]
    torch.testing.assert_close(
        images,
        torch.tensor([1.375546, -0.25, -0.915556]).view(1, 3, 1, 1).expand(1, 3, 2, 2),
    )


def test_person_filter_keeps_coco_label_one_and_aligned_native_boxes() -> None:
    rows = np.array(
        [
            [10, 20, 70, 85, 0.99, 0],  # Not a COCO person.
            [10, 30, 10, 60, 0.98, 1],  # Collapsed after clipping.
            [20, 60, 70, 90, 0.95, 2],  # Bicycle.
            [13, 41, 79, 96, 0.25, 1],  # Exactly on the inclusive cutoff.
            [15, 18, 23, 29, 0.24, 1],
        ],
        np.float32,
    )
    result = _SCRIPT["person_predictions"](rows)
    np.testing.assert_array_equal(result.boxes_xyxy, [[13, 41, 79, 96]])
    np.testing.assert_array_equal(result.scores, [0.25])
    assert result.keypoints_xyc is None
    assert _SCRIPT["person_predictions"](rows[:3]).boxes_xyxy.shape == (0, 4)


def test_parity_matches_reordered_boxes_and_detects_a_missing_person() -> None:
    expected = Predictions(
        np.array([[10, 20, 30, 60], [70, 40, 120, 90]]), np.array([0.8, 0.6])
    )
    observed = Predictions(
        np.array([[70, 40, 119, 90], [10, 20, 30, 60]]), np.array([0.62, 0.79])
    )
    compare = _SCRIPT["prediction_parity"]

    parity = compare([(expected, observed)])
    assert parity.passed
    assert parity.metrics["matched_iou_p05"] == pytest.approx(0.981)
    assert parity.metrics["abs_score_difference"]["p95"] == pytest.approx(0.0195)

    incomplete = Predictions(observed.boxes_xyxy[:1], observed.scores[:1])
    parity = compare([(expected, incomplete)])
    assert not parity.passed
    assert parity.metrics["pt32_retained_fraction"] == 0.5


def test_pt32_jsonl_pins_empty_frames_and_rejects_modified_predictions(
    tmp_path: Path,
) -> None:
    identity = _SCRIPT["header"]("meiji/cohort", "b" * 64, Backend.PT32)
    key = FrameKey("video_000/clip_003", "cam0", 120)
    raw = tmp_path / "predictions.jsonl"
    _SCRIPT["write_predictions"](
        raw, identity, {key: Predictions(np.empty((0, 4)), np.empty(0))}
    )
    write_json_atomic(
        tmp_path / "accuracy.json",
        {"weights_sha256": "a" * 64, "predictions_sha256": dual_sha256(raw)},
    )
    raw_text = raw.read_text()
    _, row = map(json.loads, raw_text.splitlines())
    assert row == {
        "clip_id": "video_000/clip_003",
        "camera_id": "cam0",
        "source_frame": 120,
        "boxes_xyxy": [],
        "scores": [],
        "keypoints_xyc": None,
    }
    predictions = _SCRIPT["read_pt32_predictions"](tmp_path, identity, "a" * 64)
    assert predictions[key].boxes_xyxy.shape == (0, 4)
    raw.write_text(raw_text.replace("120", "121"))
    with pytest.raises(ValueError, match="do not match"):
        _SCRIPT["read_pt32_predictions"](tmp_path, identity, "a" * 64)


def test_cli_rejects_an_excluded_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rfdetr.py",
            "--model",
            "rfdetr-2xl",
            "--weights",
            str(tmp_path / "absent.pth"),
            "--output-root",
            str(tmp_path),
            "export",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        _SCRIPT["main"]()
    assert caught.value.code == 2
    assert "invalid choice: 'rfdetr-2xl'" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_large_loader_rejects_an_unpinned_checkpoint(tmp_path: Path) -> None:
    weights = tmp_path / "rf-detr-large-2026.pth"
    weights.write_bytes(b"unrelated checkpoint")
    with pytest.raises(ValueError, match="mismatched pinned checkpoint"):
        _SCRIPT["Model"].open(weights)


def test_existing_large_engine_accepts_only_empty_retired_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = _SCRIPT["Model"](tmp_path / "weights.pth")
    shape = (1, 3, 704, 704)
    settings = model.export_settings(1) | {
        "rfdetr_plus_commit": None,
        "rfdetr_plus_version": None,
    }
    export = ExportManifest(
        model="rfdetr-l",
        settings=settings,
        sources={"model": model.sha256},
        onnx_files={ONNX_FILE: "b" * 64},
        inputs={"input": TensorSpec(dtype="float32", shape=shape)},
        outputs={
            "dets": TensorSpec(dtype="float32", shape=(1, 300, 4)),
            "labels": TensorSpec(dtype="float32", shape=(1, 300, 91)),
        },
    )
    engine = EngineManifest(
        export_sha256="c" * 64,
        plan_sha256="d" * 64,
        environment=EngineEnvironment(
            tensorrt_version="10.14.1", device_name="GPU", compute_capability=(12, 0)
        ),
        profiles={"input": InputProfile(minimum=shape, optimum=shape, maximum=shape)},
    )
    monkeypatch.setattr(
        EngineBundle, "read", lambda path: EngineBundle(path, export, engine)
    )
    assert _SCRIPT["load_engine"](model, tmp_path, 1).engine.plan_sha256 == "d" * 64

    export = export.model_copy(
        update={"settings": settings | {"rfdetr_plus_version": "nonempty"}}
    )
    with pytest.raises(ValueError, match="export settings mismatch"):
        _SCRIPT["load_engine"](model, tmp_path, 1)


def test_speed_batches_use_manifest_order_and_full_image_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = tuple(f"frame_{frame}" for frame in range(95, -1, -1))
    fixed = SimpleNamespace(
        sha256=_SCRIPT["SPEED_MANIFEST_SHA256"],
        manifest={"id": "meiji_4cam/person-model-speed-v1"},
        rows=lambda index: [{"id": key} for key in ids],
        frame=lambda key: np.full((1, 1, 3), int(key.split("_")[1]), np.uint8),
    )
    speed_batches = _SCRIPT["speed_batches"]
    monkeypatch.setitem(
        speed_batches.__globals__,
        "import_module",
        lambda name: SimpleNamespace(InputSet=lambda directory: fixed),
    )
    _, batches = speed_batches(tmp_path, 2)
    assert len(batches) == 48
    assert tuple(key for batch in batches for key in batch.ids) == ids
    assert [image[0, 0, 0] for image in batches[0].data] == [95, 94]
    assert all(batch.item_count == batch.processed_items == 2 for batch in batches)

    fixed.sha256 = "0" * 64
    with pytest.raises(ValueError, match="Unexpected fixed speed manifest SHA-256"):
        speed_batches(tmp_path, 2)


def test_speed_reference_rejects_a_changed_prediction_file(tmp_path: Path) -> None:
    pins = {"input_manifest_sha256": _SCRIPT["SPEED_MANIFEST_SHA256"]}
    ids = ["frame_a"]
    output = tmp_path / "speed-b1-predictions.json"
    rows = {"frame_a": {"boxes": [[1, 2, 3, 4]], "scores": [0.5]}}
    write_json_atomic(output, {**pins, "input_ids": ids, "rows": rows})
    write_json_atomic(
        tmp_path / "speed-b1.json",
        {
            "status": "ok",
            "case": {**pins, "name": "rfdetr-l", "backend": "PT32", "batch_size": 1},
            "weights": {"model": {"sha256": "a" * 64}},
            "parity": {"metrics": {"predictions_sha256": dual_sha256(output)}},
        },
    )
    read = _SCRIPT["read_speed_reference"]
    assert read(tmp_path, ids, "a" * 64) == rows
    write_json_atomic(output, {**pins, "input_ids": ids, "rows": {}})
    with pytest.raises(ConditionUnavailable, match="checksum differs") as caught:
        read(tmp_path, ids, "a" * 64)
    assert caught.value.status is Status.PARITY_UNAVAILABLE


def test_failed_engine_conversion_is_passed_to_the_benchmark(tmp_path: Path) -> None:
    write_json_atomic(
        tmp_path / "engine_build.json",
        {
            "phase": "engine_build",
            "wall_ms": None,
            "failure": {
                "status": "oom",
                "phase": "engine_build",
                "message": "builder OOM",
            },
        },
    )
    (result,) = _SCRIPT["speed_conversion"](tmp_path)
    assert result.phase is Conversion.ENGINE_BUILD
    assert result.failure.status is Status.OOM
    assert result.failure.message == "builder OOM"
