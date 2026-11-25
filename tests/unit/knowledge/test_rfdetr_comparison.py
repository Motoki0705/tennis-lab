"""Test the RF-DETR experiment adapter without upstream models or weights."""

from __future__ import annotations

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.tennis_scene.evaluation.player_pose.bundle import pinned_file
from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, Predictions
from src.utils.inference_benchmark import (
    Backend,
    Batch,
    BenchmarkProtocol,
    BenchmarkReport,
    Conversion,
    Status,
)
from src.utils.inference_benchmark_inputs import TimingSchedule
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

_RUNS = Path(__file__).resolve().parents[3] / "knowledge/runs"
_SCRIPT = runpy.run_path(str(_RUNS / "run-meiji-player-model-comparison/rfdetr.py"))


@pytest.mark.parametrize(
    ("index_path", "job_count"),
    [
        ("run-meiji-rfdetr-large-accuracy/results.json", 3),
        ("run-meiji-player-model-comparison/rfdetr-speed-initial.json", 4),
        ("run-meiji-rfdetr-large-speed/results.json", 4),
    ],
    ids=["accuracy", "initial-speed", "matched-speed"],
)
def test_queue_logs_can_be_read_without_queue_history(
    tmp_path: Path, index_path: str, job_count: int
) -> None:
    conditions = json.loads((_RUNS / index_path).read_bytes())["conditions"]
    rows = conditions.values() if isinstance(conditions, dict) else conditions
    jobs = [row["job"] for row in rows]
    assert len(jobs) == job_count

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    retained = tmp_path / "retained"
    retained.mkdir()
    artifact_root = checkout / ".worktree-outputs"
    artifact_root.symlink_to(retained, target_is_directory=True)
    expected_dir = Path(
        "parent-2492-issue-2487-rfdetr/player-model-comparison/queue-logs"
    )
    (retained / expected_dir).mkdir(parents=True)

    for job in jobs:
        log = job["log"]
        relative = Path(log["path"]).relative_to(".worktree-outputs")
        assert relative == expected_dir / Path(job["id"]).with_suffix(".log").name
        content = f"Completed {job['id']}\n".encode()
        (retained / relative).write_bytes(content)
        pin = {"path": str(relative), "sha256": sha256(content).hexdigest()}
        assert pinned_file(artifact_root, pin).read_bytes() == content

    assert not (checkout / ".training_queue").exists()


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


def test_prediction_jsonl_preserves_empty_frames(tmp_path: Path) -> None:
    identity = _SCRIPT["header"]("meiji/cohort", "b" * 64, Backend.PT16)
    key = FrameKey("video_000/clip_003", "cam0", 120)
    raw = tmp_path / "predictions.jsonl"
    _SCRIPT["write_predictions"](
        raw, identity, {key: Predictions(np.empty((0, 4)), np.empty(0))}
    )
    saved_identity, row = map(json.loads, raw.read_text().splitlines())
    assert saved_identity == identity
    assert row == {
        "clip_id": "video_000/clip_003",
        "camera_id": "cam0",
        "source_frame": 120,
        "boxes_xyxy": [],
        "scores": [],
        "keypoints_xyc": None,
    }


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


def test_speed_batches_use_pinned_timing_order_and_full_image_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ids = tuple(f"frame_{frame}" for frame in range(95, -1, -1))
    protocol = BenchmarkProtocol(batch_sizes=(1, 2))
    schedule = TimingSchedule(
        protocol, tuple(zip(ids[::2], ids[1::2], strict=True)), "e" * 64
    )

    def require_schedule(index: str, batch_size: int) -> TimingSchedule:
        assert (index, batch_size) == ("frames", 2)
        return schedule

    fixed = SimpleNamespace(
        pins=SimpleNamespace(schedule=require_schedule),
        frame=lambda key: np.full((1, 1, 3), int(key.split("_")[1]), np.uint8),
    )
    speed_batches = _SCRIPT["speed_batches"]
    monkeypatch.setitem(
        speed_batches.__globals__,
        "import_module",
        lambda name: SimpleNamespace(InputSet=lambda directory: fixed),
    )
    pins, actual_schedule, batches = speed_batches(tmp_path, 2)
    assert pins is fixed.pins
    assert actual_schedule is schedule
    assert len(batches) == 48
    assert tuple(key for batch in batches for key in batch.ids) == ids
    assert [image[0, 0, 0] for image in batches[0].data] == [95, 94]
    assert all(batch.item_count == batch.processed_items == 2 for batch in batches)


def test_speed_cli_passes_pinned_schedule_and_explicit_parity_policy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main = _SCRIPT["main"]
    model = _SCRIPT["Model"]
    protocol = BenchmarkProtocol(batch_sizes=(1, 2), warmup=2)
    schedule = TimingSchedule(protocol, (("image",),), "c" * 64)
    pins = SimpleNamespace(
        manifest={"id": "meiji_4cam/person-model-speed-v1"},
        manifest_sha256="a" * 64,
        timing_sha256="b" * 64,
    )
    batches = [Batch(("image",), (np.zeros((1, 1, 3), np.uint8),))]
    reason = "PT32計測はスコープ外（ユーザー判断 2026-11-25）"
    threads = {"torch": 2, "opencv": 12}

    def inspect_benchmark(case, inputs, adapters, **options):
        assert case.backend is Backend.PT16
        assert inputs is batches
        assert tuple(adapters) == (Backend.PT16,)
        assert case.settings["host_threads"] == threads["torch"] == 4
        assert case.settings["opencv_threads"] == threads["opencv"] == 1
        assert options == {
            "device": torch.device("cuda:0"),
            "dedicated_stream": True,
            "protocol": protocol,
            "conversion": (),
            "parity_not_evaluated_reason": reason,
        }
        return BenchmarkReport(case, protocol, (), parity_not_evaluated_reason=reason)

    monkeypatch.delenv("TENNIS_REPRO_DIR", raising=False)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: threads.update(torch=n))
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads["torch"])
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: threads.update(opencv=n))
    monkeypatch.setattr(cv2, "getNumThreads", lambda: threads["opencv"])
    monkeypatch.setattr(model, "open", lambda path, _: model(path))
    monkeypatch.setitem(main.__globals__, "benchmark", inspect_benchmark)
    monkeypatch.setitem(
        main.__globals__, "speed_batches", lambda *_: (pins, schedule, batches)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rfdetr.py",
            "--weights",
            "weights.pth",
            "--output-root",
            str(tmp_path),
            "speed",
            "--backend",
            "pt16",
            "--parity",
            "not_measured",
            "--batch-size",
            "1",
            "--speed-inputs",
            "inputs",
        ],
    )
    assert main() == 0
    path = tmp_path / "rfdetr-l/pt16/speed-b1/report.json"
    content = path.read_bytes()
    result = json.loads(content)
    assert result["status"] == "ok"
    assert result["parity_status"] == "not_evaluated"
    assert result["parity_not_evaluated_reason"] == reason
    assert result["case"]["input_manifest_sha256"] == "a" * 64
    assert result["case"]["settings"]["timing_protocol_sha256"] == "b" * 64
    assert result["case"]["settings"]["timed_input_ids_sha256"] == "c" * 64
    with pytest.raises(FileExistsError):
        main()
    assert path.read_bytes() == content


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--backend", "pt32", "--parity", "not_measured"], "invalid choice: 'pt32'"),
        (["--backend", "pt16"], "the following arguments are required: --parity"),
    ],
)
def test_speed_cli_rejects_pt32_and_requires_explicit_parity(
    args: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rfdetr.py",
            "--weights",
            "weights.pth",
            "--output-root",
            str(tmp_path),
            "speed",
            "--batch-size",
            "1",
            "--speed-inputs",
            "inputs",
            *args,
        ],
    )
    with pytest.raises(SystemExit) as caught:
        _SCRIPT["main"]()
    assert caught.value.code == 2
    assert message in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


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
