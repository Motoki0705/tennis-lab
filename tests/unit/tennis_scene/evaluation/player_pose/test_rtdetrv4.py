"""CPU fixtures for the RT-DETRv4 comparison adapter's image and box contracts."""

import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from torch import nn
from torchvision.ops import box_convert

from src.tennis_scene.evaluation.player_pose.scoring import FrameKey, ReferenceFrame

RUN = (
    Path(__file__).resolve().parents[5]
    / "knowledge/runs/run-meiji-player-model-comparison"
)
PARITY_REASON = "PT32計測はスコープ外（ユーザー判断 2026-11-25）"


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(str(RUN))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return import_module("rtdetrv4")


def test_loads_release_serialization_and_selects_ema_weights(adapter, tmp_path):
    class Deployable(nn.Linear):
        def deploy(self):
            return self

    path = tmp_path / "official.pth"
    torch.save(
        {
            "ema": {"module": {"weight": torch.tensor([[1.0, -2.0]])}},
            "model": {"weight": torch.tensor([[3.0, 4.0]])},
        },
        path,
        _use_new_zipfile_serialization=False,
    )
    assets = adapter.Assets("rtdetrv4-l", tmp_path, path)
    config = SimpleNamespace(model=Deployable(2, 1, bias=False))

    loaded = adapter._load_pt_model(assets, config)

    assert loaded.model(torch.tensor([[5.0, 3.0]])).item() == -1.0


def test_cached_positional_embeddings_move_and_do_not_enter_state_dict(adapter):
    encoder = nn.Module()
    encoder.use_encoder_idx = (1, 3)
    first, second = torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, -8.0, 2.0]])
    encoder.pos_embed1 = first
    encoder.pos_embed3 = second

    adapter._register_encoder_positions(encoder)

    assert encoder.pos_embed1 is first
    assert encoder.pos_embed3 is second
    assert not encoder.state_dict()
    encoder.to("meta")
    assert encoder.pos_embed1.device.type == "meta"
    assert encoder.pos_embed3.device.type == "meta"


def test_prepare_preserves_wh_and_uses_rgb_unit_range(adapter):
    frame: NDArray[np.uint8] = np.empty((2, 5, 3), np.uint8)
    frame[:] = (9, 55, 241)

    pixels, sizes = adapter.preprocess([frame, np.zeros((4, 1, 3), np.uint8)])

    assert pixels.shape == (2, 3, 640, 640)
    assert sizes.tolist() == [[5, 2], [1, 4]]
    torch.testing.assert_close(pixels[0, :, 80, 420], torch.tensor([241, 55, 9]) / 255)
    assert torch.count_nonzero(pixels[1]) == 0


def test_only_person_rows_survive_with_clipped_native_xyxy_and_matching_scores(adapter):
    labels = np.array([[4, 0, 0, 0], [1, 2, 3, 4]])
    boxes = np.array(
        [
            [[0, 0, 100, 40], [-3, 8, 120, 45], [20, 3, 30, 15], [0, 43, 90, 58]],
            [[1, 1, 4, 6]] * 4,
        ],
        np.float32,
    )
    scores = np.array([[0.98, 0.81, 0.20, 0.19], [0.99, 0.97, 0.91, 0.87]], np.float32)

    first, second = adapter.person_predictions(
        labels, boxes, scores, np.array([[100, 40], [11, 50]])
    )

    np.testing.assert_array_equal(first.boxes_xyxy, [[0, 8, 100, 40], [20, 3, 30, 15]])
    np.testing.assert_array_equal(first.scores, np.array([0.81, 0.20], np.float32))
    assert first.keypoints_xyc is second.keypoints_xyc is None
    assert second.boxes_xyxy.shape == (0, 4)
    assert second.scores.shape == (0,)


def test_nonfinite_coordinates_are_rejected_before_clipping(adapter):
    with pytest.raises(ValueError, match="Non-finite RT-DETRv4"):
        adapter.person_predictions(
            np.array([[0]]),
            np.array([[[0, 0, np.inf, 5]]]),
            np.array([[0.99]]),
            np.array([[10, 10]]),
        )


def test_detector_wires_native_sizes_to_postprocessing_and_declares_forward(adapter):
    calls = []

    class Stages:
        def stage(self, stage, operation):
            calls.append(stage.value)
            return operation()

        def measure_forward(self, backend, operation):
            calls.append(backend.value)
            return operation()

    class Postprocess(nn.Module):
        def forward(self, output, sizes):
            boxes = box_convert(output["pred_boxes"], in_fmt="cxcywh", out_fmt="xyxy")
            return (
                torch.zeros((2, 1), dtype=torch.long),
                boxes * sizes.repeat(1, 2)[:, None],
                output["pred_logits"].sigmoid(),
            )

    def model(pixels):
        assert pixels.shape == (2, 3, 640, 640)
        return torch.tensor([[0.0], [2.0]]), torch.tensor([[[0.5, 0.5, 0.5, 0.5]]] * 2)

    detector = adapter.Detector(
        adapter.Backend.PT32, torch.device("cpu"), model, Postprocess()
    )
    first, second = detector(
        [np.zeros((40, 20, 3), np.uint8), np.zeros((12, 100, 3), np.uint8)], Stages()
    )

    np.testing.assert_array_equal(first.boxes_xyxy, [[5, 10, 15, 30]])
    np.testing.assert_array_equal(second.boxes_xyxy, [[25, 3, 75, 9]])
    np.testing.assert_allclose([first.scores[0], second.scores[0]], [0.5, 0.880797])
    assert calls == ["preprocess", "h2d", "PT32", "postprocess", "d2h", "postprocess"]


@pytest.mark.parametrize(
    ("mode", "inputs"),
    [
        (
            "accuracy",
            ["--benchmark", "cohort", "--manifest-sha256", "pin", "--source", "media"],
        ),
        (
            "speed",
            ["--inputs", "frames", "--inputs-sha256", "pin", "--batch-size", "2"],
        ),
    ],
)
def test_measurement_rejects_pt32_and_requires_explicit_parity_status(
    adapter, mode, inputs, capsys
):
    parser = adapter._parser()
    args = [
        "--model-id",
        "rtdetrv4-l",
        "--upstream",
        "checkout",
        "--weights",
        "model.pth",
        mode,
        "--output",
        "result",
        "--device",
        "cuda:0",
        *inputs,
    ]
    parity = ["--parity-not-evaluated-reason", PARITY_REASON]

    valid = parser.parse_args([*args, "--backend", "pt16", *parity])
    assert valid.backend is adapter.Backend.PT16
    assert valid.parity_not_evaluated_reason == PARITY_REASON

    with pytest.raises(SystemExit):
        parser.parse_args([*args, "--backend", "pt32", *parity])
    assert "invalid choice" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        parser.parse_args([*args, "--backend", "trt16"])
    assert "--parity-not-evaluated-reason" in capsys.readouterr().err


def test_queue_suite_runs_only_pt16_and_trt16(tmp_path, monkeypatch):
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with Path('invocations.jsonl').open('a') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if 'export' in args or 'build' in args:\n"
        "    engine = Path(args[args.index('--engine-dir') + 1])\n"
        "    engine.mkdir(parents=True, exist_ok=True)\n"
        "    (engine / ('export.json' if 'export' in args else 'engine.json')).touch()\n"
    )
    interpreter.chmod(0o700)
    repro = tmp_path / "repro"
    repro.mkdir()
    monkeypatch.setenv("TENNIS_RUN_ID", "test-queue-job")
    monkeypatch.setenv("TENNIS_REPRO_DIR", str(repro))

    command = [
        "bash",
        str(RUN / "rtdetrv4_run.sh"),
        "rtdetrv4-l",
        "speed",
        "speed-pin",
        "eval-pin",
        "run-output",
    ]
    subprocess.run(command, cwd=tmp_path, check=True)

    invocations = [
        json.loads(line)
        for line in (tmp_path / "invocations.jsonl").read_text().splitlines()
    ]
    assert [args[9] for args in invocations] == [
        "accuracy",
        "speed",
        "speed",
        "export",
        "build",
        "accuracy",
        "speed",
        "speed",
    ]
    measures = [args for args in invocations if "--backend" in args]
    assert [args[args.index("--backend") + 1] for args in measures] == ["pt16"] * 3 + [
        "trt16"
    ] * 3
    assert all("--pt32-dir" not in args for args in invocations)
    assert all(
        args[args.index("--parity-not-evaluated-reason") + 1] == PARITY_REASON
        for args in measures
    )
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, cwd=tmp_path, check=True)
    assert len((tmp_path / "invocations.jsonl").read_text().splitlines()) == len(
        invocations
    )


def test_accuracy_receipt_records_unmeasured_parity(adapter, tmp_path, monkeypatch):
    key = FrameKey("clip", "cam0", 6)
    boxes = np.array([[1.0, 2.0, 3.0, 6.0]])
    manifest = {
        "benchmark_id": "toy",
        "source_benchmark": {"manifest_sha256": "source"},
        "clips": [
            {
                "clip_id": key.clip_id,
                "width": 4,
                "height": 7,
                "source_videos": [{"camera_id": key.camera_id}],
                "evaluation_frames": [key.source_frame],
            }
        ],
    }
    references = {
        key: ReferenceFrame(
            np.array([2]),
            boxes,
            np.zeros((1, 17, 2)),
            np.zeros((1, 17), dtype=bool),
        )
    }
    monkeypatch.setattr(adapter, "load_manifest", lambda *_args, **_kwargs: manifest)
    monkeypatch.setattr(adapter, "pinned_file", lambda *_args: tmp_path / "video")
    monkeypatch.setattr(
        adapter, "load_reference_frames", lambda *_args, **_kwargs: references
    )
    monkeypatch.setattr(
        adapter,
        "load_detector",
        lambda *_args: lambda _rows: [adapter.Predictions(boxes, np.array([0.9]))],
    )
    monkeypatch.setattr(
        adapter.torch.cuda, "get_device_name", lambda _device: "fixture"
    )
    monkeypatch.setenv("TENNIS_RUN_ID", "fixture-job")

    class Reader:
        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            pass

        def read(self, frame):
            assert frame == key.source_frame
            return np.zeros((7, 4, 3), np.uint8)

    monkeypatch.setattr(adapter, "RandomAccessVideoReader", Reader)
    args = SimpleNamespace(
        model_id="rtdetrv4-l",
        backend=adapter.Backend.PT16,
        engine_dir=None,
        benchmark=tmp_path,
        source=tmp_path,
        manifest_sha256="a" * 64,
        output=tmp_path,
        parity_not_evaluated_reason=PARITY_REASON,
    )

    adapter._accuracy(
        args,
        adapter.Assets(args.model_id, tmp_path, tmp_path / "weight"),
        torch.device("cuda:0"),
    )

    receipt = json.loads((tmp_path / "accuracy.json").read_text())
    assert receipt["condition"]["backend"] == "pt16"
    assert receipt["score"]["overall"]["detection"]["recall_by_iou"] == {
        "0.50": 1.0,
        "0.75": 1.0,
    }
    assert receipt["parity_status"] == "not_evaluated"
    assert receipt["parity_not_evaluated_reason"] == PARITY_REASON


def test_diagnostic_keeps_prepared_image_order_in_both_batch_sizes(adapter):
    module = import_module("rtdetrv4_diagnose")
    keys = (FrameKey("clip-a", "cam0", 2), FrameKey("clip-b", "cam1", 8))
    pixels = torch.tensor([[2.0], [6.0]])
    sizes = torch.tensor([[12, 20], [24, 16]])
    references = {
        key: ReferenceFrame(
            np.array([4]),
            np.array([[x, 1, x + 2, 4]], dtype=np.float32),
            np.zeros((1, 17, 2)),
            np.zeros((1, 17), dtype=bool),
        )
        for key, x in zip(keys, (2.0, 6.0), strict=True)
    }

    def forward(prepared):
        x = prepared[:, 0]
        boxes = torch.stack(
            (x, torch.ones_like(x), x + 2, torch.full_like(x, 4)), dim=-1
        )
        return x[:, None], boxes[:, None, :]

    def decode(output, native_sizes):
        logits, boxes = output
        assert len(native_sizes) == len(logits)
        return torch.zeros_like(logits, dtype=torch.long), boxes, logits.sigmoid()

    for batch_size in (1, 2):
        status, raw = module._compare_batches(
            forward,
            SimpleNamespace(decode=decode),
            pixels,
            sizes,
            keys,
            references,
            batch_size,
        )
        np.testing.assert_array_equal(raw["pred_logits"], [[2.0], [6.0]])
        assert status["pseudo_gt"] == {"0.50": 1.0, "0.75": 1.0}
