"""Exercise the Sapiens2 adapter boundary with CPU-only pose inputs."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, TypeVar

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from safetensors.torch import save_file

from src.utils.inference_benchmark import Backend, ConditionUnavailable, Stage, Status
from src.utils.models.tensorrt_artifacts import InputProfile
from src.utils.schema.player import COCO_KP_NAMES

Value = TypeVar("Value")


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).resolve().parents[5]
    monkeypatch.syspath_prepend(
        str(root / "knowledge/runs/run-meiji-topdown-pose-comparison")
    )
    return importlib.import_module("sapiens2")


def test_sapiens2_coarse_face_and_opposite_wrists_follow_official_mapping(
    adapter: ModuleType,
) -> None:
    channels = (0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14)
    mirrored = (0, 2, 1, 4, 3, 6, 5, 8, 7, 41, 62, 10, 9, 12, 11, 14, 13)
    flips = list(range(308))
    for channel, other in zip(channels, mirrored, strict=True):
        flips[channel] = other
    names = dict(zip(channels, COCO_KP_NAMES, strict=True))
    meta = {
        "num_keypoints": 308,
        "coco_wholebody_to_goliath_mapping": dict(enumerate(channels)),
        "keypoint_id2name": names,
        "flip_indices": flips,
    }

    assert adapter._coco_flip_channels(meta) == mirrored
    names[0] = "tip_of_nose"  # A separate, dense face landmark.
    with pytest.raises(ValueError, match="every COCO17 joint"):
        adapter._coco_flip_channels(meta)
    names[0] = "nose"
    flips[62] = 62
    with pytest.raises(ValueError, match="laterality"):
        adapter._coco_flip_channels(meta)


class RecordingStages:
    def __init__(self) -> None:
        self.forwards: list[tuple[Backend, str]] = []

    def stage(
        self, stage: Stage, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        assert stage is not Stage.FORWARD
        return operation()

    def measure_forward(
        self, backend: Backend, operation: Callable[[], Value], *, name: str = ""
    ) -> Value:
        self.forwards.append((backend, name))
        return operation()


def test_sapiens2_fixed_boxes_keep_row_order_and_restore_pixels_with_flip(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = importlib.import_module("pose_contract")
    monkeypatch.setattr(adapter, "HEATMAP_CHW", (308, 3, 5))
    image: NDArray[np.uint8] = np.zeros((240, 320, 3), dtype=np.uint8)
    first = np.array([20, 40, 60, 120], dtype=np.float32)
    second = np.array([65, 10, 145, 42], dtype=np.float32)

    def official_pipeline(data: dict[str, Any]) -> dict[str, Any]:
        assert data["img"] is image
        scale: tuple[float, float]
        if data["bbox"][0, 0] == 20:
            # A tall box with padding gives the native crop [2.5, 30, 77.5, 130].
            center, scale = (40, 80), (75, 100)
        else:
            # A wide box gives native crop x=[55,155] and height=100/0.75.
            center, scale = (105, 26), (100, 400 / 3)
        marker = float(data["bbox"][0, 0])
        data["bbox"][:] = -1
        return {
            "inputs": torch.tensor([[[marker, 1, 2]]]),
            "data_samples": {
                "meta": {
                    "bbox_center": np.array([center]),
                    "bbox_scale": np.array([scale]),
                    "input_size": (768, 1024),
                },
            },
        }

    def decode(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert np.isfinite(maps).all() and (maps.max(axis=(1, 2)) > 0).all()
        y, x = np.unravel_index(maps.reshape(len(maps), -1).argmax(axis=1), (3, 5))
        return np.stack((x * 192, y * 512), axis=-1)[None], maps.max(axis=(1, 2))[None]

    def run(inputs: torch.Tensor) -> torch.Tensor:
        is_flipped = inputs[0, 0, 0, 0] == 2
        maps = torch.full((len(inputs), 308, 3, 5), -1.0)
        for index, pixels in enumerate(inputs):
            marker = pixels[0, 0, -1 if is_flipped else 0].item()
            offset = 0 if marker == 20 else 1
            if is_flipped:
                maps[index, 41, 1, 3 - offset] = 10
                maps[index, 62, 2, 1 - offset] = 4
            else:
                maps[index, 62, 1, 1 + offset] = 6
                maps[index, 41, 2, 3 + offset] = 2
            maps[index, 0, 0, 2] = 0
            maps[index, 5, 0, 0] = float("nan")
            maps[index, 6, 0, 0] = float("inf")
        return maps

    flipped_channels = (0, 2, 1, 4, 3, 6, 5, 8, 7, 41, 62, 10, 9, 12, 11, 14, 13)
    official = adapter.OfficialPose(
        torch.nn.Identity,
        {},
        official_pipeline,
        torch.nn.Identity(),
        decode,
        flipped_channels,
    )
    rows = (contract.PoseInput(image, first), contract.PoseInput(image, second))
    stages = RecordingStages()

    def infer(batch: tuple[Any, ...]) -> NDArray[np.float32]:
        result: NDArray[np.float32] = adapter._run_batch(
            batch,
            stages,
            official,
            run,
            Backend.PT32,
            torch.device("cpu"),
            torch.float32,
            torch.tensor(adapter.COCO17_CHANNELS),
            torch.tensor(flipped_channels),
        ).keypoints_xyc
        return result

    result = infer(rows)
    assert result.shape == (2, 17, 3) and result.dtype == np.float32
    np.testing.assert_allclose(result[:, 9], [[21.25, 80, 8], [105, 26, 8]])
    np.testing.assert_allclose(result[:, 10], [[58.75, 130, 3], [155, 26 + 200 / 3, 3]])
    assert np.isnan(result[:, 0:2, :2]).all()
    np.testing.assert_array_equal(result[:, 0:2, 2], [[0, -1], [0, -1]])
    assert np.isnan(result[:, 5:7]).all()
    np.testing.assert_array_equal(infer(rows[1:]), result[1:])
    assert (
        stages.forwards == [(Backend.PT32, "original"), (Backend.PT32, "flipped")] * 2
    )
    np.testing.assert_array_equal(first, [20, 40, 60, 120])
    np.testing.assert_array_equal(second, [65, 10, 145, 42])
    assert not image.any()


def test_sapiens2_checkpoint_identity_and_strict_load_keep_fp32_rope(
    adapter: ModuleType, tmp_path: Path
) -> None:
    class SmallPose(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 1, bias=False)
            self.backbone = torch.nn.Module()
            self.backbone.rope_embed = torch.nn.Module()
            self.backbone.rope_embed.register_buffer("periods", torch.ones(1))

    path = tmp_path / "weights.safetensors"
    state = {
        "linear.weight": torch.tensor([[3.0, 7.0]]),
        "backbone.rope_embed.periods": torch.tensor([1.234567]),
    }
    save_file(state, path)
    blob = path.read_bytes()
    spec = adapter.Checkpoint("1b", "revision", sha256(blob).hexdigest(), len(blob), 3)
    adapter.verify_checkpoint(spec, path)
    with pytest.raises(ConditionUnavailable, match="pinned SHA-256") as error:
        adapter.verify_checkpoint(replace(spec, sha256="0" * 64), path)
    assert error.value.status is Status.UNAVAILABLE

    official = adapter.OfficialPose(
        lambda config: SmallPose(), {}, None, None, None, ()
    )
    model = adapter.load_model(official, path, torch.device("cpu"), torch.float16)
    torch.testing.assert_close(
        model.linear.weight, torch.tensor([[3, 7]], dtype=torch.float16)
    )
    torch.testing.assert_close(
        model.backbone.rope_embed.periods,
        state["backbone.rope_embed.periods"],
        atol=0,
        rtol=0,
    )
    assert not model.training and not model.linear.weight.requires_grad
    save_file({"linear.weight": state["linear.weight"]}, path)
    with pytest.raises(
        ConditionUnavailable, match="Missing key.*rope_embed.periods"
    ) as error:
        adapter.load_model(official, path, torch.device("cpu"), torch.float16)
    assert error.value.status is Status.UNAVAILABLE


def test_trt_without_engine_cannot_enter_pytorch(
    adapter: ModuleType, tmp_path: Path
) -> None:
    instance = adapter.make_adapter(
        model_id="sapiens2-5b",
        backend=Backend.TRT16,
        checkpoint=tmp_path / "pose.safetensors",
        device="cuda:0",
    )
    with pytest.raises(ConditionUnavailable, match="TensorRT bundle") as error:
        instance.load()
    assert error.value.status is Status.UNAVAILABLE


def test_sapiens2_trt_disables_the_builds_detailed_nvtx_at_inference(
    adapter: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context = SimpleNamespace(nvtx_verbosity="DETAILED")
    shape = (1, 3, 2, 2)
    profile = InputProfile(minimum=shape, optimum=shape, maximum=shape)
    bound = SimpleNamespace(engine=SimpleNamespace(profiles={"pixels": profile}))
    artifact = SimpleNamespace(for_model=lambda **_: bound)

    class Runner:
        _context = context

        def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            assert self._context.nvtx_verbosity == "NONE"
            return {"heatmaps": inputs["pixels"] + 2}

    monkeypatch.setattr(
        adapter, "EngineBundle", SimpleNamespace(read=lambda _: artifact)
    )
    monkeypatch.setattr(adapter, "TensorRTRunner", lambda *_: Runner())
    monkeypatch.setattr(
        adapter,
        "tensorrt_module",
        lambda: SimpleNamespace(ProfilingVerbosity=SimpleNamespace(NONE="NONE")),
    )
    forward = adapter._engine_forward(
        "sapiens2-1b",
        adapter.CHECKPOINTS["sapiens2-1b"],
        tmp_path,
        torch.device("cuda:0"),
    )
    torch.testing.assert_close(forward(torch.zeros(shape)), torch.full(shape, 2.0))
