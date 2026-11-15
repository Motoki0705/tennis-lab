"""Real runner/store + a CPU model: recording is strictly provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from src.submodules.models import BaseInferenceModel
from src.tasks.ball_detection.inference.predictor import BallDetectionPredictor
from src.tasks.ball_detection.model_io.adapters import BallModelIOAdapter
from src.tasks.ball_detection.model_io.contracts import BallModelInputSpec
from src.tasks.base.model_io import bind_model_io
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    SourceVideo,
)
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.utils.inference_recording import InferenceRecorder, span


@dataclass(frozen=True)
class Output:
    values: NDArray[np.float32]


class CpuModel(BaseInferenceModel[int, Output]):
    def __init__(self, weights: Path):
        super().__init__("cpu")
        self.weights = weights
        self.loads = 0

    def recording_assets(self) -> dict[str, Path]:
        return {"checkpoint": self.weights}

    def _load_impl(self) -> None:
        self.loads += 1

    def _unload_impl(self) -> None:
        pass

    def _predict_impl(self, request: int) -> Output:
        with span("inference", batch_shape=[3, 2]):
            if request < 0:
                raise ValueError("injected inference failure")
            return Output(np.array([[1., 7.], [5., -3.], [9., 2.]], np.float32) * request)


class Component:
    io = ComponentIO("values", int, Output, {}, "values")

    def __init__(self, model: CpuModel):
        self.model = model

    def process(self, inputs: int) -> Output:
        try:
            return cast(Output, self.model.predict(inputs))
        finally:
            self.model.unload()


@dataclass(frozen=True)
class Assembler:
    request: int = 2
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: object) -> int:
        return self.request


def test_recording_preserves_artifact_bytes_execution_key_and_cache(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixed CPU model weights")
    model = CpuModel(weights)
    component = Component(model)
    context = AssemblyContext(ClipSource("video/clip", (SourceVideo("cam0", tmp_path / "cam0.mp4", "fixed", 3, 30., 2, 2),)), "cam0")
    node = ComponentNode("values/cam0", component, component.io, Assembler(), {}, context, {}, "fixed-main")
    plain = ComponentRunner([node], ClipStore(tmp_path / "plain", {"input": "fixed"}))
    plain.run()
    with InferenceRecorder(tmp_path / "record.json"):
        recorded = ComponentRunner([node], ClipStore(tmp_path / "recorded", {"input": "fixed"}))
        recorded.run()
    reference = plain.references["values/cam0"]
    assert recorded.references["values/cam0"] == reference
    for path in (plain.store.root / reference.path).parent.iterdir():
        assert path.read_bytes() == (recorded.store.root / reference.path).parent.joinpath(path.name).read_bytes()
    np.testing.assert_array_equal(recorded.output("values/cam0").values, [[2., 14.], [10., -6.], [18., 4.]])
    assert model.loads == 2
    for mode in ("execute", "load"):
        with InferenceRecorder(tmp_path / f"{mode}.json"):
            cached = ComponentRunner([replace(node, source=mode)], ClipStore(recorded.store.root, {"input": "fixed"}))
            cached.run()
        record = json.loads((tmp_path / f"{mode}.json").read_text())
        assert record["models"] == []
        assert next(e for e in record["events"] if e["stage"] == "node")["attributes"]["outcome"] == ("cached" if mode == "execute" else "loaded")
        assert cached.references == recorded.references
    assert model.loads == 2


def test_runner_failure_records_unload_and_does_not_publish(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixed")
    component = Component(CpuModel(weights))
    context = AssemblyContext(ClipSource("clip", (SourceVideo("cam0", tmp_path / "v.mp4", "fixed", 1, 30., 2, 2),)))
    node = ComponentNode("values", component, component.io, Assembler(-1), {}, context, {}, "fixed-main")
    runner = ComponentRunner([node], ClipStore(tmp_path / "store", {"input": "fixed"}))
    with pytest.raises(ValueError, match="injected"), InferenceRecorder(tmp_path / "failed.json"):
        runner.run()
    record = json.loads((tmp_path / "failed.json").read_text())
    assert record["status"] == "failed"
    assert record["models"][0]["load_count"] == record["models"][0]["unload_count"] == 1
    assert runner.statuses == {"values": "failed"} and runner.store.references() == {}
    assert next(e for e in record["events"] if e["stage"] == "node")["status"] == "error"


def test_ball_records_the_mdd_forward_shape_instead_of_rgb_shape(tmp_path: Path) -> None:
    class MddModel(torch.nn.Module):
        in_channels = 2
        num_classes = 1

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            assert images.shape == (2, 2, 5, 6, 8)
            return torch.zeros_like(images[:, :1])

    adapter = BallModelIOAdapter(
        BallModelInputSpec(model_name="recording_test", input_mode="mdd", input_layout="bcthw",
                           in_channels=2, num_classes=1, configured_frames=5, image_size_hw=None,
                           minimum_spatial_size=None, mdd_gain=1., mdd_offset=0.),
        expected_model_type=MddModel, minimum_frames=1,
    )
    predictor = BallDetectionPredictor(bind_model_io(MddModel(), adapter), torch.device("cpu"), subpixel_refine=False)
    rgb = torch.linspace(0., 1., 2 * 5 * 3 * 6 * 8).reshape(2, 5, 3, 6, 8)
    with InferenceRecorder(tmp_path / "ball.json") as recorder:
        prediction = predictor.predict(rgb)
    forward = next(e for e in recorder.events if e["stage"] == "inference")
    assert forward["attributes"]["batch_shape"] == [2, 2, 5, 6, 8]
    assert forward["attributes"]["batch_items"] == 2
    assert prediction.coords.shape == (2, 5, 2)
