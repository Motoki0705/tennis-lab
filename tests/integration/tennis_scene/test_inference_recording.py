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
from src.tennis_scene.pipeline.execution_plan import ClipExecution, ExecutionPlan
from src.tennis_scene.pipeline.runner import (
    ComponentNode,
    ComponentRunner,
    PlanExecutor,
)
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


def test_multi_clip_executor_records_the_selected_clip_and_camera(tmp_path: Path) -> None:
    weights = tmp_path / "weights"
    weights.write_bytes(b"fixed")
    clips = []
    for name, camera, request in (("first", "cam2", 2), ("second", "cam0", 3)):
        context = AssemblyContext(ClipSource(name, (SourceVideo(camera, tmp_path / f"{name}.mp4", "fixed", 3, 30., 2, 2),)), camera)
        component = Component(CpuModel(weights))
        node = ComponentNode("values", component, component.io, Assembler(request), {}, context, {}, "fixed-main")
        clips.append(ClipExecution(name, [node], ClipStore(tmp_path / name, {"input": name})))
    executor = PlanExecutor(ExecutionPlan(clips), select_ready=lambda ready: ready[-1])
    with InferenceRecorder(tmp_path / "multi.json"):
        executor.run()
    events = json.loads((tmp_path / "multi.json").read_text())["events"]
    nodes = [e["attributes"] for e in events if e["stage"] == "node"]
    assert [(e["clip_id"], e["camera_id"], e["outcome"]) for e in nodes] == [
        ("second", "cam0", "executed"), ("first", "cam2", "executed")]
    loads = [e["attributes"] for e in events if e["stage"] == "model_load"]
    assert [(e["clip_id"], e["camera_id"]) for e in loads] == [("second", "cam0"), ("first", "cam2")]
    np.testing.assert_array_equal(executor.output(("first", "values")).values, [[2., 14.], [10., -6.], [18., 4.]])
    np.testing.assert_array_equal(executor.output(("second", "values")).values, [[3., 21.], [15., -9.], [27., 6.]])


def test_aflink_records_actual_cpu_load_and_preserves_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from src.tasks.person_tracking import strongsort_offline as offline

    checkpoint = tmp_path / "aflink.pth"
    torch.save(offline.AFLinkNetwork().state_dict(), checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(offline, "AF_WEIGHT_SHA256", digest)
    boxes: NDArray[np.float32] = np.zeros((2, 5, 4), np.float32)
    boxes[0, :2] = [10., 20., 30., 40.]
    boxes[1, 3:] = [12., 21., 32., 41.]
    observed = np.array([[True, True, False, False, False], [False, False, False, True, True]])
    expected = offline.AFLink(checkpoint).links(boxes, observed)
    with InferenceRecorder(tmp_path / "aflink.json"):
        actual = offline.AFLink(checkpoint).links(boxes, observed)
    assert actual == expected
    record = json.loads((tmp_path / "aflink.json").read_text())
    model = record["models"][0]
    assert (model["model"], model["device"], model["load_count"], model["unload_count"]) == ("AFLink", "cpu", 1, 0)
    assert model["weights"]["checkpoint"]["sha256"] == digest
    batch = next(e for e in record["events"] if e["stage"] == "inference")
    assert batch["attributes"]["batch_shape"] == batch["attributes"]["second_input_shape"] == [1, 1, 30, 3]


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
