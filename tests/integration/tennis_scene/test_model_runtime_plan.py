"""Real plan/store execution with small fake models and no video or GPU I/O."""

from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
    SourceVideo,
)
from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ComponentNode,
    ExecutionPlan,
)
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    ModelBorrower,
    ModelSpec,
)
from src.tennis_scene.pipeline.runner import ComponentRunner, PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import ClipStore


@dataclass(frozen=True)
class Reading:
    camera: str
    values: NDArray[np.int64]


@dataclass(frozen=True)
class Inputs:
    camera: str
    upstream: tuple[Reading, ...]


class Assembler:
    version = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Reading]) -> Inputs:
        return Inputs(context.camera_id or "global", tuple(artifacts.values()))


class Model:
    def __init__(self, name: str, value: int, events: list[tuple[str, str]]) -> None:
        self.name, self.value, self.events = name, value, events
        self.loaded = True

    def unload(self) -> None:
        self.events.append(("release", self.name))
        self.loaded = False


class Component(InferenceComponent[Inputs, Reading]):
    def __init__(self, name: str, models: Mapping[str, ModelSpec[Model]], dependencies: Mapping[str, str],
                 events: list[tuple[str, str]], failure: type[BaseException] | None = None) -> None:
        self.name, self.models, self.events, self.failure = name, models, events, failure
        self.io = ComponentIO(name.split("/")[0], Inputs, Reading,
            {port: InputPort("reading") for port in dependencies}, "reading")

    def model_specs(self) -> Mapping[str, ModelSpec[Model]]:
        return self.models

    def process_with_models(self, inputs: Inputs, models: ModelBorrower) -> Reading:
        value = int(inputs.camera[-1]) if inputs.camera != "global" else 0
        for alias in self.models:
            borrowed: Model = models.borrow(alias)
            assert borrowed.loaded
            value += borrowed.value
        self.events.append(("node", self.name))
        if self.failure is not None:
            raise self.failure("fake node stopped")
        value += sum(int(reading.values[0]) for reading in inputs.upstream)
        return Reading(inputs.camera, np.array([value], np.int64))


class Harness:
    def __init__(self, root: Path, cameras: int) -> None:
        self.events: list[tuple[str, str]] = []
        self.references: list[weakref.ReferenceType[Model]] = []
        self.lifetimes: list[tuple[str, str, str]] = []
        self.specs = {}
        for value, name in enumerate(("court", "dino", "ball", "vitpose", "clip", "hmr2", "gvhmr", "body"), start=1):
            path = root / f"{name}.weights"
            path.write_bytes(name.encode())
            self.specs[name] = ModelSpec(name, {"checkpoint": path}, name, "cpu", "float32", {},
                self.factory(name, value), release=lambda model: model.unload())
        self.store = ClipStore(root / "store", {"clip_id": "clip"}, memory_entries=0)
        videos = tuple(SourceVideo(f"cam{i}", root / f"cam{i}.mp4", f"media{i}", 2, 30., 64, 36) for i in range(cameras))
        self.source = ClipSource("clip", videos)
        self.nodes: list[ComponentNode] = []
        # Interleave cameras deliberately. The runtime, not declaration order,
        # must group each model's peers before the next model family loads.
        for video in videos:
            cam = video.camera_id
            self.add(f"court/{cam}", ("court",), {}, cam)
            self.add(f"dino/{cam}", ("dino",), {}, cam)
            self.add(f"ball/{cam}", ("ball",), {}, cam)
            self.add(f"tracking/{cam}", ("vitpose", "clip"), {"detections": f"dino/{cam}"}, cam)
        self.add("association", ("clip",), {f"tracking_{v.camera_id}": f"tracking/{v.camera_id}" for v in videos})
        self.add("mesh", ("hmr2", "gvhmr"), {"identities": "association"})
        self.add("placement", ("body",), {"mesh": "mesh"})

    def factory(self, name: str, value: int) -> Any:
        def create() -> Model:
            self.events.append(("create", name))
            model = Model(name, value, self.events)
            self.references.append(weakref.ref(model))
            return model
        return create

    def add(self, name: str, aliases: tuple[str, ...], bindings: dict[str, str], camera: str | None = None) -> None:
        component = Component(name, {alias: self.specs[alias] for alias in aliases}, bindings, self.events)
        self.nodes.append(ComponentNode(name, component, component.io, Assembler(), bindings,
            AssemblyContext(self.source, camera), {}, "fake-v1"))

    @contextmanager
    def model_event(self, action: Any, model: str, **attributes: Any) -> Iterator[None]:
        assert attributes["dtype"] == "float32" and attributes["device"] == "cpu"
        self.lifetimes.append((action, model, attributes["instance"]))
        yield

    def plan(self, *, source: str = "execute", overwrite: bool = False) -> ExecutionPlan:
        nodes = tuple(replace(node, source=source) for node in self.nodes)
        return ExecutionPlan([ClipExecution("clip", nodes, self.store, overwrite)])


@pytest.mark.parametrize("cameras", [1, 5])
def test_camera_count_does_not_increase_loads_or_overlap_all_models(tmp_path: Path, cameras: int) -> None:
    harness = Harness(tmp_path, cameras)
    plan = harness.plan()
    executor = PlanExecutor(plan, model_event=harness.model_event)
    executor.run()
    assert Counter(name for action, name, _ in harness.lifetimes if action == "load") == dict.fromkeys(harness.specs, 1)
    assert Counter((name, instance) for action, name, instance in harness.lifetimes if action == "load") == Counter(
        (name, instance) for action, name, instance in harness.lifetimes if action == "unload")
    live, peak = 0, 0
    for action, name in harness.events:
        live += int(action == "create") - int(action == "release")
        peak = max(peak, live)
        if action == "release":
            last_user = "association" if name == "clip" else "mesh" if name in {"hmr2", "gvhmr"} else "placement" if name == "body" else f"{'tracking' if name == 'vitpose' else name}/cam{cameras - 1}"
            assert harness.events.index(("node", last_user)) < harness.events.index((action, name))
    assert live == 0 and peak <= 2
    assert all(reference() is None for reference in harness.references)
    for i in range(cameras):
        # DINO=2, pose=4, CLIP=5; the camera contributes at each fake node.
        np.testing.assert_array_equal(executor.output(("clip", f"tracking/cam{i}")).values, [11 + 2 * i])
    # Association adds CLIP=5, mesh adds HMR2+GVHMR=13, body adds 8.
    np.testing.assert_array_equal(executor.output(("clip", "placement")).values,
                                  [sum(11 + 2 * i for i in range(cameras)) + 26])


@pytest.mark.parametrize("source,overwrite", [("execute", False), ("load", False), ("load", True)])
def test_all_cache_hit_and_load_only_create_no_models(tmp_path: Path, source: str, overwrite: bool) -> None:
    harness = Harness(tmp_path, 3)
    PlanExecutor(harness.plan()).run()
    harness.events.clear()
    # A fresh plan retains real resource declarations. Only the artifact
    # decision may prevent loading; fake components do not special-case cache.
    plan = harness.plan(source=source, overwrite=overwrite)
    PlanExecutor(plan, model_event=harness.model_event).run()
    assert not harness.events and not harness.lifetimes
    assert set(plan.statuses.values()) == {"cached" if source == "execute" else "loaded"}


def test_mixed_cache_reuses_model_for_only_missing_cameras(tmp_path: Path) -> None:
    harness = Harness(tmp_path, 5)
    # Publish a non-prefix subset using the same plan and identities.
    seed = harness.plan()
    PlanExecutor(seed).run(targets=[("clip", "court/cam1"), ("clip", "court/cam3")])
    harness.events.clear()
    plan = harness.plan()
    executor = PlanExecutor(plan, model_event=harness.model_event)
    executor.run()
    assert [name for action, name in harness.events if action == "node" and name.startswith("court/")] == ["court/cam0", "court/cam2", "court/cam4"]
    assert Counter(name for action, name, _ in harness.lifetimes if action == "load")["court"] == 1
    assert plan.statuses[("clip", "court/cam1")] == plan.statuses[("clip", "court/cam3")] == "cached"


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_node_failure_unloads_before_a_new_execution(tmp_path: Path, failure: type[BaseException]) -> None:
    harness = Harness(tmp_path, 3)
    plan = harness.plan()
    failing = plan.nodes[("clip", "tracking/cam0")].component
    assert isinstance(failing, Component)
    failing.failure = failure
    with pytest.raises(failure, match="fake node stopped") as caught:
        PlanExecutor(plan, model_event=harness.model_event).run()
    assert caught.value.__traceback__ is not None
    assert plan.statuses[("clip", "tracking/cam0")] == "failed"
    assert all(reference() is None for reference in harness.references)
    assert Counter((name, instance) for action, name, instance in harness.lifetimes if action == "load") == Counter(
        (name, instance) for action, name, instance in harness.lifetimes if action == "unload")
    failing.failure = None
    fresh = harness.plan()
    PlanExecutor(fresh, model_event=harness.model_event).run()
    assert "failed" not in fresh.statuses.values()
    assert all(reference() is None for reference in harness.references)


def test_single_clip_partial_targets_close_each_drain_without_reexecution(tmp_path: Path) -> None:
    harness = Harness(tmp_path, 3)
    runner = ComponentRunner(harness.nodes, harness.store, model_event=harness.model_event)
    first = runner.run(["court/cam0"])["court/cam0"]
    assert all(reference() is None for reference in harness.references)
    runner.run(["court/cam2"])
    assert runner.references["court/cam0"] == first
    assert [event for event in harness.events if event == ("node", "court/cam0")] == [("node", "court/cam0")]
    assert [action for action, name, _ in harness.lifetimes if name == "court"] == ["load", "unload", "load", "unload"]
