"""Single-decision cache semantics and clip-local dependencies in one plan."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.artifacts import json_value
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
    NodeKey,
)
from src.tennis_scene.pipeline.runner import ComponentRunner, PlanExecutor
from src.tennis_scene.pipeline.storage.clip_store import (
    ArtifactReader,
    ArtifactRef,
    ClipStore,
)


@dataclass(frozen=True)
class Reading:
    clip_id: str
    values: NDArray[np.float32]


@dataclass(frozen=True)
class Request:
    clip_id: str
    previous: Reading | None


@dataclass(frozen=True)
class Assembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Mapping[str, Any]) -> Request:
        return Request(context.source.clip_id, artifacts.get("previous"))


class Model:
    def __init__(self, offset: float, *, downstream: bool = False) -> None:
        self.calls = 0
        self.offset = offset
        self.io = ComponentIO("reading", Request, Reading, {"previous": InputPort("reading")} if downstream else {}, "reading")

    def process(self, inputs: Request) -> Reading:
        self.calls += 1
        if inputs.previous is None:
            values = np.array([self.offset, self.offset + 3], np.float32)
        else:
            assert inputs.previous.clip_id == inputs.clip_id
            values = inputs.previous.values * 2
        return Reading(inputs.clip_id, values)


class RecordingStore(ClipStore):
    def __init__(self, root: Path, clip_id: str) -> None:
        super().__init__(root, {"clip_id": clip_id}, memory_entries=0)
        self.lookups: Counter[str] = Counter()
        self.loads: Counter[str] = Counter()

    def active(self, node: str) -> ArtifactRef | None:
        self.lookups[node] += 1
        return super().active(node)

    def load(self, reference: ArtifactRef, codec: ArtifactReader[Any]) -> Any:
        self.loads[reference.artifact_id] += 1
        return super().load(reference, codec)


def clip(tmp_path: Path, clip_id: str = "a", offset: float = 2) -> ClipExecution:
    source = ClipSource(clip_id, (SourceVideo("cam0", tmp_path / f"{clip_id}.mp4", clip_id, 2, 30., 64, 36),))
    context = AssemblyContext(source)
    first, second, separate = Model(offset), Model(offset, downstream=True), Model(offset + 10)
    nodes = (
        ComponentNode("first", first, first.io, Assembler(), {}, context,
                      {"model": {"kind": "convnext"}, "preprocessing": {"channels": 2}}, "test1"),
        ComponentNode("second", second, second.io, Assembler(), {"previous": "first"}, context, {}, "test1"),
        ComponentNode("separate", separate, separate.io, Assembler(), {}, context, {}, "test1"),
    )
    return ClipExecution(clip_id, nodes, RecordingStore(tmp_path / clip_id, clip_id))


@pytest.mark.parametrize("case,expected", [
    ("empty", ("executed", "executed", "executed")),
    ("cached", ("cached", "cached", "cached")),
    ("partial", ("cached", "executed", "executed")),
    ("overwrite", ("executed", "executed", "executed")),
    ("load", ("loaded", "loaded", "loaded")),
    ("overwrite_load", ("loaded", "loaded", "loaded")),
    ("model", ("executed", "executed", "cached")),
    ("preprocessing", ("executed", "executed", "cached")),
])
def test_each_node_is_decided_and_processed_once(tmp_path: Path, case: str, expected: tuple[str, ...]) -> None:
    spec = clip(tmp_path)
    if case != "empty":
        seed = replace(spec, targets=("first",)) if case == "partial" else spec
        PlanExecutor(ExecutionPlan([seed])).run()
    before = [node.component.calls for node in spec.nodes]
    if case in ("model", "preprocessing"):
        settings = {"model": {"kind": "x3d_csn_dpt" if case == "model" else "convnext"},
                    "preprocessing": {"channels": 5 if case == "preprocessing" else 2}}
        spec = replace(spec, nodes=(replace(spec.nodes[0], settings=settings), *spec.nodes[1:]))
    if case in ("load", "overwrite_load"):
        spec = replace(spec, nodes=tuple(replace(node, source="load") for node in spec.nodes))
    spec = replace(spec, overwrite=case in ("overwrite", "overwrite_load"))
    spec.store.lookups.clear()
    spec.store.loads.clear()
    plan = ExecutionPlan([spec])
    executor = PlanExecutor(plan)
    completed: list[str] = []
    executor.run(on_clip_complete=completed.append)
    executor.run(on_clip_complete=completed.append)  # Draining again must not inspect completed nodes.
    executor.execute(("a", "first"))
    assert [plan.statuses[("a", name)] for name in ("first", "second", "separate")] == list(expected)
    assert spec.store.lookups == {"first": 1, "second": 1, "separate": 1}
    assert [node.component.calls - count for node, count in zip(spec.nodes, before, strict=True)] == [int(s == "executed") for s in expected]
    assert len(plan.decisions) == 3
    assert completed == ["a"]
    expected_loads = [int(expected[0] != "executed") + int(expected[1] == "executed"),
                      int(expected[1] != "executed"), int(expected[2] != "executed")]
    assert spec.store.loads == {plan.references[("a", name)].artifact_id: count
                                for name, count in zip(("first", "second", "separate"), expected_loads, strict=True) if count}
    np.testing.assert_array_equal(executor.output(("a", "second")).values, [4, 10])


def test_missing_load_artifact_is_required_and_decided_once(tmp_path: Path) -> None:
    spec = clip(tmp_path)
    spec = replace(spec, nodes=tuple(replace(node, source="load") for node in spec.nodes))
    plan = ExecutionPlan([spec])
    executor = PlanExecutor(plan)
    with pytest.raises(FileNotFoundError, match="artifact missing"):
        executor.run()
    with pytest.raises(FileNotFoundError, match="artifact missing"):
        plan.decide(("a", "first"))
    assert spec.store.lookups == {"first": 1} and not spec.store.loads
    assert all(node.component.calls == 0 for node in spec.nodes)


def test_single_clip_api_keeps_references_and_never_reexecutes_completed_targets(tmp_path: Path) -> None:
    spec = clip(tmp_path)
    runner = ComponentRunner(spec.nodes, spec.store, overwrite=True)
    references = runner.run(targets=["first"])
    first = references["first"]
    runner.run(targets=["second"])
    assert references is runner.references
    assert references["first"] == first and set(references) == {"first", "second"}
    assert [node.component.calls for node in spec.nodes] == [1, 1, 0]
    assert spec.store.lookups == {"first": 1, "second": 1}


@pytest.mark.parametrize("initial_targets", [None, [("a", "second"), ("a", "separate")]])
def test_failed_completion_is_recorded_once_and_does_not_block_other_clip(
    tmp_path: Path, initial_targets: list[NodeKey] | None,
) -> None:
    clips = [clip(tmp_path, "a"), clip(tmp_path, "b", 7)]
    plan = ExecutionPlan(clips)
    executor = PlanExecutor(plan)
    calls: list[str] = []
    failure = ValueError("clip inputs changed before publication")

    def publish(clip_id: str) -> None:
        calls.append(clip_id)
        if clip_id == "a":
            raise failure

    with pytest.raises(ValueError, match="inputs changed") as caught:
        executor.run(targets=initial_targets, on_clip_complete=publish)
    assert caught.value is failure
    # This call used to retry A during the initial all-clip notification scan,
    # so B's independent nodes could never execute.
    executor.run(targets=[("b", "second"), ("b", "separate")], on_clip_complete=publish)
    executor.run(on_clip_complete=publish)
    assert calls == ["a", "b"]
    assert executor.completion_attempts == {"a", "b"}
    assert executor.completion_results["a"].status == "failed"
    assert executor.completion_results["a"].error_type == "ValueError"
    assert executor.completion_results["a"].error == str(failure)
    assert executor.completion_results["b"].status == "succeeded"
    assert executor.completion_results["b"].error is None
    assert plan.clip_complete("a") and plan.clip_complete("b")
    assert set(plan.statuses.values()) == {"executed"}
    assert all(node.component.calls == 1 for spec in clips for node in spec.nodes)


@pytest.mark.parametrize("source", ["execute", "load"])
def test_corrupt_cached_or_loaded_artifact_fails_without_recalculation(tmp_path: Path, source: str) -> None:
    spec = clip(tmp_path)
    seed = ExecutionPlan([spec])
    PlanExecutor(seed).run()
    first = seed.references[("a", "first")]
    (spec.store.root / first.path).parent.joinpath("array_0000.npy").write_bytes(b"corrupt")
    spec = replace(spec, nodes=tuple(replace(node, source=source) for node in spec.nodes))
    spec.store.lookups.clear()
    plan = ExecutionPlan([spec])
    executor = PlanExecutor(plan)
    with pytest.raises(ValueError, match="checksum"):
        executor.run()
    with pytest.raises(ValueError, match="not ready"):
        executor.execute(("a", "first"))
    assert spec.store.lookups == {"first": 1}
    assert [node.component.calls for node in spec.nodes] == [1, 1, 1]
    assert plan.statuses == {("a", "first"): "failed"} and not plan.references


@pytest.mark.parametrize("grouped", [False, True])
def test_two_clips_keep_stores_and_dependencies_separate_under_ready_policy(tmp_path: Path, grouped: bool) -> None:
    clips = [clip(tmp_path, "a", 2), clip(tmp_path, "b", 7)]
    plan = ExecutionPlan(clips)
    selected: list[NodeKey] = []
    priorities = {"first": 0, "second": 1, "separate": 2}

    def choose(ready: tuple[NodeKey, ...]) -> NodeKey:
        key = min(ready, key=lambda k: (priorities[k[1]], k[0])) if grouped else ready[0]
        selected.append(key)
        return key

    executor = PlanExecutor(plan, select_ready=choose)
    executor.run()
    expected = ([(c, n) for n in ("first", "second", "separate") for c in ("a", "b")] if grouped else
                [(c, n) for c in ("a", "b") for n in ("first", "second", "separate")])
    assert selected == expected
    for spec, values in zip(clips, ([4, 10], [14, 20]), strict=True):
        output = executor.output((spec.clip_id, "second"))
        assert output.clip_id == spec.clip_id
        np.testing.assert_array_equal(output.values, values)
        reference = plan.references[(spec.clip_id, "second")]
        assert spec.store.descriptor(reference)["dependencies"] == {"previous": json_value(plan.references[(spec.clip_id, "first")])}
        assert spec.store.lookups == {"first": 1, "second": 1, "separate": 1}
    assert plan.references[("a", "first")] != plan.references[("b", "first")]
    for spec in clips:
        spec.store.lookups.clear()
    reordered = ExecutionPlan(list(reversed(clips)))
    PlanExecutor(reordered).run()
    assert reordered.references == plan.references
    assert set(reordered.statuses.values()) == {"cached"}
    assert all(node.component.calls == 1 for spec in clips for node in spec.nodes)


def test_policy_cannot_select_unready_node(tmp_path: Path) -> None:
    spec = clip(tmp_path)
    executor = PlanExecutor(ExecutionPlan([spec]), select_ready=lambda _: ("a", "second"))
    with pytest.raises(ValueError, match="not ready"):
        executor.run()
    assert not spec.store.lookups and all(node.component.calls == 0 for node in spec.nodes)


@pytest.mark.parametrize("invalid,match", [
    ("duplicate_clip", "Duplicate clip"),
    ("shared_store", "separate clip stores"),
    ("wrong_context", "source clip disagrees"),
    ("foreign_producer", "Missing producer"),
    ("cycle", "Cyclic"),
    ("unknown_target", "Unknown component target"),
])
def test_invalid_plan_is_rejected_before_execution(tmp_path: Path, invalid: str, match: str) -> None:
    a, b = clip(tmp_path, "a"), clip(tmp_path, "b")
    if invalid == "duplicate_clip":
        b = replace(b, clip_id="a")
    elif invalid == "shared_store":
        b = replace(b, store=a.store)
    elif invalid == "wrong_context":
        b = replace(b, nodes=a.nodes)
    elif invalid == "foreign_producer":
        b = replace(b, nodes=b.nodes[1:])  # a.first cannot satisfy b.second.
    elif invalid == "cycle":
        b = replace(b, nodes=(replace(b.nodes[1], bindings={"previous": "second"}),))
    else:
        b = replace(b, targets=("missing",))
    with pytest.raises(ValueError, match=match):
        ExecutionPlan([a, b])
    assert not a.store.lookups and all(node.component.calls == 0 for node in a.nodes)
