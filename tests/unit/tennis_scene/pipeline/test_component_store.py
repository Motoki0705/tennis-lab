"""Declared execution, immutable publication and memory/disk input equivalence."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    ClipSource,
    ComponentIO,
    InputPort,
    SourceVideo,
)
from src.tennis_scene.pipeline.runner import ComponentNode, ComponentRunner
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_index import indexed_scene_path


@dataclass(frozen=True)
class Values:
    numbers: NDArray[np.float32]
    camera_ids: tuple[str, ...]


@dataclass(frozen=True)
class Input:
    previous: Values | None


@dataclass(frozen=True)
class Assembler:
    version: int = 1

    def assemble(self, context: AssemblyContext, artifacts: Any) -> Input:
        return Input(artifacts.get("previous"))


class Model:
    def __init__(self, *, downstream: bool = False) -> None:
        self.calls = 0
        self.io = ComponentIO("values", Input, Values, {"previous": InputPort("values")} if downstream else {}, "values")

    def process(self, inputs: Input) -> Values:
        self.calls += 1
        numbers = np.arange(4, dtype=np.float32) if inputs.previous is None else inputs.previous.numbers * 2
        return Values(numbers, ("cam0", "cam1"))


def context(tmp_path: Path) -> AssemblyContext:
    return AssemblyContext(ClipSource("clip", (SourceVideo("cam0", tmp_path / "cam0.mp4", "media_hash", 4, 30., 640, 480),)))


def node(name: str, model: Model, ctx: AssemblyContext, dependency: str | None = None) -> ComponentNode:
    return ComponentNode(name, model, model.io, Assembler(), {} if dependency is None else {"previous": dependency}, ctx, {}, "implementation1")


def test_restart_and_memory_have_identical_typed_inputs_and_skip_models(tmp_path: Path) -> None:
    first, second = Model(), Model(downstream=True)
    nodes = [node("first", first, context(tmp_path)), node("second", second, context(tmp_path), "first")]
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    run = ComponentRunner(nodes, store)
    run.run()
    np.testing.assert_array_equal(run.output("second").numbers, np.arange(4) * 2)
    assert not run.output("second").numbers.flags.writeable
    second_run = ComponentRunner([replace(n, source="load") for n in nodes], ClipStore(store.root, {"clip": "clip"}, memory_entries=0))
    second_run.run()
    assert second_run.output("second").camera_ids == ("cam0", "cam1")
    np.testing.assert_array_equal(second_run.output("second").numbers, run.output("second").numbers)
    assert first.calls == second.calls == 1
    assert set(second_run.statuses.values()) == {"loaded"}


def test_changes_recompute_only_descendants(tmp_path: Path) -> None:
    first, second, separate = Model(), Model(downstream=True), Model()
    ctx = context(tmp_path)
    nodes = [node("first", first, ctx), node("second", second, ctx, "first"), node("separate", separate, ctx)]
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    ComponentRunner(nodes, store).run()
    changed = [replace(nodes[0], settings={"weight": "new"}), *nodes[1:]]
    run = ComponentRunner(changed, store)
    run.run()
    assert (first.calls, second.calls, separate.calls) == (2, 2, 1)
    assert run.statuses["separate"] == "cached"


def test_loaded_dependency_cannot_silently_change(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    nodes = [node("first", Model(), ctx), node("second", Model(downstream=True), ctx, "first")]
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    ComponentRunner(nodes, store).run()
    with pytest.raises(ValueError, match="dependencies changed"):
        ComponentRunner([replace(nodes[0], settings={"revision": 2}), replace(nodes[1], source="load")], store).run()


def test_missing_and_incompatible_inputs_are_rejected_before_execution(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    model = Model(downstream=True)
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    with pytest.raises(ValueError, match="Missing producer"):
        ComponentRunner([node("second", model, ctx, "missing")], store)
    wrong = replace(node("first", Model(), ctx), io=ComponentIO("wrong", Input, Values, {}, "other"))
    with pytest.raises(ValueError, match="Incompatible"):
        ComponentRunner([wrong, node("second", model, ctx, "first")], store)
    assert model.calls == 0


def test_failed_publish_does_not_replace_index_and_corruption_is_rejected(tmp_path: Path) -> None:
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    codec = ArtifactCodec(Values)
    ref = store.publish("first", Values(np.zeros(2, np.float32), ("cam0",)), codec,
        schema="values", version=1, identity={"version": 1}, dependencies={}, provenance={"origin": "test"})
    with pytest.raises(TypeError, match="Object arrays"):
        store.publish("first", Values(np.array([object()], dtype=object), ("cam0",)), codec,
            schema="values", version=1, identity={"version": 2}, dependencies={}, provenance={"origin": "test"})
    assert store.active("first") == ref
    assert not list((store.root / "components/first").glob(".writing-*"))
    (store.root / ref.path).parent.joinpath("array_0000.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        store.load(ref, codec)


def test_source_mismatch_cannot_reuse_another_clip(tmp_path: Path) -> None:
    ClipStore(tmp_path, {"clip": "a"})
    with pytest.raises(ValueError, match="source/schema"):
        ClipStore(tmp_path, {"clip": "b"})


def test_replacing_implementation_uses_same_io_and_invalidates_its_outputs(tmp_path: Path) -> None:
    class ChangedModel(Model):
        def process(self, inputs: Input) -> Values:
            self.calls += 1
            return Values(np.arange(4, dtype=np.float32) + 10, ("cam0", "cam1"))
    old, new, downstream = Model(), ChangedModel(), Model(downstream=True)
    ctx = context(tmp_path)
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    original = [node("first", old, ctx), node("second", downstream, ctx, "first")]
    ComponentRunner(original, store).run()
    replaced = [node("first", new, ctx), original[1]]
    runner = ComponentRunner(replaced, store)
    runner.run()
    np.testing.assert_array_equal(runner.output("second").numbers, (np.arange(4) + 10) * 2)
    assert old.calls == new.calls == 1 and downstream.calls == 2


def test_upstream_replacement_invalidates_scene_export_and_legacy_stale_index(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    nodes = [node("first", Model(), ctx), node("second", Model(downstream=True), ctx, "first")]
    store = ClipStore(tmp_path / "store", {"clip": "clip"})
    runner = ComponentRunner(nodes, store)
    runner.run()
    export = store.root / "exports" / "fixture"
    export.mkdir(parents=True)
    scene = export / "scene.npz"
    metadata = export / "scene.metadata.json"
    scene.write_bytes(b"scene fixture")
    metadata.write_text("{}")
    store.record_export("scene", {"scene": scene, "metadata": metadata},
                        {"second": runner.references["second"]})
    assert indexed_scene_path(store.index_path) == scene
    old_export = json.loads(store.index_path.read_text())["exports"]["scene"]

    ComponentRunner([replace(nodes[0], settings={"revision": 2})], store).run()

    current = json.loads(store.index_path.read_text())
    assert current["exports"] == {}
    with pytest.raises(ValueError, match="no completed scene export"):
        indexed_scene_path(store.index_path)
    with pytest.raises(ValueError, match="superseded"):
        store.record_export("scene", {"scene": scene, "metadata": metadata},
                            {"second": runner.references["second"]})

    # Read old indexes defensively too: an export may have been left published
    # before invalidation-on-publish existed, even when its direct scene ref remains.
    current["exports"]["scene"] = old_export
    write_json_atomic(store.index_path, current)
    with pytest.raises(ValueError, match="superseded"):
        indexed_scene_path(store.index_path)
