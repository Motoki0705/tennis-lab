"""Execute declared components without stage names or model-specific assembly."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    Component,
    ComponentIO,
    InputAssembler,
)
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec


@dataclass(frozen=True)
class ComponentNode:
    name: str
    component: Component[Any, Any]
    io: ComponentIO[Any, Any]
    assembler: InputAssembler[Any]
    bindings: Mapping[str, str]
    context: AssemblyContext
    settings: Mapping[str, Any]
    implementation: str
    source: str = "execute"

    def __post_init__(self) -> None:
        if self.source not in ("execute", "load"):
            raise ValueError("Component source must be execute or load")
        if set(self.bindings) != set(self.io.inputs):
            raise ValueError(f"Bindings disagree with declared inputs: {self.name}")


class ComponentRunner:
    def __init__(self, nodes: Sequence[ComponentNode], store: ClipStore, *, overwrite: bool = False) -> None:
        self.nodes = {node.name: node for node in nodes}
        if len(self.nodes) != len(nodes):
            raise ValueError("Duplicate component node")
        self.store, self.overwrite = store, overwrite
        self.statuses: dict[str, str] = {}
        self.seconds: dict[str, float] = {}
        self.references: dict[str, ArtifactRef] = {}
        self.active_node: str | None = None
        self.order = self._plan()

    def _plan(self) -> tuple[str, ...]:
        order: list[str] = []
        visiting: set[str] = set()
        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Cyclic component graph at {name}")
            if name in order:
                return
            visiting.add(name)
            node = self.nodes[name]
            for port, dependency in node.bindings.items():
                if dependency not in self.nodes:
                    raise ValueError(f"Missing producer {dependency} for {name}.{port}")
                declared, produced = node.io.inputs[port], self.nodes[dependency].io
                if (declared.schema, declared.version) != (produced.output_schema, produced.version):
                    raise ValueError(f"Incompatible artifact binding {name}.{port}")
                visit(dependency)
            visiting.remove(name)
            order.append(name)
        for name in self.nodes:
            visit(name)
        return tuple(order)

    def run(self, targets: Sequence[str] | None = None) -> dict[str, ArtifactRef]:
        required: set[str] = set()
        def collect(name: str) -> None:
            if name not in self.nodes:
                raise ValueError(f"Unknown component target: {name}")
            if name in required:
                return
            required.add(name)
            for parent in self.nodes[name].bindings.values():
                collect(parent)
        for target in self.nodes if targets is None else targets:
            collect(target)
        for name in self.order:
            if name not in required:
                continue
            node = self.nodes[name]
            self.active_node = name
            started = time.monotonic()
            dependencies = {port: self.references[producer] for port, producer in node.bindings.items()}
            identity = {"component": node.io.name, "schema": node.io.output_schema, "version": node.io.version,
                "implementation": node.implementation, "component_class": f"{type(node.component).__module__}.{type(node.component).__qualname__}",
                "assembler": f"{type(node.assembler).__module__}.{type(node.assembler).__qualname__}",
                "assembler_version": node.assembler.version, "settings": json_value(node.settings),
                "dependencies": json_value(dependencies), "scope": node.context.camera_id,
                "source_sha256": self.store.source_key}
            reference = self.store.active(name)
            if node.source == "load":
                if reference is None:
                    raise FileNotFoundError(f"Required component artifact missing: {name}")
                descriptor = self.store.descriptor(reference)
                if (reference.schema, reference.version) != (node.io.output_schema, node.io.version) or descriptor["node"] != name:
                    raise ValueError(f"Loaded component contract mismatch: {name}")
                if descriptor["dependencies"] != json_value(dependencies):
                    raise ValueError(f"Loaded artifact dependencies changed: {name}")
                if descriptor["provenance"].get("origin") == "component" and reference.execution_key != document_digest(identity):
                    raise ValueError(f"Loaded component identity changed: {name}")
                self.store.load(reference, ArtifactCodec(node.io.output_type))
                self.statuses[name] = "loaded"
            elif reference is not None and not self.overwrite and reference.execution_key == document_digest(identity):
                self.store.load(reference, ArtifactCodec(node.io.output_type))
                self.statuses[name] = "cached"
            else:
                artifacts = {port: self.store.load(ref, ArtifactCodec(self.nodes[node.bindings[port]].io.output_type)) for port, ref in dependencies.items()}
                inputs = node.assembler.assemble(node.context, artifacts)
                if not isinstance(inputs, node.io.input_type):
                    raise TypeError(f"Assembler returned wrong input type for {name}")
                try:
                    output = node.component.process(inputs)
                    reference = self.store.publish(name, output, ArtifactCodec(node.io.output_type),
                        schema=node.io.output_schema, version=node.io.version, identity=identity,
                        dependencies=dependencies, provenance={"origin": "component", "implementation": node.implementation})
                    self.statuses[name] = "executed"
                except Exception:
                    self.statuses[name] = "failed"
                    raise
            if reference is None:
                raise RuntimeError("Component did not publish an artifact")
            self.references[name] = reference
            self.seconds[name] = time.monotonic() - started
        self.active_node = None
        return self.references

    def output(self, node: str) -> Any:
        return self.store.load(self.references[node], ArtifactCodec(self.nodes[node].io.output_type))
