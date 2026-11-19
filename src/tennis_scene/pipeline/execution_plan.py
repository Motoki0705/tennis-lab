"""Clip-scoped dependency plans and memoized cache/load decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.tennis_scene.pipeline.artifacts import document_digest, json_value
from src.tennis_scene.pipeline.contracts import (
    AssemblyContext,
    Component,
    ComponentIO,
    InputAssembler,
)
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    model_identities,
)
from src.tennis_scene.pipeline.model_assets import asset_hashes
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore

NodeKey = tuple[str, str]


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
    required_import_identity: Mapping[str, Any] | None = None
    # A guarded component must authorize adoption even when its process() is skipped.
    # Only its own output, with the complete execution identity, can be reused.
    require_component: Callable[[], None] | None = None
    # Soft barriers across the selected component families, including cameras
    # and clips. They order execution without adding artifact inputs or targets.
    after_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in ("execute", "load"):
            raise ValueError("Component source must be execute or load")
        if set(self.bindings) != set(self.io.inputs):
            raise ValueError(f"Bindings disagree with declared inputs: {self.name}")
        if self.required_import_identity is not None and self.source != "load":
            raise ValueError("Pinned external inputs must be load-only")

    def require_importable(self) -> None:
        if self.require_component is not None:
            raise ValueError(f"{self.name} requires component origin")

    def validate_import_identity(self, identity: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
        if provenance.get("origin") != "component":
            self.require_importable()
        if self.required_import_identity is not None and (
                provenance.get("origin") != "import"
                or any(identity.get(key) != json_value(value) for key, value in self.required_import_identity.items())):
            raise ValueError(f"Loaded import identity changed: {self.name}")

    def validate_output(self, value: Any) -> None:
        validator = getattr(self.component, "validate_output", None)
        if callable(validator):
            validator(value, self.settings)

    def validate_artifact(self, descriptor: Mapping[str, Any]) -> None:
        validator = getattr(self.component, "validate_artifact", None)
        if callable(validator):
            validator(descriptor, self.settings)


@dataclass(frozen=True)
class ClipExecution:
    """One clip's nodes, explicit store and requested dependency closure."""

    clip_id: str
    nodes: Sequence[ComponentNode]
    store: ClipStore
    overwrite: bool = False
    targets: Sequence[str] | None = None


@dataclass(frozen=True)
class NodeDecision:
    key: NodeKey
    action: Literal["execute", "load", "cached"]
    identity: Mapping[str, Any]
    dependencies: Mapping[str, ArtifactRef]
    reference: ArtifactRef | None


class ExecutionPlan:
    """Graph keyed by (clip ID, node), independent of model scheduling.

    Decisions are deferred until dependencies have published their references,
    then retained for the lifetime of the plan. Scheduling never enters the
    execution identity; changing clip order or the ready-node policy is safe.
    """

    def __init__(self, clips: Sequence[ClipExecution]) -> None:
        self.clips: dict[str, ClipExecution] = {}
        self.nodes: dict[NodeKey, ComponentNode] = {}
        roots: set[Path] = set()
        for clip in clips:
            if clip.clip_id in self.clips:
                raise ValueError(f"Duplicate clip ID: {clip.clip_id}")
            if clip.store.root in roots:
                raise ValueError("Different clips must use separate clip stores")
            roots.add(clip.store.root)
            self.clips[clip.clip_id] = clip
            for node in clip.nodes:
                key = (clip.clip_id, node.name)
                if key in self.nodes:
                    raise ValueError(f"Duplicate component node: {key}")
                if node.context.source.clip_id != clip.clip_id:
                    raise ValueError(f"Component source clip disagrees with plan: {key}")
                self.nodes[key] = node
        self.dependencies = {
            key: {port: (key[0], producer) for port, producer in node.bindings.items()}
            for key, node in self.nodes.items()
        }
        families: dict[str, list[NodeKey]] = {}
        for key, node in self.nodes.items():
            families.setdefault(node.name.split('/')[0], []).append(key)
        self.scheduling_dependencies = {
            key: tuple(parent for family in node.after_components for parent in families.get(family, ()))
            for key, node in self.nodes.items()
        }
        producers: dict[tuple[str, int], list[NodeKey]] = {}
        for key, node in self.nodes.items():
            producers.setdefault((node.io.output_schema, node.io.version), []).append(key)
        self.ordering_dependencies = {
            key: tuple(dict.fromkeys(parent for port in node.io.after_all_inputs
                for parent in producers.get((node.io.inputs[port].schema, node.io.inputs[port].version), ())))
            for key, node in self.nodes.items()
        }
        self.order = self._dependency_order()
        targets = [(clip.clip_id, name) for clip in clips
                   for name in (tuple(n.name for n in clip.nodes) if clip.targets is None else clip.targets)]
        self.required = self.target_nodes(targets)
        self.references: dict[NodeKey, ArtifactRef] = {}
        self.statuses: dict[NodeKey, str] = {}
        self.seconds: dict[NodeKey, float] = {}
        self.decisions: dict[NodeKey, NodeDecision] = {}
        self._decision_errors: dict[NodeKey, Exception] = {}
        self.active_node: NodeKey | None = None
        self.exports: dict[str, Path] = {}

    def _dependency_order(self) -> tuple[NodeKey, ...]:
        order: list[NodeKey] = []
        visited: set[NodeKey] = set()
        visiting: set[NodeKey] = set()

        def visit(key: NodeKey) -> None:
            if key in visiting:
                raise ValueError(f"Cyclic component graph at {key}")
            if key in visited:
                return
            visiting.add(key)
            node = self.nodes[key]
            for port, dependency in self.dependencies[key].items():
                if dependency not in self.nodes:
                    raise ValueError(f"Missing producer {dependency[1]} for {key[1]}.{port}")
                declared, produced = node.io.inputs[port], self.nodes[dependency].io
                if (declared.schema, declared.version) != (produced.output_schema, produced.version):
                    raise ValueError(f"Incompatible artifact binding {key[1]}.{port}")
                visit(dependency)
            for dependency in (*self.scheduling_dependencies[key], *self.ordering_dependencies[key]):
                visit(dependency)
            visiting.remove(key)
            visited.add(key)
            order.append(key)

        for key in self.nodes:
            visit(key)
        return tuple(order)

    def target_nodes(self, targets: Sequence[NodeKey]) -> frozenset[NodeKey]:
        required: set[NodeKey] = set()

        def collect(key: NodeKey) -> None:
            if key not in self.nodes:
                raise ValueError(f"Unknown component target: {key}")
            if key in required:
                return
            required.add(key)
            for dependency in self.dependencies[key].values():
                collect(dependency)

        for target in targets:
            collect(target)
        return frozenset(required)

    def ready(self, required: frozenset[NodeKey] | None = None) -> tuple[NodeKey, ...]:
        selected = self.required if required is None else required
        pending_families = {self.nodes[key].name.split('/')[0] for key in selected if key not in self.references}
        return tuple(key for key in self.order if key in selected and key not in self.statuses
                     and all(parent in self.references for parent in self.dependencies[key].values())
                     and not any(family in pending_families for family in self.nodes[key].after_components)
                     and all(parent in self.references for parent in self.ordering_dependencies[key] if parent in selected))

    def clip_complete(self, clip_id: str) -> bool:
        if clip_id not in self.clips:
            raise ValueError(f"Unknown clip: {clip_id}")
        return all(key in self.references for key in self.required if key[0] == clip_id)

    def block_clip(self, clip_id: str) -> tuple[NodeKey, ...]:
        """A failed clip cannot publish, so withdraw all of its unfinished work."""
        blocked = tuple(key for key in self.order if key[0] == clip_id and key in self.required and key not in self.statuses)
        self.statuses.update(dict.fromkeys(blocked, "blocked"))
        return blocked

    def decide(self, key: NodeKey) -> NodeDecision:
        """Inspect a node's adopted artifact once, after all its inputs exist."""
        if key in self.decisions:
            return self.decisions[key]
        if key in self._decision_errors:
            raise self._decision_errors[key]
        if any(parent not in self.references for parent in self.dependencies[key].values()):
            raise ValueError(f"Component dependencies are not ready: {key}")
        try:
            decision = self._decide(key)
        except Exception as error:
            self._decision_errors[key] = error
            raise
        self.decisions[key] = decision
        return decision

    def _decide(self, key: NodeKey) -> NodeDecision:
        node, clip = self.nodes[key], self.clips[key[0]]
        if node.source != 'load' and isinstance(node.component, InferenceComponent):
            specs = node.component.model_specs()
            expected = node.settings.get('models')
            if specs or expected:
                if not isinstance(expected, Mapping):
                    raise ValueError(f'Inference node has no bound model declaration: {key}')
                hashes = asset_hashes(node.settings)
                def bound(path: Path) -> dict[str, Any]:
                    if path not in hashes:
                        raise ValueError(f'Inference model declaration contains an unbound asset: {path}')
                    return {'path': str(path), 'sha256': hashes[path]}
                if json_value(model_identities(specs, bound)) != json_value(expected):
                    raise ValueError(f'Inference model declaration differs from bound node identity: {key}')
        dependencies = {port: self.references[parent] for port, parent in self.dependencies[key].items()}
        identity = {"component": node.io.name, "schema": node.io.output_schema, "version": node.io.version,
            "implementation": node.implementation, "component_class": f"{type(node.component).__module__}.{type(node.component).__qualname__}",
            "assembler": f"{type(node.assembler).__module__}.{type(node.assembler).__qualname__}",
            "assembler_version": node.assembler.version, "settings": json_value(node.settings),
            "dependencies": json_value(dependencies), "scope": node.context.camera_id,
            "source_sha256": clip.store.source_key}
        reference = clip.store.active(key[1])
        cache_hit = reference is not None and not clip.overwrite and reference.execution_key == document_digest(identity)
        action: Literal["execute", "load", "cached"] = "cached" if cache_hit else "execute"
        if node.source == "load" or (cache_hit and node.require_component is not None):
            if node.require_component is not None:
                node.require_component()
            if reference is None:
                raise FileNotFoundError(f"Required component artifact missing: {key[1]}")
            descriptor = clip.store.descriptor(reference)
            if (reference.schema, reference.version) != (node.io.output_schema, node.io.version) or descriptor["node"] != key[1]:
                raise ValueError(f"Loaded component contract mismatch: {key[1]}")
            if descriptor["dependencies"] != json_value(dependencies):
                raise ValueError(f"Loaded artifact dependencies changed: {key[1]}")
            node.validate_import_identity(descriptor["identity"], descriptor["provenance"])
            node.validate_artifact(descriptor)
            if descriptor["provenance"].get("origin") == "component" and (
                    reference.execution_key != document_digest(identity) or descriptor["identity"] != identity):
                raise ValueError(f"Loaded component identity changed: {key[1]}")
            action = "load" if node.source == "load" else "cached"
        return NodeDecision(key, action, identity, dependencies, reference)
