"""Execute ready nodes and finalize clips using a shared dependency plan."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.tennis_scene.pipeline.execution_plan import (
    ClipExecution,
    ExecutionPlan,
    NodeDecision,
    NodeKey,
)
from src.tennis_scene.pipeline.execution_plan import ComponentNode as ComponentNode
from src.tennis_scene.pipeline.inference_runtime import (
    InferenceComponent,
    InferenceRuntime,
    ModelBorrower,
    ModelEventHook,
)
from src.tennis_scene.pipeline.model_assets import asset_hashes
from src.tennis_scene.pipeline.storage.clip_store import ArtifactRef, ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SceneResult

ReadySelector = Callable[[tuple[NodeKey, ...]], NodeKey]


@dataclass(frozen=True)
class ClipCompletionResult:
    """Callback outcome, separate from completion of the clip's nodes."""

    status: Literal["succeeded", "failed"]
    error: str | None = None
    error_type: str | None = None


class PlanExecutor:
    def __init__(self, plan: ExecutionPlan, *, select_ready: ReadySelector | None = None,
                 model_event: ModelEventHook | None = None) -> None:
        self.plan = plan
        self.select_ready = select_ready
        self.model_event = model_event
        self._completion_attempts: set[str] = set()
        self.completion_results: dict[str, ClipCompletionResult] = {}

    @property
    def completion_attempts(self) -> frozenset[str]:
        return frozenset(self._completion_attempts)

    def run(self, *, targets: Sequence[NodeKey] | None = None,
            on_clip_complete: Callable[[str], None] | None = None) -> dict[NodeKey, ArtifactRef]:
        references: dict[NodeKey, ArtifactRef] = self.plan.references
        required = self.plan.required if targets is None else self.plan.target_nodes(targets)
        if not required <= self.plan.required:
            raise ValueError("Targets are outside the execution plan")
        def notify(clip_id: str) -> None:
            if on_clip_complete is not None and clip_id not in self._completion_attempts and self.plan.clip_complete(clip_id):
                # A failed or interrupted publication must not be retried when
                # the caller resumes this executor for an independent clip.
                self._completion_attempts.add(clip_id)
                try:
                    on_clip_complete(clip_id)
                except BaseException as error:
                    self.completion_results[clip_id] = ClipCompletionResult("failed", str(error), type(error).__name__)
                    raise
                else:
                    self.completion_results[clip_id] = ClipCompletionResult("succeeded")
        with self._runtime(required) as runtime:
            for clip_id in self.plan.clips:
                notify(clip_id)
            while required - references.keys():
                ready = self.plan.ready(required)
                if not ready:
                    raise ValueError("Execution plan is blocked by a failed component")
                key = runtime.select_ready(ready) if self.select_ready is None else self.select_ready(ready)
                if key not in ready:
                    raise ValueError(f"Selected component is not ready: {key}")
                self._execute(key, runtime, required)
                notify(key[0])
        return references

    def _runtime(self, required: frozenset[NodeKey]) -> InferenceRuntime:
        declarations = {}
        for key in self.plan.order:
            if key not in required or key in self.plan.statuses:
                continue
            node = self.plan.nodes[key]
            declarations[key] = (node.component.model_specs()
                                 if node.source == "execute" and isinstance(node.component, InferenceComponent) else {})
        return InferenceRuntime(declarations,
            expected_assets={key: asset_hashes(self.plan.nodes[key].settings) for key in declarations},
            model_event=self.model_event)

    def execute(self, key: NodeKey) -> ArtifactRef:
        """Run or load one ready node; completed nodes are never reconsidered."""
        if key not in self.plan.required:
            raise ValueError("Targets are outside the execution plan")
        with self._runtime(frozenset({key})) as runtime:
            return self._execute(key, runtime, frozenset({key}))

    def _execute(self, key: NodeKey, runtime: InferenceRuntime, required: frozenset[NodeKey]) -> ArtifactRef:
        plan = self.plan
        if key in plan.references:
            return plan.references[key]
        if key not in plan.ready(required):
            raise ValueError(f"Component is not ready: {key}")
        plan.active_node = key
        started = time.monotonic()
        try:
            with runtime.node(key) as models:
                output, decision = self._execute_node(key, models)
            node, store = plan.nodes[key], plan.clips[key[0]].store
            reference = decision.reference
            if decision.action == "execute":
                reference = store.publish(key[1], output, ArtifactCodec(node.io.output_type),
                    schema=node.io.output_schema, version=node.io.version, identity=decision.identity,
                    dependencies=decision.dependencies, provenance={"origin": "component", "implementation": node.implementation})
                status = "executed"
            else:
                status = "loaded" if decision.action == "load" else "cached"
            if reference is None:
                raise RuntimeError("Completed component has no artifact reference")
            # Unload/recorder/cleanup must succeed before dependencies or clip
            # publication can observe this node as completed, including resume.
            plan.references[key] = reference
            plan.statuses[key] = status
            plan.active_node = None
            return reference
        except BaseException:
            plan.statuses[key] = "failed"
            raise
        finally:
            plan.seconds[key] = time.monotonic() - started

    def _execute_node(self, key: NodeKey, models: ModelBorrower) -> tuple[Any, NodeDecision]:
        plan = self.plan
        node, store = plan.nodes[key], plan.clips[key[0]].store
        decision = plan.decide(key)
        reference = decision.reference
        if decision.action in ("load", "cached"):
            if reference is None:
                raise RuntimeError("Reused component has no artifact reference")
            value = store.load(reference, ArtifactCodec(node.io.output_type))
            node.validate_output(value)
            output = None
        else:
            artifacts = {port: store.load(ref, ArtifactCodec(plan.nodes[plan.dependencies[key][port]].io.output_type))
                         for port, ref in decision.dependencies.items()}
            inputs = node.assembler.assemble(node.context, artifacts)
            if not isinstance(inputs, node.io.input_type):
                raise TypeError(f"Assembler returned wrong input type for {key[1]}")
            output = (node.component.process_with_models(inputs, models)
                      if isinstance(node.component, InferenceComponent) else node.component.process(inputs))
            node.validate_output(output)
        return output, decision

    def output(self, key: NodeKey) -> Any:
        return self.plan.clips[key[0]].store.load(self.plan.references[key], ArtifactCodec(self.plan.nodes[key].io.output_type))

    def finalize_clip(self, clip_id: str) -> tuple[SceneResult, Path]:
        """Export a completed clip independently of any other clip's outcome."""
        plan = self.plan
        if not plan.clip_complete(clip_id) or (clip_id, "scene_assembly") not in plan.references:
            raise ValueError(f"Cannot finalize an incomplete clip: {clip_id}")
        key = (clip_id, "scene_assembly")
        scene = self.output(key)
        if not isinstance(scene, SceneResult):
            raise TypeError("scene_assembly must produce a SceneResult")
        if clip_id not in plan.exports:
            plan.exports[clip_id] = export_scene(plan.clips[clip_id].store, scene, plan.references[key])
        return scene, plan.exports[clip_id]


class ComponentRunner:
    """Existing single-clip API, backed by the same plan as multi-clip callers."""

    def __init__(self, nodes: Sequence[ComponentNode], store: ClipStore, *, overwrite: bool = False,
                 select_ready: ReadySelector | None = None, model_event: ModelEventHook | None = None) -> None:
        self.nodes = {node.name: node for node in nodes}
        self.store, self.overwrite = store, overwrite
        self.clip_id = nodes[0].context.source.clip_id if nodes else "standalone"
        self.plan = ExecutionPlan([ClipExecution(self.clip_id, tuple(nodes), store, overwrite)])
        self.executor = PlanExecutor(self.plan, select_ready=select_ready, model_event=model_event)
        self.order = tuple(name for _, name in self.plan.order)
        self._statuses: dict[str, str] = {}
        self._seconds: dict[str, float] = {}
        self._references: dict[str, ArtifactRef] = {}

    def _sync(self) -> None:
        # Retain the existing mapping objects for callers holding run()'s result.
        self._statuses.update({name: status for (_, name), status in self.plan.statuses.items()})
        self._seconds.update({name: seconds for (_, name), seconds in self.plan.seconds.items()})
        self._references.update({name: reference for (_, name), reference in self.plan.references.items()})

    @property
    def statuses(self) -> dict[str, str]:
        self._sync()
        return self._statuses

    @property
    def seconds(self) -> dict[str, float]:
        self._sync()
        return self._seconds

    @property
    def references(self) -> dict[str, ArtifactRef]:
        self._sync()
        return self._references

    @property
    def active_node(self) -> str | None:
        return None if self.plan.active_node is None else self.plan.active_node[1]

    def run(self, targets: Sequence[str] | None = None) -> dict[str, ArtifactRef]:
        try:
            self.executor.run(targets=None if targets is None else [(self.clip_id, name) for name in targets])
        finally:
            self._sync()
        return self.references

    def output(self, node: str) -> Any:
        return self.executor.output((self.clip_id, node))

    def finalize_clip(self) -> tuple[SceneResult, Path]:
        return self.executor.finalize_clip(self.clip_id)
