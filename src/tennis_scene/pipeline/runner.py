"""Execute ready nodes and finalize clips using a shared dependency plan."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.tennis_scene.pipeline.batching import (
    BatchAdapter,
    BatchConfig,
    BatchJob,
    SourceMemory,
    WorkingMemory,
    drain_batch_jobs,
)
from src.tennis_scene.pipeline.errors import is_clip_failure
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
from src.tennis_scene.pipeline.storage.clip_store import (
    ArtifactRef,
    ClipStore,
    artifact_memory_bytes,
)
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.tennis_scene.pipeline.storage.scene_export import export_scene
from src.tennis_scene.schema import SceneResult
from src.utils.inference_recording import measured, recording_context, span

ReadySelector = Callable[[tuple[NodeKey, ...]], NodeKey]


@dataclass(frozen=True)
class ClipCompletionResult:
    """Callback outcome, separate from completion of the clip's nodes."""

    status: Literal["succeeded", "failed"]
    error: str | None = None
    error_type: str | None = None


class PlanExecutor:
    def __init__(self, plan: ExecutionPlan, *, select_ready: ReadySelector | None = None,
                 model_event: ModelEventHook | None = None,
                 node_event: Callable[[NodeKey], AbstractContextManager[Any]] | None = None,
                 batching: BatchConfig | None = None) -> None:
        self.plan = plan
        self.select_ready = select_ready
        self.model_event = model_event
        self.node_event = node_event
        self.batching = BatchConfig() if batching is None else batching
        self.working_memory = WorkingMemory(self.batching.max_working_bytes)
        self.clip_errors: dict[str, ClipCompletionResult] = {}
        self._completion_attempts: set[str] = set()
        self.completion_results: dict[str, ClipCompletionResult] = {}

    @property
    def completion_attempts(self) -> frozenset[str]:
        return frozenset(self._completion_attempts)

    def run(self, *, targets: Sequence[NodeKey] | None = None,
            on_clip_complete: Callable[[str], None] | None = None,
            continue_on_error: bool = False,
            on_clip_error: Callable[[str, Exception], None] | None = None) -> dict[NodeKey, ArtifactRef]:
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
        def failed(clip_id: str, error: Exception) -> None:
            self.clip_errors[clip_id] = ClipCompletionResult("failed", str(error), type(error).__name__)
            if continue_on_error:
                self.plan.block_clip(clip_id)
                runtime.discard(tuple(key for key in required if key[0] == clip_id))
            if on_clip_error is not None:
                on_clip_error(clip_id, error)

        with self._runtime(required) as runtime:
            for clip_id in self.plan.clips:
                notify(clip_id)
            while required - self.plan.statuses.keys():
                with span("planning", operation="select_ready"):
                    ready = self.plan.ready(required)
                    if not ready:
                        raise ValueError("Execution plan is blocked by a failed component")
                    key = runtime.select_ready(ready) if self.select_ready is None else self.select_ready(ready)
                    if key not in ready:
                        raise ValueError(f"Selected component is not ready: {key}")
                    event = nullcontext() if self.node_event is None else self.node_event(key)
                    peers = self._batch_peers(key, ready, runtime)
                if peers:
                    with event:
                        self._execute_batch(peers, runtime, notify, failed, continue_on_error=continue_on_error)
                    continue
                try:
                    with event:
                        self._execute(key, runtime, required=required, continue_on_error=continue_on_error)
                    notify(key[0])
                except Exception as error:
                    failed(key[0], error)
                    if not continue_on_error or not runtime.healthy or not is_clip_failure(error):
                        raise
        if not continue_on_error and required - references.keys():
            raise ValueError("Execution plan is blocked by a failed component")
        return references

    def _batch_adapter(self, key: NodeKey) -> BatchAdapter | None:
        node = self.plan.nodes[key]
        if node.source == "execute" and isinstance(node.component, InferenceComponent):
            adapter: BatchAdapter | None = node.component.batch_adapter()
            return adapter
        return None

    def _batch_peers(self, key: NodeKey, ready: tuple[NodeKey, ...], runtime: InferenceRuntime
                     ) -> dict[NodeKey, BatchAdapter]:
        if not self.batching.enabled or (adapter := self._batch_adapter(key)) is None:
            return {}
        affinity = runtime.batch_affinity(key, adapter.batch_alias)
        counts: Counter[str] = Counter({key[0]: 1})
        peers = [(0, key, adapter)]
        for candidate in ready:
            if candidate == key or counts[candidate[0]] >= self.batching.max_sources:
                continue
            if (other := self._batch_adapter(candidate)) is None:
                continue
            if (type(other) is type(adapter)
                    and (other.batch_model, other.batch_alias) == (adapter.batch_model, adapter.batch_alias)
                    and runtime.batch_affinity(candidate, other.batch_alias) == affinity):
                peers.append((counts[candidate[0]], candidate, other))
                counts[candidate[0]] += 1
        peers.sort(key=lambda peer: peer[0])
        return {candidate: other for _, candidate, other in peers[:self.batching.max_sources]}

    def _execute_batch(self, peers: dict[NodeKey, BatchAdapter], runtime: InferenceRuntime,
                       notify: Callable[[str], None], failed: Callable[[str, Exception], None],
                       *, continue_on_error: bool) -> None:
        plan = self.plan
        starts: dict[NodeKey, float] = {}
        decisions: dict[NodeKey, NodeDecision] = {}
        jobs: list[BatchJob] = []

        def fail(key: NodeKey, error: Exception) -> None:
            if key not in plan.references:
                plan.statuses[key] = "failed"
            plan.seconds[key] = time.monotonic() - starts[key]
            failed(key[0], error)
            if not continue_on_error or not runtime.healthy or not is_clip_failure(error):
                raise error

        def complete(key: NodeKey, output: Any) -> None:
            with span("node", operation="batch_complete", source=plan.nodes[key].source) as event:
                plan.nodes[key].validate_output(output)
                runtime.discard((key,))
                self._publish(key, output, decisions[key])
                event["outcome"] = plan.statuses[key]
            plan.seconds[key] = time.monotonic() - starts[key]
            notify(key[0])

        try:
            with runtime.batch(tuple(peers)) as borrowers:
                for key, adapter in peers.items():
                    if key in plan.statuses:
                        continue
                    starts[key] = time.monotonic()
                    plan.active_node = key
                    node = plan.nodes[key]
                    memory = SourceMemory(self.working_memory)
                    try:
                        with recording_context(clip_id=key[0], camera_id=node.context.camera_id, node=key[1]), span(
                            "node", operation="batch_open", source=node.source,
                        ) as event:
                            decisions[key] = decision = plan.decide(key)
                            if decision.action != "execute":
                                self._load_reused(key, decision, memory)
                                memory.close()
                                runtime.discard((key,))
                                self._publish(key, None, decision)
                                event["outcome"] = plan.statuses[key]
                                plan.seconds[key] = time.monotonic() - starts[key]
                                notify(key[0])
                                continue
                            source = adapter.open_batch(key[0], self._assemble(key, decision, memory), borrowers[key], memory)
                            jobs.append(BatchJob(key, node.context.camera_id, adapter, borrowers[key], source, memory))
                            del source
                    except Exception as error:
                        memory.close()
                        for job in jobs:
                            if job.key[0] == key[0]:
                                job.close()
                        fail(key, error)
                for job in jobs:
                    if job.key in plan.statuses:
                        job.close()
                drain_batch_jobs([job for job in jobs if job.source is not None], self.batching,
                                 self.working_memory, complete=complete, fail=fail)
        finally:
            for job in jobs:
                job.close()
            plan.active_node = None

    @measured("planning")
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
            return self._execute(key, runtime, required=frozenset({key}))

    def _execute(self, key: NodeKey, runtime: InferenceRuntime, *, required: frozenset[NodeKey] | None = None,
                 continue_on_error: bool = False) -> ArtifactRef:
        if key in self.plan.references:
            return self._execute_recorded(key, runtime, required=required, continue_on_error=continue_on_error)
        node = self.plan.nodes[key]
        with recording_context(clip_id=key[0], camera_id=node.context.camera_id, node=key[1]), span("node", source=node.source) as event:
            try:
                return self._execute_recorded(key, runtime, required=required, continue_on_error=continue_on_error)
            finally:
                event["outcome"] = self.plan.statuses.get(key, "failed")

    def _execute_recorded(self, key: NodeKey, runtime: InferenceRuntime, *, required: frozenset[NodeKey] | None,
                          continue_on_error: bool) -> ArtifactRef:
        plan = self.plan
        if key in plan.references:
            adopted: ArtifactRef = plan.references[key]
            return adopted
        if key not in plan.ready(required):
            raise ValueError(f"Component is not ready: {key}")
        plan.active_node = key
        started = time.monotonic()
        memory = SourceMemory(self.working_memory) if self.batching.enabled else None
        try:
            with runtime.node(key, continue_on_error=continue_on_error) as models:
                output, decision = self._execute_node(key, models, memory)
            reference = self._publish(key, output, decision)
            plan.active_node = None
            return reference
        except BaseException:
            plan.statuses[key] = "failed"
            raise
        finally:
            if memory is not None:
                memory.close()
            plan.seconds[key] = time.monotonic() - started

    def _publish(self, key: NodeKey, output: Any, decision: NodeDecision) -> ArtifactRef:
        plan = self.plan
        node, store = plan.nodes[key], plan.clips[key[0]].store
        reference: ArtifactRef | None = decision.reference
        if decision.action == "execute":
            reference = store.publish(key[1], output, ArtifactCodec(node.io.output_type),
                schema=node.io.output_schema, version=node.io.version, identity=decision.identity,
                dependencies=decision.dependencies, provenance={"origin": "component", "implementation": node.implementation})
        if reference is None:
            raise RuntimeError("Completed component has no artifact reference")
        # Unload/recorder/cleanup must succeed before publication.
        plan.references[key] = reference
        plan.statuses[key] = {"execute": "executed", "load": "loaded", "cached": "cached"}[decision.action]
        return reference

    def _load_reused(self, key: NodeKey, decision: NodeDecision, memory: SourceMemory | None) -> None:
        node, store = self.plan.nodes[key], self.plan.clips[key[0]].store
        if decision.reference is None:
            raise RuntimeError("Reused component has no artifact reference")
        node.validate_output(self._read_artifact(store, decision.reference, node.io.output_type, memory))

    @staticmethod
    def _read_artifact(store: ClipStore, reference: ArtifactRef, output_type: type[Any],
                       memory: SourceMemory | None) -> Any:
        codec = ArtifactCodec(output_type)
        if memory is None:
            return store.load(reference, codec)
        return store.load(reference, codec, reserve=memory.reserve)

    def _assemble(self, key: NodeKey, decision: NodeDecision, memory: SourceMemory | None) -> Any:
        plan = self.plan
        node, store = plan.nodes[key], plan.clips[key[0]].store
        artifacts = {port: self._read_artifact(store, ref, plan.nodes[plan.dependencies[key][port]].io.output_type, memory)
                     for port, ref in decision.dependencies.items()}
        with span("preprocess", operation="input_assembly"):
            inputs = node.assembler.assemble(node.context, artifacts)
        if not isinstance(inputs, node.io.input_type):
            raise TypeError(f"Assembler returned wrong input type for {key[1]}")
        return inputs

    def _execute_node(self, key: NodeKey, models: ModelBorrower, memory: SourceMemory | None) -> tuple[Any, NodeDecision]:
        plan = self.plan
        node = plan.nodes[key]
        decision = plan.decide(key)
        if decision.action != "execute":
            self._load_reused(key, decision, memory)
            return None, decision
        inputs = self._assemble(key, decision, memory)
        estimate = getattr(node.component, "estimate_working_bytes", None)
        if memory is not None and callable(estimate):
            memory.reserve(estimate(inputs))
        with span("postprocess", operation=key[1].partition("/")[0]):
            output = (node.component.process_with_models(inputs, models)
                      if isinstance(node.component, InferenceComponent) else node.component.process(inputs))
        node.validate_output(output)
        return output, decision

    def output(self, key: NodeKey) -> Any:
        return self.plan.clips[key[0]].store.load(self.plan.references[key], ArtifactCodec(self.plan.nodes[key].io.output_type))

    def finalize_clip(self, clip_id: str, *, memory: SourceMemory | None = None) -> tuple[SceneResult, Path]:
        """Export a completed clip independently of any other clip's outcome."""
        plan = self.plan
        if not plan.clip_complete(clip_id) or (clip_id, "scene_assembly") not in plan.references:
            raise ValueError(f"Cannot finalize an incomplete clip: {clip_id}")
        key = (clip_id, "scene_assembly")
        scene = self._read_artifact(plan.clips[clip_id].store, plan.references[key], plan.nodes[key].io.output_type, memory)
        if not isinstance(scene, SceneResult):
            raise TypeError("scene_assembly must produce a SceneResult")
        if memory is not None:
            memory.reserve(artifact_memory_bytes(scene))
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
