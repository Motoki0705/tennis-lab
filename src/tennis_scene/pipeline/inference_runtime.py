"""Explicit ownership of lazy inference models for one execution-plan drain."""

from __future__ import annotations

import gc
import json
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from src.tennis_scene.pipeline.artifacts import json_value
from src.utils.checksum import dual_sha256

ModelT = TypeVar("ModelT")
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
NodeKey = tuple[str, str]


@dataclass(frozen=True)
class ModelKey:
    kind: str
    weights: tuple[tuple[str, str], ...]
    architecture: str
    device: str
    dtype: str
    settings: str


@dataclass(frozen=True)
class ModelSpec(Generic[ModelT]):
    """A model declaration, containing no loaded weights or clip/camera state.

    Factories return an unloaded object when initialization needs cleanup on
    failure. Eager checkpoint readers may use the factory itself.
    """

    kind: str
    weights: Mapping[str, Path]
    architecture: str
    device: str
    dtype: str
    settings: Mapping[str, Any]
    factory: Callable[[], ModelT] = field(repr=False, compare=False)
    initialize: Callable[[ModelT], None] = field(default=lambda _: None, repr=False, compare=False)
    release: Callable[[ModelT], None] = field(default=lambda _: None, repr=False, compare=False)
    _settings_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from types import MappingProxyType

        if not all((self.kind, self.architecture, self.device, self.dtype)):
            raise ValueError("Model kind, architecture, device and dtype are required")
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        settings_json = json.dumps(json_value(self.settings), sort_keys=True, separators=(",", ":"), allow_nan=False)
        object.__setattr__(self, "_settings_json", settings_json)
        object.__setattr__(self, "settings", MappingProxyType(json.loads(settings_json)))

    @property
    def key(self) -> ModelKey:
        return self._key_with(dual_sha256)

    def _key_with(self, digest: Callable[[Path], str]) -> ModelKey:
        # Paths locate assets but do not define model equivalence.
        return ModelKey(self.kind, tuple(sorted((name, digest(path)) for name, path in self.weights.items())),
                        self.architecture, self.device, self.dtype,
                        self._settings_json)


class ModelEventHook(Protocol):
    """Matches C1's model_event context-manager API without importing C1."""

    def __call__(self, action: Literal["load", "unload"], model: str, *, weights: Mapping[str, Path],
                 device: str, dtype: str, instance: str) -> AbstractContextManager[Any]: ...


def release_inference_memory(device: str) -> None:
    """Drop cycles and idle CUDA allocations, without initializing CUDA."""
    import torch

    gc.collect()
    if torch.device(device).type == "cuda" and torch.cuda.is_initialized():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


@dataclass
class _OwnedModel:
    spec: ModelSpec[Any]
    value: Any
    instance: str


class ModelBorrower:
    """A node may borrow only its declared aliases, and cannot unload them."""

    def __init__(self, runtime: InferenceRuntime, node: NodeKey) -> None:
        self._runtime, self._node = runtime, node

    def borrow(self, alias: str) -> Any:
        return self._runtime.borrow(self._node, alias)


class InferenceRuntime:
    """Own models until their last remaining declared node finishes.

    Undecided cache candidates consume demand but never load a model. Explicit
    load-only nodes declare no demand. A context boundary always closes residual
    ownership, including on BaseException and partial-plan drains.
    """

    def __init__(self, declarations: Mapping[NodeKey, Mapping[str, ModelSpec[Any]]], *,
                 model_event: ModelEventHook | None = None,
                 cleanup: Callable[[str], None] = release_inference_memory) -> None:
        self._remaining = {node: dict(specs) for node, specs in declarations.items()}
        self._owned: dict[ModelKey, _OwnedModel] = {}
        self._keys: dict[NodeKey, dict[str, ModelKey]] = {}
        self._assets: dict[Path, tuple[tuple[int, ...], str]] = {}
        self._active: NodeKey | None = None
        self._closed = False
        self._serial = 0
        self._model_event, self._cleanup = model_event, cleanup

    @property
    def loaded_keys(self) -> frozenset[ModelKey]:
        return frozenset(self._owned)

    def keys_for(self, node: NodeKey) -> frozenset[ModelKey]:
        if node not in self._keys:
            self._keys[node] = {alias: spec._key_with(self._weight_hash) for alias, spec in self._remaining.get(node, {}).items()}
        return frozenset(self._keys[node].values())

    def _weight_hash(self, path: Path) -> str:
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._assets.get(path)
        if cached is not None and cached[0] != signature:
            raise ValueError(f"Model assets changed during the inference plan: {path}")
        if cached is None:
            cached = (signature, dual_sha256(path))
            self._assets[path] = cached
        return cached[1]

    def select_ready(self, ready: tuple[NodeKey, ...]) -> NodeKey:
        """Drain resident models' ready peers, CPU dependencies, then declaration order."""
        resident = self.loaded_keys
        return min(ready, key=lambda node: (
            0 if resident & self.keys_for(node) else 1 if not self.keys_for(node) else 2,
            len(self.keys_for(node) - resident), ready.index(node)))

    def __enter__(self) -> InferenceRuntime:
        if self._closed:
            raise RuntimeError("Inference runtime is closed")
        return self

    def __exit__(self, kind: Any, error: Any, tb: Any) -> None:
        if error is not None:
            traceback.clear_frames(tb)
        self.close()

    @contextmanager
    def node(self, node: NodeKey) -> Iterator[ModelBorrower]:
        if self._closed or self._active is not None or node not in self._remaining:
            raise RuntimeError(f"Cannot enter inference node: {node}")
        self._active = node
        try:
            yield ModelBorrower(self, node)
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            self.close()
            raise
        else:
            self._remaining.pop(node)
            self._keys.pop(node, None)
            for key in tuple(self._owned):
                if not any(key in self.keys_for(other) for other in self._remaining):
                    self._release(key)
        finally:
            self._active = None

    def _event(self, action: Literal["load", "unload"], owned: _OwnedModel) -> AbstractContextManager[Any]:
        spec = owned.spec
        return nullcontext() if self._model_event is None else self._model_event(
            action, spec.kind, weights=spec.weights, device=spec.device, dtype=spec.dtype, instance=owned.instance)

    def borrow(self, node: NodeKey, alias: str) -> Any:
        if self._closed or self._active != node:
            raise RuntimeError("Models may be borrowed only during their node")
        if alias not in self._remaining[node]:
            raise ValueError(f"Undeclared inference model: {node}.{alias}")
        spec = self._remaining[node][alias]
        # Pin the remaining plan before loading; reused declarations may be
        # used by another drain later, with a different weight file version.
        for other in self._remaining:
            self.keys_for(other)
        key = self._keys[node][alias]
        if spec._key_with(self._weight_hash) != key:
            raise ValueError(f"Model assets changed during the inference plan: {node}.{alias}")
        if key not in self._owned:
            self._serial += 1
            owned = _OwnedModel(spec, None, f"{id(self)}:{self._serial}")
            try:
                with self._event("load", owned):
                    # Recheck content at the load boundary: small same-size
                    # rewrites can share stat timestamps on coarse filesystems.
                    if spec.key != key:
                        raise ValueError(f"Model assets changed before loading: {node}.{alias}")
                    owned.value = spec.factory()
                    self._owned[key] = owned  # Own partially initialized models too.
                    spec.initialize(owned.value)
                    if spec._key_with(self._weight_hash) != key:
                        raise ValueError(f"Model assets changed while loading: {node}.{alias}")
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                if key in self._owned:
                    self._release(key)
                else:
                    self._cleanup(spec.device)
                raise
        return self._owned[key].value

    def _release(self, key: ModelKey) -> None:
        owned = self._owned.pop(key)
        released = False

        def release() -> None:
            nonlocal released
            released = True
            try:
                owned.spec.release(owned.value)
            finally:
                owned.value = None
                self._cleanup(owned.spec.device)

        try:
            with self._event("unload", owned):
                release()
        finally:
            if not released:
                release()  # Recorder failures must not prevent model cleanup.

    def close(self) -> None:
        self._closed = True
        errors: list[BaseException] = []
        for key in tuple(self._owned):
            try:
                self._release(key)
            except BaseException as error:
                traceback.clear_frames(error.__traceback__)
                errors.append(error)
        self._remaining.clear()
        self._keys.clear()
        self._assets.clear()
        if errors:
            raise BaseExceptionGroup("Inference model cleanup failed", errors)


class InferenceComponent(ABC, Generic[InputT, OutputT]):
    """The direct component API uses the same ownership as a one-node plan."""

    @abstractmethod
    def model_specs(self) -> Mapping[str, ModelSpec[Any]]: ...

    @abstractmethod
    def process_with_models(self, inputs: InputT, models: ModelBorrower) -> OutputT: ...

    def process(self, inputs: InputT) -> OutputT:
        node = ("standalone", type(self).__name__)
        with InferenceRuntime({node: self.model_specs()}) as runtime, runtime.node(node) as models:
            return self.process_with_models(inputs, models)
