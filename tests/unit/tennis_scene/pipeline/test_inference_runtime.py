"""Content equivalence and ownership transitions without real model assets."""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime, ModelSpec


def spec(tmp_path: Path, **changes: Any) -> ModelSpec[Any]:
    weights = tmp_path / "model.bin"
    weights.write_bytes(b"model-a")
    return replace(ModelSpec("ball", {"checkpoint": weights}, "convnext", "cpu", "float32",
                             {"resize": [360, 640], "threshold": .5}, object), **changes)


@pytest.mark.parametrize("change", ["kind", "weights", "architecture", "device", "dtype", "resize", "threshold"])
def test_different_effective_model_settings_never_share(tmp_path: Path, change: str) -> None:
    original = spec(tmp_path)
    if change == "weights":
        other = tmp_path / "other.bin"
        other.write_bytes(b"model-b")
        updated = replace(original, weights={"checkpoint": other})
    elif change in {"resize", "threshold"}:
        updated = replace(original, settings={**original.settings, change: [720, 1280] if change == "resize" else .7})
    else:
        changes: dict[str, Any] = {change: {"kind": "court", "architecture": "x3d_csn_dpt", "device": "cuda:1", "dtype": "bfloat16"}[change]}
        updated = replace(original, **changes)
    assert original.key != updated.key


def test_content_not_path_and_mapping_order_define_equivalence(tmp_path: Path) -> None:
    original = spec(tmp_path)
    copy = tmp_path / "copy.bin"
    copy.write_bytes(b"model-a")
    updated = replace(original, weights={"checkpoint": copy}, settings={"threshold": .5, "resize": [360, 640]})
    assert original.key == updated.key
    assert original.key.weights == (("checkpoint", "73f95cf180a19624e4be9a711fd53a90dadfffa410cc9cd2ba1999454a5b99b8"),)


def test_key_snapshots_settings_and_rehashes_replaced_file(tmp_path: Path) -> None:
    settings = {"resize": [360, 640]}
    original = spec(tmp_path, settings=settings)
    settings["resize"][0] = 1
    assert '360' in original.key.settings
    previous = original.key
    original.weights["checkpoint"].write_bytes(b"model-z")
    assert original.key != previous


class Resource:
    def __init__(self) -> None:
        self.loaded = False


def test_lazy_borrow_last_demand_and_recorded_actual_lifetimes(tmp_path: Path) -> None:
    events: list[tuple[str, str]] = []
    references: list[weakref.ReferenceType[Resource]] = []
    instances: list[str] = []

    @contextmanager
    def event(action: Any, model: str, **attributes: Any) -> Iterator[None]:
        assert attributes["device"] == "cpu" and attributes["dtype"] == "float32"
        assert attributes["weights"] == {"checkpoint": tmp_path / "model.bin"}
        events.append((action, "start"))
        instances.append(attributes["instance"])
        yield
        events.append((action, "end"))

    def create() -> Resource:
        resource = Resource()
        references.append(weakref.ref(resource))
        return resource

    request = spec(tmp_path, factory=create, initialize=lambda r: setattr(r, "loaded", True),
                   release=lambda r: setattr(r, "loaded", False))
    copy = tmp_path / "same-weights.bin"
    copy.write_bytes(b"model-a")
    a, b, cached = ("clip", "a"), ("clip", "b"), ("clip", "cached")
    with InferenceRuntime({a: {"model": request}, b: {"different_alias": replace(request, weights={"checkpoint": copy})}, cached: {"model": request}},
                          model_event=event, cleanup=lambda _: None) as runtime:
        assert not events
        with runtime.node(a) as borrowed:
            assert borrowed.borrow("model").loaded
            assert borrowed.borrow("model") is references[0]()
        assert runtime.select_ready((b, cached)) == b  # Same content at another path is a resident peer.
        with runtime.node(b) as borrowed:
            assert borrowed.borrow("different_alias") is references[0]()
        assert len(references) == 1 and runtime.loaded_keys
        with runtime.node(cached):
            pass  # Reuse an artifact without borrowing its declared model.
        assert not runtime.loaded_keys and references[0]() is None
    assert events == [("load", "start"), ("load", "end"), ("unload", "start"), ("unload", "end")]
    assert len(set(instances)) == 1


@pytest.mark.parametrize("use_optional", [False, True])
def test_unused_optional_assets_are_not_required_for_scheduling_or_cleanup(tmp_path: Path, use_optional: bool) -> None:
    active = spec(tmp_path)
    absent = replace(active, weights={"checkpoint": tmp_path / "not-installed.bin"})
    a, b = ("clip", "active"), ("clip", "optional")
    with InferenceRuntime({a: {"model": active}, b: {"optional": absent}}, cleanup=lambda _: None) as runtime:
        assert runtime.select_ready((a, b)) == a
        with runtime.node(a) as models:
            models.borrow("model")
        assert not runtime.loaded_keys
        assert runtime.select_ready((b,)) == b
        if use_optional:
            with pytest.raises(FileNotFoundError), runtime.node(b) as models:
                models.borrow("optional")
        else:
            with runtime.node(b):
                pass


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_exception_clears_owned_weights_even_with_retained_traceback(tmp_path: Path, failure: type[BaseException]) -> None:
    references: list[weakref.ReferenceType[Resource]] = []
    releases: list[bool] = []

    def create() -> Resource:
        resource = Resource()
        references.append(weakref.ref(resource))
        return resource

    request = spec(tmp_path, factory=create, release=lambda r: releases.append(True))
    node, later = ("clip", "fails"), ("clip", "later")

    def process(models: Any) -> None:
        resource = models.borrow("model")
        assert resource is not None
        raise failure("inference stopped")

    with pytest.raises(failure, match="inference stopped") as caught, \
            InferenceRuntime({node: {"model": request}, later: {"model": request}}, cleanup=lambda _: None) as runtime, \
            runtime.node(node) as borrowed:
        process(borrowed)
    assert caught.value.__traceback__ is not None
    assert releases == [True] and not runtime.loaded_keys
    assert references[0]() is None


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_partial_initialization_is_owned_and_released(tmp_path: Path, failure: type[BaseException]) -> None:
    releases: list[bool] = []
    def initialize(resource: Resource) -> None:
        resource.loaded = True
        raise failure("bad weights")

    request = spec(tmp_path, factory=Resource, initialize=initialize,
                   release=lambda r: releases.append(r.loaded))
    key = ("clip", "load_fails")
    with pytest.raises(failure, match="bad weights"), InferenceRuntime({key: {"model": request}}, cleanup=lambda _: None) as runtime, runtime.node(key) as models:
        models.borrow("model")
    assert releases == [True] and not runtime.loaded_keys


def test_cleanup_failure_still_releases_other_models(tmp_path: Path) -> None:
    releases: list[str] = []
    references: list[weakref.ReferenceType[Resource]] = []
    def create() -> Resource:
        resource = Resource()
        references.append(weakref.ref(resource))
        return resource

    def broken(resource: object) -> None:
        releases.append("first")
        raise RuntimeError("cannot unload")
    first = spec(tmp_path, factory=create, release=broken)
    second = replace(first, kind="other", release=lambda _: releases.append("second"))
    key, later = ("clip", "node"), ("clip", "later")
    runtime = InferenceRuntime({key: {"a": first, "b": second}, later: {"a": first, "b": second}}, cleanup=lambda _: None)
    with runtime.node(key) as models:
        models.borrow("a")
        models.borrow("b")
    with pytest.raises(BaseExceptionGroup, match="cleanup failed") as caught:
        runtime.close()
    assert len(caught.value.exceptions) == 1
    assert releases == ["first", "second"] and not runtime.loaded_keys
    assert all(reference() is None for reference in references)
    runtime.close()


def test_borrower_cannot_access_other_nodes_or_undeclared_models(tmp_path: Path) -> None:
    key = ("clip", "node")
    with InferenceRuntime({key: {"allowed": spec(tmp_path)}}, cleanup=lambda _: None) as runtime:
        with runtime.node(key) as models, pytest.raises(ValueError, match="Undeclared"):
            models.borrow("other")
        with pytest.raises(RuntimeError, match="only during their node"):
            models.borrow("allowed")


def test_weight_change_after_plan_pinning_fails_instead_of_mixing_models(tmp_path: Path) -> None:
    request = spec(tmp_path)
    first, next_node = ("clip", "first"), ("clip", "next")
    with InferenceRuntime({first: {"model": request}, next_node: {"model": request}}, cleanup=lambda _: None) as runtime:
        with runtime.node(first) as models:
            models.borrow("model")
        request.weights["checkpoint"].write_bytes(b"changed-model")
        with pytest.raises(ValueError, match="assets changed"), runtime.node(next_node) as models:
            models.borrow("model")
        assert not runtime.loaded_keys


def test_recorder_failure_cannot_skip_unloading(tmp_path: Path) -> None:
    released: list[bool] = []

    @contextmanager
    def event(action: Any, model: str, **attributes: Any) -> Iterator[None]:
        if action == "unload":
            raise RuntimeError("recorder failed")
        yield

    request = spec(tmp_path, release=lambda _: released.append(True))
    node = ("clip", "node")
    with pytest.raises(RuntimeError, match="recorder failed"), \
            InferenceRuntime({node: {"model": request}}, model_event=event, cleanup=lambda _: None) as runtime, \
            runtime.node(node) as models:
        models.borrow("model")
    assert released == [True] and not runtime.loaded_keys
