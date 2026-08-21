"""Unit tests for explicit shared training-model compilation."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from src.tasks.base.configuration import CompileConfig
from src.tasks.base.model_io import BoundModelIO
from src.tasks.base.training.compilation import (
    CompilationTargetError,
    compile_modules,
)

pytestmark = pytest.mark.unit


def _config(*, enabled: bool = True) -> CompileConfig:
    return CompileConfig(
        enabled=enabled,
        backend="inductor",
        mode="reduce-overhead",
        fullgraph=False,
        dynamic=False,
    )


def test_disabled_compile_does_not_require_targets() -> None:
    assert compile_modules({}, _config(enabled=False)) == ()


def test_compile_preserves_identity_state_dict_and_bound_model_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []

    def fake_compile(callable_: Any, **kwargs: object) -> Any:
        compile_calls.append(dict(kwargs))
        return callable_

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = nn.Linear(4, 2)
    model_identity = id(model)
    state_keys = tuple(model.state_dict())
    bound = BoundModelIO(model=model, adapter=cast(Any, object()))

    compiled = compile_modules(
        {"model": model, "model_io_alias": bound.model},
        _config(),
    )

    assert compiled == ("model",)
    assert compile_calls == [
        {
            "backend": "inductor",
            "mode": "reduce-overhead",
            "fullgraph": False,
            "dynamic": False,
        }
    ]
    assert id(model) == model_identity
    assert bound.model is model
    assert tuple(model.state_dict()) == state_keys


def test_primary_and_discriminator_are_compiled_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[Any] = []

    def fake_compile(callable_: Any, **kwargs: object) -> Any:
        del kwargs
        compile_calls.append(callable_)
        return callable_

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = nn.Linear(2, 2)
    discriminator = nn.Linear(2, 1)

    compiled = compile_modules(
        {"model": model, "discriminator": discriminator},
        _config(),
    )

    assert compiled == ("model", "discriminator")
    assert len(compile_calls) == 2


def test_enabled_compile_requires_at_least_one_target() -> None:
    with pytest.raises(CompilationTargetError, match="at least one"):
        compile_modules({}, _config())


@pytest.mark.parametrize("name", ["", " model", "model "])
def test_compile_rejects_invalid_target_name(name: str) -> None:
    with pytest.raises(CompilationTargetError, match="names"):
        compile_modules({name: nn.Linear(1, 1)}, _config())


def test_compile_rejects_non_module_target() -> None:
    with pytest.raises(CompilationTargetError, match="must be an nn.Module"):
        compile_modules({"model": cast("nn.Module", object())}, _config())
