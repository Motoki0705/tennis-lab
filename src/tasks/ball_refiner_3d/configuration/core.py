"""Strict parsing shared by task configuration boundaries."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any, TypeVar, cast, get_type_hints

from omegaconf import DictConfig, OmegaConf

from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

T = TypeVar("T")


def _epoch_count(steps: int, interval: int) -> int:
    if type(steps) is not int or type(interval) is not int or steps < 1 or interval < 1:
        raise ValueError("Require positive integer update count and epoch interval")
    return (steps + interval - 1) // interval


def _epoch_index(step: int, interval: int) -> int:
    if type(step) is not int or type(interval) is not int or step < 0 or interval < 1:
        raise ValueError("Require a nonnegative update index and positive interval")
    return step // interval


for _name, _resolver in (
    ("refiner_epochs", _epoch_count),
    ("refiner_epoch_index", _epoch_index),
):
    if not OmegaConf.has_resolver(_name):
        OmegaConf.register_new_resolver(_name, _resolver)


def parse_section(cls: type[T], raw: dict[str, Any]) -> T:
    """Reject missing/unknown fields and nonfinite numbers before constructing."""
    expected = {field.name for field in fields(cast(Any, cls))}
    if set(raw) != expected:
        raise ValueError(
            f"{cls.__name__}: missing={expected - set(raw)}, unknown={set(raw) - expected}"
        )
    types = get_type_hints(cls)
    for key, value in raw.items():
        allowed = (int, float) if types[key] is float else (types[key],)
        if type(value) not in allowed:
            raise TypeError(
                f"Invalid type for {cls.__name__}.{key}: {type(value).__name__}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Nonfinite {cls.__name__}.{key}")
    return cls(**raw)


def resolved_config(
    config: DictConfig, expected: set[str]
) -> tuple[dict[str, Any], PathResolver]:
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(
            f"Require exactly the configuration sections {sorted(expected)}"
        )
    raw = cast(dict[str, Any], raw)
    roots = RuntimePathRoots.from_mapping(raw["paths"], repository_root=PROJECT_ROOT)
    return raw, PathResolver(roots)
