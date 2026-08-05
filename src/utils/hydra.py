"""Typed wrapper around :func:`hydra.main`.

``hydra.main`` is an untyped decorator factory, so every CLI entry point used to
re-declare a private ``hydra_main`` shim purely to keep mypy/pyright happy. This
module is the single shared implementation; import it instead of copying the
wrapper into each script.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import Any, TypeVar, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.configuration.paths import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

R = TypeVar("R")
HydraEntrypoint = Callable[[DictConfig], R]
HydraCLI = Callable[[], R]
BoundaryValidator = Callable[[DictConfig], None]

_BOUNDARY_VALIDATORS: dict[str, BoundaryValidator] = {}
HYDRA_PATH_RESOLVER = "tennis_lab_path"

__all__ = [
    "BoundaryValidator",
    "HydraCLI",
    "HydraEntrypoint",
    "HYDRA_PATH_RESOLVER",
    "RegisteredBoundaryValidator",
    "hydra_main",
    "register_boundary_validator",
    "registered_boundary_validators",
    "resolve_hydra_path",
    "validate_boundary",
]


@dataclass(frozen=True, slots=True)
class RegisteredBoundaryValidator:
    """One runtime registry entry bound to the callable that is invoked."""

    name: str
    callable_symbol: str
    validator: BoundaryValidator


def _callable_symbol(function: BoundaryValidator) -> str:
    module = getattr(function, "__module__", None)
    qualified_name = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not module:
        raise TypeError("Boundary validators must declare a non-empty __module__.")
    if not isinstance(qualified_name, str) or not qualified_name:
        raise TypeError("Boundary validators must declare a non-empty __qualname__.")
    return f"{module}.{qualified_name}"


def register_boundary_validator(name: str, validator: BoundaryValidator) -> None:
    """Register the sole strict validator for a named Hydra runtime boundary."""
    if not name:
        raise ValueError("Hydra validation boundary name must not be empty.")
    if name in _BOUNDARY_VALIDATORS:
        raise ValueError(f"Hydra validation boundary {name!r} is already registered.")
    _callable_symbol(validator)
    _BOUNDARY_VALIDATORS[name] = validator


def registered_boundary_validators() -> MappingProxyType[
    str, RegisteredBoundaryValidator
]:
    """Return an immutable snapshot including each actual callable binding."""
    return MappingProxyType(
        {
            name: RegisteredBoundaryValidator(
                name=name,
                callable_symbol=_callable_symbol(validator),
                validator=validator,
            )
            for name, validator in _BOUNDARY_VALIDATORS.items()
        }
    )


def validate_boundary(name: str, config: DictConfig) -> None:
    """Invoke a registered boundary validator, failing if none is registered."""
    try:
        validator = _BOUNDARY_VALIDATORS[name]
    except KeyError as error:
        raise RuntimeError(
            f"No strict configuration validator is registered for Hydra boundary {name!r}."
        ) from error
    validator(config)


def hydra_main(
    *args: Any,
    validation_boundary: str | None = None,
    **kwargs: Any,
) -> Callable[[HydraEntrypoint[R]], HydraCLI[R]]:
    """Wrap :func:`hydra.main`, optionally validating before the user function."""

    def decorate(function: HydraEntrypoint[R]) -> HydraCLI[R]:
        target = function
        if validation_boundary is not None:

            @wraps(function)
            def validated(config: DictConfig) -> R:
                validate_boundary(validation_boundary, config)
                return function(config)

            target = validated
        hydra_decorated = hydra.main(*args, **kwargs)(target)
        return cast(HydraCLI[R], hydra_decorated)

    return decorate


def resolve_hydra_path(role: str, *parts: str) -> str:
    """Resolve Hydra metadata paths using the shared runtime path authority.

    The sole registered resolver has two explicit forms::

        ${tennis_lab_path:project}
        ${tennis_lab_path:output,${paths.output_root},${run.output_dir}}

    ``project`` is derived directly from :data:`PROJECT_ROOT`. ``output``
    treats its first part as the configured output root and resolves all
    remaining parts through :class:`PathResolver`. No process-CWD fallback or
    second project-root calculation exists.
    """
    if type(role) is not str:
        raise ValueError("Hydra path role must be exactly a string.")
    if any(type(part) is not str or not part for part in parts):
        raise ValueError("Hydra path parts must be non-empty strings.")
    try:
        path_role = PathRole(role)
    except ValueError as error:
        raise ValueError(
            "Hydra path resolver supports only the explicit project/output roles; "
            f"got {role!r}."
        ) from error
    if path_role not in {PathRole.PROJECT, PathRole.OUTPUT}:
        raise ValueError(
            "Hydra path resolver supports only the explicit project/output roles; "
            f"got {role!r}."
        )

    if path_role is PathRole.PROJECT:
        role_root = "."
        relative_parts: Sequence[str] = parts
    else:
        if not parts:
            raise ValueError("Hydra output resolution requires paths.output_root.")
        role_root, *remaining_parts = parts
        relative_parts = remaining_parts

    roots_mapping = {
        f"{candidate_role.value}_root": (
            role_root if candidate_role is path_role else "."
        )
        for candidate_role in PathRole
    }
    roots = RuntimePathRoots.from_mapping(
        roots_mapping,
        repository_root=PROJECT_ROOT,
    )
    resolver = PathResolver(roots)
    if not relative_parts:
        return str(roots.root(path_role))
    return str(resolver.resolve(path_role, *relative_parts))


OmegaConf.register_new_resolver(HYDRA_PATH_RESOLVER, resolve_hydra_path)
