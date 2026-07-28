"""Strict registries for configuration-selectable dataset algorithms."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

AlgorithmT = TypeVar("AlgorithmT")


@dataclass(frozen=True)
class AlgorithmDefinition(Generic[AlgorithmT]):
    """One named implementation that may be selected from configuration."""

    name: str
    implementation: AlgorithmT
    description: str

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ValueError("Algorithm names must be non-empty and normalized.")
        if not self.description.strip():
            raise ValueError(f"Algorithm {self.name!r} requires a description.")


class AlgorithmRegistry(Generic[AlgorithmT]):
    """Resolve only explicitly registered algorithms without a fallback."""

    def __init__(
        self,
        *,
        namespace: str,
        definitions: Iterable[AlgorithmDefinition[AlgorithmT]],
    ) -> None:
        if not namespace or namespace.strip() != namespace:
            raise ValueError("Algorithm registry namespaces must be normalized.")
        indexed: dict[str, AlgorithmDefinition[AlgorithmT]] = {}
        for definition in definitions:
            if definition.name in indexed:
                raise ValueError(
                    f"Duplicate algorithm {definition.name!r} in {namespace!r}."
                )
            indexed[definition.name] = definition
        if not indexed:
            raise ValueError(f"Algorithm registry {namespace!r} must not be empty.")
        self._namespace = namespace
        self._definitions: Mapping[str, AlgorithmDefinition[AlgorithmT]] = (
            MappingProxyType(indexed)
        )

    @property
    def namespace(self) -> str:
        """Return the configuration namespace governed by this registry."""
        return self._namespace

    def names(self) -> tuple[str, ...]:
        """Return the deterministic set of selectable algorithm names."""
        return tuple(sorted(self._definitions))

    def definitions(self) -> tuple[AlgorithmDefinition[AlgorithmT], ...]:
        """Return definitions ordered by their configuration name."""
        return tuple(self._definitions[name] for name in self.names())

    def resolve(self, name: str) -> AlgorithmT:
        """Resolve an exact name or fail with the available choices."""
        try:
            return self._definitions[name].implementation
        except KeyError as error:
            choices = ", ".join(self.names())
            raise ValueError(
                f"Unknown {self._namespace} algorithm {name!r}; "
                f"available choices: {choices}."
            ) from error

    def describe(self, name: str) -> AlgorithmDefinition[AlgorithmT]:
        """Return metadata for one exact algorithm name."""
        try:
            return self._definitions[name]
        except KeyError as error:
            choices = ", ".join(self.names())
            raise ValueError(
                f"Unknown {self._namespace} algorithm {name!r}; "
                f"available choices: {choices}."
            ) from error
