"""Typed augmentation registry for ground-court procedural samples.

The initial experiment registers only ``identity``.  Augmentations receive a
small immutable rendered sample and return a new one, making future blur,
dropout, noise, and warp implementations independently testable and avoiding
an augmentation name silently selecting a different operation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AugmentableGroundCourtSample:
    """Image and geometry payload passed through an augmentation."""

    image: Tensor  # [1,H,W] binary or augmented line evidence
    keypoints: Tensor  # [M,14,2], output pixels
    visibility: Tensor  # [M,14], bool
    centers: Tensor  # [M,2], output pixels
    instance_ids: Tensor  # [M], int64

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[0] != 1:
            raise ValueError("Augmentable image must have shape [1,H,W].")
        if self.keypoints.ndim != 3 or self.keypoints.shape[-2:] != (14, 2):
            raise ValueError("Augmentable keypoints must have shape [M,14,2].")
        count = self.keypoints.shape[0]
        if self.visibility.shape != (count, 14) or self.visibility.dtype != torch.bool:
            raise ValueError("Augmentable visibility must have shape [M,14] and bool dtype.")
        if self.centers.shape != (count, 2):
            raise ValueError("Augmentable centers must have shape [M,2].")
        if self.instance_ids.shape != (count,) or self.instance_ids.dtype != torch.long:
            raise ValueError("Augmentable instance_ids must be int64 [M].")


class GroundCourtAugmentation(Protocol):
    """Callable augmentation contract used by the procedural dataset."""

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample: ...


@dataclass(frozen=True, slots=True)
class GroundCourtAugmentationConfig:
    """Name and typed parameter mapping for one registered augmentation."""

    name: str = "identity"
    params: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("augmentation name must be a non-empty string.")
        if not isinstance(self.params, Mapping):
            raise TypeError("augmentation params must be a mapping.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> GroundCourtAugmentationConfig:
        """Parse ``{'type': ..., 'params': {...}}`` with no unknown fields.

        ``name`` and ``type`` are equivalent spellings; a config must select
        exactly one of them.
        """

        if not isinstance(value, Mapping):
            raise TypeError("augmentation config must be a mapping.")
        unknown = set(value) - {"name", "type", "params"}
        if unknown:
            raise ValueError(f"Unknown augmentation config fields: {sorted(unknown)}")
        if "name" in value and "type" in value:
            raise ValueError("augmentation config cannot define both name and type.")
        name = value.get("name", value.get("type", "identity"))
        params = value.get("params", {})
        if not isinstance(name, str):
            raise TypeError("augmentation config name must be a string.")
        if not isinstance(params, Mapping):
            raise TypeError("augmentation config params must be a mapping.")
        return cls(name=name, params=params)


@dataclass(frozen=True, slots=True)
class IdentityAugmentation:
    """Baseline no-op augmentation."""

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        _ = rng
        return sample


@dataclass(frozen=True, slots=True)
class ComposeAugmentation:
    """Apply an ordered tuple of typed augmentations."""

    augmentations: tuple[GroundCourtAugmentation, ...]

    def __call__(
        self, sample: AugmentableGroundCourtSample, rng: np.random.Generator
    ) -> AugmentableGroundCourtSample:
        result = sample
        for augmentation in self.augmentations:
            result = augmentation(result, rng)
            if not isinstance(result, AugmentableGroundCourtSample):
                raise TypeError(
                    "Ground-court augmentation must return "
                    "AugmentableGroundCourtSample."
                )
        return result


_AugmentationFactory = Callable[[Mapping[str, object]], GroundCourtAugmentation]
_AUGMENTATION_FACTORIES: dict[str, _AugmentationFactory] = {}


def register_augmentation(
    name: str,
) -> Callable[[_AugmentationFactory], _AugmentationFactory]:
    """Register a named factory; duplicate names fail during import/configuration."""

    normalized = name.strip() if isinstance(name, str) else ""
    if not normalized:
        raise ValueError("augmentation registry names must be non-empty strings.")

    def decorator(factory: _AugmentationFactory) -> _AugmentationFactory:
        if normalized in _AUGMENTATION_FACTORIES:
            raise ValueError(f"Augmentation already registered: {normalized!r}.")
        _AUGMENTATION_FACTORIES[normalized] = factory
        return factory

    return decorator


@register_augmentation("identity")
def _build_identity(params: Mapping[str, object]) -> GroundCourtAugmentation:
    if params:
        raise ValueError("identity augmentation does not accept parameters.")
    return IdentityAugmentation()


def build_augmentation(
    config: GroundCourtAugmentationConfig | Mapping[str, object] | str | None = None,
) -> GroundCourtAugmentation:
    """Build a registered augmentation and reject unknown names immediately."""

    if config is None:
        resolved = GroundCourtAugmentationConfig()
    elif isinstance(config, GroundCourtAugmentationConfig):
        resolved = config
    elif isinstance(config, str):
        resolved = GroundCourtAugmentationConfig(name=config)
    elif isinstance(config, Mapping):
        resolved = GroundCourtAugmentationConfig.from_mapping(config)
    else:
        raise TypeError("Unsupported ground-court augmentation config.")
    try:
        factory = _AUGMENTATION_FACTORIES[resolved.name]
    except KeyError as error:
        available = ", ".join(sorted(_AUGMENTATION_FACTORIES))
        raise ValueError(
            f"Unknown ground-court augmentation {resolved.name!r}; available: {available}."
        ) from error
    return factory(resolved.params)


def build_augmentations(
    configs: Sequence[
        GroundCourtAugmentationConfig | Mapping[str, object] | str
    ] = (),
) -> GroundCourtAugmentation:
    """Build an ordered augmentation pipeline, defaulting to identity."""

    built = tuple(build_augmentation(config) for config in configs)
    if not built:
        return IdentityAugmentation()
    return ComposeAugmentation(built)


__all__ = [
    "AugmentableGroundCourtSample",
    "ComposeAugmentation",
    "GroundCourtAugmentation",
    "GroundCourtAugmentationConfig",
    "IdentityAugmentation",
    "build_augmentation",
    "build_augmentations",
    "register_augmentation",
]
