"""Read an :class:`AssociationConfig` from a mapping (``configs/association.yaml``).

Every field must be present and no unknown field is accepted, so a renamed or
forgotten parameter stops the load instead of taking a default.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from src.tasks.player_association.appearance.affinity import AppearanceAffinityConfig
from src.tasks.player_association.association.associate import AssociationConfig
from src.tasks.player_association.geometry.affinity import GeometryAffinityConfig
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tasks.player_association.geometry.region import PlayRegionConfig
from src.tasks.player_association.geometry.switches import SwitchConfig

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "association.yaml"


def _build(kind: type, values: Any, where: str) -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{where} must be a mapping")
    names = {field.name for field in fields(kind)}
    if set(values) != names:
        raise ValueError(f"{where}: missing {sorted(names - set(values))}, unknown {sorted(set(values) - names)}")
    return kind(**values)


def association_config(values: Mapping[str, Any]) -> AssociationConfig:
    """Build the config; ``appearance: null`` selects geometry-only association."""
    nested = {"footpoints": FootpointConfig, "switches": SwitchConfig, "geometry": GeometryAffinityConfig,
              "appearance": AppearanceAffinityConfig, "region": PlayRegionConfig}
    names = {field.name for field in fields(AssociationConfig)}
    if set(values) != names:
        raise ValueError(f"Association config: missing {sorted(names - set(values))}, unknown {sorted(set(values) - names)}")
    built: dict[str, Any] = {key: (None if key == "appearance" and value is None else _build(nested[key], value, key)) if key in nested else value
             for key, value in values.items()}
    return AssociationConfig(**built)


def load_association_config(path: Path = DEFAULT_CONFIG, *, overrides: Mapping[str, Any] | None = None) -> AssociationConfig:
    """Load ``path``; ``overrides`` replace top-level fields (for example ``appearance: None``)."""
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"{path} does not hold a mapping")
    unknown = set(overrides or {}) - set(values)
    if unknown:
        raise ValueError(f"Overrides name unknown fields {sorted(unknown)}")
    return association_config({**values, **(overrides or {})})
