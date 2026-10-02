"""Read an :class:`AssociationConfig` from a mapping (the fitted default or an explicit option).

Every field must be present and no unknown field is accepted, so a renamed or
forgotten parameter stops the load instead of taking a default. The file holds
the method's parameters; ``players_per_side`` is a property of the clip
(singles or doubles), so the caller always passes it and the file must not.
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
from src.utils.paths import PROJECT_ROOT

LEGACY_CONFIG = PROJECT_ROOT / "src/tasks/player_association/configs/association.yaml"
DEFAULT_CONFIG = PROJECT_ROOT / "src/tasks/player_association/configs/association_i964_r14_lovo_a.yaml"


def _build(kind: type, values: Any, where: str) -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{where} must be a mapping")
    names = {field.name for field in fields(kind)}
    if set(values) != names:
        raise ValueError(f"{where}: missing {sorted(names - set(values))}, unknown {sorted(set(values) - names)}")
    return kind(**values)


def association_config(values: Mapping[str, Any], *, players_per_side: int) -> AssociationConfig:
    """Build the config; ``appearance: null`` selects geometry-only association."""
    nested = {"footpoints": FootpointConfig, "switches": SwitchConfig, "geometry": GeometryAffinityConfig,
              "appearance": AppearanceAffinityConfig, "region": PlayRegionConfig}
    names = {field.name for field in fields(AssociationConfig)} - {"players_per_side"}
    if set(values) != names:
        raise ValueError(f"Association config: missing {sorted(names - set(values))}, unknown {sorted(set(values) - names)}")
    built: dict[str, Any] = {key: (None if key == "appearance" and value is None else _build(nested[key], value, key)) if key in nested else value
             for key, value in values.items()}
    return AssociationConfig(players_per_side=players_per_side, **built)


def load_association_config(path: Path = DEFAULT_CONFIG, *, players_per_side: int,
                            overrides: Mapping[str, Any] | None = None) -> AssociationConfig:
    """Load ``path``; ``overrides`` replace top-level fields (for example ``appearance: None``)."""
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"{path} does not hold a mapping")
    if overrides is not None:
        unknown = set(overrides) - set(values)
        if unknown:
            raise ValueError(f"Overrides name unknown fields {sorted(unknown)}")
        values = {**values, **overrides}
    return association_config(values, players_per_side=players_per_side)
