"""Shared PLCS configuration value objects below runtime boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
)
from src.utils.configuration import (
    PathResolver,
    RuntimePathRoots,
)
from src.utils.paths import PROJECT_ROOT

__all__ = ["PLCSPathConfig"]


@dataclass(frozen=True, slots=True)
class PLCSPathConfig:
    """Seven shared runtime roots for every PLCS execution boundary."""

    resolver: PathResolver

    @classmethod
    def from_config(cls, value: object) -> PLCSPathConfig:
        root = as_config_mapping(value, path="configuration")
        paths = require_config_mapping(root, "paths", path="configuration")
        roots = RuntimePathRoots.from_mapping(paths, repository_root=PROJECT_ROOT)
        return cls(resolver=PathResolver(roots))
