"""Base config-dispatch machinery for observation augmentation.

This module extracts the shared structure between PLCS keypoint augmentation
(:class:`PLCSObservationAugmentation`) and BLCS ball augmentation
(:class:`BLCSBallObservationAugmentation`).

Both classes:
- parse the same six required config blocks (``temporal_jitter``,
  ``burst_dropout``, ``false_positive``, ``edge_degradation``,
  ``speed_conditioned``, plus the ``uv_scale``/``gaussian_noise``/
  ``visibility_dropout`` blocks) into strict mappings;
- guard each ``_apply_*`` dispatch with explicitly typed ``enabled`` and
  ``prob`` fields;
- delegate the numerical work to the primitives in
  :mod:`src.utils.data.augmentation`.

Subclasses define ``forward`` (the per-task RNG draw order and which sample
keys to read) and the entity-key routing (PLCS uses ``human``/``court`` keys,
BLCS uses ``ball``/``court`` keys).  All RNG-consuming primitives live in
``src.utils.data.augmentation`` so subclasses preserve exact numerical
behavior simply by calling the same helpers in the same order.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Generic, TypeVar, cast

import torch
from torch import Tensor

from src.tasks.base.configuration import (
    as_config_mapping,
    require_config_mapping,
    require_config_value,
)
from src.utils.configuration import SemanticConfigurationError
from src.utils.data.augmentation import parse_float_range

__all__ = ["BaseObservationAugmentation"]

SampleT = TypeVar("SampleT")


class BaseObservationAugmentation(ABC, Generic[SampleT]):
    """Shared config parsing and dispatch guards for observation augmentation.

    Subclasses implement :meth:`forward` (defining the per-task RNG draw order
    and entity-key routing) and may override the ``_*_config`` helpers to inject
    task-specific default config blocks.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(as_config_mapping(config, path="augmentation"))
        self.enabled = bool(
            require_config_value(self.config, "enabled", bool, path="augmentation")
        )

        self.uv_scale_cfg = self._uv_scale_config()
        self.gaussian_cfg = self._gaussian_config()
        self.visibility_dropout_cfg = self._visibility_dropout_config()
        self.temporal_jitter_cfg = dict(
            require_config_mapping(self.config, "temporal_jitter", path="augmentation")
        )
        self.burst_dropout_cfg = dict(
            require_config_mapping(self.config, "burst_dropout", path="augmentation")
        )
        self.false_positive_cfg = dict(
            require_config_mapping(self.config, "false_positive", path="augmentation")
        )
        self.edge_degradation_cfg = dict(
            require_config_mapping(self.config, "edge_degradation", path="augmentation")
        )
        self.speed_conditioned_cfg = dict(
            require_config_mapping(
                self.config, "speed_conditioned", path="augmentation"
            )
        )

    # -- config block builders (overridable for task-specific defaults) --------

    @abstractmethod
    def _uv_scale_config(self) -> dict[str, Any]:
        """Return the resolved ``uv_scale`` config block."""

    @abstractmethod
    def _gaussian_config(self) -> dict[str, Any]:
        """Return the resolved ``gaussian_noise`` config block."""

    @abstractmethod
    def _visibility_dropout_config(self) -> dict[str, Any]:
        """Return the resolved ``visibility_dropout`` config block."""

    # -- shared dispatch guard --------------------------------------------------

    @staticmethod
    def _active(cfg: Mapping[str, Any], reference: Tensor) -> bool:
        """Return True when the config block is enabled and sampled for use.

        ``enabled`` and ``prob`` are required and exactly typed. Sampling occurs
        only when enabled, preserving the original short-circuit RNG behavior.
        """
        mapping = as_config_mapping(cfg, path="augmentation.block")
        enabled = bool(
            require_config_value(mapping, "enabled", bool, path="augmentation.block")
        )
        probability = float(
            cast(
                "float | int",
                require_config_value(
                    mapping, "prob", (float, int), path="augmentation.block"
                ),
            )
        )
        if not 0.0 <= probability <= 1.0:
            raise SemanticConfigurationError(
                f"augmentation.block.prob must be within [0, 1]; got {probability}."
            )
        if not enabled:
            return False
        if probability == 0.0:
            return False
        if probability == 1.0:
            return True
        return bool(torch.rand((), device=reference.device).item() < probability)

    @staticmethod
    def _parse_scale_range(
        cfg: Mapping[str, Any],
        name: str = "augmentation.uv_scale.scale_range",
    ) -> tuple[float, float]:
        """Parse and validate a positive ``scale_range`` from a uv_scale block."""
        scale_min, scale_max = parse_float_range(cfg["scale_range"], name)
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError(f"{name} values must be positive.")
        return scale_min, scale_max

    # -- forward ---------------------------------------------------------------

    @abstractmethod
    def forward(self, sample: SampleT) -> SampleT:
        """Return an augmented sample (task-specific key routing/draw order)."""
