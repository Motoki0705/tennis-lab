"""Base config-dispatch machinery for observation augmentation.

This module extracts the shared structure between PLCS keypoint augmentation
(:class:`PLCSObservationAugmentation`) and BLCS ball augmentation
(:class:`BLCSBallObservationAugmentation`).

Both classes:
- parse the same six config blocks (``temporal_jitter``, ``burst_dropout``,
  ``false_positive``, ``edge_degradation``, ``speed_conditioned``, plus the
  ``uv_scale``/``gaussian_noise``/``visibility_dropout`` blocks) via
  :func:`src.utils.data.augmentation._as_dict`;
- guard each ``_apply_*`` dispatch with
  ``if not _enabled(cfg) or not _should_apply(_prob(cfg), ref): return ...``;
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
from typing import Any

from torch import Tensor

from src.utils.data.augmentation import (
    _as_dict,
    _enabled,
    _prob,
    _should_apply,
    parse_float_range,
)

__all__ = ["BaseObservationAugmentation"]


class BaseObservationAugmentation(ABC):
    """Shared config parsing and dispatch guards for observation augmentation.

    Subclasses implement :meth:`forward` (defining the per-task RNG draw order
    and entity-key routing) and may override the ``_*_config`` helpers to inject
    task-specific default config blocks.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = _as_dict(config)
        self.enabled = bool(self.config.get("enabled", True))

        self.uv_scale_cfg = self._uv_scale_config()
        self.gaussian_cfg = self._gaussian_config()
        self.visibility_dropout_cfg = self._visibility_dropout_config()
        self.temporal_jitter_cfg = _as_dict(self.config.get("temporal_jitter"))
        self.burst_dropout_cfg = _as_dict(self.config.get("burst_dropout"))
        self.false_positive_cfg = _as_dict(self.config.get("false_positive"))
        self.edge_degradation_cfg = _as_dict(self.config.get("edge_degradation"))
        self.speed_conditioned_cfg = _as_dict(self.config.get("speed_conditioned"))

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

        Mirrors the guard duplicated across every ``_apply_*`` method:
        ``_enabled(cfg) and _should_apply(_prob(cfg), reference)``.
        Calls ``_should_apply`` (which may draw from the RNG) only when enabled,
        preserving the original short-circuit RNG behavior.
        """
        return _enabled(cfg) and _should_apply(_prob(cfg), reference)

    @staticmethod
    def _parse_scale_range(
        cfg: Mapping[str, Any],
        name: str = "augmentation.uv_scale.scale_range",
    ) -> tuple[float, float]:
        """Parse and validate a positive ``scale_range`` from a uv_scale block."""
        scale_min, scale_max = parse_float_range(cfg.get("scale_range", [1.0, 1.0]), name)
        if scale_min <= 0 or scale_max <= 0:
            raise ValueError(f"{name} values must be positive.")
        return scale_min, scale_max

    # -- forward ---------------------------------------------------------------

    @abstractmethod
    def forward(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Return an augmented sample (task-specific key routing/draw order)."""
