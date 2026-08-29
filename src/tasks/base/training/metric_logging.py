"""Strict contracts for metric visibility and epoch aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import torch
from torch import Tensor

MetricValue: TypeAlias = int | float | Tensor
MetricStage: TypeAlias = str

_STAGES = ("train", "val", "test")


class MetricContractError(ValueError):
    """Raised when metric production and the visibility contract disagree."""


def format_metric_threshold(value: float) -> str:
    """Return a compact, round-trip-safe threshold for a metric key.

    Python's float representation is injective for finite values: distinct
    floats never collapse to the same rendered value. Removing only the
    redundant ``.0`` suffix preserves established keys such as ``0.5m`` and
    ``15deg`` while retaining configured precision such as ``0.5004m``.
    """
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise MetricContractError("Metric thresholds must be finite and > 0.")
    rendered = repr(threshold)
    return rendered[:-2] if rendered.endswith(".0") else rendered


@dataclass(frozen=True, slots=True)
class ScalarMetricStatistic:
    """One scalar metric's additive numerator and denominator.

    Ratios are deliberately not reduced at batch scope: merging these sufficient
    statistics first makes the final value invariant to DataLoader partitioning.
    """

    numerator: Tensor
    denominator: Tensor

    def __post_init__(self) -> None:
        for name, value in (
            ("numerator", self.numerator),
            ("denominator", self.denominator),
        ):
            if not isinstance(value, Tensor):
                raise MetricContractError(
                    f"Metric statistic {name} must be a Tensor, got "
                    f"{type(value).__name__}."
                )
            if value.numel() != 1:
                raise MetricContractError(
                    f"Metric statistic {name} must be scalar, got shape "
                    f"{tuple(value.shape)}."
                )
            if not bool(torch.isfinite(value.detach()).item()):
                raise MetricContractError(
                    f"Metric statistic {name} must be finite."
                )
        if self.numerator.device != self.denominator.device:
            raise MetricContractError(
                "Metric statistic numerator and denominator must share a device."
            )
        if bool((self.denominator.detach() < 0).item()):
            raise MetricContractError(
                "Metric statistic denominator must be greater than or equal to zero."
            )
        if bool((self.denominator.detach() == 0).item()) and bool(
            (self.numerator.detach() != 0).item()
        ):
            raise MetricContractError(
                "A zero-denominator metric statistic must have a zero numerator."
            )

    @classmethod
    def from_mean(cls, value: Tensor, *, weight: int) -> ScalarMetricStatistic:
        """Represent an already averaged scalar with an explicit sample weight."""
        if type(weight) is not int or weight <= 0:
            raise MetricContractError("Metric statistic weight must be an int > 0.")
        if not isinstance(value, Tensor) or value.numel() != 1:
            shape = tuple(value.shape) if isinstance(value, Tensor) else None
            raise MetricContractError(
                f"Mean metric value must be a scalar Tensor, got shape {shape}."
            )
        denominator = value.new_tensor(float(weight))
        return cls(numerator=value * denominator, denominator=denominator)

    def compute(self) -> Tensor | None:
        """Return the ratio, or ``None`` when no evaluable item was observed."""
        if bool((self.denominator.detach() == 0).item()):
            return None
        return self.numerator / self.denominator


def compute_scalar_metric_statistics(
    statistics: Mapping[str, ScalarMetricStatistic],
    *,
    zero_denominator_value: float | None,
) -> dict[str, Tensor]:
    """Reduce scalar statistics, optionally materializing undefined batch ratios."""
    computed: dict[str, Tensor] = {}
    for key, statistic in statistics.items():
        if not isinstance(statistic, ScalarMetricStatistic):
            raise MetricContractError(
                f"Metric statistic {key!r} must be ScalarMetricStatistic, got "
                f"{type(statistic).__name__}."
            )
        value = statistic.compute()
        if value is not None:
            computed[key] = value
        elif zero_denominator_value is not None:
            computed[key] = statistic.numerator.new_tensor(zero_denominator_value)
    return computed


class MetricStatisticsAccumulator:
    """Merge additive scalar statistics, retaining a denominator for every key."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard all accumulated sufficient statistics."""
        self._numerators: dict[str, float] = {}
        self._denominators: dict[str, float] = {}

    def update(self, statistics: Mapping[str, ScalarMetricStatistic]) -> None:
        """Merge one batch of statistics; dynamic keys remain independently weighted."""
        for key, statistic in statistics.items():
            if not isinstance(statistic, ScalarMetricStatistic):
                raise MetricContractError(
                    f"Metric statistic {key!r} must be ScalarMetricStatistic, got "
                    f"{type(statistic).__name__}."
                )
            numerator = float(statistic.numerator.detach().cpu().item())
            denominator = float(statistic.denominator.detach().cpu().item())
            self._numerators[key] = self._numerators.get(key, 0.0) + numerator
            self._denominators[key] = (
                self._denominators.get(key, 0.0) + denominator
            )

    def compute(self) -> dict[str, float]:
        """Return defined full-epoch ratios, omitting keys with no denominator."""
        return {
            key: numerator / self._denominators[key]
            for key, numerator in self._numerators.items()
            if self._denominators[key] > 0.0
        }


@dataclass(frozen=True, slots=True)
class StageMetricContract:
    """Headline and progress-bar allowlists for one Lightning stage."""

    stage: MetricStage
    headline_keys: tuple[str, ...]
    progress_bar_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise MetricContractError(f"Unknown metric stage: {self.stage!r}.")
        if len(set(self.headline_keys)) != len(self.headline_keys):
            raise MetricContractError(
                f"Metric contract for {self.stage!r} contains duplicate headline keys."
            )
        if len(set(self.progress_bar_keys)) != len(self.progress_bar_keys):
            raise MetricContractError(
                f"Metric contract for {self.stage!r} contains duplicate progress-bar keys."
            )
        unknown_progress = set(self.progress_bar_keys) - set(self.headline_keys)
        if unknown_progress:
            rendered = ", ".join(sorted(unknown_progress))
            raise MetricContractError(
                f"Progress-bar keys must also be headline keys for {self.stage!r}: "
                f"{rendered}."
            )


@dataclass(frozen=True, slots=True)
class MetricPartition:
    """A disjoint headline/diagnostic view of computed metrics."""

    headline: dict[str, MetricValue]
    diagnostics: dict[str, MetricValue]


@dataclass(frozen=True, slots=True)
class MetricLoggingContract:
    """Strict train/val/test visibility contract for one model family."""

    name: str
    stages: tuple[StageMetricContract, ...]

    def __post_init__(self) -> None:
        registered = [contract.stage for contract in self.stages]
        duplicates = sorted(
            stage for stage in set(registered) if registered.count(stage) > 1
        )
        if duplicates:
            rendered = ", ".join(duplicates)
            raise MetricContractError(
                f"Metric logging contract {self.name!r} registers stages more than "
                f"once: {rendered}."
            )
        missing = sorted(set(_STAGES) - set(registered))
        if missing:
            rendered = ", ".join(missing)
            raise MetricContractError(
                f"Metric logging contract {self.name!r} is missing stages: {rendered}."
            )

    def for_stage(self, stage: MetricStage) -> StageMetricContract:
        """Return the exact stage contract, rejecting unknown stage names."""
        if stage not in _STAGES:
            raise MetricContractError(f"Unknown metric stage: {stage!r}.")
        for contract in self.stages:
            if contract.stage == stage:
                return contract
        raise MetricContractError(
            f"Metric logging contract {self.name!r} has no {stage!r} stage."
        )

    def partition(
        self,
        stage: MetricStage,
        metrics: Mapping[str, MetricValue],
    ) -> MetricPartition:
        """Split computed metrics, requiring every configured headline key."""
        contract = self.for_stage(stage)
        missing = [key for key in contract.headline_keys if key not in metrics]
        if missing:
            rendered = ", ".join(missing)
            raise MetricContractError(
                f"{self.name} {stage} metrics are missing required headline keys: "
                f"{rendered}."
            )
        headline = {key: metrics[key] for key in contract.headline_keys}
        diagnostics = {
            key: value for key, value in metrics.items() if key not in headline
        }
        return MetricPartition(headline=headline, diagnostics=diagnostics)

    def is_progress_bar_metric(self, stage: MetricStage, key: str) -> bool:
        """Return whether a headline belongs on the stage progress bar."""
        return key in self.for_stage(stage).progress_bar_keys


class WeightedMetricAccumulator:
    """Accumulate scalar batch metrics with explicit per-key sample weights."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Discard all accumulated values and weights."""
        self._weighted_sums: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    @staticmethod
    def _scalar(value: object, *, key: str) -> float:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise MetricContractError(
                    f"Metric {key!r} must be scalar, got shape {tuple(value.shape)}."
                )
            return float(value.detach().cpu().item())
        if not isinstance(value, Real):
            raise MetricContractError(
                f"Metric {key!r} must be a real scalar, got "
                f"{type(value).__name__}."
            )
        return float(value)

    def update(self, metrics: Mapping[str, MetricValue], *, weight: int) -> None:
        """Add a batch; dynamic diagnostic keys retain their own denominators."""
        if type(weight) is not int or weight <= 0:
            raise MetricContractError("Metric accumulator weight must be an int > 0.")
        for key, value in metrics.items():
            scalar = self._scalar(value, key=key)
            self._weighted_sums[key] = (
                self._weighted_sums.get(key, 0.0) + scalar * weight
            )
            self._weights[key] = self._weights.get(key, 0.0) + weight

    def compute(self) -> dict[str, float]:
        """Return the batch-size-weighted mean for every observed key."""
        return {
            key: weighted_sum / self._weights[key]
            for key, weighted_sum in self._weighted_sums.items()
        }


def uniform_metric_logging_contract(
    name: str,
    *,
    headline_keys: tuple[str, ...],
    progress_bar_keys: tuple[str, ...],
) -> MetricLoggingContract:
    """Build a trajectory contract whose canonical keys match across stages."""
    return MetricLoggingContract(
        name=name,
        stages=tuple(
            StageMetricContract(
                stage=stage,
                headline_keys=headline_keys,
                progress_bar_keys=progress_bar_keys if stage != "test" else (),
            )
            for stage in _STAGES
        ),
    )


def evaluation_only_metric_logging_contract(
    name: str,
    *,
    headline_keys: tuple[str, ...],
    progress_bar_keys: tuple[str, ...],
) -> MetricLoggingContract:
    """Build a tracking contract with loss-only training visibility."""
    return MetricLoggingContract(
        name=name,
        stages=(
            StageMetricContract(stage="train", headline_keys=()),
            StageMetricContract(
                stage="val",
                headline_keys=headline_keys,
                progress_bar_keys=progress_bar_keys,
            ),
            StageMetricContract(stage="test", headline_keys=headline_keys),
        ),
    )


__all__ = [
    "MetricContractError",
    "MetricLoggingContract",
    "MetricPartition",
    "MetricStatisticsAccumulator",
    "MetricStage",
    "MetricValue",
    "ScalarMetricStatistic",
    "StageMetricContract",
    "WeightedMetricAccumulator",
    "compute_scalar_metric_statistics",
    "evaluation_only_metric_logging_contract",
    "uniform_metric_logging_contract",
]
