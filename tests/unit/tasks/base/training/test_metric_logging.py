"""Tests for the strict shared metric visibility contract."""

from __future__ import annotations

import pytest
import torch

from src.tasks.base.training.metric_logging import (
    MetricContractError,
    MetricLoggingContract,
    MetricStatisticsAccumulator,
    ScalarMetricStatistic,
    StageMetricContract,
    WeightedMetricAccumulator,
    compute_scalar_metric_statistics,
    format_metric_threshold,
    uniform_metric_logging_contract,
)


def _stage(stage: str) -> StageMetricContract:
    return StageMetricContract(stage=stage, headline_keys=("quality",))


def test_metric_threshold_formatter_is_round_trip_safe_and_preserves_defaults() -> None:
    assert format_metric_threshold(0.5) == "0.5"
    assert format_metric_threshold(15.0) == "15"
    assert format_metric_threshold(0.5004) == "0.5004"
    assert float(format_metric_threshold(0.5004)) == 0.5004


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("inf"), float("nan")])
def test_metric_threshold_formatter_rejects_invalid_values(threshold: float) -> None:
    with pytest.raises(MetricContractError, match="finite and > 0"):
        format_metric_threshold(threshold)


def test_contract_rejects_unknown_duplicate_and_missing_stages() -> None:
    with pytest.raises(MetricContractError, match="Unknown metric stage"):
        StageMetricContract(stage="predict", headline_keys=())

    with pytest.raises(MetricContractError, match="registers stages more than once"):
        MetricLoggingContract(
            name="duplicate",
            stages=(_stage("train"), _stage("train"), _stage("test")),
        )

    with pytest.raises(MetricContractError, match="missing stages: test"):
        MetricLoggingContract(
            name="missing",
            stages=(_stage("train"), _stage("val")),
        )


def test_stage_contract_rejects_duplicate_or_non_headline_progress_keys() -> None:
    with pytest.raises(MetricContractError, match="duplicate headline"):
        StageMetricContract(stage="train", headline_keys=("quality", "quality"))
    with pytest.raises(MetricContractError, match="must also be headline"):
        StageMetricContract(
            stage="train",
            headline_keys=("quality",),
            progress_bar_keys=("detail",),
        )


def test_partition_requires_headlines_and_keeps_diagnostics_disjoint() -> None:
    contract = uniform_metric_logging_contract(
        "trajectory",
        headline_keys=("quality",),
        progress_bar_keys=("quality",),
    )

    partition = contract.partition("test", {"quality": 1.0, "axis_error": 2.0})

    assert partition.headline == {"quality": 1.0}
    assert partition.diagnostics == {"axis_error": 2.0}
    with pytest.raises(MetricContractError, match="missing required headline"):
        contract.partition("val", {"axis_error": 2.0})
    with pytest.raises(MetricContractError, match="Unknown metric stage"):
        contract.partition("predict", {"quality": 1.0})


def test_weighted_accumulator_matches_uneven_batch_reduction_and_resets() -> None:
    accumulator = WeightedMetricAccumulator()
    accumulator.update(
        {"quality": torch.tensor(1.0), "dynamic_detail": 4.0},
        weight=2,
    )
    accumulator.update({"quality": torch.tensor(7.0)}, weight=1)

    assert accumulator.compute() == {
        "quality": pytest.approx(3.0),
        "dynamic_detail": pytest.approx(4.0),
    }

    accumulator.reset()
    assert accumulator.compute() == {}


def test_weighted_accumulator_rejects_non_scalar_values_and_invalid_weights() -> None:
    accumulator = WeightedMetricAccumulator()
    with pytest.raises(MetricContractError, match="must be scalar"):
        accumulator.update({"quality": torch.ones(2)}, weight=1)
    with pytest.raises(MetricContractError, match="int > 0"):
        accumulator.update({"quality": 1.0}, weight=0)


def test_statistics_accumulator_merges_each_dynamic_ratio_independently() -> None:
    accumulator = MetricStatisticsAccumulator()
    accumulator.update(
        {
            "quality": ScalarMetricStatistic(torch.tensor(2.0), torch.tensor(2.0)),
            "reference_index_0": ScalarMetricStatistic(
                torch.tensor(6.0), torch.tensor(2.0)
            ),
        }
    )
    accumulator.update(
        {
            "quality": ScalarMetricStatistic(torch.tensor(9.0), torch.tensor(3.0)),
            "reference_index_1": ScalarMetricStatistic(
                torch.tensor(8.0), torch.tensor(1.0)
            ),
        }
    )

    assert accumulator.compute() == {
        "quality": pytest.approx(2.2),
        "reference_index_0": pytest.approx(3.0),
        "reference_index_1": pytest.approx(8.0),
    }
    accumulator.reset()
    assert accumulator.compute() == {}


def test_statistics_omit_undefined_epoch_ratios_but_support_batch_wrappers() -> None:
    statistic = ScalarMetricStatistic(torch.tensor(0.0), torch.tensor(0.0))
    accumulator = MetricStatisticsAccumulator()
    accumulator.update({"optional": statistic})

    assert accumulator.compute() == {}
    assert compute_scalar_metric_statistics(
        {"optional": statistic}, zero_denominator_value=None
    ) == {}
    computed = compute_scalar_metric_statistics(
        {"optional": statistic}, zero_denominator_value=0.0
    )
    torch.testing.assert_close(computed["optional"], torch.tensor(0.0))


def test_scalar_statistic_rejects_invalid_sufficient_statistics() -> None:
    with pytest.raises(MetricContractError, match="must be scalar"):
        ScalarMetricStatistic(torch.ones(2), torch.tensor(1.0))
    with pytest.raises(MetricContractError, match="finite"):
        ScalarMetricStatistic(torch.tensor(float("nan")), torch.tensor(1.0))
    with pytest.raises(MetricContractError, match="greater than or equal to zero"):
        ScalarMetricStatistic(torch.tensor(0.0), torch.tensor(-1.0))
    with pytest.raises(MetricContractError, match="zero numerator"):
        ScalarMetricStatistic(torch.tensor(1.0), torch.tensor(0.0))
