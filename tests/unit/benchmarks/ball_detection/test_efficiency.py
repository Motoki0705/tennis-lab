"""Benchmarks retain their units and do not double-count shared storage."""

import pytest
import torch
from torch import nn

from tests.benchmarks.ball_detection.efficiency import (
    conv_linear_macs,
    quantiles,
    tensor_bytes,
)


def test_quantiles_require_real_samples_and_preserve_milliseconds() -> None:
    assert quantiles([10.0, 20.0, 30.0, 40.0, 50.0]) == {
        "p50": 30.0,
        "p90": 46.0,
        "p95": 48.0,
    }
    with pytest.raises(ValueError):
        quantiles([])
    with pytest.raises(ValueError):
        quantiles([float("nan")])


def test_model_optimizer_bytes_count_storage_once_including_views() -> None:
    x = torch.empty(12, dtype=torch.float32)
    assert (
        tensor_bytes(
            {
                "parameter": x,
                "alias": x.view(3, 4),
                "moment": torch.empty(12, dtype=torch.float32),
            }
        )
        == 96
    )


def test_conv3d_counts_groups_and_temporal_kernel_without_profiler_flops() -> None:
    model = nn.Conv3d(4, 8, (3, 3, 3), groups=2, padding=1, bias=False)
    result = conv_linear_macs(model, lambda: model(torch.zeros(2, 4, 5, 7, 11)))
    assert result["conv_linear_macs"] == 2 * 8 * 5 * 7 * 11 * 2 * 27
    assert "MSDA sampling (not zero)" in result["excludes"]
