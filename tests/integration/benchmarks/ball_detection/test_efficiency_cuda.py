"""Execute the CUDA timing/profiler harness on a tiny module, not a model sweep."""

import pytest
import torch
from torch import nn

from tests.benchmarks.ball_detection.efficiency import benchmark

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA timing/profiling"
)


def test_cuda_phase_sampling_profiler_and_new_backward_graphs(tmp_path):
    model = nn.Linear(8, 8).cuda()
    inputs = (torch.randn(3, 8, device="cuda", requires_grad=True),)
    result = benchmark(
        model,
        inputs,
        lambda value: value.float().square().mean(),
        frames=3,
        microbatches=2,
        compile_mode="off",
        fullgraph=True,
        profiler_path=tmp_path / "trace.json",
    )
    assert len(result["raw"]["training"]) == len(result["raw"]["inference"]) == 30
    assert result["frames_per_optimizer_update"] == 6
    assert result["raw"]["training"][-1]["optimizer_bytes"] > 0
    assert all(
        row["total"]["cuda_ms"] > 0 and row["phases"]["backward"]["cuda_ms"] > 0
        for row in result["raw"]["training"]
    )
    assert result["summary"]["update_wall_ms"]["p95"] > 0
    assert (tmp_path / "trace.json").is_file() and (
        tmp_path / "trace-ops.txt"
    ).is_file()
