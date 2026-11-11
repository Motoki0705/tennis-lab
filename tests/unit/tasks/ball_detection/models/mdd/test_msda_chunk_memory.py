"""Chunk backward must not zero and sum a complete image-memory tensor per slice."""

from __future__ import annotations

import copy
from dataclasses import replace

import torch
from torch import Tensor, nn

from src.tasks.ball_detection.models.mdd.multiscale_query import (
    DeformableMHCQueryBlock,
    PyramidMemory,
)
from tests.unit.tasks.ball_detection.models.mdd.test_multiscale_query import (
    multiscale_config,
)


class ReadoutDouble(nn.Module):
    """Frame-local differentiable readout; the real CUDA implementation has its own tests."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(
        self,
        query: Tensor,
        references: Tensor,
        values: Tensor,
        shapes: Tensor,
        starts: Tensor,
        mask: Tensor,
    ) -> Tensor:
        return (
            self.query(query)
            + self.value(values.mean(1))[:, None]
            + references.sum((-1, -2))[:, :, None]
        )


def test_chunks_match_outputs_all_gradients_without_full_memory_zero_buffers(
    monkeypatch,
) -> None:
    torch.set_num_threads(2)
    config = replace(multiscale_config(), msda_chunk_size=3)
    full = DeformableMHCQueryBlock(config)
    monkeypatch.setattr(full, "cross", ReadoutDouble(config.dim))
    chunked = copy.deepcopy(full)
    chunked.chunk_size = 1
    streams = torch.randn(1, 3, 4, config.dim, requires_grad=True)
    memory = torch.randn(1, 3, 17, config.dim, requires_grad=True)
    other_streams = streams.detach().clone().requires_grad_()
    other_memory = memory.detach().clone().requires_grad_()
    shapes = torch.tensor([[2, 4], [1, 4], [1, 4], [1, 1]])
    starts = torch.tensor([0, 8, 12, 16])
    mask = torch.zeros(3, 17, dtype=torch.bool)
    ratios = torch.ones(3, 4, 2)
    times = torch.tensor([[0.0, 0.03, 0.07]])
    first = full(streams, PyramidMemory(memory, shapes, starts, mask, ratios), times)
    second = chunked(
        other_streams, PyramidMemory(other_memory, shapes, starts, mask, ratios), times
    )
    torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-6)
    probe = torch.randn_like(first)
    (first * probe).sum().backward()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profile:
        (second * probe).sum().backward()
    torch.testing.assert_close(streams.grad, other_streams.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(memory.grad, other_memory.grad, rtol=1e-4, atol=1e-5)
    for (name, a), (other, b) in zip(
        full.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert name == other and a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-4, atol=1e-5, msg=name)
    full_bytes = memory.numel() * memory.element_size()
    full_zeros = [
        event
        for event in profile.events()
        if event.name == "aten::zeros" and event.cpu_memory_usage == full_bytes
    ]
    assert not full_zeros, (
        "Per-chunk SliceBackward allocated whole-image-memory zero buffers"
    )
