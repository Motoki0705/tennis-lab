"""Fresh-process CUDA initialization tests for PLCS execution."""

from typing import cast

import pytest
import torch
from pytest import MonkeyPatch

from src.synthetic_data_generation.dataset.plcs.execution import (
    CUDAPLCSExecutionBackend,
)
from src.synthetic_data_generation.dataset.plcs.rendering.contracts import (
    PLCSForegroundCompositor,
)


class _ResetOnlyCompositor:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reset_stage(self) -> None:
        self.events.append("compositor-reset")


def test_cuda_reset_initializes_allocator_before_resetting_peak_stats(
    monkeypatch: MonkeyPatch,
) -> None:
    events: list[str] = []
    allocator_initialized = False

    def initialize_allocator(
        size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        nonlocal allocator_initialized
        assert size == 1
        assert dtype is torch.uint8
        assert device == torch.device("cuda:0")
        allocator_initialized = True
        events.append("allocator-initialized")
        return torch.tensor([0], dtype=torch.uint8)

    def reset_peak(device: torch.device) -> None:
        assert allocator_initialized
        assert device == torch.device("cuda:0")
        events.append("peak-reset")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", initialize_allocator)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    compositor = _ResetOnlyCompositor(events)

    backend = CUDAPLCSExecutionBackend()
    backend.reset_stage(
        configured_device="cuda:0",
        compositor=cast(PLCSForegroundCompositor, compositor),
    )

    assert backend.torch_device == torch.device("cuda:0")
    assert events == ["compositor-reset", "allocator-initialized", "peak-reset"]


def test_cuda_reset_still_rejects_unavailable_cuda(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    compositor = _ResetOnlyCompositor([])

    with pytest.raises(RuntimeError, match="requires available CUDA"):
        CUDAPLCSExecutionBackend().reset_stage(
            configured_device="cuda:0",
            compositor=cast(PLCSForegroundCompositor, compositor),
        )

    assert compositor.events == []
