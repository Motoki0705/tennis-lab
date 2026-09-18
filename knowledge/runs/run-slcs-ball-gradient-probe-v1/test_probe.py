"""Tiny CPU-only checks; no dataset/checkpoint loading."""

import importlib.util
from pathlib import Path

import torch


def test_gradient_geometry_and_unused_entries() -> None:
    spec = importlib.util.spec_from_file_location(
        "gradient_probe", Path(__file__).with_name("probe.py")
    )
    assert spec is not None and spec.loader is not None
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    first = torch.tensor([1.0, 2.0], requires_grad=True)
    unused = torch.tensor([3.0], requires_grad=True)
    objectives = {
        "a": (first * torch.tensor([3.0, 4.0])).sum(),
        "b": (first * torch.tensor([-4.0, 3.0])).sum(),
        "zero": (first * 0).sum(),
    }
    result = probe.gradient_summary(objectives, [first, unused])
    assert result["norms"] == {"a": 5.0, "b": 5.0, "zero": 0.0}
    assert result["pairs"]["a__b"] == {"dot": 0.0, "cosine": 0.0}
    assert result["pairs"]["a__zero"] == {"dot": 0.0, "cosine": None}
    assert result["unused_as_zero"]["a"] == {"unused_tensors": 1, "unused_elements": 1}
    assert first.grad is None and unused.grad is None
    assert result["element_count"] == 3
