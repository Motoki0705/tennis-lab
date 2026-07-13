"""Tests for spatial DINO patch-token compression."""

from __future__ import annotations

import pytest
import torch

from src.tasks.slcs.models.components.dino_adapter import DinoTokenEncoder


def test_bilinear_downsample_precedes_channel_projection() -> None:
    encoder = DinoTokenEncoder(
        input_dim=1,
        dim=4,
        grid_h=4,
        grid_w=4,
        downsample_factor=2,
    )
    observed_inputs: list[torch.Tensor] = []
    encoder.proj.register_forward_pre_hook(
        lambda _module, args: observed_inputs.append(args[0].detach().clone())
    )
    tokens = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16, 1)

    output = encoder(tokens)

    assert output.shape == (1, 1, 4, 4)
    assert observed_inputs[0].shape == (1, 1, 4, 1)
    torch.testing.assert_close(
        observed_inputs[0].flatten(), torch.tensor([2.5, 4.5, 10.5, 12.5])
    )


def test_downsample_factor_requires_divisible_grid() -> None:
    with pytest.raises(ValueError, match="divisible"):
        DinoTokenEncoder(
            input_dim=8,
            dim=16,
            grid_h=3,
            grid_w=4,
            downsample_factor=2,
        )
