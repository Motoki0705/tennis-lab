"""Spatial scale, finite temporal support and true-clip-edge chunk contracts."""

from dataclasses import replace

import pytest
import torch

from src.tasks.ball_detection.models.mdd.cnn_encoder import build_mdd_encoder
from src.tasks.ball_detection.models.mdd.x3d_csn import X3DCSNEncoder
from tests.support.tasks.ball_detection.temporal import small_mhc_model


@pytest.mark.parametrize("height,width", [(17, 25), (65, 97)])
def test_irregular_spatial_features_preserve_all_frames(height: int, width: int) -> None:
    torch.set_num_threads(2)
    config = small_mhc_model(frames=5)
    encoder = build_mdd_encoder(config)
    assert isinstance(encoder, X3DCSNEncoder)
    levels = encoder(torch.randn(2, 5, 5, height, width))
    assert [tuple(level.shape) for level in levels] == [
        (2, 5, channels, (height + scale - 1) // scale, (width + scale - 1) // scale)
        for channels, scale in zip((8, 8, 8, 16), (8, 16, 32, 64), strict=True)
    ]


def test_every_feature_has_exact_full_temporal_support_without_normalization_leak() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(83)
    config = small_mhc_model(frames=25, residual_blocks=(0, 1, 2, 3, 3, 2))
    encoder = build_mdd_encoder(config).double().eval()
    assert encoder.temporal_radius == 11
    x = torch.randn(1, 5, 25, 17, 25, dtype=torch.float64, requires_grad=True)
    for level, radius in zip(encoder(x), (3, 6, 9, 11), strict=True):
        # An asymmetric scalar avoids a normalization-invariant channel sum.
        coefficients = torch.arange(1, level[0, 12].numel() + 1, dtype=torch.float64).reshape_as(level[0, 12])
        grad, = torch.autograd.grad((level[0, 12] * coefficients).sum(), x, retain_graph=True)
        active = grad.abs().sum(dim=(0, 1, 3, 4)).nonzero().flatten().tolist()
        assert active == list(range(12 - radius, 13 + radius)), (radius, active)


@pytest.mark.parametrize("chunk_size,checkpointing", [(1, False), (6, False), (5, True)])
def test_halo_eleven_matches_full_outputs_and_all_gradients_at_true_edges(
    chunk_size: int, checkpointing: bool,
) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(87)
    config = small_mhc_model(frames=27, residual_blocks=(0, 1, 2, 3, 3, 2))
    full = build_mdd_encoder(config).double()
    chunked = build_mdd_encoder(replace(config, frame_chunk_size=chunk_size, activation_checkpointing=checkpointing)).double()
    chunked.load_state_dict(full.state_dict(), strict=True)
    x = torch.randn(1, 5, 27, 17, 25, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    expected, actual = full(x), chunked(y)
    for first, second in zip(expected, actual, strict=True):
        torch.testing.assert_close(first, second, rtol=1e-8, atol=1e-9)
    weights = torch.linspace(.3, 1.7, 27, dtype=torch.float64)[None, :, None, None, None]
    sum((level.square() * weights).mean() for level in expected).backward()
    sum((level.square() * weights).mean() for level in actual).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-9)
    for (name, first), (other, second) in zip(full.named_parameters(), chunked.named_parameters(), strict=True):
        assert name == other and first.grad is not None and second.grad is not None
        torch.testing.assert_close(first.grad, second.grad, rtol=1e-7, atol=1e-8, msg=name)
