"""The imported encoder preserves frame normalization and finite time support."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from src.tasks.ball_detection.models.mdd.cnn_encoder import build_mdd_encoder
from src.tasks.ball_detection.models.mdd.x3d_csn import FrameGroupNorm
from tests.support.tasks.ball_detection.temporal import small_model


def test_normalization_never_leaks_between_frames() -> None:
    norm = FrameGroupNorm(1, 3).double()
    x = torch.arange(2 * 3 * 5 * 2 * 4, dtype=torch.float64).reshape(2, 3, 5, 2, 4).square()
    expected = torch.stack([F.group_norm(x[:, :, t], 1, norm.weight, norm.bias, norm.eps) for t in range(5)], dim=2)
    torch.testing.assert_close(norm(x), expected, rtol=1e-12, atol=1e-12)


def test_feature_support_and_axis_order() -> None:
    torch.manual_seed(83)
    config = replace(small_model("dpt", frames=25), encoder_variant="x3d_csn", temporal_mixing="depthwise3d",
                     residual_blocks=(0, 1, 2, 3, 3, 2))
    encoder = build_mdd_encoder(config).double().eval()
    x = torch.randn(1, 5, 25, 17, 25, dtype=torch.float64, requires_grad=True)
    for level, radius, channels, scale in zip(encoder(x), (3, 6, 9, 11), (8, 8, 8, 16), (8, 16, 32, 64), strict=True):
        assert tuple(level.shape) == (1, 25, channels, (17 + scale - 1) // scale, (25 + scale - 1) // scale)
        weights = torch.arange(1, level[0, 12].numel() + 1, dtype=torch.float64).reshape_as(level[0, 12])
        grad, = torch.autograd.grad((level[0, 12] * weights).sum(), x, retain_graph=True)
        assert grad.abs().sum(dim=(0, 1, 3, 4)).nonzero().flatten().tolist() == list(range(12 - radius, 13 + radius))


@pytest.mark.parametrize("changes", [dict(temporal_mixing="dense3d"), dict(residual_blocks=(1, 1, 1, 1, 1, 1))])
def test_incompatible_encoder_settings_fail(changes: dict) -> None:
    config = replace(small_model("dpt"), encoder_variant="x3d_csn", temporal_mixing="depthwise3d")
    with pytest.raises(ValueError):
        replace(config, **changes)
