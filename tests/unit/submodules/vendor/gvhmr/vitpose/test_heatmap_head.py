"""Tests for the vendored ViTPose heatmap head (small config, CPU)."""

import pytest
import torch

from src.submodules.vendor.gvhmr.vitpose.heatmap_head import (
    TopdownHeatmapSimpleHead,
    ViTPoseHeadConfig,
)


def _config(
    *,
    num_deconv_layers: int = 2,
    num_deconv_filters: tuple[int, ...] = (8, 8),
    num_deconv_kernels: tuple[int, ...] = (4, 4),
) -> ViTPoseHeadConfig:
    return ViTPoseHeadConfig(
        in_channels=8,
        out_channels=17,
        num_deconv_layers=num_deconv_layers,
        num_deconv_filters=num_deconv_filters,
        num_deconv_kernels=num_deconv_kernels,
        final_conv_kernel=1,
        num_conv_layers=0,
        num_conv_kernels=(),
    )


class TestTopdownHeatmapSimpleHead:
    def test_forward_upsamples_by_deconvs(self):
        head = TopdownHeatmapSimpleHead(_config()).eval()
        out = head(torch.rand(1, 8, 16, 12))
        # two stride-2 deconvs: 16x12 -> 32x24 -> 64x48
        assert out.shape == (1, 17, 64, 48)

    def test_zero_deconv_layers_identity(self):
        head = TopdownHeatmapSimpleHead(
            _config(
                num_deconv_layers=0,
                num_deconv_filters=(),
                num_deconv_kernels=(),
            )
        ).eval()
        out = head(torch.rand(1, 8, 16, 12))
        assert out.shape == (1, 17, 16, 12)

    def test_invalid_deconv_config_raises(self):
        with pytest.raises(ValueError, match="num_deconv_layers"):
            _config(
                num_deconv_layers=2,
                num_deconv_filters=(8,),
                num_deconv_kernels=(4, 4),
            )
