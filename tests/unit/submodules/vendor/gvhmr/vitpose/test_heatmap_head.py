"""Tests for the vendored ViTPose heatmap head (small config, CPU)."""

import pytest
import torch

from src.submodules.vendor.gvhmr.vitpose.heatmap_head import TopdownHeatmapSimpleHead


class TestTopdownHeatmapSimpleHead:
    def test_forward_upsamples_by_deconvs(self):
        head = TopdownHeatmapSimpleHead(
            in_channels=8,
            out_channels=17,
            num_deconv_layers=2,
            num_deconv_filters=(8, 8),
            num_deconv_kernels=(4, 4),
            extra={"final_conv_kernel": 1},
        ).eval()
        out = head(torch.rand(1, 8, 16, 12))
        # two stride-2 deconvs: 16x12 -> 32x24 -> 64x48
        assert out.shape == (1, 17, 64, 48)

    def test_zero_deconv_layers_identity(self):
        head = TopdownHeatmapSimpleHead(
            in_channels=8,
            out_channels=17,
            num_deconv_layers=0,
            num_deconv_filters=(),
            num_deconv_kernels=(),
            extra={"final_conv_kernel": 1},
        ).eval()
        out = head(torch.rand(1, 8, 16, 12))
        assert out.shape == (1, 17, 16, 12)

    def test_invalid_deconv_config_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            TopdownHeatmapSimpleHead(
                in_channels=8,
                out_channels=17,
                num_deconv_layers=2,
                num_deconv_filters=(8,),
                num_deconv_kernels=(4, 4),
            )
