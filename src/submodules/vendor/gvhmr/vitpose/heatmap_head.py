# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: ViTAE-Transformer/ViTPose, commit c050ed29112da7704797cc1a65af0234b525010d
# https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/heads/topdown_heatmap_simple_head.py
# Deconvolution geometry: mmpose/models/heads/topdown_heatmap_base_head.py
# Modified for tennis-lab: inference only, validated config, native torch layers;
# no MMCV registry, training losses, input transforms, or optional upsampling.
# License text: ../LICENSE.ViTPose
"""Apache ViTPose simple head with the released checkpoint's module names."""

import torch
from torch import nn

from src.submodules.configuration import ViTPoseHeadConfig


class TopdownHeatmapSimpleHead(nn.Module):
    """Upsample a single ViT feature map and project it to joint heatmaps."""

    def __init__(self, config: ViTPoseHeadConfig) -> None:
        super().__init__()
        channels = config.in_channels
        upsampling: list[nn.Module] = []
        # Kernel 3 needs output_padding=1; all supported kernels double H/W.
        geometry = {2: (0, 0), 3: (1, 1), 4: (1, 0)}
        for width, kernel in zip(
            config.num_deconv_filters, config.num_deconv_kernels, strict=True
        ):
            padding, output_padding = geometry[kernel]
            upsampling.extend(
                (
                    nn.ConvTranspose2d(
                        channels, width, kernel, stride=2, padding=padding,
                        output_padding=output_padding, bias=False,
                    ),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                )
            )
            channels = width
        self.deconv_layers = nn.Sequential(*upsampling) if upsampling else nn.Identity()

        if config.final_conv_kernel == 0:
            self.final_layer = nn.Identity()
        else:
            projection: list[nn.Module] = []
            for kernel in config.num_conv_kernels:
                projection.extend(
                    (
                        nn.Conv2d(channels, channels, kernel, padding=(kernel - 1) // 2),
                        nn.BatchNorm2d(channels),
                        nn.ReLU(inplace=True),
                    )
                )
            projection.append(
                nn.Conv2d(
                    channels, config.out_channels, config.final_conv_kernel,
                    padding=config.final_conv_kernel // 2,
                )
            )
            # A lone Conv2d must retain final_layer.weight, without a .0 suffix.
            self.final_layer = (
                projection[0] if len(projection) == 1 else nn.Sequential(*projection)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.final_layer(self.deconv_layers(features))


__all__ = ["TopdownHeatmapSimpleHead", "ViTPoseHeadConfig"]
