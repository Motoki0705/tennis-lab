# Copyright (c) OpenMMLab. All rights reserved.
"""Top-down heatmap head (trimmed from ViTPose's TopdownHeatmapSimpleHead).

Only the inference `forward` path is kept; the module layout matches the
original so released checkpoints load with identical state-dict keys.
"""

import torch.nn as nn


class TopdownHeatmapSimpleHead(nn.Module):
    """Top-down heatmap simple head, paper ref: Bin Xiao et al. ``Simple
    Baselines for Human Pose Estimation and Tracking``.

    Consists of (>=0) deconv layers followed by a simple conv2d layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=3,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        extra=None,
    ):
        super().__init__()

        self.in_channels = in_channels

        if extra is not None and not isinstance(extra, dict):
            raise TypeError("extra should be dict or None.")

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(f"num_deconv_layers ({num_deconv_layers}) should >= 0.")

        identity_final_layer = False
        if extra is not None and "final_conv_kernel" in extra:
            assert extra["final_conv_kernel"] in [0, 1, 3]
            if extra["final_conv_kernel"] == 3:
                padding = 1
            elif extra["final_conv_kernel"] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra["final_conv_kernel"]
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[-1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get("num_conv_layers", 0)
                num_conv_kernels = extra.get("num_conv_kernels", [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2,
                        )
                    )
                    layers.append(nn.BatchNorm2d(conv_channels))
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

    def forward(self, x):
        """Forward function."""
        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    @staticmethod
    def _get_deconv_cfg(deconv_kernel):
        """Get configurations for deconv layers."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError(f"Not supported num_kernels ({deconv_kernel}).")

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            raise ValueError(f"num_layers({num_layers}) != length of num_filters({len(num_filters)})")
        if num_layers != len(num_kernels):
            raise ValueError(f"num_layers({num_layers}) != length of num_kernels({len(num_kernels)})")

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)
