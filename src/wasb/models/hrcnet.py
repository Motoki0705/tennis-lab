from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


_BLOCKS = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
}


class MobileNetDownsample(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_downsample: int = 4,
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_channels = in_ch
        if in_ch != out_ch:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_ch, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_ch

        for _ in range(max(num_downsample, 0)):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_channels,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(in_channels, momentum=bn_momentum))
            layers.append(nn.ReLU(inplace=True))

        self.down = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class ContextFusionBlock(nn.Module):
    """Two-resolution fusion block with repeated high/low interaction.

    This block updates a high-resolution branch (H, W) and a low-resolution
    branch (H_low, W_low) and lets them exchange information in both directions:

    1. High branch is updated by a stack of residual blocks.
    2. Low branch is updated by a stack of residual blocks and a transformer encoder.
    3. Low → High: low is upsampled and projected, then added to high.
    4. High → Low: high is pooled and projected, then added to low.

    The shapes must satisfy:
        high: (B, C_high, H, W)
        low : (B, C_low,  H_low, W_low)
    """

    def __init__(
        self,
        high_channels: int,
        low_channels: int,
        high_block: str = "BASIC",
        low_block: str = "BASIC",
        num_high_blocks: int = 2,
        num_low_blocks: int = 1,
        upsample_mode: str = "nearest",
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if high_block not in _BLOCKS:
            raise ValueError(f"Unknown high_block '{high_block}'.")
        if low_block not in _BLOCKS:
            raise ValueError(f"Unknown low_block '{low_block}'.")

        high_block_cls = _BLOCKS[high_block]
        low_block_cls = _BLOCKS[low_block]

        self.high_path = nn.Sequential(
            *[
                high_block_cls(high_channels, high_channels)
                for _ in range(max(num_high_blocks, 1))
            ]
        )

        self.low_cnn = nn.Sequential(
            *[
                low_block_cls(low_channels, low_channels)
                for _ in range(max(num_low_blocks, 0))
            ]
        ) if num_low_blocks > 0 else nn.Identity()
        t_kwargs = dict(transformer_kwargs or {})
        d_model = t_kwargs.pop("d_model", low_channels)
        num_heads = t_kwargs.pop("num_heads", 8)
        dim_ff = t_kwargs.pop("dim_ff", d_model * 4)
        dropout = t_kwargs.pop("dropout", 0.1)
        depth = t_kwargs.pop("depth", 2)

        if d_model != low_channels:
            raise ValueError(
                f"d_model ({d_model}) must match low_channels ({low_channels})."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            **t_kwargs,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.low_to_high = nn.Sequential(
            nn.Conv2d(low_channels, high_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(high_channels, low_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(low_channels, momentum=BN_MOMENTUM),
        )

        self.upsample_mode = upsample_mode
        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one fusion step.

        Args:
            high: High-resolution feature map, shape (B, C_high, H, W).
            low: Low-resolution feature map, shape (B, C_low, H_low, W_low).

        Returns:
            Tuple of updated (high, low) feature maps with the same shapes as inputs.
        """
        high = self.high_path(high)
        low = self.low_cnn(low)
        b, c, h, w = low.shape
        low_seq = low.flatten(2).permute(0, 2, 1)
        low_seq = self.transformer(low_seq)
        low = low_seq.permute(0, 2, 1).view(b, c, h, w)

        low_up = F.interpolate(low, size=high.shape[-2:], mode=self.upsample_mode)
        low_up = self.low_to_high(low_up)
        high = self.activation(high + low_up)

        pooled = F.adaptive_avg_pool2d(high, output_size=low.shape[-2:])
        pooled = self.high_to_low(pooled)
        low = self.activation(low + pooled)

        return high, low


class HRCNet(nn.Module):
    """High-Resolution Context Net (HRCNet).

    This architecture maintains a high-resolution convolutional branch and a
    low-resolution transformer branch. They exchange information multiple times
    in a stack of ContextFusionBlocks, in a spirit similar to HRNet but with
    only two resolutions and a transformer in the low-resolution branch.

    The overall flow is:

        input -> stem -> branch0 (high)
            branch0 -> MobileNetDownsample -> branch4 (low, 1/16 resolution)
            for each fusion stage:
                (branch0, branch4) = ContextFusionBlock(branch0, branch4)
        output head uses branch0 (high resolution).

    Only the high-resolution output is intended for downstream tasks, but the
    low-resolution features are kept for diagnostics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        high_channels: int,
        low_channels: int,
        num_stages: int = 3,
        high_block: str = "BASIC",
        low_block: str = "BASIC",
        num_high_blocks: int = 2,
        num_low_blocks: int = 1,
        upsample_mode: str = "nearest",
        downsample_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if num_stages < 1:
            raise ValueError("num_stages must be >= 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.high_channels = high_channels
        self.low_channels = low_channels
        self.num_stages = num_stages

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                high_channels,
                high_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.initial_down = MobileNetDownsample(
            in_ch=high_channels,
            out_ch=low_channels,
            **(downsample_kwargs or {}),
        )

        stages: list[ContextFusionBlock] = []
        for _ in range(num_stages):
            stages.append(
                ContextFusionBlock(
                    high_channels=high_channels,
                    low_channels=low_channels,
                    high_block=high_block,
                    low_block=low_block,
                    num_high_blocks=num_high_blocks,
                    num_low_blocks=num_low_blocks,
                    upsample_mode=upsample_mode,
                    transformer_kwargs=transformer_kwargs,
                )
            )
        self.stages = nn.ModuleList(stages)

        self.head = nn.Conv2d(
            high_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution and normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the HRCNet forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            Dict with:
                - "out": high-resolution output, shape (B, out_channels, H, W)
                - "high": final high-resolution features
                - "low": final low-resolution features
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {x.shape}.")

        high = self.stem(x)
        low = self.initial_down(high)

        for stage in self.stages:
            high, low = stage(high, low)

        out = self.head(high)

        return out

if __name__ == "__main__":
    inputs = torch.rand(1, 3, 256, 256)
    model = HRCNet(
        in_channels=3,
        out_channels=1,
        high_channels=64,
        low_channels=64,
    )
    outputs = model(inputs)
    print(outputs)
