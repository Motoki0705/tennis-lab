"""HRCNet implementation for WASB ball detection.

This variant adds:
- A ConvNeXt-style residual block option ("CONVNEXT") to `_BLOCKS`.
- A MobileNetV3-inspired downsample module (inverted residual + optional SE +
  hard-swish) as `MobileNetDownsample`.

References (for design inspiration, not runtime dependency):
- ConvNeXt: "A ConvNet for the 2020s" (CVPR 2022).
- MobileNetV3: "Searching for MobileNetV3" (2019) and torchvision implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False,
    )


# ----------------------------------------------------------------------
# ResNet-style blocks
# ----------------------------------------------------------------------


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        out = out + residual
        out = self.relu(out)
        return out


# ----------------------------------------------------------------------
# ConvNeXt-style block
# ----------------------------------------------------------------------


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last or channels_first.

    - channels_last: (B, H, W, C)  -> use F.layer_norm
    - channels_first: (B, C, H, W) -> manual per-channel normalization
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unknown data_format: {data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # channels_first
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth (a.k.a. DropPath)."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """DropPath module wrapper."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ConvNeXtBlock(nn.Module):
    """ConvNeXt-style residual block.

    Implementation (common PyTorch form):
        DwConv(7x7) -> permute (NHWC) -> LN -> Linear(4C) -> GELU
        -> Linear(C) -> (LayerScale gamma) -> permute back (NCHW)
        -> Residual + DropPath
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        # Spatial mixing (depthwise conv). Stride is supported for completeness,
        # but typical ConvNeXt blocks use stride=1 (downsampling is separate).
        self.dwconv = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=inplanes,
            bias=False,
        )

        # Channel mixing (MLP-style) in channels_last format.
        self.norm = LayerNorm(inplanes, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(inplanes, 4 * inplanes)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * inplanes, inplanes)

        # Layer scale (small init helps stability in deep nets).
        self.gamma = nn.Parameter(1e-6 * torch.ones((inplanes,)), requires_grad=True)

        # Stochastic depth (kept off by default; set externally if you want).
        self.drop_path = DropPath(0.0)

        # Residual projection if spatial/feature dims change.
        if downsample is not None:
            self.residual = downsample
        elif stride != 1 or inplanes != planes:
            self.residual = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
        else:
            self.residual = nn.Identity()

        # If planes differs from inplanes, add a final 1x1 projection on main path.
        self.out_proj = nn.Identity() if inplanes == planes else nn.Sequential(
            conv1x1(inplanes, planes, stride=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.dwconv(x)                      # (B, C, H, W)
        out = out.permute(0, 2, 3, 1)             # (B, H, W, C)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.gamma * out
        out = out.permute(0, 3, 1, 2)             # (B, C, H, W)

        out = self.out_proj(out)
        out = residual + self.drop_path(out)
        return out


_BLOCKS = {
    "BASIC": BasicBlock,
    "BOTTLENECK": Bottleneck,
    "CONVNEXT": ConvNeXtBlock,
}


# ----------------------------------------------------------------------
# MobileNetV3-inspired downsample
# ----------------------------------------------------------------------


class HardSigmoid(nn.Module):
    """Hard-sigmoid used in MobileNetV3 (piecewise-linear sigmoid)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(x + 3.0) / 6.0


class HardSwish(nn.Module):
    """Hard-swish used in MobileNetV3: x * hard_sigmoid(x)."""

    def __init__(self) -> None:
        super().__init__()
        self.hsigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hsigmoid(x)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation (SE) block (MobileNetV3-style gating)."""

    def __init__(self, channels: int, squeeze_factor: int = 4) -> None:
        super().__init__()
        squeezed = max(channels // squeeze_factor, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, squeezed, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed, channels, kernel_size=1)
        self.gate = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.avg_pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class ConvBNAct(nn.Module):
    """Conv -> BN -> Activation helper."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int = 1,
        act: Optional[nn.Module] = None,
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch, momentum=bn_momentum)
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))



class InvertedResidualV3Explicit(nn.Module):
    """Same as InvertedResidualV3, but with explicit steps for clarity."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expansion: float = 4.0,
        kernel_size: int = 3,
        use_se: bool = True,
        activation: str = "hswish",
        bn_momentum: float = BN_MOMENTUM,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")
        if kernel_size not in (3, 5):
            raise ValueError(f"kernel_size must be 3 or 5, got {kernel_size}")

        if activation == "hswish":
            act = HardSwish()
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        hidden = int(round(in_ch * expansion))
        self.use_res = stride == 1 and in_ch == out_ch

        self.expand = (
            ConvBNAct(
                in_ch,
                hidden,
                kernel_size=1,
                stride=1,
                padding=0,
                act=act,
                bn_momentum=bn_momentum,
            )
            if hidden != in_ch
            else nn.Identity()
        )

        pad = kernel_size // 2
        self.dw = ConvBNAct(
            hidden,
            hidden,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=hidden,
            act=act,
            bn_momentum=bn_momentum,
        )

        self.se = SqueezeExcite(hidden) if use_se else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = out + x
        return out


class MobileNetDownsample(nn.Module):
    """MobileNetV3-inspired downsample module.

    This replaces the old "stacked depthwise stride-2 convs" with a small stack of
    MobileNetV3-style inverted residual blocks (dw + optional SE + hard-swish).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_downsample: int = 4,
        bn_momentum: float = BN_MOMENTUM,
        expansion: float = 4.0,
        kernel_size: int = 3,
        use_se: bool = True,
        activation: str = "hswish",
    ) -> None:
        super().__init__()

        if num_downsample < 0:
            raise ValueError("num_downsample must be >= 0.")

        layers: list[nn.Module] = []

        if num_downsample == 0:
            # Only channel adaptation (if needed).
            if in_ch != out_ch:
                act = HardSwish() if activation == "hswish" else nn.ReLU(inplace=True)
                layers.append(
                    ConvBNAct(
                        in_ch,
                        out_ch,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        act=act,
                        bn_momentum=bn_momentum,
                    )
                )
            self.down = nn.Sequential(*layers) if layers else nn.Identity()
            return

        # First downsample step also adapts channels to out_ch.
        layers.append(
            InvertedResidualV3Explicit(
                in_ch=in_ch,
                out_ch=out_ch,
                stride=2,
                expansion=expansion,
                kernel_size=kernel_size,
                use_se=use_se,
                activation=activation,
                bn_momentum=bn_momentum,
            )
        )

        # Remaining downsample steps keep channels == out_ch.
        for _ in range(num_downsample - 1):
            layers.append(
                InvertedResidualV3Explicit(
                    in_ch=out_ch,
                    out_ch=out_ch,
                    stride=2,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    use_se=use_se,
                    activation=activation,
                    bn_momentum=bn_momentum,
                )
            )

        self.down = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


# ----------------------------------------------------------------------
# Two-resolution fusion
# ----------------------------------------------------------------------


class ContextFusionBlock(nn.Module):
    """Two-resolution fusion block with repeated high/low interaction.

    Updates a high-resolution branch (H, W) and a low-resolution branch
    (H_low, W_low) and lets them exchange information both directions:

    1. High: stack of residual blocks.
    2. Low : stack of residual blocks (optional) + transformer encoder.
    3. Low -> High: upsample + 1x1 proj + add.
    4. High -> Low: pool   + 1x1 proj + add.
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

        self.low_cnn = (
            nn.Sequential(
                *[
                    low_block_cls(low_channels, low_channels)
                    for _ in range(max(num_low_blocks, 0))
                ]
            )
            if num_low_blocks > 0
            else nn.Identity()
        )

        t_kwargs = dict(transformer_kwargs or {})
        d_model = t_kwargs.pop("d_model", low_channels)
        num_heads = t_kwargs.pop("num_heads", 8)
        dim_ff = t_kwargs.pop("dim_ff", d_model * 4)
        dropout = t_kwargs.pop("dropout", 0.1)
        depth = t_kwargs.pop("depth", 2)

        if d_model != low_channels:
            raise ValueError(f"d_model ({d_model}) must match low_channels ({low_channels}).")

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
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

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

    def forward(self, high: torch.Tensor, low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        high = self.high_path(high)

        low = self.low_cnn(low)
        b, c, h, w = low.shape
        low_seq = low.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        low_seq = self.transformer(low_seq)
        low = low_seq.permute(0, 2, 1).view(b, c, h, w)

        # Low -> High
        low_up = F.interpolate(low, size=high.shape[-2:], mode=self.upsample_mode)
        low_up = self.low_to_high(low_up)
        high = self.activation(high + low_up)

        # High -> Low
        pooled = F.adaptive_avg_pool2d(high, output_size=low.shape[-2:])
        pooled = self.high_to_low(pooled)
        low = self.activation(low + pooled)

        return high, low


# ----------------------------------------------------------------------
# HRCNet
# ----------------------------------------------------------------------


class HRCNet(nn.Module):
    """High-Resolution Context Net (HRCNet).

    Maintains a high-resolution convolutional branch and a low-resolution
    transformer branch. They exchange information multiple times via
    ContextFusionBlocks.

    Flow:
        input -> stem -> high
             high -> MobileNetDownsample -> low  (typically 1/16 resolution)
        repeat num_stages:
            (high, low) = ContextFusionBlock(high, low)
        head uses high.
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
            nn.Conv2d(in_channels, high_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(high_channels, high_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(high_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )

        self.initial_down = MobileNetDownsample(
            in_ch=high_channels,
            out_ch=low_channels,
            **(downsample_kwargs or {}),
        )

        self.stages = nn.ModuleList(
            [
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
                for _ in range(num_stages)
            ]
        )

        self.head = nn.Conv2d(high_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolution/linear and normalization layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the HRCNet trunk and return the final high-resolution features."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {x.shape}.")

        high = self.stem(x)
        low = self.initial_down(high)

        for stage in self.stages:
            high, low = stage(high, low)

        return high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the HRCNet forward pass and return the high-res prediction map.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).

        Returns:
            out: Tensor of shape (B, out_channels, H, W).
        """
        high = self.forward_features(x)
        out = self.head(high)
        return out


if __name__ == "__main__":
    inputs = torch.rand(1, 3, 256, 256)
    model = HRCNet(
        in_channels=3,
        out_channels=1,
        high_channels=64,
        low_channels=64,
        # Example: use ConvNeXt blocks on high branch
        # high_block="CONVNEXT",
    )
    outputs = model(inputs)
    print(outputs.shape)
