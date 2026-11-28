# Copyright (c) OpenMMLab. All rights reserved.
# Licensed under Apache-2.0. See LICENSE, NOTICE.md and origin.json.
"""MMPose CSPNeXt and RTMCC specialized to the published RTMPose-X checkpoint."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

WIDTH, HEIGHT = 288, 384
HALPE_JOINTS = 26
SIMCC_SPLIT = 2


class ConvBNAct(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel: int, *, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(outputs)
        self.activate = nn.SiLU(inplace=True)

    def forward(self, image: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(image)))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels // 2, 1)
        self.poolings = nn.ModuleList(nn.MaxPool2d(k, stride=1, padding=k // 2) for k in (5, 9, 13))
        self.conv2 = ConvBNAct(channels * 2, channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        features = self.conv1(features)
        return self.conv2(torch.cat([features, *(pool(features) for pool in self.poolings)], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        return features * self.act(self.fc(self.global_avgpool(features)))


class CSPNeXtBlock(nn.Module):
    def __init__(self, channels: int, *, add_identity: bool) -> None:
        super().__init__()
        self.add_identity = add_identity
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = nn.ModuleDict({
            'depthwise_conv': ConvBNAct(channels, channels, 5, groups=channels),
            'pointwise_conv': ConvBNAct(channels, channels, 1),
        })

    def forward(self, features: Tensor) -> Tensor:
        output = self.conv1(features)
        output = self.conv2['depthwise_conv'](output)
        output = self.conv2['pointwise_conv'](output)
        return output + features if self.add_identity else output


class CSPLayer(nn.Module):
    def __init__(self, channels: int, blocks: int, *, add_identity: bool) -> None:
        super().__init__()
        hidden = channels // 2
        self.main_conv = ConvBNAct(channels, hidden, 1)
        self.short_conv = ConvBNAct(channels, hidden, 1)
        self.final_conv = ConvBNAct(channels, channels, 1)
        self.blocks = nn.Sequential(*(CSPNeXtBlock(hidden, add_identity=add_identity) for _ in range(blocks)))
        self.attention = ChannelAttention(channels)

    def forward(self, features: Tensor) -> Tensor:
        short = self.short_conv(features)
        main = self.blocks(self.main_conv(features))
        return self.final_conv(self.attention(torch.cat((main, short), dim=1)))


class CSPNeXt(nn.Module):
    """P5 with width multiplier 1.25, depth multiplier 1.33, and only stride-32 output."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, 40, 3, stride=2), ConvBNAct(40, 40, 3), ConvBNAct(40, 80, 3))
        self.stage1 = nn.Sequential(ConvBNAct(80, 160, 3, stride=2), CSPLayer(160, 4, add_identity=True))
        self.stage2 = nn.Sequential(ConvBNAct(160, 320, 3, stride=2), CSPLayer(320, 8, add_identity=True))
        self.stage3 = nn.Sequential(ConvBNAct(320, 640, 3, stride=2), CSPLayer(640, 8, add_identity=True))
        self.stage4 = nn.Sequential(
            ConvBNAct(640, 1280, 3, stride=2),
            SpatialPyramidPooling(1280),
            CSPLayer(1280, 4, add_identity=False),
        )

    def forward(self, image: Tensor) -> Tensor:
        for layer in (self.stem, self.stage1, self.stage2, self.stage3, self.stage4):
            image = layer(image)
        return image


class ScaleNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = channels ** -0.5
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, features: Tensor) -> Tensor:
        norm = torch.norm(features, dim=2, keepdim=True) * self.scale
        return features / norm.clamp(min=1e-5) * self.g


class Scale(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, features: Tensor) -> Tensor:
        return features * self.scale


class RTMCCBlock(nn.Module):
    """Self-attention GAU without dropout, relative bias or rotary embeddings."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.expanded = 2 * channels
        self.attention_channels = 128
        self.sqrt_s = math.sqrt(self.attention_channels)
        self.o = nn.Linear(self.expanded, channels, bias=False)
        self.uv = nn.Linear(channels, 2 * self.expanded + self.attention_channels, bias=False)
        self.gamma = nn.Parameter(torch.empty(2, self.attention_channels))
        self.beta = nn.Parameter(torch.empty(2, self.attention_channels))
        self.act_fn = nn.SiLU(inplace=True)
        self.ln = ScaleNorm(channels)
        self.res_scale = Scale(channels)

    def forward(self, features: Tensor) -> Tensor:
        projected = self.act_fn(self.uv(self.ln(features)))
        gate, value, key_query = torch.split(projected, [self.expanded, self.expanded, self.attention_channels], dim=2)
        key_query = key_query.unsqueeze(2) * self.gamma[None, None, :] + self.beta
        query, key = torch.unbind(key_query, dim=2)
        kernel = torch.square(F.relu(torch.bmm(query, key.permute(0, 2, 1)) / self.sqrt_s))
        return self.res_scale(features) + self.o(gate * torch.bmm(kernel, value))


class RTMCCHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        spatial = (HEIGHT // 32) * (WIDTH // 32)
        hidden = 256
        self.final_layer = nn.Conv2d(1280, HALPE_JOINTS, 7, padding=3)
        self.mlp = nn.Sequential(ScaleNorm(spatial), nn.Linear(spatial, hidden, bias=False))
        self.gau = RTMCCBlock(hidden)
        self.cls_x = nn.Linear(hidden, WIDTH * SIMCC_SPLIT, bias=False)
        self.cls_y = nn.Linear(hidden, HEIGHT * SIMCC_SPLIT, bias=False)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        features = self.gau(self.mlp(torch.flatten(self.final_layer(features), 2)))
        return self.cls_x(features), self.cls_y(features)


class RTMPoseXNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = CSPNeXt()
        self.head = RTMCCHead()

    def forward(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        return self.head(self.backbone(pixels))
