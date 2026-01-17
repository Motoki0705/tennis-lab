"""Backbone networks for court keypoint detection."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a backbone network.

    Args:
        name: Backbone name ('resnet50', 'resnet101', 'hrnet_w32').
        pretrained: Whether to use pretrained weights.

    Returns:
        Tuple of (backbone_module, output_channels).
    """
    if name == "resnet50":
        return _build_resnet(models.resnet50, ResNet50_Weights.DEFAULT, pretrained)
    elif name == "resnet101":
        return _build_resnet(models.resnet101, ResNet101_Weights.DEFAULT, pretrained)
    elif name == "hrnet_w32":
        return _build_simple_resnet_backbone(pretrained)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def _build_resnet(
    model_fn,
    weights,
    pretrained: bool,
) -> tuple[nn.Module, int]:
    """Build a ResNet backbone without the final FC layer."""
    if pretrained:
        model = model_fn(weights=weights)
    else:
        model = model_fn(weights=None)

    # Remove avgpool and fc layers
    backbone = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )

    # ResNet50/101 layer4 output is 2048 channels
    out_channels = 2048

    return backbone, out_channels


def _build_simple_resnet_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    """Build a simple ResNet-based backbone as HRNet placeholder.

    Note: For a proper HRNet implementation, consider using timm or
    a dedicated HRNet library.
    """
    if pretrained:
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = models.resnet50(weights=None)

    backbone = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    )

    return backbone, 2048
