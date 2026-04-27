"""Reusable encoder modules for court detection models."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.tasks.court_detection.models.blocks import Conv2dWiseWiseBlock
from src.utils.models.loading import load_dino_backbone

_DINO_ENCODER_ALIASES = {
    "resnet": "resnet50",
    "swin": "swin_T_224_1k",
}


class CourtDefaultEncoder(nn.Module):
    """Default native encoder extracted from the original court U-Net."""

    feature_channels = (64, 128, 256, 512)

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = Conv2dWiseWiseBlock(self.in_channels, 64)
        self.enc2 = Conv2dWiseWiseBlock(64, 128)
        self.enc3 = Conv2dWiseWiseBlock(128, 256)
        self.bottleneck_1 = Conv2dWiseWiseBlock(256, 512)
        self.bottleneck_2 = Conv2dWiseWiseBlock(512, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                "CourtDefaultEncoder expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )

        x = self.stem(x)

        feat1 = self.enc1(x)
        x = self.pool(feat1)

        feat2 = self.enc2(x)
        x = self.pool(feat2)

        feat3 = self.enc3(x)
        x = self.pool(feat3)

        feat4 = self.bottleneck_2(self.bottleneck_1(x))
        return (feat1, feat2, feat3, feat4)


class CourtDINOEncoder(nn.Module):
    """Court encoder backed by a reusable DINO backbone loader."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        backbone_name: str = "resnet50",
        checkpoint_path: str | Path | None = None,
        dilation: bool = False,
        return_interm_indices: tuple[int, ...] = (0, 1, 2, 3),
        pretrain_img_size: int | None = None,
        use_checkpoint: bool = False,
        strict: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(
                "CourtDINOEncoder requires 3-channel RGB input, "
                f"got in_channels={in_channels}."
            )
        if tuple(return_interm_indices) != (0, 1, 2, 3):
            raise ValueError("CourtDINOEncoder expects return_interm_indices=(0, 1, 2, 3).")

        self.in_channels = int(in_channels)
        self.backbone_name = str(backbone_name)
        self.return_interm_indices = tuple(return_interm_indices)

        loaded_backbone = load_dino_backbone(
            backbone_name=self.backbone_name,
            checkpoint_path=checkpoint_path,
            dilation=dilation,
            return_interm_indices=self.return_interm_indices,
            pretrain_img_size=pretrain_img_size,
            use_checkpoint=use_checkpoint,
            strict=strict,
        )
        self.backbone = loaded_backbone.module
        self.feature_channels = loaded_backbone.metadata.feature_channels

        if strict and (
            loaded_backbone.load_result.missing_keys or loaded_backbone.load_result.unexpected_keys
        ):
            raise RuntimeError(
                "Unexpected DINO backbone load result: "
                f"missing={loaded_backbone.load_result.missing_keys}, "
                f"unexpected={loaded_backbone.load_result.unexpected_keys}"
            )
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                "CourtDINOEncoder expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )

        features = self.backbone(x)
        return tuple(
            features[str(index)]
            for index in self.return_interm_indices
        )


def build_court_encoder(
    *,
    encoder_name: str = "default",
    in_channels: int = 3,
    checkpoint_path: str | Path | None = None,
    dilation: bool = False,
    return_interm_indices: tuple[int, ...] = (0, 1, 2, 3),
    pretrain_img_size: int | None = None,
    use_checkpoint: bool = False,
    strict: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build the requested court encoder."""

    resolved_encoder_name = _DINO_ENCODER_ALIASES.get(str(encoder_name), str(encoder_name))
    if resolved_encoder_name == "default":
        return CourtDefaultEncoder(in_channels=in_channels)
    return CourtDINOEncoder(
        in_channels=in_channels,
        backbone_name=resolved_encoder_name,
        checkpoint_path=checkpoint_path,
        dilation=dilation,
        return_interm_indices=return_interm_indices,
        pretrain_img_size=pretrain_img_size,
        use_checkpoint=use_checkpoint,
        strict=strict,
        freeze_backbone=freeze_backbone,
    )


__all__ = ["CourtDINOEncoder", "CourtDefaultEncoder", "build_court_encoder"]