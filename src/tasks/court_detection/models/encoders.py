"""Reusable encoder modules for court detection models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
import torch.nn as nn

from src.tasks.court_detection.configuration import CourtEncoderConfig
from src.utils.models.blocks import Conv2dWiseWiseBlock
from src.utils.models.loading import (
    DINOv3BackboneAdapter,
    DINOv3TrainMode,
    configure_dinov3_trainability,
    load_dinov3_backbone,
)
from src.utils.models.lora import LoRAConfig

IntermediateLayerMode = Literal["uniform", "last"]


class CourtDefaultEncoder(nn.Module):
    """Default native encoder extracted from the original court U-Net."""

    feature_channels = (64, 128, 256, 512)
    requires_prepared_features = False

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = int(in_channels)

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                self.in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )
        self.enc1 = Conv2dWiseWiseBlock(self.in_channels, 64)
        self.enc2 = Conv2dWiseWiseBlock(64, 128)
        self.enc3 = Conv2dWiseWiseBlock(128, 256)
        self.bottleneck_1 = Conv2dWiseWiseBlock(256, 512)
        self.bottleneck_2 = Conv2dWiseWiseBlock(512, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)

        feat1 = self.enc1(x)
        x = self.pool(feat1)

        feat2 = self.enc2(x)
        x = self.pool(feat2)

        feat3 = self.enc3(x)
        x = self.pool(feat3)

        feat4 = self.bottleneck_2(self.bottleneck_1(x))
        return (feat1, feat2, feat3, feat4)


class CourtDINOv3Encoder(nn.Module):
    """DINOv3 ViT encoder that exposes DPT-style multi-layer feature maps."""

    requires_prepared_features = True

    def __init__(
        self,
        *,
        out_indices: Sequence[int],
        in_channels: int,
        repository_path: str | Path | None,
        checkpoint_path: str | Path | None,
        backbone_name: str | None,
        strict: bool | None,
        train_mode: DINOv3TrainMode,
        last_n_blocks: int,
        lora: LoRAConfig,
        layer_mode: IntermediateLayerMode,
        backbone: DINOv3BackboneAdapter | None,
    ) -> None:
        super().__init__()
        if backbone is None and None in (
            repository_path,
            checkpoint_path,
            backbone_name,
            strict,
        ):
            raise ValueError(
                "CourtDINOv3Encoder requires explicit DINOv3 asset settings."
            )
        self._validate_init_args(
            in_channels=in_channels,
            out_indices=out_indices,
            layer_mode=layer_mode,
        )
        self.in_channels = int(in_channels)
        self.train_mode = train_mode
        self.lora_enabled = lora.enabled
        self.layer_mode = layer_mode

        if backbone is None:
            assert repository_path is not None and checkpoint_path is not None
            assert backbone_name is not None and strict is not None
            backbone = load_dinov3_backbone(
                repository_path=Path(repository_path),
                checkpoint_path=Path(checkpoint_path),
                backbone_name=backbone_name,
                strict=strict,
            )
        self.backbone = backbone
        configure_dinov3_trainability(
            self.backbone,
            train_mode=train_mode,
            last_n_blocks=last_n_blocks,
            lora=lora,
        )
        self.patch_size = self.backbone.patch_size
        self.feature_channels = tuple([self.backbone.embed_dim] * 4)
        self.out_indices = tuple(out_indices)
        get_intermediate_layers = getattr(
            self.backbone.module,
            "get_intermediate_layers",
            None,
        )
        if not callable(get_intermediate_layers):
            raise TypeError(
                "DINOv3 backbone must expose get_intermediate_layers for DPT decoding."
            )
        self._get_intermediate_layers = cast(
            Callable[..., tuple[torch.Tensor, ...]],
            get_intermediate_layers,
        )

    @staticmethod
    def _validate_init_args(
        *,
        in_channels: int,
        out_indices: Sequence[int] | None,
        layer_mode: str,
    ) -> None:
        if in_channels != 3:
            raise ValueError("CourtDINOv3Encoder requires 3-channel RGB input.")
        if layer_mode not in {"uniform", "last"}:
            raise ValueError("layer_mode must be one of ['uniform', 'last'].")
        if out_indices is not None and len(tuple(out_indices)) != 4:
            raise ValueError("out_indices must contain exactly four layer indices.")

    def train(self, mode: bool = True) -> CourtDINOv3Encoder:
        """Keep a frozen backbone deterministic while training the decoder."""
        super().train(mode)
        if self.train_mode == "frozen" and not self.lora_enabled:
            self.backbone.eval()
        return self

def _build_dinov3_encoder(
    *, in_channels: int, config: CourtEncoderConfig
) -> CourtDINOv3Encoder:
    if None in (
        config.repository_path,
        config.checkpoint_path,
        config.backbone_name,
        config.strict,
        config.train_mode,
        config.last_n_blocks,
        config.out_indices,
        config.layer_mode,
        config.lora,
    ):
        raise AssertionError("Validated DINOv3 encoder configuration is incomplete.")
    domain_lora = config.lora
    assert domain_lora is not None
    lora = LoRAConfig(
        enabled=domain_lora.enabled,
        rank=domain_lora.rank,
        alpha=domain_lora.alpha,
        dropout=domain_lora.dropout,
        target_modules=domain_lora.target_modules,
    )
    return CourtDINOv3Encoder(
        in_channels=in_channels,
        repository_path=cast("Path", config.repository_path),
        checkpoint_path=cast("Path", config.checkpoint_path),
        backbone_name=cast("str", config.backbone_name),
        strict=cast("bool", config.strict),
        train_mode=cast("DINOv3TrainMode", config.train_mode),
        last_n_blocks=cast("int", config.last_n_blocks),
        lora=lora,
        out_indices=cast("tuple[int, ...]", config.out_indices),
        layer_mode=cast("IntermediateLayerMode", config.layer_mode),
        backbone=None,
    )


def build_court_encoder(
    *,
    config: CourtEncoderConfig,
    in_channels: int,
) -> CourtEncoder:
    """Build the requested court encoder."""

    resolved_encoder_name = config.name
    if resolved_encoder_name == "default":
        return CourtDefaultEncoder(in_channels=in_channels)
    if resolved_encoder_name == "dinov3":
        return _build_dinov3_encoder(
            in_channels=in_channels,
            config=config,
        )
    raise ValueError(
        f"Unknown court encoder '{config.name}'. Supported: ['default', 'dinov3']."
    )


CourtEncoder: TypeAlias = CourtDefaultEncoder | CourtDINOv3Encoder


__all__ = ["CourtDINOv3Encoder", "CourtDefaultEncoder", "build_court_encoder"]
