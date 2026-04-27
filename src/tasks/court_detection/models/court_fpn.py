"""Configurable FPN-style court detection model."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tasks.court_detection.models.decoder import CourtFPNDecoder
from src.tasks.court_detection.models.encoders import build_court_encoder

if TYPE_CHECKING:
    from omegaconf import DictConfig


class CourtFPN(nn.Module):
    """Court detection model with pluggable encoder and shared FPN decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 7,
        *,
        encoder_name: str = "default",
        encoder_checkpoint_path: str | Path | None = None,
        encoder_dilation: bool = False,
        return_interm_indices: tuple[int, ...] = (0, 1, 2, 3),
        encoder_pretrain_img_size: int | None = None,
        encoder_use_checkpoint: bool = False,
        encoder_strict: bool = True,
        freeze_encoder: bool = True,
        decoder_channels: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.encoder_name = str(encoder_name)
        self.decoder_channels = tuple(int(channel) for channel in decoder_channels)

        self.encoder = build_court_encoder(
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            checkpoint_path=encoder_checkpoint_path,
            dilation=encoder_dilation,
            return_interm_indices=tuple(return_interm_indices),
            pretrain_img_size=encoder_pretrain_img_size,
            use_checkpoint=encoder_use_checkpoint,
            strict=encoder_strict,
            freeze_backbone=freeze_encoder,
        )
        self.decoder = CourtFPNDecoder(
            encoder_channels=self.encoder.feature_channels,
            decoder_channels=self.decoder_channels,
        )
        self.final_conv = nn.Conv2d(self.decoder.output_channels, self.num_classes, kernel_size=1)

    @classmethod
    def from_config(cls, config: DictConfig) -> CourtFPN:
        """Create the model from a composed Hydra config."""

        model_cfg = config.get("model", {}) or {}
        checkpoint_path = model_cfg.get("encoder_checkpoint_path")
        pretrain_img_size = model_cfg.get("encoder_pretrain_img_size")
        decoder_channels = tuple(
            int(channel)
            for channel in model_cfg.get("decoder_channels", [64, 128, 256, 512])
        )
        return cls(
            in_channels=int(model_cfg.get("in_channels", 3)),
            num_classes=int(model_cfg.get("num_classes", 7)),
            encoder_name=str(model_cfg.get("encoder_name", "default")),
            encoder_checkpoint_path=(
                None if checkpoint_path is None else str(checkpoint_path)
            ),
            encoder_dilation=bool(model_cfg.get("encoder_dilation", False)),
            return_interm_indices=tuple(
                int(index)
                for index in model_cfg.get("return_interm_indices", [0, 1, 2, 3])
            ),
            encoder_pretrain_img_size=(
                None if pretrain_img_size is None else int(pretrain_img_size)
            ),
            encoder_use_checkpoint=bool(model_cfg.get("encoder_use_checkpoint", False)),
            encoder_strict=bool(model_cfg.get("encoder_strict", True)),
            freeze_encoder=bool(model_cfg.get("freeze_encoder", True)),
            decoder_channels=decoder_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "CourtFPN expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )

        input_hw = x.shape[-2:]
        decoded = self.decoder(self.encoder(x))
        if decoded.shape[-2:] != input_hw:
            decoded = F.interpolate(decoded, size=input_hw, mode="bilinear", align_corners=False)
        return self.final_conv(decoded)


__all__ = ["CourtFPN"]