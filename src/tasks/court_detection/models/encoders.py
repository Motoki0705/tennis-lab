"""Reusable encoder modules for court detection models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.models.blocks import Conv2dWiseWiseBlock
from src.utils.models.loading import (
    DEFAULT_DINOV3_CHECKPOINT,
    DEFAULT_DINOV3_LORA_TARGET_MODULES,
    DEFAULT_DINOV3_REPOSITORY,
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

    def __init__(self, in_channels: int = 3) -> None:
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
        self._validate_forward_input(x)

        x = self.stem(x)

        feat1 = self.enc1(x)
        x = self.pool(feat1)

        feat2 = self.enc2(x)
        x = self.pool(feat2)

        feat3 = self.enc3(x)
        x = self.pool(feat3)

        feat4 = self.bottleneck_2(self.bottleneck_1(x))
        return (feat1, feat2, feat3, feat4)

    def _validate_forward_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "CourtDefaultEncoder expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )


class CourtDINOv3Encoder(nn.Module):
    """DINOv3 ViT encoder that exposes DPT-style multi-layer feature maps."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        repository_path: str | Path = DEFAULT_DINOV3_REPOSITORY,
        checkpoint_path: str | Path = DEFAULT_DINOV3_CHECKPOINT,
        backbone_name: str = "dinov3_vitb16",
        strict: bool = True,
        train_mode: DINOv3TrainMode = "frozen",
        last_n_blocks: int = 0,
        lora: LoRAConfig | None = None,
        out_indices: Sequence[int] | None = None,
        layer_mode: IntermediateLayerMode = "uniform",
        backbone: DINOv3BackboneAdapter | None = None,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            in_channels=in_channels,
            out_indices=out_indices,
            layer_mode=layer_mode,
        )
        self.in_channels = int(in_channels)
        self.train_mode = train_mode
        self.lora_enabled = bool(lora is not None and lora.enabled)
        self.layer_mode = layer_mode

        self.backbone = backbone or load_dinov3_backbone(
            repository_path=repository_path,
            checkpoint_path=checkpoint_path,
            backbone_name=backbone_name,
            strict=strict,
        )
        configure_dinov3_trainability(
            self.backbone,
            train_mode=train_mode,
            last_n_blocks=last_n_blocks,
            lora=lora,
        )
        self.patch_size = self.backbone.patch_size
        self.feature_channels = tuple([self.backbone.embed_dim] * 4)
        self.out_indices = self._resolve_out_indices(out_indices)

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

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_forward_input(x)
        padded_x = self._pad_to_patch_grid(x)
        patch_height = padded_x.shape[-2] // self.patch_size
        patch_width = padded_x.shape[-1] // self.patch_size
        tokens = self._extract_intermediate_tokens(padded_x)
        expected_tokens = patch_height * patch_width

        feature_maps = []
        for level_tokens in tokens:
            if level_tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    "DINOv3 patch-token count does not match the input grid: "
                    f"expected {expected_tokens}, got {level_tokens.shape[1]}."
                )
            feature_maps.append(
                level_tokens.transpose(1, 2).reshape(
                    padded_x.shape[0],
                    level_tokens.shape[-1],
                    patch_height,
                    patch_width,
                )
            )
        return cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple(feature_maps),
        )

    def _extract_intermediate_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.train_mode == "frozen" and not self.lora_enabled:
            with torch.no_grad():
                return self._call_get_intermediate_layers(x)
        return self._call_get_intermediate_layers(x)

    def _pad_to_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
        pad_h = (-x.shape[-2]) % self.patch_size
        pad_w = (-x.shape[-1]) % self.patch_size
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    def _call_get_intermediate_layers(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        get_intermediate_layers = getattr(
            self.backbone.module,
            "get_intermediate_layers",
            None,
        )
        if not callable(get_intermediate_layers):
            raise TypeError(
                "DINOv3 backbone must expose get_intermediate_layers for DPT decoding."
            )

        outputs = get_intermediate_layers(
            x,
            n=self.out_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if not isinstance(outputs, (tuple, list)):
            raise TypeError("get_intermediate_layers must return a tuple/list.")
        if len(outputs) != 4:
            raise ValueError(
                "get_intermediate_layers must return exactly four tensors, "
                f"got {len(outputs)}."
            )
        tokens = tuple(self._normalize_intermediate_output(output) for output in outputs)
        for token in tokens:
            if token.ndim != 3:
                raise ValueError(
                    "DINOv3 intermediate tokens must have shape (B, N, C), "
                    f"got {tuple(token.shape)}."
                )
            if token.shape[-1] != self.backbone.embed_dim:
                raise ValueError(
                    "DINOv3 intermediate width does not match embed_dim: "
                    f"{token.shape[-1]} != {self.backbone.embed_dim}."
                )
        return tokens

    @staticmethod
    def _normalize_intermediate_output(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if (
            isinstance(output, (tuple, list))
            and output
            and isinstance(output[0], torch.Tensor)
        ):
            return output[0]
        raise TypeError(
            "Each DINOv3 intermediate output must be a tensor or a tuple/list "
            "whose first item is a tensor."
        )

    def _resolve_out_indices(self, out_indices: Sequence[int] | None) -> tuple[int, ...]:
        if out_indices is not None:
            return tuple(int(index) for index in out_indices)

        depth = len(self.backbone.transformer_blocks())
        if depth < 4:
            raise ValueError(f"DINOv3 backbone depth must be at least 4, got {depth}.")
        if self.layer_mode == "last":
            return tuple(range(depth - 4, depth))
        return tuple(
            round(depth * fraction) - 1 for fraction in (0.25, 0.5, 0.75, 1.0)
        )

    def _validate_forward_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                "CourtDINOv3Encoder expects input with shape (B, C, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )
        if min(x.shape[-2:]) <= 0:
            raise ValueError("Input height and width must be positive.")


def _build_dinov3_encoder(
    *,
    in_channels: int,
    encoder_config: Mapping[str, Any],
) -> CourtDINOv3Encoder:
    lora = LoRAConfig.from_mapping(
        encoder_config.get("lora"),
        default_target_modules=DEFAULT_DINOV3_LORA_TARGET_MODULES,
    )
    return CourtDINOv3Encoder(
        in_channels=in_channels,
        repository_path=encoder_config.get("repository_path", DEFAULT_DINOV3_REPOSITORY),
        checkpoint_path=encoder_config.get("checkpoint_path", DEFAULT_DINOV3_CHECKPOINT),
        backbone_name=str(encoder_config.get("backbone_name", "dinov3_vitb16")),
        strict=bool(encoder_config.get("strict", True)),
        train_mode=cast(
            DINOv3TrainMode,
            str(encoder_config.get("train_mode", "frozen")),
        ),
        last_n_blocks=int(encoder_config.get("last_n_blocks", 0)),
        lora=lora,
        out_indices=encoder_config.get("out_indices"),
        layer_mode=cast(
            IntermediateLayerMode,
            str(encoder_config.get("layer_mode", "uniform")),
        ),
    )


def build_court_encoder(
    *,
    encoder_name: str = "default",
    in_channels: int = 3,
    encoder_config: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build the requested court encoder."""

    resolved_encoder_name = str(encoder_name)
    if resolved_encoder_name == "default":
        return CourtDefaultEncoder(in_channels=in_channels)
    if resolved_encoder_name == "dinov3":
        return _build_dinov3_encoder(
            in_channels=in_channels,
            encoder_config=encoder_config or {},
        )
    raise ValueError(
        f"Unknown court encoder '{encoder_name}'. Supported: ['default', 'dinov3']."
    )


__all__ = ["CourtDINOv3Encoder", "CourtDefaultEncoder", "build_court_encoder"]
