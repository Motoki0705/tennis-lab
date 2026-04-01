"""DINO-backbone pseudo-3D decoder for ball detection."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from checkpoints.DINO.scripts.load_dino_backbone import load_backbone_body_state
from src.tasks.ball_detection.input_adapter import resolve_model_in_channels

if TYPE_CHECKING:
    from omegaconf import DictConfig


class ConvBNAct3d(nn.Sequential):
    """Conv3d-BN-(ReLU6) helper used by the decoder blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidualPseudo3D(nn.Module):
    """Pseudo-3D inverted residual block with spatial then temporal mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int] = (1, 1, 1),
        expand_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        hidden_channels = int(round(in_channels * expand_ratio))
        self.use_residual = stride == (1, 1, 1) and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct3d(in_channels, hidden_channels, kernel_size=1),
            ConvBNAct3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3, 3),
                stride=stride,
                padding=(0, 1, 1),
                groups=hidden_channels,
            ),
            ConvBNAct3d(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=(1, 0, 0),
                groups=hidden_channels,
            ),
            ConvBNAct3d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                activation=False,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the pseudo-3D block and optional residual connection."""
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


def _make_decoder_stage(
    in_channels: int,
    mid_channels: int,
    out_channels: int,
    *,
    expand_ratio: float,
) -> nn.Sequential:
    """Build one 3-block decoder stage."""
    return nn.Sequential(
        InvertedResidualPseudo3D(
            in_channels=in_channels,
            out_channels=mid_channels,
            expand_ratio=expand_ratio,
        ),
        InvertedResidualPseudo3D(
            in_channels=mid_channels,
            out_channels=mid_channels,
            expand_ratio=expand_ratio,
        ),
        InvertedResidualPseudo3D(
            in_channels=mid_channels,
            out_channels=out_channels,
            expand_ratio=expand_ratio,
        ),
    )


class DINOPseudo3DBallDetector(nn.Module):
    """Ball detector with a DINO ResNet backbone and pseudo-3D decoder."""

    _FEATURE_CHANNELS = (256, 512, 1024, 2048)

    def __init__(
        self,
        *,
        in_channels: int = 3,
        num_classes: int = 1,
        backbone_checkpoint_path: str | Path = "/workspace/checkpoints/DINO/backbone_body_state.pth",
        backbone_name: str = "resnet50",
        backbone_dilation: bool = False,
        return_interm_indices: tuple[int, ...] = (0, 1, 2, 3),
        backbone_strict: bool = True,
        freeze_backbone: bool = True,
        expand_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError(
                "DINOPseudo3DBallDetector requires 3-channel RGB input, "
                f"got in_channels={in_channels}."
            )
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if tuple(return_interm_indices) != (0, 1, 2, 3):
            raise ValueError(
                "DINOPseudo3DBallDetector expects return_interm_indices=(0, 1, 2, 3)."
            )

        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.return_interm_indices = tuple(return_interm_indices)
        self.freeze_backbone = bool(freeze_backbone)

        self.backbone, load_result = load_backbone_body_state(
            checkpoint_path=Path(backbone_checkpoint_path),
            backbone=backbone_name,
            dilation=backbone_dilation,
            return_interm_indices=list(self.return_interm_indices),
            strict=backbone_strict,
        )
        if backbone_strict and (
            load_result.missing_keys or load_result.unexpected_keys
        ):
            raise RuntimeError(
                "Unexpected DINO backbone load result: "
                f"missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}"
            )
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        c1, c2, c3, c4 = self._FEATURE_CHANNELS
        self.decode4 = _make_decoder_stage(
            in_channels=c4,
            mid_channels=c4,
            out_channels=c3,
            expand_ratio=expand_ratio,
        )
        self.decode3 = _make_decoder_stage(
            in_channels=c3 + c3,
            mid_channels=c3,
            out_channels=c2,
            expand_ratio=expand_ratio,
        )
        self.decode2 = _make_decoder_stage(
            in_channels=c2 + c2,
            mid_channels=c2,
            out_channels=c1,
            expand_ratio=expand_ratio,
        )
        self.decode1 = _make_decoder_stage(
            in_channels=c1 + c1,
            mid_channels=c1,
            out_channels=64,
            expand_ratio=expand_ratio,
        )
        self.final_refine = nn.Sequential(
            ConvBNAct3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            ConvBNAct3d(64, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        self.final_conv = nn.Conv3d(64, self.num_classes, kernel_size=1)

    @classmethod
    def from_config(cls, config: DictConfig) -> "DINOPseudo3DBallDetector":
        """Create the model from a composed Hydra config."""
        model_cfg = config.get("model", {}) or {}
        return cls(
            in_channels=resolve_model_in_channels(model_cfg),
            num_classes=int(model_cfg.get("num_classes", 1)),
            backbone_checkpoint_path=str(
                model_cfg.get(
                    "backbone_checkpoint_path",
                    "/workspace/checkpoints/DINO/backbone_body_state.pth",
                )
            ),
            backbone_name=str(model_cfg.get("backbone_name", "resnet50")),
            backbone_dilation=bool(model_cfg.get("backbone_dilation", False)),
            return_interm_indices=tuple(
                int(index) for index in model_cfg.get("return_interm_indices", [0, 1, 2, 3])
            ),
            backbone_strict=bool(model_cfg.get("backbone_strict", True)),
            freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
            expand_ratio=float(model_cfg.get("decoder_expand_ratio", 4.0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the DINO backbone and pseudo-3D decoder."""
        if x.ndim != 5:
            raise ValueError(
                "DINOPseudo3DBallDetector expects input with shape (B, C, T, H, W), "
                f"got ndim={x.ndim}."
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels but received {x.shape[1]}."
            )

        batch_size, channels, timesteps, height, width = x.shape
        backbone_input = x.permute(0, 2, 1, 3, 4).contiguous().view(
            batch_size * timesteps,
            channels,
            height,
            width,
        )
        features = self.backbone(backbone_input)
        level1 = self._to_5d(features["0"], batch_size, timesteps)
        level2 = self._to_5d(features["1"], batch_size, timesteps)
        level3 = self._to_5d(features["2"], batch_size, timesteps)
        level4 = self._to_5d(features["3"], batch_size, timesteps)

        x4 = self.decode4(level4)
        x3 = self.decode3(torch.cat([self._upsample_to(x4, level3), level3], dim=1))
        x2 = self.decode2(torch.cat([self._upsample_to(x3, level2), level2], dim=1))
        x1 = self.decode1(torch.cat([self._upsample_to(x2, level1), level1], dim=1))
        full_res = torch.nn.functional.interpolate(
            x1,
            size=(timesteps, height, width),
            mode="trilinear",
            align_corners=False,
        )
        return self.final_conv(self.final_refine(full_res))

    @staticmethod
    def _to_5d(feature: torch.Tensor, batch_size: int, timesteps: int) -> torch.Tensor:
        """Convert flattened `(B*T, C, H, W)` backbone features into 5D tensors."""
        _, channels, height, width = feature.shape
        return feature.view(batch_size, timesteps, channels, height, width).permute(
            0, 2, 1, 3, 4
        ).contiguous()

    @staticmethod
    def _upsample_to(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsample `x` to the target 5D tensor resolution."""
        return torch.nn.functional.interpolate(
            x,
            size=target.shape[2:],
            mode="trilinear",
            align_corners=False,
        )


__all__ = [
    "ConvBNAct3d",
    "DINOPseudo3DBallDetector",
    "InvertedResidualPseudo3D",
]
