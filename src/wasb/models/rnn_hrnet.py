"""HRNet backbone with ConvGRU temporal head."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .hrnet import HRNet


class ConvGRUCell(nn.Module):
    """A lightweight ConvGRU cell for spatial feature maps."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv_gates = nn.Conv2d(
            input_channels + hidden_channels,
            2 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv_candidate = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor, h_prev: Tensor | None = None) -> Tensor:
        if h_prev is None:
            h_prev = torch.zeros(
                x.size(0),
                self.hidden_channels,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype,
            )

        combined = torch.cat([x, h_prev], dim=1)
        zr = torch.sigmoid(self.conv_gates(combined))
        z, r = torch.split(zr, self.hidden_channels, dim=1)
        candidate = torch.tanh(
            self.conv_candidate(torch.cat([x, r * h_prev], dim=1))
        )
        h = (1 - z) * h_prev + z * candidate
        return h


class HRNetConvGRU(nn.Module):
    """Sequence model using HRNet features with a ConvGRU head."""

    def __init__(self, cfg: DictConfig | dict[str, Any]):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        self.frames_in = int(cfg.get("frames_in", 1))
        self.frames_out = int(cfg.get("frames_out", self.frames_in))
        self.stack_channels = bool(cfg.get("stack_channels", False))
        self.expects_sequence_input = True

        backbone_cfg = cfg.get("backbone") or cfg
        if isinstance(backbone_cfg, dict):
            backbone_cfg = OmegaConf.create(backbone_cfg)

        # Use a single-frame HRNet backbone by default and reuse other settings.
        backbone_dict = OmegaConf.to_container(backbone_cfg, resolve=True)
        backbone_dict["frames_in"] = cfg.get(
            "backbone_frames_in",
            self.frames_in if self.stack_channels else 1,
        )
        backbone_dict["frames_out"] = cfg.get("backbone_frames_out", 1)
        if "out_scales" not in backbone_dict and cfg.get("out_scales") is not None:
            backbone_dict["out_scales"] = cfg.get("out_scales")
        self.backbone = HRNet(OmegaConf.create(backbone_dict))

        # Channel dimension of the deconvolved HRNet features at scale 0.
        self.feature_channels = self.backbone.final_layers[0].in_channels
        hidden_channels = cfg.get("gru_hidden_channels", self.feature_channels)
        kernel_size = cfg.get("gru_kernel_size", 3)

        self.gru_cell = ConvGRUCell(
            input_channels=self.feature_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, frames: Tensor, h_state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass with optional hidden state carry-over.

        Returns:
            pred: Heatmaps for the last ``frames_out`` steps, shape ``[B, frames_out, H, W]``.
            h_state: Final hidden state tensor that can be fed into the next call.
        """
        if frames.dim() != 5:
            raise ValueError(
                f"Expected input shape [B, T, C, H, W], got {tuple(frames.shape)}"
            )
        b, t, c, h, w = frames.shape
        frames_flat = frames.view(b * t, c, h, w)

        features = self.backbone.forward_features(frames_flat)[0]
        feat_h, feat_w = features.shape[-2:]
        features = features.view(b, t, self.feature_channels, feat_h, feat_w)

        outputs = []
        for idx in range(t):
            feat_t = features[:, idx]
            h_state = self.gru_cell(feat_t, h_state)
            logits = self.head(h_state)
            outputs.append(logits)

        pred = torch.stack(outputs, dim=1)  # [B, T, 1, H, W]
        pred = pred[:, -self.frames_out :, :, :, :]
        return pred.squeeze(2), h_state


__all__ = ["HRNetConvGRU", "ConvGRUCell"]
