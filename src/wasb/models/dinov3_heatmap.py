from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor, nn
import torch.nn.functional as F

from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16


LOGGER = logging.getLogger(__name__)


class FPNDecoder(nn.Module):
    """Simple FPN-style upsampling decoder.

    Takes a single low-resolution feature map and progressively upsamples it
    using convolutional blocks followed by 2x interpolation. The final output
    is projected to a single-channel heatmap.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        out_channels: int = 1,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must contain at least one element")

        self.upsample_mode = upsample_mode
        blocks: list[nn.Sequential] = []
        current_in = in_channels
        for ch in channels:
            ch_int = int(ch)
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_in, ch_int, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch_int),
                    nn.ReLU(inplace=True),
                )
            )
            current_in = ch_int
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(current_in, out_channels, kernel_size=1)

    def _interpolate(self, x: Tensor, *, size: tuple[int, int] | None = None) -> Tensor:
        if self.upsample_mode in ("bilinear", "bicubic"):
            return F.interpolate(
                x,
                size=size,
                scale_factor=None if size is not None else 2.0,
                mode=self.upsample_mode,
                align_corners=False,
            )
        return F.interpolate(
            x,
            size=size,
            scale_factor=None if size is not None else 2.0,
            mode=self.upsample_mode,
        )

    def forward(self, x: Tensor, target_hw: tuple[int, int]) -> Tensor:
        """Upsample features to the target spatial resolution.

        Args:
            x: Input feature map of shape ``[B, C, H_in, W_in]``.
            target_hw: Target ``(height, width)`` for the output heatmap.
        """
        h_target, w_target = target_hw
        if x.dim() != 4:
            raise ValueError(f"Expected 4D feature map, got {tuple(x.shape)}")

        for block in self.blocks:
            x = block(x)
            x = self._interpolate(x)

        if x.shape[-2] != h_target or x.shape[-1] != w_target:
            x = self._interpolate(x, size=(h_target, w_target))

        return self.out_conv(x)


class DinoV3HeatmapModel(nn.Module):
    """Single-frame DINOv3 ViT-S/16 backbone with FPN decoder to heatmaps.

    The model expects input frames of shape ``[B, 3, H, W]`` and outputs
    per-pixel heatmaps of shape ``[B, 1, H_out, W_out]``. By default,
    ``H=W=H_out=W_out=224`` but these can be configured.
    """

    def __init__(self, cfg: DictConfig | dict[str, Any]) -> None:
        super().__init__()

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.config = cfg

        image_hw = cfg.get("image_hw", [224, 224])
        if isinstance(image_hw, (list, tuple, ListConfig)):
            self.image_height = int(image_hw[0])
            self.image_width = int(image_hw[1])
        else:
            self.image_height = int(image_hw)
            self.image_width = int(image_hw)

        heatmap_hw = cfg.get("heatmap_hw", [self.image_height, self.image_width])
        if isinstance(heatmap_hw, (list, tuple, ListConfig)):
            self.heatmap_height = int(heatmap_hw[0])
            self.heatmap_width = int(heatmap_hw[1])
        else:
            self.heatmap_height = self.image_height
            self.heatmap_width = self.image_width

        self.patch_size = int(cfg.get("patch_size", 16))

        # Initialize DINOv3 backbone without pre-trained weights; if a
        # backbone checkpoint is provided, it can be loaded later via
        # ``load_backbone_checkpoint`` which is compatible with the
        # existing training script.
        self.backbone = get_dinov3_vits16(checkpoint_path=None)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 384))

        mean = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
        std = cfg.get("normalize_std", [0.229, 0.224, 0.225])
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean_tensor, persistent=False)
        self.register_buffer("pixel_std", std_tensor, persistent=False)

        decoder_channels = cfg.get("decoder_channels", [256, 128, 64, 32])
        if isinstance(decoder_channels, int):
            decoder_channels = [int(decoder_channels)]
        else:
            decoder_channels = [int(ch) for ch in decoder_channels]

        upsample_mode = str(cfg.get("upsample_mode", "bilinear"))

        self.decoder = FPNDecoder(
            in_channels=self.embed_dim,
            channels=decoder_channels,
            out_channels=1,
            upsample_mode=upsample_mode,
        )

        self._backbone_train_mode: bool | None = None
        self._backbone_frozen = False

    def _normalize(self, x: Tensor) -> Tensor:
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, frames: Tensor) -> Tensor:
        """Forward pass from frames to heatmaps.

        Args:
            frames: Input tensor of shape ``[B, 3, H, W]``.

        Returns:
            Heatmaps of shape ``[B, 1, H_out, W_out]``.
        """
        if frames.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(frames.shape)}")
        b, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"DinoV3HeatmapModel expects 3-channel RGB input, got C={c}")

        # Resize to the configured image resolution before feeding into ViT.
        if h != self.image_height or w != self.image_width:
            frames = F.interpolate(
                frames,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )

        x = self._normalize(frames)

        # Use the last transformer block features, reshaped to patch grid.
        # get_intermediate_layers returns a tuple; with n=1 it has length 1.
        features_tuple = self.backbone.get_intermediate_layers(
            x,
            n=1,
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        if isinstance(features_tuple, (tuple, list)):
            feats = features_tuple[-1]
        else:
            feats = features_tuple

        if feats.dim() != 4:
            raise ValueError(
                "Backbone features must be 4D tensor [B, C, H_p, W_p], "
                f"got {tuple(feats.shape)}"
            )

        heatmaps = self.decoder(feats, target_hw=(self.heatmap_height, self.heatmap_width))
        return heatmaps

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: torch.device | str | None = "cpu",
    ) -> None:
        """Load pre-trained DINOv3 weights into the backbone.

        This expects a checkpoint compatible with ``dinov3_vits16`` as used
        by the official DINOv3 repository. The weights are loaded via the
        local torch.hub entry point to ensure consistency.
        """
        _ = map_location  # kept for API compatibility
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        # If a directory is given (e.g., the local DINOv3 repository root), we
        # treat this as a no-op for tests and offline usage. Actual checkpoint
        # files should be passed as a concrete .pth path.
        if checkpoint_path.is_dir():
            LOGGER.info(
                "Backbone checkpoint path %s is a directory; skipping weight loading.",
                checkpoint_path,
            )
            return

        loaded_backbone = get_dinov3_vits16(checkpoint_path=str(checkpoint_path))
        self.backbone.load_state_dict(loaded_backbone.state_dict(), strict=True)
        LOGGER.info("Loaded DINOv3 backbone parameters from %s", checkpoint_path)

    def freeze_backbone(self) -> None:
        """Disable gradient updates for the DINOv3 backbone."""
        self._backbone_train_mode = self.backbone.training
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self._backbone_frozen = True
        LOGGER.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Re-enable gradient updates for the DINOv3 backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        if self._backbone_train_mode is not None:
            if self._backbone_train_mode:
                self.backbone.train()
            self._backbone_train_mode = None
        self._backbone_frozen = False
        LOGGER.info("Backbone unfrozen")
