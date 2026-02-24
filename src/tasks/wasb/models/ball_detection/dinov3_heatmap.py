"""DINOv3-based heatmap model for WASB ball detection."""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torchvision.ops import FeaturePyramidNetwork

from third_party.dinov3.api.dinov3_loader import get_dinov3_vits16

LOGGER = logging.getLogger(__name__)


class DinoV3FPNHeatmap(nn.Module):
    """DINOv3 ViT-S/16 backbone + FPN + 1x1 conv head to dense heatmaps.

    - Input:  frames  … [B, 3, H, W]
      Assumes that resizing and normalization for DINOv3 are done outside this module.
    - Pipeline:  frames → DINOv3 ViT → single-scale features →
                 conv + upsampling hierarchy → torchvision.ops.FeaturePyramidNetwork
                 → highest-resolution FPN feature → 1x1 conv head → upsample to (H, W).
    - Output:  dense heatmaps with shape [B, 1, H, W].
    """

    def __init__(
        self,
        cfg: DictConfig | dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Treat cfg in a very simple way (only reads backbone_checkpoint and fpn_out_channels)
        if cfg is None:
            cfg = {}
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.config = cfg

        # DINOv3 ViT-S/16 backbone
        checkpoint_path = cfg.get("backbone_checkpoint", None)
        self.backbone = get_dinov3_vits16(checkpoint_path=checkpoint_path)
        self.embed_dim = int(getattr(self.backbone, "embed_dim", 384))

        # FPN configuration: per-scale channels and number of scales.
        # Example (default): fpn_channels: [256, 128, 64, 32]
        fpn_channels_cfg = cfg.get("fpn_channels", [256, 128, 64])
        if isinstance(fpn_channels_cfg, int):
            fpn_channels = [int(fpn_channels_cfg)]
        else:
            fpn_channels = [int(ch) for ch in fpn_channels_cfg]
        num_scales = len(fpn_channels)

        # Hierarchical scales (magnification factors) derived from a single-scale ViT feature.
        # The number of scales is determined by fpn_channels. For N scales we use
        # [2^(N-1), ..., 2^1, 2^0] = [2**(N-1), ..., 2.0, 1.0], ordered from
        # higher to lower spatial resolution.
        self.scale_factors = [float(2 ** i) for i in reversed(range(num_scales))]

        # Per-scale conv2d -> BN -> ReLU blocks
        # (in_channels = embed_dim, out_channels = fpn_channels[i] for each scale)
        self.hierarchy_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.embed_dim, ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                )
                for ch in fpn_channels
            ]
        )

        # The main torchvision FPN module
        fpn_out_channels = int(cfg.get("fpn_out_channels", fpn_channels[0]))
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_channels,
            out_channels=fpn_out_channels,
        )

        # Final 1x1 conv head to produce a single-channel heatmap from the
        # highest-resolution FPN feature map.
        self.head = nn.Conv2d(fpn_out_channels, 1, kernel_size=1)

        self._backbone_train_mode: bool | None = None
        self._backbone_frozen = False

    def forward(self, frames: Tensor) -> Tensor:
        """Forward pass: frames → ViT features → FPN → 1x1 head → heatmaps.

        Args:
            frames: [B, 3, H, W] RGB tensor.
                    - No resizing or normalization is performed inside this module.
                    - Preprocessing for DINOv3 (resizing, mean/std normalization) must be
                      done by the caller.

        Returns:
            Tensor: Dense heatmaps of shape [B, 1, H, W].
        """
        if frames.dim() != 4:
            raise ValueError(f"Expected input shape [B, 3, H, W], got {tuple(frames.shape)}")
        b, c, h, w = frames.shape
        if c != 3:
            raise ValueError(f"DinoV3FPNBackbone expects 3-channel RGB input, got C={c}")

        # Obtain the features from the last DINOv3 ViT block as a 4D tensor
        features_tuple = self.backbone.get_intermediate_layers(
            frames,
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

        _, _, H_p, W_p = feats.shape

        # Construct hierarchical feature maps from the ViT patch grid.
        pyramid_inputs = OrderedDict()
        for idx, scale in enumerate(self.scale_factors):
            x = feats
            if scale != 1.0:
                x = F.interpolate(
                    x,
                    scale_factor=scale,
                    mode="bilinear",
                    align_corners=False,
                )
            x = self.hierarchy_convs[idx](x)
            name = f"scale_{int(scale)}x"
            pyramid_inputs[name] = x

        # Pass through the FPN.
        # OrderedDict: { "scale_*x": P_*x, ... }, ordered from higher to lower
        # spatial resolution.
        fpn_outputs = self.fpn(pyramid_inputs)

        # Use the highest-resolution FPN feature (first in the OrderedDict) as
        # input to the 1x1 conv head, then upsample to the input frame
        # resolution (H, W).
        highest_res = next(iter(fpn_outputs.values()))
        heatmap = self.head(highest_res)
        if heatmap.shape[-2] != h or heatmap.shape[-1] != w:
            heatmap = F.interpolate(
                heatmap,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

        return heatmap

    # ===== Backbone utility methods (simplified copy from the original class) =====

    def load_backbone_checkpoint(
        self,
        checkpoint_path: str | Path,
        map_location: torch.device | str | None = "cpu",
    ) -> None:
        """Load pre-trained DINOv3 weights into the backbone.

        Assumes a checkpoint that is compatible with the official DINOv3 release.
        """
        _ = map_location  # kept for API compatibility
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        # If a directory is given, do nothing (following the original behavior/specification).
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
