from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork


@dataclass
class DinoBackboneConfig:
    """Runtime configuration for the DINOv3 backbone + FPN adapter."""

    repo_dir: str = "third_party/dinov3"
    entry: str = "dinov3_vits16"
    weights: str | None = None
    freeze: bool = True
    vit_layers: tuple[int, ...] = (11,)
    fpn_channels: int = 256


class DinoFpnBackbone(nn.Module):
    """Wrap a DINOv3 ViT backbone and expose a 3-level FPN feature pyramid.

    DINOv3 ViT models emit patch tokens at a single resolution (H/patch, W/patch).
    To obtain multi-scale maps, we treat the projected token map (1/16) as the
    middle level (C4), apply upsampling to build a finer C3, and strided
    convolution to derive a coarser C5. A lightweight FPN then blends the scales.
    """

    def __init__(self, cfg: DinoBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vit = torch.hub.load(
            cfg.repo_dir,
            cfg.entry,
            source="local",
            weights=cfg.weights if cfg.weights else True,
        )
        if cfg.freeze:
            for param in self.vit.parameters():
                param.requires_grad = False

        embed_dim = getattr(self.vit, "embed_dim", None) or getattr(self.vit, "num_features", None)
        patch_size = getattr(self.vit, "patch_size", None)
        if embed_dim is None or patch_size is None:
            raise AttributeError("Loaded DINOv3 backbone missing 'embed_dim' or 'patch_size'.")
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.vit_layers: tuple[int, ...] = tuple(cfg.vit_layers)

        out_channels = int(cfg.fpn_channels)
        self.proj = nn.Conv2d(self.embed_dim, out_channels, kernel_size=1)
        self.c3_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.c5_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.lateral_c3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[out_channels, out_channels, out_channels],
            out_channels=out_channels,
        )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.freeze:
            with torch.no_grad():
                feats = self.vit.get_intermediate_layers(
                    x,
                    n=self.vit_layers,
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
        else:
            feats = self.vit.get_intermediate_layers(
                x,
                n=self.vit_layers,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        fmap = feats[-1]
        if fmap.dim() != 4:
            raise ValueError(f"Expected backbone features with 4 dims, got shape={tuple(fmap.shape)}")
        return fmap

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        fmap = self._tokens_to_map(x)  # [B, C, H/patch, W/patch]
        c4 = self.proj(fmap)

        # Build finer and coarser scales around C4.
        c3 = F.interpolate(c4, scale_factor=2.0, mode="bilinear", align_corners=False)
        c3 = self.c3_conv(c3)
        c5 = self.c5_conv(c4)

        lat3 = self.lateral_c3(c3)
        lat4 = self.lateral_c4(c4)
        lat5 = self.lateral_c5(c5)

        fpn_out = self.fpn({"0": lat3, "1": lat4, "2": lat5})
        return {
            "P3": fpn_out["0"],
            "P4": fpn_out["1"],
            "P5": fpn_out["2"],
        }

    @property
    def output_channels(self) -> int:
        return self.cfg.fpn_channels

    def train(self, mode: bool = True) -> DinoFpnBackbone:
        super().train(mode)
        if self.cfg.freeze:
            self.vit.eval()
        return self


__all__ = ["DinoBackboneConfig", "DinoFpnBackbone"]
