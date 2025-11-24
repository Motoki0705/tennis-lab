import torch
import torch.nn as nn
import torch.nn.functional as F

from .dino_backbone import DINOBackbone


class DinoVitHeatmap(nn.Module):
    """
    DINOv3 ViT backbone + simple upsampling decoder to produce heatmap at input resolution.

    - Backbone: ViT-S/16 provides a single-scale feature map at roughly (H/16, W/16).
    - Decoder: a small stack of conv + 2x upsampling blocks to reach input resolution.
    """

    def __init__(
        self,
        num_keypoints: int = 15,
        decoder_channels: list[int] | None = None,
        backbone_name: str = "dinov3_vits16",
        weights_path: str | None = "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ) -> None:
        super().__init__()
        self.backbone = DINOBackbone(backbone_name=backbone_name, weights_path=weights_path)
        c_in = self.backbone.embed_dim

        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]  # 4 upsample steps -> x16

        blocks: list[nn.Module] = []
        in_c = c_in
        for out_c in decoder_channels:
            blocks += [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ]
            in_c = out_c
        self.decoder = nn.Sequential(*blocks)
        self.head = nn.Conv2d(in_c, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone feature map at ~H/16
        fmap = self.backbone(x)
        y = self.decoder(fmap)
        # Final precise upsample to input size (robust to non-16-multiples)
        if y.shape[-2:] != x.shape[-2:]:
            y = F.interpolate(y, size=x.shape[-2:], mode="bilinear", align_corners=False)
        y = self.head(y)
        return y
