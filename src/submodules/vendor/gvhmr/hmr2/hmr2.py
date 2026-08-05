"""HMR2 image-feature model (adapted from hmr4d.network.hmr2.hmr2).

Trimmed to a plain ``nn.Module``; the default ``feat_mode`` returns the SMPL
head's 1024-d token used by GVHMR as the per-frame image feature.
"""

import torch.nn as nn

from .smpl_head import SMPLTransformerDecoderHead
from .vit import ViT


class HMR2(nn.Module):
    def __init__(self, *, mean_params_path):
        super().__init__()
        self.backbone = ViT(
            img_size=(256, 192),
            patch_size=16,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            ratio=1,
            use_checkpoint=False,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.55,
        )
        self.smpl_head = SMPLTransformerDecoderHead(mean_params_path=mean_params_path)

    def forward(self, batch, feat_mode=True):
        """
        Args:
            batch: {"img": (B, 3, 256, 256) normalized crops}
            feat_mode: when True, only the feature token output is returned,
                which is all the GVHMR pipeline needs.
        """
        # Backbone
        x = batch["img"][:, :, :, 32:-32]
        vit_feats = self.backbone(x)

        # Output head
        token_out = self.smpl_head(vit_feats, only_return_token_out=feat_mode)
        if feat_mode:
            return token_out  # (B, 1024)
        raise NotImplementedError(
            "Full HMR2 SMPL decoding is not vendored; use feat_mode=True."
        )
