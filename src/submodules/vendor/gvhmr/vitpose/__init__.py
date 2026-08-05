"""Vendored ViTPose (2D keypoint detector used by GVHMR preprocessing).

Only the ``ViTPose_huge_coco_256x192`` variant used by GVHMR is provided; the
ViT backbone implementation is shared with the vendored HMR2.
"""

import torch
import torch.nn as nn

from src.submodules.vendor.gvhmr.hmr2.vit import ViT

from .heatmap_head import TopdownHeatmapSimpleHead, ViTPoseHeadConfig


class VitPoseModel(nn.Module):
    def __init__(self, backbone, keypoint_head):
        super().__init__()
        self.backbone = backbone
        self.keypoint_head = keypoint_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.keypoint_head(x)
        return x


def build_vitpose_huge(checkpoint_path, *, head_config: ViTPoseHeadConfig):
    """Build ViTPose-H (COCO 256x192) and load the released checkpoint."""
    backbone = ViT(
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
    head = TopdownHeatmapSimpleHead(head_config)
    pose = VitPoseModel(backbone, head)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    pose.load_state_dict(state_dict, strict=True)
    return pose
