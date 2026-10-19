# Copyright (c) OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source: ViTAE-Transformer/ViTPose, commit c050ed29112da7704797cc1a65af0234b525010d
# https://github.com/ViTAE-Transformer/ViTPose/blob/c050ed29112da7704797cc1a65af0234b525010d/mmpose/models/detectors/top_down.py
# Model settings: configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_huge_coco_256x192.py
# Modified for tennis-lab: named torch sequence for TopDown.forward_dummy's
# backbone -> head path, explicit ViT-H construction and strict checkpoint load.
# Registry, neck, training, decoding and visualization are omitted.
# License text: ../LICENSE.ViTPose
"""ViTPose-H inference assembly from the official Apache-2.0 configuration."""

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from src.submodules.vendor.gvhmr.hmr2.vit import ViT

from .heatmap_head import TopdownHeatmapSimpleHead, ViTPoseHeadConfig


class VitPoseModel(nn.Sequential):
    """Backbone followed by heatmap head, with the official checkpoint prefixes."""

    def __init__(self, backbone: nn.Module, keypoint_head: nn.Module) -> None:
        super().__init__(
            OrderedDict((("backbone", backbone), ("keypoint_head", keypoint_head)))
        )


def build_vitpose_huge(
    checkpoint_path: str | Path, *, head_config: ViTPoseHeadConfig
) -> VitPoseModel:
    """Load the released COCO ViTPose-H weights without remapping or omissions."""
    model = VitPoseModel(
        backbone=ViT(
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
        ),
        keypoint_head=TopdownHeatmapSimpleHead(head_config),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model
