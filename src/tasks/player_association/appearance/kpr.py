"""Inference-only KPR Market/SOLIDER, preserving native parts and visibility.

HL3 notice and provenance: kpr_vendor/NOTICE.md. This is deliberately not a
whole-image AppearanceEncoder: KPR's pairwise visible-part distance needs its
own downstream contract, rather than an implicit flattened cosine surrogate.
"""
from __future__ import annotations

import pickle
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from torch import nn

from src.tasks.player_association.appearance.kpr_vendor.heads import (
    AfterPoolingDimReduceLayer,
    BNClassifier,
    GlobalAveragePoolingHead,
    GlobalWeightedAveragePoolingHead,
    MultiStageFusion,
    PixelToPartClassifier,
)
from src.tasks.player_association.appearance.solider_vendor.swin_transformer import (
    PatchEmbed,
    swin_base_patch4_window7_224,
)
from src.utils.checksum import dual_sha256

WEIGHT_SHA256 = 'e29bacd699a15d1d069c9d19c8804b7d35baea81e713b55e632ee546e1733a1b'
BODY_GROUPS = ((0, 1, 2, 3, 4), (7, 9), (8, 10), (13, 15), (14, 16), (5, 6, 11, 12))


class _ConfigData(dict):
    """Data-only compatibility with the checkpoint's yacs metadata."""


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ('yacs.config', 'CfgNode'):
            return _ConfigData
        return super().find_class(module, name)


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if dual_sha256(path) != WEIGHT_SHA256:
        raise ValueError('KPR requires the explicit official Market/SOLIDER checkpoint SHA-256')
    # Explicit legacy metadata compatibility, restricted to this recorded file.
    # No attempt to install its modified Torchreid or change PyTorch.
    compatibility = types.ModuleType('kpr_checkpoint_pickle')
    compatibility.Unpickler = _CheckpointUnpickler  # type: ignore[attr-defined]
    compatibility.load = pickle.load  # type: ignore[attr-defined]
    saved = torch.load(path, map_location='cpu', weights_only=False, pickle_module=compatibility)
    state = saved['state_dict']
    if not isinstance(state, dict) or not all(isinstance(k, str) and k.startswith('module.')
                                             and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError('KPR checkpoint has an unexpected state dict')
    return {k.removeprefix('module.'): v for k, v in state.items()}, dict(saved['config'])


def prompt_heatmaps(prompts: np.ndarray, negative: np.ndarray | None = None) -> np.ndarray:
    """Crop-normalized COCO17 (x,y,raw confidence) -> upstream eight prompt maps.

    Negative is (N,M,17,3); absent means explicitly no negative skeletons.
    Outside-crop joints are invisible, matching upstream inference transforms.
    Raw heatmap confidence is thresholded, never interpreted as a probability.
    """
    if prompts.ndim != 3 or prompts.shape[1:] != (17, 3) or not np.isfinite(prompts).all():
        raise ValueError('KPR prompts must be finite (N,17,3)')
    if negative is not None and (negative.ndim != 4 or negative.shape[0] != len(prompts)
                                or negative.shape[2:] != (17, 3) or not np.isfinite(negative).all()):
        raise ValueError('KPR negative prompts must be finite (N,M,17,3)')
    height, width, radius = 384, 128, 128 // 11
    kernel_1d = gaussian(2 * radius + 1, std=(2 * radius + 1) / 4)
    kernel = np.outer(kernel_1d, kernel_1d)

    def skeleton(points: np.ndarray) -> np.ndarray:
        maps: np.ndarray = np.zeros((17, height, width), np.float64)
        for joint, (x, y, confidence) in enumerate(points):
            if confidence <= .3 or not (0 <= x < 1 and 0 <= y < 1):
                continue
            px, py = int(x * width), int(y * height)
            top, bottom = min(radius, py), min(radius, height - 1 - py)
            left, right = min(radius, px), min(radius, width - 1 - px)
            maps[joint, py - top:py + bottom + 1, px - left:px + right + 1] = kernel[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
        return maps

    result = []
    for n, pose in enumerate(prompts):
        joints = skeleton(pose)
        positive = np.stack([joints[list(group)].max(0) for group in BODY_GROUPS])
        other: np.ndarray = np.zeros((1, height, width), np.float64)
        if negative is not None:
            for pose in negative[n]:
                other[0] = np.maximum(other[0], skeleton(pose).max(0))
        foreground = np.concatenate((other, positive))
        background = np.clip(1 - foreground.sum(0, keepdims=True), 0, 1)
        result.append(np.concatenate((background, foreground)).astype(np.float32))
    return np.stack(result) if result else np.empty((0, 8, height, width), np.float32)


class _PromptedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = swin_base_patch4_window7_224(img_size=(384, 128), semantic_weight=-1., drop_path_rate=.1)
        self.parts_embed = nn.Parameter(torch.zeros(8, 1, 128))
        self.patch_embed = model.patch_embed
        self.masks_patch_embed = PatchEmbed(in_channels=8, embed_dims=128, conv_type='Conv2d', kernel_size=4,
                                           stride=4, norm_cfg={'type': 'LN'}, init_cfg=None)
        self.model = model
        # These are aliases, not separately registered checkpoint parameters.
        self.output_norms = tuple(model.output_norms[i] for i in range(4))

    def forward(self, images: torch.Tensor, prompts: torch.Tensor) -> dict[int, torch.Tensor]:
        features, shape = self.patch_embed(images)
        tokens, _ = self.masks_patch_embed(prompts)
        features = self.model.drop_after_pos(features + tokens)
        outputs = {}
        for index, (stage, norm) in enumerate(zip(self.model.stages, self.output_norms, strict=True)):
            features, shape, out, out_shape = stage(features, shape)
            out = norm(out)
            outputs[index] = out.view(-1, *out_shape, self.model.num_features[index]).permute(0, 3, 1, 2).contiguous()
        return outputs


class KprInference(nn.Module):
    """Fixed public checkpoint graph; native six 512-d descriptors and masks."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone_appearance_feature_extractor = _PromptedBackbone()
        self.msf = MultiStageFusion(input_dim=1920, output_dim=1024)
        self.global_pooling_head = nn.AdaptiveAvgPool2d(1)
        self.foreground_attention_pooling_head = GlobalAveragePoolingHead(512)
        self.background_attention_pooling_head = GlobalAveragePoolingHead(512)
        self.parts_attention_pooling_head = GlobalWeightedAveragePoolingHead(512)
        self.pixel_classifier = PixelToPartClassifier(1024, 5)
        self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(1024, 512)
        self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(1024, 512)
        self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(1024, 512)
        self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(1024, 512)
        self.global_identity_classifier = BNClassifier(512, 751)
        self.background_identity_classifier = BNClassifier(512, 751)
        self.foreground_identity_classifier = BNClassifier(512, 751)
        self.concat_parts_identity_classifier = BNClassifier(2560, 751)
        self.parts_identity_classifier = nn.ModuleList([BNClassifier(512, 751) for _ in range(5)])

    def forward(self, images: torch.Tensor, prompts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.msf(self.backbone_appearance_feature_extractor(images, prompts))
        probability = F.softmax(self.pixel_classifier(features), dim=1)
        parts = probability[:, 1:]
        foreground = parts.max(dim=1)[0]
        visible = F.one_hot(probability.argmax(1), 6).permute(0, 3, 1, 2).amax(dim=(2, 3)).bool()
        foreground_features = self.foreground_attention_pooling_head(features, foreground.unsqueeze(1)).flatten(1, 2)
        foreground_features = self.foreground_after_pooling_dim_reduce(foreground_features)
        bn_foreground, _ = self.foreground_identity_classifier(foreground_features)
        parts_features = self.parts_after_pooling_dim_reduce(self.parts_attention_pooling_head(features, parts))
        embeddings = torch.cat((bn_foreground.unsqueeze(1), parts_features), 1)
        visibility = torch.cat((visible.amax(1, keepdim=True), visible[:, 1:]), 1)
        return F.normalize(embeddings, p=2, dim=-1), visibility, torch.cat((foreground.unsqueeze(1), parts), 1)


class KprEncoder:
    input_size = (384, 128)
    name = 'kpr_market_solider_parts'

    def __init__(self, checkpoint: Path, device: str = 'cpu') -> None:
        self.device = device
        state, self.source_config = load_checkpoint(checkpoint)
        self.model = KprInference()
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)

    def extract(self, crops: torch.Tensor, prompts: np.ndarray,
                negative: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if crops.ndim != 4 or tuple(crops.shape[1:]) != (3, 384, 128) or len(crops) != len(prompts) \
                or not torch.isfinite(crops).all() or bool(((crops < 0) | (crops > 1)).any()):
            raise ValueError('KPR crops must be finite RGB [0,1] (N,3,384,128), aligned to prompts')
        heatmaps = torch.from_numpy(prompt_heatmaps(prompts, negative)).to(self.device)
        with torch.inference_mode():
            embedded, visible, masks = self.model((crops.to(self.device) - .5) / .5, heatmaps)
        if embedded.shape != (len(crops), 6, 512) or visible.shape != (len(crops), 6) \
                or not torch.isfinite(embedded).all() or not torch.isfinite(masks).all():
            raise ValueError('KPR output differs from the native part contract')
        return embedded.cpu(), visible.cpu(), masks.cpu()
