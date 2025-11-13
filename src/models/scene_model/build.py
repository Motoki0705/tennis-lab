"""Builders and adapters for assembling SceneModel instances."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn

from .config import SceneConfig
from .head import BBoxHead
from .model import SceneModel
from ..utils.load_dinov3 import load_dinov3


def build_scene_model(
    model_cfg: DictConfig | Mapping[str, Any],
    debug_cfg: DictConfig | Mapping[str, Any] | None = None,
) -> SceneModel:
    """Instantiate SceneModel and apply optional backbone/head overrides declared in config."""
    cfg = _to_dict(model_cfg)
    debug = _to_dict(debug_cfg)
    scene_cfg = copy.deepcopy(cfg.get("scene", {}))
    scene = SceneConfig(**scene_cfg)
    model = SceneModel(scene)
    backbone_cfg = cfg.get("backbone", {})
    head_cfg = cfg.get("head", {})
    model.backbone = _build_backbone(backbone_cfg, model.backbone, debug, scene.D_model)
    model.head = _build_head(head_cfg, model.head, scene.D_model)
    return model


def _build_backbone(
    cfg: Mapping[str, Any],
    default_backbone: nn.Module,
    debug_cfg: Mapping[str, Any],
    model_dim: int,
) -> nn.Module:
    cfg_dict = _to_dict(cfg)
    backend_type = cfg_dict.get("type", "native")
    if debug_cfg.get("minimal"):
        return default_backbone
    if backend_type.startswith("dinov3"):
        adapter = _load_dinov3_backbone(cfg_dict)
        if adapter is not None:
            embed_dim = getattr(adapter, "embed_dim", None)
            if embed_dim is None:
                raise ValueError("DINOv3 adapter did not expose embed_dim")
            if embed_dim != model_dim:
                msg = (
                    "DINOv3 backbone embed_dim "
                    f"({embed_dim}) does not match SceneModel D_model ({model_dim})"
                )
                raise ValueError(msg)
            return adapter
    return default_backbone


def _build_head(
    cfg: Mapping[str, Any],
    default_head: nn.Module,
    model_dim: int,
) -> nn.Module:
    cfg_dict = _to_dict(cfg)
    head_type = cfg_dict.get("type", "scene")
    if head_type == "bbox":
        num_classes = int(cfg_dict.get("num_classes", 2))
        return BBoxHead(model_dim, num_classes=num_classes)
    return default_head


def _load_dinov3_backbone(cfg: Mapping[str, Any]) -> nn.Module | None:
    arch = cfg.get("type", None)
    weights_path = cfg.get("weights_path", None)
    
    dinov3 = load_dinov3(
        arch=arch,
        weights_path=weights_path,
    )
    return Dinov3BackboneAdapter(dinov3)


class Dinov3BackboneAdapter(nn.Module):
    """Adapter that exposes torch.hub dinov3 ViTs under the SceneModel backbone interface."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = getattr(
            backbone, "embed_dim", getattr(backbone, "num_features", 0)
        )
        if self.embed_dim == 0:
            raise ValueError("Could not infer embed_dim from dinov3 backbone")

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """Run the DINOv3 encoder and reshape outputs to `[B, T, ...]` tensors."""
        if frames.ndim != 5:
            msg = "frames must be [B,T,3,H,W]"
            raise ValueError(msg)
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        features = self.backbone.get_intermediate_layers(
            flat,
            n=1,
            return_class_token=True,
            return_extra_tokens=False,
        )
        patches, cls_tokens = features[0]
        patches = patches.reshape(B, T, patches.shape[1], self.embed_dim)
        cls_tokens = cls_tokens.reshape(B, T, self.embed_dim)
        return patches, cls_tokens


def _to_dict(cfg: DictConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return cfg
    if isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


__all__ = ["build_scene_model", "Dinov3BackboneAdapter"]
