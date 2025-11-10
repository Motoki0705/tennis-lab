"""Builders and adapters for assembling SceneModel instances."""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from omegaconf import DictConfig, OmegaConf

from .config import SceneConfig
from .head import BBoxHead
from .model import SceneModel

_MINIMAL_OVERRIDES: dict[str, Any] = {
    "img_size": 64,
    "patch_size": 8,
    "vit_depth": 1,
    "vit_heads": 2,
    "D_model": 64,
    "temporal_depth": 1,
    "temporal_heads": 2,
    "max_T_hint": 4,
    "num_queries": 4,
    "decoder_layers": 1,
    "num_points": 2,
    "window_k": 1,
}


def build_scene_model(
    model_cfg: DictConfig | Mapping[str, Any],
    debug_cfg: DictConfig | Mapping[str, Any] | None = None,
) -> SceneModel:
    """Instantiate SceneModel and apply optional backbone/head overrides declared in config."""

    cfg = _to_dict(model_cfg)
    debug = _to_dict(debug_cfg)
    scene_cfg = copy.deepcopy(cfg.get("scene", {}))
    if debug.get("minimal"):
        scene_cfg.update(_MINIMAL_OVERRIDES)
    scene = SceneConfig(**scene_cfg)
    model = SceneModel(scene)
    backbone_cfg = cfg.get("backbone", {})
    head_cfg = cfg.get("head", {})
    model.backbone = _build_backbone(backbone_cfg, model.backbone, debug)
    model.head = _build_head(head_cfg, model.head, scene.D_model)
    return model


def _build_backbone(
    cfg: Mapping[str, Any],
    default_backbone: nn.Module,
    debug_cfg: Mapping[str, Any],
) -> nn.Module:
    cfg_dict = _to_dict(cfg)
    backend_type = cfg_dict.get("type", "native")
    if debug_cfg.get("minimal"):
        return default_backbone
    if backend_type.startswith("dinov3"):
        adapter = _load_dinov3_backbone(cfg_dict)
        if adapter is not None:
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
    arch = cfg.get("type")
    weights_path = cfg.get("weights_path")
    if weights_path is not None:
        resolved = Path(weights_path).expanduser()
        if not resolved.exists():
            warnings.warn(f"DINOv3 weights not found at {resolved}", stacklevel=2)
            return None
        weights_path = str(resolved)
    hub_kwargs: dict[str, Any] = {"source": "local"}
    if weights_path is not None:
        hub_kwargs["weights"] = weights_path
    try:
        dinov3 = torch.hub.load("third_party/dinov3", arch, **hub_kwargs)
    except Exception as exc:  # pragma: no cover - actual import tested via adapter
        warnings.warn(f"Failed to load {arch} from third_party/dinov3: {exc}", stacklevel=2)
        return None
    if not hasattr(dinov3, "get_intermediate_layers"):
        warnings.warn(f"{arch} missing get_intermediate_layers; falling back to native backbone", stacklevel=2)
        return None
    return Dinov3BackboneAdapter(dinov3)


class Dinov3BackboneAdapter(nn.Module):
    """Adapter that exposes torch.hub dinov3 ViTs under the SceneModel backbone interface."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "embed_dim", getattr(backbone, "num_features", 0))
        if self.embed_dim == 0:
            raise ValueError("Could not infer embed_dim from dinov3 backbone")

    def forward(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        if frames.ndim != 5:
            msg = "frames must be [B,T,3,H,W]"
            raise ValueError(msg)
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        device = flat.device
        self.backbone = self.backbone.to(device)
        features = self.backbone.get_intermediate_layers(
            flat,
            n=1,
            return_class_token=True,
            return_extra_tokens=False,
        )
        patches, cls_tokens = features[0]
        patches = patches.view(B, T, patches.shape[1], self.embed_dim)
        cls_tokens = cls_tokens.view(B, T, self.embed_dim)
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
