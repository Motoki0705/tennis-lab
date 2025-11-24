from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .architecture import DinoFpnHeatmapNet, DinoFpnModelConfig
from .backbone import DinoBackboneConfig
from .decoder import HeatmapDecoderConfig


def _to_dict(data: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(data, DictConfig):
        container = OmegaConf.to_container(data, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Expected DictConfig to convert into dict.")
        return container
    return dict(data)


def create_model(*, backbone: Mapping[str, Any] | DictConfig, decoder: Mapping[str, Any] | DictConfig, **_: Any):
    """Instantiate the DINOv3+FPN heatmap network."""

    backbone_cfg = DinoBackboneConfig(**_to_dict(backbone))
    decoder_cfg = HeatmapDecoderConfig(**_to_dict(decoder))
    model_cfg = DinoFpnModelConfig(backbone=backbone_cfg, decoder=decoder_cfg)
    return DinoFpnHeatmapNet(model_cfg)
