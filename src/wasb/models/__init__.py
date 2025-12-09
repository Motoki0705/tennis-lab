"""Models for WASB tennis dataset generation."""

from __future__ import annotations

from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .clip_segmenter import ClipSegmenter, RuleBasedClipSegmenter
from .hrnet import HRNet
from .hrcnet import HRCNet
from .rnn_hrnet import TemporalConvGRUModel
from .trajectory_completer import (
    BiLSTMCompleter,
    CompletionResult,
    HybridCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    create_completer,
)


def _hrnet_handlers() -> tuple[Callable, Callable]:
    def prepare_frames(frames):
        if getattr(frames, "dim", None) is not None and frames.dim() == 5:
            b, t, c, h, w = frames.shape
            return frames.view(b, t * c, h, w)
        return frames

    def extract_heatmaps(outputs):
        if isinstance(outputs, Tensor):
            return outputs  
        raise TypeError("Unsupported model output type for HRNet.")

    return prepare_frames, extract_heatmaps


def _temporal_conv_gru_handlers() -> tuple[Callable, Callable]:
    def prepare_frames(frames):
        if getattr(frames, "dim", None) is not None and frames.dim() != 5:
            raise ValueError(
                f"HRNetConvGRU expects input shape [B, T, C, H, W], got {getattr(frames, 'shape', None)}"
            )
        return frames

    def extract_heatmaps(outputs):
        if isinstance(outputs, tuple):
            pred, _ = outputs
            if not isinstance(pred, Tensor):
                raise TypeError("First element of HRNetConvGRU output must be a Tensor.")
            return pred
        raise TypeError("Unsupported model output type for HRNetConvGRU.")

    return prepare_frames, extract_heatmaps


def _build_hrnet_backbone(model_cfg: DictConfig | dict[str, Any]) -> tuple[HRNet, int]:
    """Build an HRNet backbone from model config.

    The model config is expected to contain either:

    - ``backbone: "hrnet"`` and a corresponding ``hrnet: {...}`` section,
      or
    - an old-style ``backbone: {name: hrnet, ...}`` dict (kept for
      compatibility, but not used in new configs).
    """
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)

    backbone_selector = model_cfg.get("backbone", "hrnet")

    # New style: backbone is a string key selecting a sub-config.
    if isinstance(backbone_selector, str):
        if backbone_selector not in model_cfg:
            raise KeyError(
                f"Model config missing sub-config for backbone '{backbone_selector}'."
            )
        backbone_cfg = model_cfg[backbone_selector]
    else:
        # Fallback for old-style dict backbone configs.
        backbone_cfg = (
            OmegaConf.create(backbone_selector)
            if isinstance(backbone_selector, dict)
            else backbone_selector
        )
        backbone_dict = OmegaConf.to_container(backbone_cfg, resolve=True)
        backbone_cfg = OmegaConf.create(backbone_dict)

    hrnet = HRNet(backbone_cfg)
    feature_channels = hrnet.final_layers[0].in_channels
    return hrnet, feature_channels


def _build_hrcnet_backbone(model_cfg: DictConfig | dict[str, Any]) -> tuple[HRCNet, int]:
    """Build an HRCNet backbone from model config.

    The model config is expected to contain either:

    - ``backbone: "hrcnet"`` and a corresponding ``hrcnet: {...}`` section,
      or
    - an old-style ``backbone: {name: hrcnet, ...}`` dict (kept for
      compatibility, but not used in new configs).
    """
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)

    backbone_selector = model_cfg.get("backbone", "hrcnet")

    if isinstance(backbone_selector, str):
        if backbone_selector not in model_cfg:
            raise KeyError(
                f"Model config missing sub-config for backbone '{backbone_selector}'."
            )
        backbone_cfg = model_cfg[backbone_selector]
    else:
        backbone_cfg = backbone_selector

    if isinstance(backbone_cfg, dict):
        backbone_cfg = OmegaConf.create(backbone_cfg)

    backbone_dict = OmegaConf.to_container(backbone_cfg, resolve=True)

    frames_in = int(backbone_dict.get("frames_in", 1))
    frames_out = int(backbone_dict.get("frames_out", 1))

    in_channels = 3 * frames_in
    out_channels = frames_out
    hrcnet = HRCNet(
        in_channels=in_channels,
        out_channels=out_channels,
        high_channels=backbone_dict.get("high_channels", 64),
        low_channels=backbone_dict.get("low_channels", 64),
        num_stages=backbone_dict.get("num_stages", 3),
        high_block=backbone_dict.get("high_block", "BASIC"),
        low_block=backbone_dict.get("low_block", "BASIC"),
        num_high_blocks=backbone_dict.get("num_high_blocks", 2),
        num_low_blocks=backbone_dict.get("num_low_blocks", 1),
        upsample_mode=backbone_dict.get("upsample_mode", "nearest"),
        downsample_kwargs=backbone_dict.get("downsample_kwargs", {}),
        transformer_kwargs=backbone_dict.get("transformer_kwargs", {}),
    )
    feature_channels = hrcnet.high_channels
    return hrcnet, feature_channels


def _build_temporal_conv_gru(cfg: DictConfig | dict[str, Any]):
    """Build TemporalConvGRUModel with a configured backbone injected."""
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)

    model_cfg = cfg
    if "model" in cfg:
        model_cfg = cfg["model"]

    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)

    backbone_selector = model_cfg.get("backbone", "hrnet")

    # New style: backbone is a string key selecting a sub-config ("hrnet" /
    # "hrcnet"). Old style dict with "name" field is still accepted.
    if isinstance(backbone_selector, str):
        backbone_name = str(backbone_selector).lower()
    else:
        backbone_cfg = (
            OmegaConf.create(backbone_selector)
            if isinstance(backbone_selector, dict)
            else backbone_selector
        )
        backbone_dict = OmegaConf.to_container(backbone_cfg, resolve=True)
        backbone_name = str(backbone_dict.get("name", "hrnet")).lower()

    if backbone_name == "hrnet":
        backbone, feature_channels = _build_hrnet_backbone(model_cfg)
    elif backbone_name == "hrcnet":
        backbone, feature_channels = _build_hrcnet_backbone(model_cfg)
    else:
        raise ValueError(f"Unsupported backbone name for HRNetConvGRU: {backbone_name}")

    frames_in = int(model_cfg.get("frames_in", 1))
    frames_out = int(model_cfg.get("frames_out", frames_in))
    stack_channels = bool(model_cfg.get("stack_channels", False))

    hidden_cfg = model_cfg.get("gru_hidden_channels", feature_channels)
    if isinstance(hidden_cfg, int):
        hidden_dims = [int(hidden_cfg)]
    else:
        hidden_dims = [int(h) for h in hidden_cfg]

    kernel_size = int(model_cfg.get("gru_kernel_size", 3))

    model = TemporalConvGRUModel(
        backbone=backbone,
        feature_channels=feature_channels,
        frames_in=frames_in,
        frames_out=frames_out,
        stack_channels=stack_channels,
        gru_hidden_channels=hidden_dims,
        gru_kernel_size=kernel_size,
    )

    return model, _temporal_conv_gru_handlers()


__factory: dict[str, Any] = {
    "hrnet": lambda cfg: (HRNet(cfg), _hrnet_handlers()),
    "hrcnet": lambda cfg: (
        HRCNet(
            in_channels=3 * cfg.get("frames_in", 1),
            out_channels=cfg.get("frames_out", 1),
            high_channels=cfg.get("high_channels", 64),
            low_channels=cfg.get("low_channels", 64),
            num_stages=cfg.get("num_stages", 3),
            high_block=cfg.get("high_block", "BASIC"),
            low_block=cfg.get("low_block", "BASIC"),
            num_high_blocks=cfg.get("num_high_blocks", 2),
            num_low_blocks=cfg.get("num_low_blocks", 1),
            upsample_mode=cfg.get("upsample_mode", "nearest"),
            downsample_kwargs=cfg.get("downsample_kwargs", {}),
            transformer_kwargs=cfg.get("transformer_kwargs", {}),
        ),
        _hrnet_handlers(),
    ),
    "temporal_conv_gru": _build_temporal_conv_gru,
}


def build_model(cfg: DictConfig | dict[str, Any]):
    """Build a model instance and its IO handlers from config."""
    model_cfg = cfg["model"] if "model" in cfg else cfg
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)
    model_name = model_cfg.get("name")
    if model_name not in __factory:
        raise KeyError(f"invalid model: {model_name}")

    if model_name in ("hrnet", "hrcnet", "temporal_conv_gru"):
        return __factory[model_name](model_cfg)

    raise KeyError(f"Unsupported model: {model_name}")


__all__ = [
    "ClipSegmenter",
    "RuleBasedClipSegmenter",
    "TrajectoryCompleter",
    "PhysicsInterpolator",
    "BiLSTMCompleter",
    "HybridCompleter",
    "CompletionResult",
    "create_completer",
    "build_model",
    "TemporalConvGRUModel",
]
