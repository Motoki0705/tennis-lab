"""Models for WASB tennis dataset generation."""

from __future__ import annotations

from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from .clip_segmenter import ClipSegmenter, RuleBasedClipSegmenter
from .dinov3_heatmap import DinoV3FPNHeatmap
from .hrnet import HRNet
from .hrcnet import HRCNet
from .temporal_conv_gru import TemporalConvGRUModel
from .trajectory_completer import (
    BiLSTMCompleter,
    CompletionResult,
    HybridCompleter,
    IterativeRefinementCompleter,
    PhysicsInterpolator,
    TrajectoryCompleter,
    TransformerCompleter,
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


def _dinov3_heatmap_handlers() -> tuple[Callable, Callable]:
    def prepare_frames(frames):
        if getattr(frames, "dim", None) is None:
            raise ValueError("DinoV3FPNHeatmap expects a tensor input for frames.")
        if frames.dim() == 5:
            # [B, T, C, H, W] -> use the last frame in the sequence.
            return frames[:, -1]
        if frames.dim() == 4:
            return frames
        raise ValueError(
            "DinoV3FPNHeatmap expects frames with shape [B, C, H, W] "
            "or [B, T, C, H, W], got "
            f"{getattr(frames, 'shape', None)}"
        )

    def extract_heatmaps(outputs):
        # DinoV3FPNHeatmap already returns dense heatmaps as a Tensor.
        if isinstance(outputs, Tensor):
            return outputs
        raise TypeError("Unsupported model output type for DinoV3FPNHeatmap.")

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

    # backbone must be a string key selecting a sub-config ("hrnet").
    if not isinstance(backbone_selector, str):
        raise TypeError("backbone must be a string selector (e.g. 'hrnet').")
    if backbone_selector not in model_cfg:
        raise KeyError(
            f"Model config missing sub-config for backbone '{backbone_selector}'."
        )
    backbone_cfg = model_cfg[backbone_selector]

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

    if not isinstance(backbone_selector, str):
        raise TypeError("backbone must be a string selector (e.g. 'hrcnet').")
    if backbone_selector not in model_cfg:
        raise KeyError(
            f"Model config missing sub-config for backbone '{backbone_selector}'."
        )
    backbone_cfg = model_cfg[backbone_selector]

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

    # backbone must be a string key selecting a sub-config ("hrnet" / "hrcnet").
    if not isinstance(backbone_selector, str):
        raise TypeError("backbone must be a string selector (e.g. 'hrnet' or 'hrcnet').")
    backbone_name = str(backbone_selector).lower()

    if backbone_name == "hrnet":
        backbone, feature_channels = _build_hrnet_backbone(model_cfg)
    elif backbone_name == "hrcnet":
        backbone, feature_channels = _build_hrcnet_backbone(model_cfg)
    else:
        raise ValueError(f"Unsupported backbone name for TemporalConvGRUModel: {backbone_name}")

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
    "dinov3_heatmap": lambda cfg: (DinoV3FPNHeatmap(cfg), _dinov3_heatmap_handlers()),
}


def build_model(cfg: DictConfig | dict[str, Any]):
    """Build a model instance and its IO handlers from config."""
    model_cfg = cfg["model"] if "model" in cfg else cfg
    if isinstance(model_cfg, dict):
        model_cfg = OmegaConf.create(model_cfg)
    model_name = model_cfg.get("name")
    if model_name not in __factory:
        raise KeyError(f"invalid model: {model_name}")

    if model_name in ("hrnet", "hrcnet", "temporal_conv_gru", "dinov3_heatmap"):
        return __factory[model_name](model_cfg)

    raise KeyError(f"Unsupported model: {model_name}")


__all__ = [
    "ClipSegmenter",
    "RuleBasedClipSegmenter",
    "TrajectoryCompleter",
    "PhysicsInterpolator",
    "BiLSTMCompleter",
    "HybridCompleter",
    "IterativeRefinementCompleter",
    "TransformerCompleter",
    "CompletionResult",
    "create_completer",
    "build_model",
    "TemporalConvGRUModel",
    "DinoV3FPNHeatmap",
]
