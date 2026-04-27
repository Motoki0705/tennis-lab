"""Reusable DINO backbone loading helpers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from checkpoints.DINO.scripts.load_dino_backbone import load_backbone_body_state

_DINO_RESNET_BACKBONE_CHANNELS = {
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
}
_DINO_SWIN_BACKBONE_CHANNELS = {
    "swin_T_224_1k": (96, 192, 384, 768),
    "swin_B_224_22k": (128, 256, 512, 1024),
    "swin_B_384_22k": (128, 256, 512, 1024),
    "swin_L_224_22k": (192, 384, 768, 1536),
    "swin_L_384_22k": (192, 384, 768, 1536),
}

_DEFAULT_RESNET_CHECKPOINT = Path("checkpoints/DINO/backbone_body_state.pth")
_DEFAULT_SWIN_CHECKPOINT = Path(
    "checkpoints/DINO/swin_backbone_state_checkpoint0027_5scale.pth"
)


@dataclass(frozen=True)
class DINOBackboneMetadata:
    """Metadata needed to construct downstream heads for a DINO backbone."""

    name: str
    family: str
    feature_channels: tuple[int, int, int, int]
    default_checkpoint_path: Path


@dataclass(frozen=True)
class LoadedDINOBackbone:
    """Loaded DINO backbone plus metadata about the resolved artifact."""

    module: nn.Module
    load_result: nn.modules.module._IncompatibleKeys
    metadata: DINOBackboneMetadata
    checkpoint_path: Path


def get_dino_backbone_metadata(backbone_name: str) -> DINOBackboneMetadata:
    """Return family and feature metadata for a supported DINO backbone."""

    if backbone_name in _DINO_RESNET_BACKBONE_CHANNELS:
        return DINOBackboneMetadata(
            name=backbone_name,
            family="resnet",
            feature_channels=_DINO_RESNET_BACKBONE_CHANNELS[backbone_name],
            default_checkpoint_path=_DEFAULT_RESNET_CHECKPOINT,
        )
    if backbone_name in _DINO_SWIN_BACKBONE_CHANNELS:
        return DINOBackboneMetadata(
            name=backbone_name,
            family="swin",
            feature_channels=_DINO_SWIN_BACKBONE_CHANNELS[backbone_name],
            default_checkpoint_path=_DEFAULT_SWIN_CHECKPOINT,
        )
    raise ValueError(f"Unsupported DINO backbone: {backbone_name}")


def resolve_dino_backbone_checkpoint_path(
    backbone_name: str,
    checkpoint_path: str | Path | None,
) -> Path:
    """Resolve the checkpoint path for a supported DINO backbone."""

    if checkpoint_path is not None and str(checkpoint_path).strip():
        return Path(checkpoint_path)
    return get_dino_backbone_metadata(backbone_name).default_checkpoint_path


def load_dino_backbone(
    *,
    backbone_name: str,
    checkpoint_path: str | Path | None = None,
    dilation: bool = False,
    return_interm_indices: tuple[int, ...] = (0, 1, 2, 3),
    pretrain_img_size: int | None = None,
    use_checkpoint: bool = False,
    strict: bool = True,
) -> LoadedDINOBackbone:
    """Load a DINO backbone in a task-agnostic form."""

    metadata = get_dino_backbone_metadata(backbone_name)
    resolved_checkpoint_path = resolve_dino_backbone_checkpoint_path(
        backbone_name,
        checkpoint_path,
    )
    resolved_return_interm_indices = list(return_interm_indices)

    if metadata.family == "resnet":
        module, load_result = load_backbone_body_state(
            checkpoint_path=resolved_checkpoint_path,
            backbone=backbone_name,
            dilation=dilation,
            return_interm_indices=resolved_return_interm_indices,
            strict=strict,
        )
        return LoadedDINOBackbone(
            module=module,
            load_result=load_result,
            metadata=metadata,
            checkpoint_path=resolved_checkpoint_path,
        )

    load_swin_backbone_state = _load_swin_backbone_loader()
    resolved_pretrain_img_size = (
        int(pretrain_img_size)
        if pretrain_img_size is not None
        else _infer_swin_pretrain_img_size(backbone_name)
    )
    module, load_result = load_swin_backbone_state(
        resolved_checkpoint_path,
        backbone=backbone_name,
        pretrain_img_size=resolved_pretrain_img_size,
        dilation=dilation,
        return_interm_indices=resolved_return_interm_indices,
        use_checkpoint=use_checkpoint,
        strict=strict,
    )
    return LoadedDINOBackbone(
        module=module,
        load_result=load_result,
        metadata=metadata,
        checkpoint_path=resolved_checkpoint_path,
    )


def _infer_swin_pretrain_img_size(backbone_name: str) -> int:
    return int(backbone_name.split("_")[-2])


def _load_swin_backbone_loader():
    try:
        module = importlib.import_module("checkpoints.DINO.scripts.load_dino_swin_backbone")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Swin DINO backbone loader. "
            "Check that the DINO Swin builder sources are available in this workspace."
        ) from exc
    return module.load_swin_backbone_state


__all__ = [
    "DINOBackboneMetadata",
    "LoadedDINOBackbone",
    "get_dino_backbone_metadata",
    "load_dino_backbone",
    "resolve_dino_backbone_checkpoint_path",
]