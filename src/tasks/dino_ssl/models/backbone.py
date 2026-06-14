"""Load the vendored DINOv3 ViT backbone and attach LoRA adapters.

The backbone weights are loaded from ``third_party/dinov3`` and remain subject
to the DINOv3 License Agreement.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.tasks.dino_ssl._dinov3 import (
    DEFAULT_CHECKPOINT,
    DINOV3_REPOSITORY,
    ensure_dinov3_importable,
)


def build_dinov3_vit(
    *,
    backbone_name: str = "dinov3_vitb16",
    checkpoint_path: str | Path | None = None,
    load_pretrained: bool = True,
) -> nn.Module:
    """Instantiate a DINOv3 ViT and optionally load pretrained weights."""
    ensure_dinov3_importable()
    backbone = torch.hub.load(
        str(DINOV3_REPOSITORY),
        backbone_name,
        source="local",
        pretrained=False,
    )
    if load_pretrained:
        checkpoint = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(
                f"DINOv3 checkpoint missing or empty: {checkpoint}. "
                "Download the pretrained weights before SSL fine-tuning."
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, Mapping) and "model" in state:
            state = state["model"]
        if not isinstance(state, Mapping):
            raise TypeError("DINOv3 checkpoint must contain a state-dict mapping.")
        result = backbone.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Unexpected DINOv3 checkpoint load result: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}."
            )
    return backbone


def apply_lora(backbone: nn.Module, lora_cfg: Any) -> nn.Module:
    """Wrap ``backbone`` with LoRA adapters; freeze all non-LoRA weights."""
    config = LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        target_modules=list(lora_cfg.target_modules),
        bias=str(lora_cfg.get("bias", "none")),
    )
    return get_peft_model(backbone, config)


def count_trainable_parameters(module: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


__all__ = [
    "build_dinov3_vit",
    "apply_lora",
    "count_trainable_parameters",
]
