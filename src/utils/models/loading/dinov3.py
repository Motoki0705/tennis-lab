"""Shared DINOv3 patch-token backbone loading and trainability helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from src.utils.models.lora import LoRAConfig, apply_lora

DINOv3TrainMode = Literal["frozen", "last_n_blocks", "full"]
# Attention (qkv/proj) and MLP (fc1/fc2) projections inside each DINOv3 block.
DEFAULT_DINOV3_LORA_TARGET_MODULES: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")


class DINOv3BackboneAdapter(nn.Module):
    """Expose validated patch tokens from the dynamic DINOv3 hub API."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        dynamic_module = cast(Any, module)
        embed_dim = dynamic_module.embed_dim
        patch_size = dynamic_module.patch_size
        if not isinstance(embed_dim, int):
            raise TypeError("DINOv3 backbone embed_dim must be an integer.")
        if isinstance(patch_size, tuple):
            if len(patch_size) != 2 or len(set(patch_size)) != 1:
                raise ValueError("DINOv3 patch size must be square.")
            patch_size = patch_size[0]
        if not isinstance(patch_size, int):
            raise TypeError("DINOv3 backbone patch_size must be an integer.")
        if patch_size <= 0:
            raise ValueError("DINOv3 backbone patch_size must be positive.")

        self.module = module
        self.embed_dim = embed_dim
        self.patch_size = patch_size

    def transformer_blocks(self) -> tuple[nn.Module, ...]:
        """Return the backbone transformer blocks used by ``last_n_blocks``."""
        blocks = getattr(self.module, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise TypeError(
                "DINOv3 backbone must expose blocks as ModuleList or Sequential "
                "when train_mode='last_n_blocks'."
            )
        return tuple(blocks)


def require_dinov3_patch_tokens(
    outputs: object,
    *,
    expected_batch_size: int,
    expected_embed_dim: int,
    expected_num_tokens: int | None = None,
    context: str = "DINOv3 forward_features",
) -> torch.Tensor:
    """Decode and validate dynamic DINO output at an explicit model-I/O boundary."""
    if not isinstance(outputs, Mapping):
        raise TypeError(f"{context} must return a mapping.")
    try:
        patch_tokens = outputs["x_norm_patchtokens"]
    except KeyError as error:
        raise KeyError(f"{context} is missing required x_norm_patchtokens.") from error
    if not isinstance(patch_tokens, torch.Tensor):
        raise TypeError(f"{context} x_norm_patchtokens must be a tensor.")
    if patch_tokens.ndim != 3:
        raise ValueError(
            f"{context} patch tokens must have shape (B, N, C), "
            f"got {tuple(patch_tokens.shape)}."
        )
    if patch_tokens.shape[0] != expected_batch_size:
        raise ValueError(
            f"{context} batch size {patch_tokens.shape[0]} does not match "
            f"expected {expected_batch_size}."
        )
    if expected_num_tokens is not None and patch_tokens.shape[1] != expected_num_tokens:
        raise ValueError(
            f"{context} token count {patch_tokens.shape[1]} does not match "
            f"expected {expected_num_tokens}."
        )
    if patch_tokens.shape[2] != expected_embed_dim:
        raise ValueError(
            f"{context} embedding width {patch_tokens.shape[2]} does not match "
            f"expected {expected_embed_dim}."
        )
    return patch_tokens


def load_dinov3_backbone(
    *,
    repository_path: Path,
    checkpoint_path: Path,
    backbone_name: str,
    strict: bool,
) -> DINOv3BackboneAdapter:
    """Load a DINOv3 backbone from explicit, already-resolved runtime paths."""
    if not isinstance(repository_path, Path) or not repository_path.is_absolute():
        raise ValueError("repository_path must be an absolute pathlib.Path.")
    if not isinstance(checkpoint_path, Path) or not checkpoint_path.is_absolute():
        raise ValueError("checkpoint_path must be an absolute pathlib.Path.")
    if not repository_path.is_dir():
        raise FileNotFoundError(f"DINOv3 repository not found: {repository_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint_path}")

    backbone: nn.Module = torch.hub.load(
        str(repository_path),
        backbone_name,
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(
            "DINOv3 checkpoint must contain a state-dict mapping, "
            f"got {type(state).__name__}."
        )

    load_result = backbone.load_state_dict(state, strict=strict)
    if strict and (load_result.missing_keys or load_result.unexpected_keys):
        raise RuntimeError(
            "Unexpected DINOv3 checkpoint load result: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}."
        )
    return DINOv3BackboneAdapter(backbone)


def apply_dinov3_lora(
    backbone: DINOv3BackboneAdapter,
    lora: LoRAConfig,
) -> list[str]:
    """Freeze the backbone and attach trainable LoRA adapters to its blocks.

    The base backbone weights are frozen; only the injected LoRA factors remain
    trainable. Returns the qualified names of the wrapped linear layers.
    """
    if not lora.enabled:
        raise ValueError("apply_dinov3_lora requires an enabled LoRAConfig.")
    backbone.requires_grad_(False)
    wrapped: list[str] = apply_lora(
        backbone.module,
        rank=lora.rank,
        alpha=lora.alpha,
        dropout=lora.dropout,
        target_modules=lora.target_modules,
    )
    return wrapped


def configure_dinov3_trainability(
    backbone: DINOv3BackboneAdapter,
    *,
    train_mode: DINOv3TrainMode,
    last_n_blocks: int,
    lora: LoRAConfig | None,
) -> None:
    """Configure ``frozen``, ``last_n_blocks``, ``full``, or LoRA training.

    When ``lora`` is provided and enabled, the base backbone is frozen and LoRA
    adapters are injected regardless of ``train_mode`` (only the adapters train).
    """
    if lora is not None and lora.enabled:
        apply_dinov3_lora(backbone, lora)
        return

    if train_mode not in {"frozen", "last_n_blocks", "full"}:
        raise ValueError(
            "DINOv3 train_mode must be one of "
            "['frozen', 'last_n_blocks', 'full'], "
            f"got {train_mode!r}."
        )

    backbone.requires_grad_(train_mode == "full")
    if train_mode != "last_n_blocks":
        if last_n_blocks < 0:
            raise ValueError("last_n_blocks must be non-negative.")
        return

    blocks = backbone.transformer_blocks()
    if last_n_blocks <= 0:
        raise ValueError(
            "last_n_blocks must be positive when train_mode='last_n_blocks'."
        )
    if last_n_blocks > len(blocks):
        raise ValueError(
            f"last_n_blocks={last_n_blocks} exceeds backbone depth={len(blocks)}."
        )

    backbone.requires_grad_(False)
    for block in blocks[-last_n_blocks:]:
        block.requires_grad_(True)
    for name in ("norm", "norm_cls"):
        final_norm = getattr(backbone.module, name, None)
        if isinstance(final_norm, nn.Module):
            final_norm.requires_grad_(True)


__all__ = [
    "DEFAULT_DINOV3_LORA_TARGET_MODULES",
    "DINOv3BackboneAdapter",
    "DINOv3TrainMode",
    "apply_dinov3_lora",
    "configure_dinov3_trainability",
    "load_dinov3_backbone",
    "require_dinov3_patch_tokens",
]
