"""Shared DINOv3 patch-token backbone loading and trainability helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch
from torch import nn

from src.utils.paths import resolve_project_path

DEFAULT_DINOV3_REPOSITORY = Path("third_party/dinov3")
DEFAULT_DINOV3_CHECKPOINT = Path(
    "third_party/dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
)
DINOv3TrainMode = Literal["frozen", "last_n_blocks", "full"]


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

    def forward_features(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return normalized patch tokens with shape ``(B, N, C)``."""
        outputs = cast(Any, self.module).forward_features(inputs)
        if not isinstance(outputs, Mapping):
            raise TypeError("DINOv3 forward_features must return a mapping.")
        patch_tokens = outputs.get("x_norm_patchtokens")
        if not isinstance(patch_tokens, torch.Tensor):
            raise TypeError(
                "DINOv3 forward_features did not return tensor patch tokens."
            )
        if patch_tokens.ndim != 3:
            raise ValueError(
                "DINOv3 patch tokens must have shape (B, N, C), "
                f"got {tuple(patch_tokens.shape)}."
            )
        if patch_tokens.shape[-1] != self.embed_dim:
            raise ValueError(
                "DINOv3 patch-token width does not match embed_dim: "
                f"{patch_tokens.shape[-1]} != {self.embed_dim}."
            )
        return {"x_norm_patchtokens": patch_tokens}

    def transformer_blocks(self) -> tuple[nn.Module, ...]:
        """Return the backbone transformer blocks used by ``last_n_blocks``."""
        blocks = getattr(self.module, "blocks", None)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            raise TypeError(
                "DINOv3 backbone must expose blocks as ModuleList or Sequential "
                "when train_mode='last_n_blocks'."
            )
        return tuple(blocks)


def load_dinov3_backbone(
    *,
    repository_path: str | Path = DEFAULT_DINOV3_REPOSITORY,
    checkpoint_path: str | Path = DEFAULT_DINOV3_CHECKPOINT,
    backbone_name: str = "dinov3_vitb16",
    strict: bool = True,
) -> DINOv3BackboneAdapter:
    """Load a DINOv3 backbone from the vendored repository and local weights."""
    repository = resolve_project_path(repository_path)
    checkpoint = resolve_project_path(checkpoint_path)
    if not repository.is_dir():
        raise FileNotFoundError(f"DINOv3 repository not found: {repository}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")

    backbone: nn.Module = torch.hub.load(
        str(repository),
        backbone_name,
        source="local",
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
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


def configure_dinov3_trainability(
    backbone: DINOv3BackboneAdapter,
    *,
    train_mode: DINOv3TrainMode,
    last_n_blocks: int = 0,
) -> None:
    """Configure ``frozen``, ``last_n_blocks``, or ``full`` backbone training."""
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
    "DEFAULT_DINOV3_CHECKPOINT",
    "DEFAULT_DINOV3_REPOSITORY",
    "DINOv3BackboneAdapter",
    "DINOv3TrainMode",
    "configure_dinov3_trainability",
    "load_dinov3_backbone",
]
