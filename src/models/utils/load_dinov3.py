"""Utilities for loading DINOv3 models."""

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_dinov3(
    arch: str = "dinov3_vits16",
    weights_path: str
    | None = "third_party/dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    **hub_kwargs: Any,
) -> nn.Module:
    """Load a DINOv3 model from torch.hub.

    Args:
        arch (str): Model architecture (default: 'dinov3_vits16')
        weights_path (str | None): Path to pretrained weights (default: standard checkpoint path)
        **hub_kwargs (Any): Additional arguments for torch.hub.load()

    Returns:
        nn.Module: Loaded DINOv3 model

    Raises:
        FileNotFoundError: If weights file is specified but not found
        AttributeError: If model lacks required 'get_intermediate_layers' method
        RuntimeError: For other loading failures

    """
    hub_kwargs.update({"source": "local"})
    if weights_path is not None:
        resolved = Path(weights_path).expanduser()
        if not resolved.exists():
            raise FileNotFoundError(f"DINOv3 weights not found at {resolved}")
        hub_kwargs["weights"] = str(resolved)

    try:
        dinov3 = torch.hub.load("third_party/dinov3", arch, **hub_kwargs)
        if not hasattr(dinov3, "get_intermediate_layers"):
            raise AttributeError(
                f"{arch} missing required 'get_intermediate_layers' method"
            )
        return dinov3
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {arch} from third_party/dinov3: {exc}"
        ) from exc
