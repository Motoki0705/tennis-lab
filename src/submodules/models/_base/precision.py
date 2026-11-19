"""Autocast dtypes for DINO and ViTPose inference."""

from __future__ import annotations

import torch

from src.submodules.configuration import InferencePrecision


def autocast_dtype(precision: InferencePrecision, device: torch.device) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "bf16":
        if device.type == "cpu":
            return torch.bfloat16
        if device.type == "cuda":
            with torch.cuda.device(device):
                if torch.cuda.is_bf16_supported(including_emulation=False):
                    return torch.bfloat16
        raise ValueError(f"bf16 autocast requires CPU or native CUDA bfloat16 support, got {device}")
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError(f"fp16 autocast requires CUDA, got {device}")
        return torch.float16
    raise ValueError(f"Unsupported inference precision: {precision!r}")
