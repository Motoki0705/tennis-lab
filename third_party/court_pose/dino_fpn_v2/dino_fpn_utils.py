"""Utility helpers for adapting Lightning checkpoints to the DINO FPN v2 court pose model."""

from __future__ import annotations

from collections import OrderedDict

import torch


def load_lightning_state_dict(ckpt_path: str) -> OrderedDict[str, torch.Tensor]:
    """Load a PyTorch Lightning checkpoint and return its ``state_dict`` on CPU."""

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" not in ckpt:
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return OrderedDict(ckpt)
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")
    return OrderedDict(ckpt["state_dict"])


def strip_prefix(state_dict: OrderedDict[str, torch.Tensor], prefix: str = "model.") -> OrderedDict[str, torch.Tensor]:
    """Remove a common prefix from state-dict keys while keeping others untouched."""

    out = OrderedDict()
    for key, value in state_dict.items():
        if prefix and key.startswith(prefix):
            out[key[len(prefix) :]] = value
        else:
            out[key] = value
    return out


def _partition_keys(
    target_sd: dict[str, torch.Tensor],
    src_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set[str], set[str]]:
    """Split state-dict keys into matching / missing / unexpected categories."""

    target_keys = set(target_sd.keys())
    src_keys = set(src_sd.keys())
    common = target_keys & src_keys
    filtered = {k: src_sd[k] for k in common}
    missing = target_keys - src_keys
    unexpected = src_keys - target_keys
    return filtered, missing, unexpected


def align_and_load(
    module: torch.nn.Module,
    src_state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> None:
    """Load a checkpoint into ``module`` while printing a concise load report."""

    target_state = module.state_dict()
    filtered, missing, unexpected = _partition_keys(target_state, src_state_dict)

    log = [
        "[dino_fpn_v2_utils] Loading weights",
        f"  target params:   {len(target_state)}",
        f"  provided params: {len(src_state_dict)}",
        f"  matched load:    {len(filtered)}",
        f"  missing in src:  {len(missing)}",
        f"  unexpected src:  {len(unexpected)}",
    ]
    print("\n".join(log))

    load_result = module.load_state_dict(filtered, strict=strict)
    if hasattr(load_result, "missing_keys") and hasattr(load_result, "unexpected_keys"):
        print("[dino_fpn_v2_utils] torch.load_state_dict report:")
        print(f"  missing_keys:   {len(load_result.missing_keys)}")
        print(f"  unexpected_keys:{len(load_result.unexpected_keys)}")


__all__ = ["align_and_load", "load_lightning_state_dict", "strip_prefix"]
