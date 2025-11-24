# coat_util.py
"""
Utilities for adapting a PyTorch Lightning checkpoint to a pure DinoVitHeatmap model.

What you get
------------
- load_lightning_state_dict(ckpt_path): read a .ckpt and return 'state_dict'.
- strip_prefix(state_dict, prefix="model."): remove 'model.' from keys saved by a LightningModule.
- align_and_load(nn_module, state_dict, strict=False): load only intersecting keys into your module
  and print a concise report of loaded / missing / unexpected params.

Typical usage
-------------
from coat_util import load_lightning_state_dict, strip_prefix, align_and_load
raw_sd = load_lightning_state_dict("path/to/your.ckpt")
adapted_sd = strip_prefix(raw_sd, prefix="model.")
align_and_load(model, adapted_sd, strict=False)
"""

from __future__ import annotations

from collections import OrderedDict

import torch


def load_lightning_state_dict(ckpt_path: str) -> OrderedDict[str, torch.Tensor]:
    """Load the 'state_dict' from a PyTorch Lightning .ckpt file (CPU map)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        # Fallback: if it's a raw state dict (non-Lightning)
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return OrderedDict(ckpt)
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")
    return OrderedDict(ckpt["state_dict"])


def strip_prefix(state_dict: OrderedDict[str, torch.Tensor], prefix: str = "model.") -> OrderedDict[str, torch.Tensor]:
    """Remove a leading prefix (e.g., 'model.') from keys."""
    out = OrderedDict()
    for k, v in state_dict.items():
        out[k[len(prefix) :]] = v if k.startswith(prefix) else v
    return out


def _partition_keys(
    target_sd: dict[str, torch.Tensor],
    src_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set, set]:
    """Return (filtered_to_intersection, missing_in_src, unexpected_in_src)."""
    tgt = set(target_sd.keys())
    src = set(src_sd.keys())
    inter = tgt & src
    filtered = {k: src_sd[k] for k in inter}
    return filtered, (tgt - src), (src - tgt)


def align_and_load(
    nn_module,
    src_state_dict: dict[str, torch.Tensor],
    strict: bool = False,
) -> None:
    """
    Load `src_state_dict` into `nn_module` with a readable report.
    Filters to the intersection of keys by default (strict=False recommended).
    """
    tgt_sd = nn_module.state_dict()
    filtered, missing, unexpected = _partition_keys(tgt_sd, src_state_dict)

    print(
        "[coat_util] Loading weights:\n"
        f"  target params:   {len(tgt_sd)}\n"
        f"  provided params: {len(src_state_dict)}\n"
        f"  matched load:    {len(filtered)}\n"
        f"  missing in src:  {len(missing)}\n"
        f"  unexpected src:  {len(unexpected)}"
    )

    result = nn_module.load_state_dict(filtered, strict=strict)
    if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
        print("[coat_util] torch.load_state_dict report:")
        print(f"  missing_keys:    {len(result.missing_keys)}")
        print(f"  unexpected_keys: {len(result.unexpected_keys)}")
