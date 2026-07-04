"""Vendored HMR2 (feature extractor for GVHMR)."""

import torch

from .hmr2 import HMR2


def load_hmr2(checkpoint_path):
    """Build HMR2.0a and load the released checkpoint (backbone + smpl_head)."""
    model = HMR2()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
    keys = [k for k in state_dict if k.split(".")[0] in ["backbone", "smpl_head"]]
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)

    return model
