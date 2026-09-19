"""Deterministic input ablations for measuring RGB use and detector robustness."""

from __future__ import annotations

import torch

from src.tasks.slcs.data.augmentation import INPUT_CONDITIONS


def condition_inputs(
    batch: dict[str, torch.Tensor], mode: str
) -> dict[str, torch.Tensor]:
    """Change observations only; the held-out targets/masks remain identical."""
    if mode not in INPUT_CONDITIONS:
        raise ValueError(f"Unknown SLCS input condition {mode!r}")
    if mode == "full":
        return batch
    if mode == "detector_gap_no_rgb":
        return condition_inputs(condition_inputs(batch, "detector_gap"), "no_rgb")
    names = {
        "player_kp",
        "player_kp_vis",
        "player_valid",
        "ball_uv",
        "ball_vis",
        "court_kp",
        "court_vis",
        "dino_tokens",
        "dino_padding_mask",
    }
    out = {
        key: value.clone() if key in names else value for key, value in batch.items()
    }
    if mode == "no_rgb":
        out["dino_tokens"].zero_()
        out["dino_padding_mask"].fill_(True)
    elif mode == "rgb_only":
        if out["dino_padding_mask"].all(dim=1).any():
            raise ValueError("RGB-only evaluation requires RGB tokens for every sample")
        for name in names - {"dino_tokens", "dino_padding_mask"}:
            out[name].zero_()
    else:
        for sample in range(out["padding_mask"].shape[0]):
            length = int((~out["padding_mask"][sample]).sum())
            start, end = length // 3, max(length // 3 + 1, 2 * length // 3)
            for key in ("player_kp", "player_kp_vis", "player_valid"):
                out[key][sample, :, start:end] = 0
            for key in ("ball_uv", "ball_vis"):
                out[key][sample, start:end] = 0
    return out
