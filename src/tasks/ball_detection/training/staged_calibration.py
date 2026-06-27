"""OOM-probe calibration of the per-T physical batch size B(T) (issue #579).

For each clip length ``T`` we grow the physical batch ``B`` until a forward +
backward pass no longer fits in GPU memory, then back off by a safety margin.
``B*T`` ends up roughly constant. The largest allocation per step is bounded by
``token_budget`` frames so the probe never pushes the 16 GB WSL2 box into the
system-memory fallback that freezes the host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from src.tasks.ball_detection.models import build_ball_detection_model, to_model_input

if TYPE_CHECKING:
    from collections.abc import Sequence


def _image_hw(config: Any) -> tuple[int, int]:
    model_cfg = config.get("model", {}) or {}
    data_cfg = config.get("data", {}) or {}
    size = model_cfg.get("image_size") or data_cfg.get("image_size") or [288, 512]
    return int(size[0]), int(size[1])


def _fits(model: Any, model_cfg: Any, batch: int, t: int, hw: tuple[int, int], device: torch.device) -> bool:
    """Whether one fwd+bwd of ``(batch, t, 3, H, W)`` fits without OOM."""
    height, width = hw
    try:
        frames = torch.randn(batch, t, 3, height, width, device=device, requires_grad=False)
        logits = model(to_model_input(frames, model_cfg))
        loss = logits.float().pow(2).mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
        del frames, logits, loss
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return False


def probe_batch_size_by_t(
    config: Any,
    t_values: Sequence[int],
    *,
    device: torch.device,
    token_budget: int = 24,
    safety: float = 0.9,
) -> dict[int, int]:
    """Return the calibrated ``{T: B(T)}`` map.

    ``token_budget`` caps ``B*T`` (largest probed allocation). ``safety`` scales
    the largest fitting batch down to leave headroom for optimizer state and
    fragmentation.
    """
    model_cfg = config.get("model", {}) or {}
    model = build_ball_detection_model(config).to(device)
    model.train()
    hw = _image_hw(config)

    result: dict[int, int] = {}
    for raw_t in t_values:
        t = int(raw_t)
        cap = max(1, token_budget // t)
        best = 0
        for batch in range(1, cap + 1):
            if _fits(model, model_cfg, batch, t, hw, device):
                best = batch
            else:
                break
        if best == 0:
            best = 1  # always allow at least one sample; may still OOM at runtime
            result[t] = 1
        else:
            result[t] = max(1, int(best * safety))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


__all__ = ["probe_batch_size_by_t"]
